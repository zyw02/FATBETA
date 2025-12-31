import logging
import math
import operator
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from quan.func import SwithableBatchNorm
from util import AverageMeter
from util.utils import model_profiling, calibrate_batchnorm_state, accuracy, update_meter, set_global_seed
from util.qat import profile_layerwise_quantization_metric, freeze_layers, set_bit_width, auxiliary_quantized_loss, remove_hook_for_quantized_layers, set_forward_hook_for_quantized_layers
from util.mpq import sample_one_mixed_policy, sample_max_cands, sample_min_cands
from util.dist import master_only, logger_info

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def compute_overall_loss(outputs, teacher_outputs, targets, criterion, model, quantization_error_minimization=False, QE_loss_weight=.5, disable_smallest_regularization=True, configs=None):
    task_loss = loss_forward(outputs, teacher_outputs, targets, criterion)

    if quantization_error_minimization or disable_smallest_regularization:
        QE_loss, distribution_loss = auxiliary_quantized_loss(model, 
                                                           quantization_error_minimization=quantization_error_minimization, 
                                                           fairness_regularization=disable_smallest_regularization
                                                           )
    else:
        QE_loss, distribution_loss = 0, 0

    QE_loss *= QE_loss_weight

    adaptive_region_weight_decay = getattr(configs, 'adaptive_region_weight_decay', configs.weight_decay)
    distribution_loss *= (adaptive_region_weight_decay - configs.weight_decay)

    return task_loss + QE_loss + distribution_loss, QE_loss, distribution_loss


@master_only
def show_training_info(meters, target_bits, nr_random_sample, mode):
    iters = len(meters) if mode == 'training' else 1
    for i in range(iters):
            logger.info('==> %s Top1: %.3f    Top5: %.3f    Loss: %.3f', meters[i]['name'],
                        meters[i]['top1'].avg, meters[i]['top5'].avg, meters[i]['loss'].avg)
    for i in range(iters):
            # print(f'==> %s Top1: %.3f    Top5: %.3f    Loss: %.3f', meters[i]['name'],
            #             meters[i]['top1'].avg, meters[i]['top5'].avg, meters[i]['loss'].avg)
            print(f'==> {meters[i]["name"]} '
                  f'Top1: {meters[i]["top1"].avg:.3f}    '
                  f'Top5: {meters[i]["top5"].avg:.3f}    '
                  f'Loss: {meters[i]["loss"].avg:.3f}')


@master_only
def update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, nr_random_sample, optimizer, optimizer_q, mode='training'):
    iters = len(meters) if mode == 'training' else 1
    for m in monitors:
        for i in range(iters):
            # if meters[i]['top1'].avg == 0.:
            #     continue
            p = meters[i]['name'] + ' '
            m.update(epoch, batch_idx + 1, steps_per_epoch, p + 'Training', {
                'Loss': meters[i]['loss'],
                'QE Loss': meters[i]['QE_loss'], 
                'Distribution Loss': meters[i]['dist_loss'], 
                'IDM Loss': meters[i]['IDM_loss'], 
                'Top1': meters[i]['top1'],
                'Top5': meters[i]['top5'],
                'LR': optimizer.param_groups[0]['lr'],
                'QLR': optimizer_q.param_groups[0]['lr'] if optimizer_q is not None else 0
            })
        
        if mode == 'finetuning':
            continue

def loss_forward(outputs, teacher_outputs, targets, criterion):
    loss = criterion(outputs, targets)

    if teacher_outputs is not None:
        loss = 1/2 * loss + 1/2 * F.kl_div(F.log_softmax(outputs, dim=-1), F.softmax(teacher_outputs, dim=-1), reduction='batchmean')
    
    return loss

def get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min):
    if mode == 'training':
        if not sample_current_max and not sample_current_min:
            num_fixed_sample = len(target_bits)
            fixed_name = [f'Bits [{target_bits[i]}]' for i in range(num_fixed_sample)]
            num_fixed_sample = 0
        else:
            num_fixed_sample = sample_current_min + sample_current_max
            if num_fixed_sample == 2:
                fixed_name = ['Max', 'Min']
            else:
                fixed_name = ['Max'] if not sample_current_min else ['Min']
        meters = [{
            'name': fixed_name[i] if i < num_fixed_sample else f'Mixed {i - num_fixed_sample}', 
            'loss': AverageMeter(),
            'QE_loss': AverageMeter(),
            'dist_loss': AverageMeter(),
            'IDM_loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'batch_time': AverageMeter()
        } for i in range(num_fixed_sample + nr_random_sample)]
    else:
        meters = [{
            'name': 'Finetune',
            'loss': AverageMeter(),
            'QE_loss': AverageMeter(),
            'dist_loss': AverageMeter(),
            'IDM_loss': AverageMeter(),
            'top1': AverageMeter(),
            'top5': AverageMeter(),
            'batch_time': AverageMeter()
        }]

        num_fixed_sample = 1
    
    return meters, num_fixed_sample

def alexnet_train(train_loader, model, epochs,test_loader):
    import torch.optim as optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5, verbose=True)
    model.features[3].bits = (8, 8)
    model.features[6].bits = (8, 8)
    model.features[8].bits = (8, 8)
    model.features[10].bits = (8, 8)
    model = model.to(device)
    # 训练循环
    best_acc = 0.0
    from tqdm import tqdm
    for epoch in range(epochs):
        # 训练阶段（带批次进度条）
        model.train()
        train_loss = 0.0

        # 创建训练进度条（每个epoch独立）
        with tqdm(train_loader,
                  desc=f"Epoch {epoch + 1}/{epochs}",
                  unit="batch",
                  bar_format="{l_bar}{bar:20}{r_bar}") as pbar:

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 更新指标
                train_loss += loss.item()

                # 实时更新进度条右侧信息（当前batch的loss）
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # 计算平均训练损失（进度条关闭后）
        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # 计算验证损失
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算验证指标
        val_loss /= len(test_loader)
        val_acc = 100 * correct / total

        # 在进度条关闭后的同一行末尾显示完整指标（使用 ANSI 光标控制）
        print(
            f"\033[K"  # 清除当前行
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f" | Val Acc: {val_acc:.2f}%",
            end="\n"  # 换行准备下个epoch
        )
        time.sleep(1)
        if (epoch + 1) % 20 == 0 and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'D:/Program Files/pyprj/retraining-free-quantization/trained_models/alex_w8a8_noema.pt')
        # 或者直接更新进度条右侧信息（更紧凑）
        # tqdm.get_lock().acquire()  # 确保线程安全
        # tqdm._instances.clear()  # 清除旧实例避免多行显示
        # tqdm.get_lock().release()
        # tqdm.set_postfix(
        #     Train_Loss=f"{train_loss:.4f}",
        #     Val_Loss=f"{val_loss:.4f}",
        #     Val_Acc=f"{val_acc:.2f}%"
        # )




def train(train_loader, model, criterion, optimizer, epoch, monitors, configs, model_ema=None, nr_random_sample=2, mode='training', soft_criterion=None, teacher_model=None, optimizer_q=None, annealing_schedule=None, freezing_annealing_schedule=None, IDM_weight=0.01, scaler=None):
    assert mode in ['finetuning', 'training']

    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    if getattr(configs, 'sandwich_training', False):
        sample_current_max, sample_current_min = True, True
    else:
        sample_current_max, sample_current_min = False, False
    
    sample_current_max = True
    
    print("Bit-width candidates:", target_bits)
    
    meters, num_fixed_sample = get_meters(mode, target_bits, nr_random_sample, sample_current_max, sample_current_min)

    total_sample = len(train_loader.sampler)
    batch_size = configs.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    information_distortion_mitigation = getattr(configs, 'information_distortion_mitigation', False)
    if information_distortion_mitigation:
        assert sample_current_max

    # logger_info(logger, 'Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    # print('Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    num_updates = epoch * len(train_loader)
    seed = num_updates
    set_global_seed(seed + 1)
    model.train()
    if model_ema:
        model_ema.ema.train()

    T = 2 if epoch <= int(configs.epochs * 0.72) else 15

    ''' 2025年5月6日 10:26 去掉freeze_layers
    if configs.enable_dynamic_bit_training and \
         epoch > 5 and (epoch + 1) % T == 0:

        freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
        freezing_metric = profile_layerwise_quantization_metric(model=model)
        freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio, 
                      progressive=False, logger=logger, org_cands=configs.target_bits
                      )
        # logger_info(logger=logger, msg= f'Current freezing ratio: {freezing_ratio}')
        print(f'Current freezing ratio: {freezing_ratio}')
    '''
    '''2025年5月9日，09:12 加上freeze_layers'''
    '''2025年5月30日，09:22 注释freeze_layers，对照起见，因为BSAM量化网络没有用freezing'''
    if configs.enable_dynamic_bit_training and \
            epoch > 5 and (epoch + 1) % T == 0:
        freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
        freezing_metric = profile_layerwise_quantization_metric(model=model)
        freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio,
                      progressive=False, logger=logger, org_cands=configs.target_bits
                      )
        # logger_info(logger=logger, msg= f'Current freezing ratio: {freezing_ratio}')
        print(f'Current freezing ratio: {freezing_ratio}')

    if teacher_model is not None:
        teacher_model.eval()
        print("Training with KD...")
    
    total_subnets = num_fixed_sample + nr_random_sample
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 动态搜索体现在哪里
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        external_teacher_outputs = None
        if teacher_model is not None and soft_criterion is not None:
            with torch.no_grad():
                external_teacher_outputs = teacher_model(inputs)

        QE_loss_weight = annealing_schedule(num_updates) # We use a scheduler for the weights of QE loss according to QAT Oscillations Overcoming [ICML'22]. 

        if sample_current_max:
            start_time = time.time()
            # 这里选择权重位宽和激活值位宽
            sample_max_cands(model, configs)

            if information_distortion_mitigation:
                target_features = []
                hooks = set_forward_hook_for_quantized_layers(model, target_features, is_max=True)

            max_outputs = model(inputs)

            loss, QE_loss, dist_loss = compute_overall_loss(max_outputs, external_teacher_outputs, targets, criterion, model, quantization_error_minimization=False, 
                                                                configs=configs, disable_smallest_regularization=True)

            loss.backward()

            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

            teacher_outputs = max_outputs.clone().detach()
            
            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[0], loss, QE_loss, dist_loss, 0, 
                        acc1, acc5, inputs.size(0), time.time() - start_time)
                
        weight_conf_pool = []

        for iter_idx in range(nr_random_sample):
            start_time = time.time()

            w_conf, a_conf, min_w_index = sample_one_mixed_policy(model, configs)
            weight_conf_pool.append(w_conf)
            
            if information_distortion_mitigation:
                distorted_features = []
                hooks = set_forward_hook_for_quantized_layers(model, distorted_features, is_max=False)

            outputs = model(inputs)

            loss, QE_loss, dist_loss = compute_overall_loss(outputs, teacher_outputs, targets, criterion, model, quantization_error_minimization=epoch>40, 
                                                                QE_loss_weight=QE_loss_weight, disable_smallest_regularization=True, configs=configs)

            IDM_loss = 0
            if information_distortion_mitigation:
                remove_hook_for_quantized_layers(hooks)

                IDM_loss = sum([F.mse_loss(s, t).sum() if s is not None else 0 for s, t in zip(distorted_features, target_features)])
                loss += (IDM_loss * IDM_weight)
            
            loss.backward()
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[iter_idx+num_fixed_sample], loss, QE_loss, dist_loss, IDM_loss, 
                        acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)

        nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()
        if optimizer_q is not None:
            optimizer_q.step()

        num_updates += 1

        if model_ema is not None:
            model_ema.update(model)

        if (batch_idx + 1) % configs.log.print_freq == 0:
            update_monitors(monitors, meters, target_bits, epoch, batch_idx, steps_per_epoch, nr_random_sample, optimizer, optimizer_q, mode=mode)
            logger_info(logger, "="*115)

    # show_training_info(meters, target_bits, nr_random_sample, mode=mode)
    
    return meters[0]['top1'].avg, meters[0]['top5'].avg, meters[0]['loss'].avg


def alex_train_2(train_loader, test_loader, model, criterion, optimizer, epoch, monitors, configs, model_ema=None, nr_random_sample=2,
          mode='training', soft_criterion=None, teacher_model=None, optimizer_q=None, annealing_schedule=None,
          freezing_annealing_schedule=None, IDM_weight=0.01, scaler=None):
    assert mode in ['finetuning', 'training']

    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    if getattr(configs, 'sandwich_training', False):
        sample_current_max, sample_current_min = True, True
    else:
        sample_current_max, sample_current_min = False, False

    sample_current_max = True

    # print("Bit-width candidates:", target_bits,flush=True)


    total_sample = len(train_loader.sampler)
    batch_size = configs.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)


    # logger_info(logger, 'Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    # print('Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    num_updates = epoch * len(train_loader)
    seed = num_updates
    set_global_seed(seed + 1)
    model.train()
    total_acc1 = 0.0
    total_acc5 = 0.0
    train_loss = 0.0
    T = 5 if epoch <= int(configs.epochs * 0.75) else 15

    if configs.enable_dynamic_bit_training and \
            epoch > 30 and (epoch + 1) % T == 0:
        freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
        freezing_metric = profile_layerwise_quantization_metric(model=model)
        freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio,
                      progressive=False, logger=logger, org_cands=configs.target_bits
                      )
        # logger_info(logger=logger, msg= f'Current freezing ratio: {freezing_ratio}')
        print(f'Current freezing ratio: {freezing_ratio}',flush=True)
        time.sleep(1)


    from tqdm import tqdm
    pbar = tqdm(total=len(train_loader) + 1, desc=f"Epoch {epoch + 1}",
                postfix={"train_loss": "N/A", "val_loss": "N/A", "val_acc": "N/A"})

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 动态搜索体现在哪里
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()


        QE_loss_weight = annealing_schedule(
            num_updates)  # We use a scheduler for the weights of QE loss according to QAT Oscillations Overcoming [ICML'22].

        if sample_current_max:
            start_time = time.time()
            # 这里选择权重位宽和激活值位宽
            sample_max_cands(model, configs)


            max_outputs = model(inputs)

            loss, QE_loss, dist_loss = compute_overall_loss(max_outputs, None, targets, criterion,
                                                            model, quantization_error_minimization=False,
                                                            configs=configs, disable_smallest_regularization=True)

            loss.backward()



            acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
            total_acc1 += acc1
            total_acc5 += acc5
            train_loss += loss.item()
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "train_loss": f"{train_loss / (batch_idx + 1):.4f}",
                "val_loss": "N/A",
                "val_acc": "N/A"
            }, refresh=True)

        weight_conf_pool = []
        nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()
        if optimizer_q is not None:
            optimizer_q.step()

        num_updates += 1
        '''2025/5/10 取消model_ema_update for alexnet
        if model_ema is not None:
            model_ema.update(model)
        '''
    avg_train_loss = train_loss / len(train_loader)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_targets in test_loader:
            val_inputs = val_inputs.cuda()
            val_targets = val_targets.cuda()

            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()

            _, predicted = torch.max(val_outputs.data, 1)
            val_total += val_targets.size(0)
            val_correct += (predicted == val_targets).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100 * val_correct / val_total

    # ============== 最终指标显示 ==============
    pbar.set_postfix({
        "train_loss": f"{avg_train_loss:.4f}",
        "val_loss": f"{avg_val_loss:.4f}",
        "val_acc": f"{val_acc:.2f}%"
    }, refresh=True)
    pbar.update(1)
    pbar.close()
    time.sleep(1)

def train_one_epoch_no_freezing_no_sample(model, train_loader, test_loader, criterion, optimizer, epoch, configs,
                              lr_scheduler = None, optimizer_q = None, annealing_schedule=None,freezing_annealing_schedule=None):
    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    if getattr(configs, 'sandwich_training', False):
        sample_current_max, sample_current_min = True, True
    else:
        sample_current_max, sample_current_min = False, False

    sample_current_max = True

    # print("Bit-width candidates:", target_bits,flush=True)

    total_sample = len(train_loader.sampler)
    batch_size = configs.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    # logger_info(logger, 'Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    # print('Training: %d samples (%d per mini-batch)', total_sample, batch_size)
    num_updates = epoch * len(train_loader)
    seed = num_updates
    set_global_seed(seed + 1)
    model.train()
    total_acc1 = 0.0
    total_acc5 = 0.0
    train_loss = 0.0
    # T = 2 if epoch <= int(configs.epochs * 0.75) else 15
    #
    # if configs.enable_dynamic_bit_training and \
    #         epoch > 5 and (epoch + 1) % T == 0:
    #     freezing_ratio = freezing_annealing_schedule((epoch - 5) // 2)
    #     freezing_metric = profile_layerwise_quantization_metric(model=model)
    #     freeze_layers(metric=freezing_metric, model=model, ratio=freezing_ratio,
    #                   progressive=False, logger=None, org_cands=configs.target_bits
    #                   )
    #     # logger_info(logger=logger, msg= f'Current freezing ratio: {freezing_ratio}')
    #     print(f'Current freezing ratio: {freezing_ratio}', flush=True)
    #     time.sleep(1)

    from tqdm import tqdm
    pbar = tqdm(total=len(train_loader) + 1, desc=f"Epoch {epoch + 1}",
                postfix={"train_loss": "N/A", "val_loss": "N/A", "val_acc": "N/A"})

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 动态搜索体现在哪里
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)



        QE_loss_weight = annealing_schedule(
            num_updates)  # We use a scheduler for the weights of QE loss according to QAT Oscillations Overcoming [ICML'22].


        start_time = time.time()
        # 这里选择权重位宽和激活值位宽
        sample_max_cands(model, configs)
        # optimizer.base_optimizer.zero_grad()
        optimizer.zero_grad()

        # '''控制变量 取消optimizer_q 记得改127行
        if optimizer_q is not None:
            optimizer_q.zero_grad()
        # '''

        max_outputs = model(inputs)

        # 先不用smooth_cross_entropy
        loss, _, _ = compute_overall_loss(max_outputs, None, targets, criterion,
                                                        model, quantization_error_minimization=False,
                                                        configs=configs, disable_smallest_regularization=True)
        # loss = criterion(max_outputs, targets)
        loss.backward()
        # optimizer.base_optimizer.step()


        acc1, acc5 = accuracy(max_outputs.data, targets.data, topk=(1, 5))
        total_acc1 += acc1
        total_acc5 += acc5
        train_loss += loss.item()
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            "train_loss": f"{train_loss / (batch_idx + 1):.4f}",
            "val_loss": "N/A",
            "val_acc": "N/A"
        }, refresh=True)

        weight_conf_pool = []

        nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        # ''''''
        if optimizer_q is not None:
            optimizer_q.step()

        num_updates += 1
        '''2025/5/10 取消model_ema_update for alexnet
        if model_ema is not None:
            model_ema.update(model)
        '''
    avg_train_loss = train_loss / len(train_loader)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_targets in test_loader:
            val_inputs = val_inputs.cuda()
            val_targets = val_targets.cuda()

            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()

            _, predicted = torch.max(val_outputs.data, 1)
            val_total += val_targets.size(0)
            val_correct += (predicted == val_targets).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100 * val_correct / val_total

    # ============== 最终指标显示 ==============
    pbar.set_postfix({
        "train_loss": f"{avg_train_loss:.4f}",
        "val_loss": f"{avg_val_loss:.4f}",
        "val_acc": f"{val_acc:.2f}%"
    }, refresh=True)
    pbar.update(1)
    pbar.close()
    time.sleep(1)

def validate(data_loader, model, criterion, epoch, monitors, configs, nr_random_sample=3, alpha=1, train_loader=None, eval_predefined_arch=None, bops_limit=1e10, train_mode=False):
    target_bits = configs.target_bits

    criterion = torch.nn.CrossEntropyLoss().cuda()

    meters = [{
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'QE_loss': AverageMeter(),
        'dist_loss': AverageMeter(),
        'IDM_loss': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(len(target_bits) + nr_random_sample)]

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size

    logger_info(logger, msg=f'Validation: {total_sample} samples ({batch_size} per mini-batch)')

    model.eval()

    def _eval(_loader, meter):
        for batch_idx, (inputs, targets) in enumerate(_loader):
            inputs = inputs.to(configs.device)
            targets = targets.to(configs.device)
            start_time = time.time()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))

            update_meter(meter, loss, None, None, None, acc1, acc5, inputs.size(0), time.time() - start_time, configs.world_size)
    
    if train_mode:
        logger_info(logger, msg='Using training mode...')
        model.train()

    if eval_predefined_arch == None:
        from .policy import MIN_POLICY
        eval_predefined_arch = [
            MIN_POLICY
        ]
    
    meters = [{
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'QE_loss': AverageMeter(),
        'dist_loss': AverageMeter(),
        'IDM_loss': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(len(eval_predefined_arch))]

    for idx, arch in enumerate(eval_predefined_arch): 
        w_configs, a_configs = arch[-2], arch[-1]
        if arch[0] == -1:
            sample_min_cands(model, configs)
        elif arch[0] == 32:
            pass
        else:
            set_bit_width(model, w_configs, a_configs)
        
        with torch.no_grad():
            if configs.post_training_batchnorm_calibration:
                assert train_loader is not None

                calibrate_batchnorm_state(model, loader=train_loader, reset=True, distributed_training=True, num_batch=7000//torch.distributed.get_world_size()//configs.dataloader.batch_size)
            
            _eval(data_loader, meters[idx])
            bops, size = model_profiling(model=model, return_layers=False)

            logger_info(logger, msg=f"Arch {idx}, BitOPs {round(bops, 2)} G, Size {round(size, 2)} MB, Top-1 Acc. {round(meters[idx]['top1'].avg, 2)}")
    
    return [meters[idx]['top1'].avg for idx in range(len(eval_predefined_arch))]


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch