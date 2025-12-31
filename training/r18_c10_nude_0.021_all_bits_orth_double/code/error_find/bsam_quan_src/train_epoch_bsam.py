import math
import torch
import torch.nn as nn
from util.utils import set_global_seed, accuracy
from util.qat import profile_layerwise_quantization_metric, freeze_layers, set_bit_width, auxiliary_quantized_loss, remove_hook_for_quantized_layers, set_forward_hook_for_quantized_layers
from util.mpq import sample_one_mixed_policy, sample_max_cands, sample_min_cands
from process import compute_overall_loss
import time

def train_with_bsam_one_epoch(model, train_loader, test_loader, criterion, optimizer, epoch, configs,
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
        # optimizer.zero_grad()

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
        optimizer.to_max_point(zero_grad=True) # 修改模型权重，并执行梯度清零
        #
        max_outputs = model(inputs)

        # 先不用smooth_cross_entropy
        lossa, _, _ = compute_overall_loss(max_outputs, None, targets, criterion,
                                          model, quantization_error_minimization=False,
                                          configs=configs, disable_smallest_regularization=True)
        # lossa = criterion(max_outputs, targets)
        lossa.backward()
        optimizer.to_min_point(zero_grad=True)
        #
        max_outputs = model(inputs)
        #
        # # 先不用smooth_cross_entropy
        lossb, _, _ = compute_overall_loss(max_outputs, None, targets, criterion,
                                          model, quantization_error_minimization=False,
                                          configs=configs, disable_smallest_regularization=True)
        # lossb = criterion(max_outputs, targets)
        lossb.backward()
        # opt_max_min_step这一步已经完成了对w参数的更新，并在最后清空了梯度，因此无需手动清零梯度
        optimizer.opt_max_min_step(epoch, zero_grad=True)

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


        with torch.no_grad():
            # lr_ = lr_scheduler.step()
            optimizer.update_rho_t()
        # optimizer.step()
        # ''''''
        # 现在的写法是w和s同步更新，改为先w更新再s更新试一下
        if optimizer_q is not None:
            optimizer_q.zero_grad()
        max_outputs = model(inputs)
        #
        # # 先不用smooth_cross_entropy
        lossc, _, _ = compute_overall_loss(max_outputs, None, targets, criterion,
                                           model, quantization_error_minimization=False,
                                           configs=configs, disable_smallest_regularization=True)
        if optimizer_q is not None:
            optimizer_q.step() # 根据当前学习率更新参数
        optimizer.zero_grad()
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
    if (epoch+1) % 5 == 0:
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
    val_acc = 100 * val_correct / (val_total+1e-6)

    # ============== 最终指标显示 ==============
    pbar.set_postfix({
        "train_loss": f"{avg_train_loss:.4f}",
        "val_loss": f"{avg_val_loss:.4f}",
        "val_acc": f"{val_acc:.2f}%"
    }, refresh=True)
    pbar.update(1)
    pbar.close()
    time.sleep(1)