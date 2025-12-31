import math
import torch
import torch.nn as nn
from util.utils import set_global_seed, accuracy
from util.qat import profile_layerwise_quantization_metric, freeze_layers, set_bit_width, auxiliary_quantized_loss, remove_hook_for_quantized_layers, set_forward_hook_for_quantized_layers
from util.mpq import sample_one_mixed_policy, sample_max_cands, sample_min_cands
from process import compute_overall_loss
import time

def get_scale_for_param(model, param):
    # 遍历模型的所有模块
    for module in model.modules():
        # 检查模块是否包含与param相关的scale参数
        # 这里需要根据你的QAT实现调整条件
        if hasattr(module, 'weight') and module.weight is param:
            if hasattr(module, 'quan_w_fn'):
                return module.quan_w_fn.s[0]  # 返回对应的scale参数
    return None  # 如果未找到对应scale

def train_cubesam_epoch(model, train_loader, test_loader, criterion, optimizer, epoch, configs,
                              lr_scheduler = None, optimizer_q = None, annealing_schedule=None,freezing_annealing_schedule=None, rho = 1.0):
    target_bits = configs.target_bits
    target_bits.sort()
    target_bits.reverse()

    if getattr(configs, 'sandwich_training', False):
        sample_current_max, sample_current_min = True, True
    else:
        sample_current_max, sample_current_min = False, False

    sample_current_max = True

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
        optimizer.zero_grad()
        optimizer_q.zero_grad()
        # === Step 1: Forward + Compute Clean Loss ===
        clean_outputs = model(inputs)
        # clean_loss, _, _ = compute_overall_loss(clean_outputs, None, targets, criterion,
        #                                                 model, quantization_error_minimization=False,
        #                                                 configs=configs, disable_smallest_regularization=True)
        loss_clean = criterion(clean_outputs, targets)

        # === Step 2: Compute Gradient Norm Regularizer ===
        res = []
        optimizer_param_ids = {id(p) for p in optimizer.param_groups[0]['params']}

        grads = torch.autograd.grad(
            loss_clean,
            [p for p in model.parameters()
             if p.requires_grad and id(p) in optimizer_param_ids],  # 改用 ID 比对 [7,8](@ref)
            create_graph=True)


        # === Step 3: Gradient norm regularization term ===
        grad_norm_sq = sum((g ** 2).sum() for g in grads)
        # === Step 4: Generate L_inf SAM perturbations ===
        epsilon_list = []
        # weight_params = optimizer.param_groups[0]['params']
        # for param, grad in zip(weight_params, grads):
        #     scale = get_scale_for_param(model,param)  # 你提前实现好的映射函数
        #     if scale is None:
        #         epsilon = torch.zeros_like(param)
        #     else:
        #         # scale = scale.view_as(param) if scale.numel() == 1 else scale.reshape_as(param)
        #         if scale.numel() == 1:
        #             # 标量直接赋值，无需重塑
        #             scale = scale.item()  # 转为Python数值
        #         else:
        #             scale = scale.reshape_as(param)  # 仅当元素数量匹配时重塑
        #         epsilon = rho * scale * grad.sign()
        #     # epsilon_list.append(epsilon)
        #         with torch.no_grad():
        #             setattr(param, 'orig_data', param.data.clone())
        #             param.data.add_(epsilon)  # w ← w + ε
        #
        # # === 新增：清空梯度 ===
        # optimizer.zero_grad()
        # optimizer_q.zero_grad()
        #
        # # === Step 5: Forward on perturbed weights ===
        # output_adv = model(inputs)
        # loss_adv = criterion(output_adv, targets)
        #
        # # === Step 6: Restore original weights ===
        # with torch.no_grad():
        #     for param in weight_params:
        #         param.data = getattr(param, 'orig_data')
        #         delattr(param, 'orig_data')
        #
        # # === Step 7: Backprop total loss ===
        # loss_total = loss_clean + grad_norm_sq + 1e-2*loss_adv
        loss_total = loss_clean + 1e-3*grad_norm_sq
        # loss_total = loss_clean
        loss_total.backward()

        # === Step 8: Optimizer steps ===
        optimizer.step()
        optimizer_q.step()

        train_loss += loss_total.item()
        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            "train_loss": f"{train_loss / (batch_idx + 1):.4f}",
            "val_loss": "N/A",
            "val_acc": "N/A"
        }, refresh=True)

        weight_conf_pool = []



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