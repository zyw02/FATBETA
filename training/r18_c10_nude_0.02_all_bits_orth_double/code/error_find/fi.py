import torch
import torch.nn as nn
import yaml
import os
import time
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy

import torch.distributed as dist
from model import create_model
from util import (ProgressMonitor, TensorBoardMonitor,
                  get_config, init_logger, set_global_seed, setup_print, load_checkpoint, save_checkpoint,
                  preprocess_model, init_dataloader)
from util.mpq import sample_min_cands, switch_bit_width
from util.greedy_search import search, reset_bit_cands
from util.model_ema import ModelEma
from util.qat import get_quantized_layers
from util.loss_ops import DistributionLoss
from util.utils import create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend, tbmonitor_add_scalars
from util.weight_schd import CosineSched
from quan import find_modules_to_quantize, replace_module_by_names
from data.cifar import Cifar10
from tqdm import tqdm
import numpy as np
import copy
import logging
import math
from scipy.stats import poisson

bsam_choices = {}

normal_choices = {}


g_cnt = 0


def vec_q_seu_simulation(x, model, q_layer, q_layer_name, p_flip, cnt):
    """高效模拟量化权重的位翻转故障注入 (向量化优化版本)"""
    # 设置随机种子 - 使用PyTorch内部生成器
    generator = torch.Generator(device='cuda')
    generator.manual_seed(19 + cnt)

    # 直接从模型获取权重并移动到GPU
    with torch.no_grad():
        fp32_w = model.get_parameter(q_layer_name + '.weight').detach()

    # 确定量化参数
    if q_layer.bits is None:
        bits = 8
    else:
        bits = q_layer.bits[0]
    scale = q_layer.quan_w_fn.s[0]
    is_signed = not q_layer.quan_w_fn.all_positive
    is_symmetric = q_layer.quan_w_fn.symmetric if is_signed else False

    # 计算量化范围
    if is_signed:
        if is_symmetric:
            thd_neg, thd_pos = -2 ** (bits - 1) + 1, 2 ** (bits - 1) - 1
        else:
            thd_neg, thd_pos = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    else:
        thd_neg, thd_pos = 0, 2 ** bits - 1

    # ==================== 向量化核心操作 ====================
    # 1. 权重预处理 (直接在GPU上)
    original_shape = fp32_w.shape
    int_w = (fp32_w / scale).round().clamp(thd_neg, thd_pos).to(torch.int32)

    # 2. 转换为无符号表示
    bit_mask = (1 << bits) - 1
    uint_w = int_w & bit_mask if is_signed else int_w
    n_weights = uint_w.numel()

    # 3. 向量化比特展开 (使用位并行技术)
    bits_matrix = uint_w.unsqueeze(-1).bitwise_and(
        1 << torch.arange(bits, device='cuda')
    ).ne(0).reshape(-1, bits)

    # 4. 泊松采样翻转比特数
    total_bits = n_weights * bits
    lambda_flip = p_flip * total_bits
    n_flips = torch.poisson(torch.tensor(lambda_flip,device='cuda'), generator=generator).item()

    # 5. 随机翻转比特 (向量化操作)
    if n_flips > 0:
        # 创建随机索引
        flip_idx = torch.randperm(total_bits, generator=generator, device='cuda')[:int(n_flips)]
        # 直接通过索引翻转比特
        bits_matrix.flatten()[flip_idx] = ~bits_matrix.flatten()[flip_idx]

    # 6. 向量化比特重组
    bit_values = 1 << torch.arange(bits, device='cuda')
    flipped_uint = (bits_matrix * bit_values).sum(dim=1).to(torch.int32)

    # 7. 处理有符号数
    if is_signed:
        sign_mask = (1 << (bits - 1))
        sign_bits = (flipped_uint & sign_mask).bool()
        flipped_ints = torch.where(sign_bits, flipped_uint - (1 << bits), flipped_uint)
    else:
        flipped_ints = flipped_uint

    # ==================== 结束核心操作 ====================

    # 恢复权重并更新模型
    flipped_fp32_w = flipped_ints.float() * scale
    flipped_fp32_w = flipped_fp32_w.view(original_shape)

    # 调试信息

    print(f'{cnt} layer has been injected with {int(n_flips)} errors')

    return flipped_fp32_w

def q_seu_simulation(x, model, q_layer, q_layer_name, p_flip, cnt):

    """模拟量化权重的位翻转故障注入
    Args:
        model: 目标模型
        q_layer: 目标量化层
        p_flip: 位翻转概率
        cnt: 随机种子偏移量
    """
    np.random.seed(19 + cnt)
    torch.manual_seed(19 + cnt)

    # 获取权重并反量化到整数域
    fp32_w = model.state_dict()[q_layer_name + '.weight']
    original_shape = fp32_w.shape
    fp32_w = fp32_w.flatten()
    scale = q_layer.quan_w_fn.s[0]
    fp32_w = fp32_w / scale  # 反量化到整数域

    # 确定量化参数
    if q_layer.bits is None:
        bits = 8
    else:
        bits = q_layer.bits[0]
    is_signed = not q_layer.quan_w_fn.all_positive
    is_symmetric = q_layer.quan_w_fn.symmetric if is_signed else False

    # 计算量化范围
    if q_layer.quan_w_fn.all_positive:
        thd_neg, thd_pos = 0, 2 ** bits - 1
    else:
        if is_symmetric:
            thd_neg, thd_pos = -2 ** (bits - 1) + 1, 2 ** (bits - 1) - 1
        else:
            thd_neg, thd_pos = -2 ** (bits - 1), 2 ** (bits - 1) - 1

    # 转换为整数并截断到量化范围
    int_w = fp32_w.clamp(min=thd_neg, max=thd_pos).round().to(torch.int32)

    # ==================== 核心部分：聚合所有比特并进行翻转 ====================
    n_weights = int_w.numel()
    total_bits = n_weights * bits

    # 1. 将所有权重转换为无符号整数表示
    if is_signed:
        # 将有符号整数转换为无符号表示（取低bits位）
        uint_w = int_w & ((1 << bits) - 1)
    else:
        uint_w = int_w

    # 2. 创建全比特序列 (N * bits)
    all_bits = torch.zeros(total_bits, dtype=torch.bool,device='cuda')

    # 3. 填充比特序列（每个权重的bits位）
    for i in range(n_weights):
        val = uint_w[i].item()
        for b in range(bits):
            bit_pos = i * bits + b
            all_bits[bit_pos] = (val >> b) & 1

    # 4. 根据泊松分布确定翻转位数
    lambda_flip = p_flip * total_bits
    n_flips = np.random.poisson(lambda_flip)

    if n_flips > 0:
        # 5. 随机选择要翻转的位位置
        flip_positions = torch.randperm(total_bits)[:n_flips]

        # 6. 执行位翻转
        for pos in flip_positions:
            all_bits[pos] = ~all_bits[pos]

    # 7. 将比特序列重组为整数权重
    flipped_uint = torch.zeros(n_weights, dtype=torch.int32,device='cuda')
    for i in range(n_weights):
        val = 0
        for b in range(bits):
            bit_pos = i * bits + b
            if all_bits[bit_pos]:
                val |= (1 << b)
        flipped_uint[i] = val

    # 8. 将无符号整数转换回有符号表示（如果需要）
    if is_signed:
        # 计算符号位位置
        sign_bit = 1 << (bits - 1)
        # 创建掩码用于检测负数
        is_negative = (flipped_uint & sign_bit) != 0
        # 对有符号数进行符号扩展
        flipped_ints = torch.where(
            is_negative,
            flipped_uint - (1 << bits),
            flipped_uint
        )
    else:
        flipped_ints = flipped_uint

    # ==================== 结束核心部分 ====================

    # 将翻转后的整数转换回浮点并恢复原始形状
    flipped_fp32_w = flipped_ints.float() * scale
    flipped_fp32_w = flipped_fp32_w.reshape(original_shape)

    # 更新模型权重
    print(f'{cnt} layer has been injected with errors')
    return flipped_fp32_w




def seu_simulation(tensor, p, cnt):
    """
    对PyTorch张量的权重进行单粒子翻转模拟。

    参数：
        tensor (torch.Tensor): 输入的浮点数张量（float32）
        p (float): 每一位的翻转概率，范围在1e-6到1e-2之间

    返回：
        torch.Tensor: 翻转后的张量，保持原始形状
    """
    np.random.seed(19+cnt)
    assert tensor.dtype == torch.float32, "输入张量必须是float32类型"
    assert 1e-9 <= p <= 1e-1, "翻转率p必须在1e-6到1e-2之间"

    # 展平张量为1D
    flat_tensor = tensor.flatten()
    n = flat_tensor.numel()  # 元素总数
    N = n * 32  # 总位数

    # 使用泊松分布估计翻转的位数
    k = np.random.poisson(N * p)
    if k == 0:
        return tensor.clone()

    # 随机选择要翻转的位索引
    bit_indices = np.random.choice(N, size=min(k, N), replace=False)
    if cnt % 2 == 0:
        bsam_choices[cnt / 2] = bit_indices
    else:
        normal_choices[cnt / 2] = bit_indices
    # 将张量转换为NumPy uint32数组以进行位操作
    uint_tensor = flat_tensor.cpu().numpy().view(np.uint32).copy()

    # 对选中的位进行翻转
    for bit_index in bit_indices:
        i = bit_index // 32  # 权重索引
        j = bit_index % 32  # 位位置
        mask = np.uint32(1 << j)
        uint_tensor[i] ^= mask  # 异或操作翻转该位

    # 转换回float32并恢复原始形状
    flipped_tensor = torch.from_numpy(uint_tensor.view(np.float32)).reshape(tensor.shape)
    return flipped_tensor

def alexnet_fi(model, p, cnt):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            flipped_tensor = seu_simulation(model.state_dict()[name+'.weight'], p, cnt)
            mod.weight.data.copy_(flipped_tensor.cuda())

def q_alexnet_fi(model, p, cnt):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            flipped_tensor = vec_q_seu_simulation(model.state_dict()[name+'.weight'], model, mod, name, p, cnt)
            mod.weight.data.copy_(flipped_tensor.cuda())


def eval(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    nan_times = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader,desc='Evaluating'):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            # has_nan = torch.isnan(outputs).any().item()
            # if (has_nan):
            #     nan_times += 1
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    global g_cnt
    script_dir = Path.cwd()
    configs = get_config(default_file=script_dir / 'template.yaml')
    # set_global_seed(seed=0)

    model = create_model('alexnet', pre_trained=configs.pre_trained)
    model = preprocess_model(model, configs)
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    model.features[3].bits = (6, 6)
    model.features[6].bits = (6, 6)
    model.features[8].bits = (6, 6)
    model.features[10].bits = (6, 6)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            print(mod.quan_w_fn)
    # model.load_state_dict(torch.load('./trained_models/alexnet_ema985_w8a8.pt',weights_only=True))
    filename = 'alexnet_cubesam_timm_w6a6_0.006_0.01.pt'
    model.load_state_dict(torch.load('./trained_models/cube/'+filename, weights_only=True))

    # model.load_state_dict(torch.load('./trained_models/alexnet_bsam_restart_0.01_0.001.pt', weights_only=True))
    # model.load_state_dict(torch.load('./trained_models/qwsam/alexnet_qbsam_timm_w6a6_0.01_.pt', weights_only=True))
    # model.load_state_dict(torch.load('./trained_models/qwsam/alexnet_qbsam_timm_w6a6_0.02_.pt', weights_only=True))
    # model.load_state_dict(torch.load('./trained_models/qwsam/alexnet_qbsam_timm_w6a6_0.002_.pt', weights_only=True))
    # model.load_state_dict(torch.load('./trained_models/qwsam/alexnet_qbsam_timm_w6a6_0.003_.pt', weights_only=True))
    # model.load_state_dict(torch.load('./trained_models/qwsam/alexnet_qbsam_timm_w6a6_0.004_.pt', weights_only=True))
    # model.load_state_dict(torch.load('./trained_models/qwsam/alexnet_qbsam_timm_w6a6_0.005_.pt', weights_only=True))

    # model.load_state_dict(torch.load('./trained_models/qsam/_alexnet_qbsam_timm_w6a6_minsharpness_max_0.02_.pt', weights_only=True))
    model = model.cuda()
    # ------------- data --------------
    dataset = Cifar10(32, 0)
    test_loader = dataset.test
    # acc, _ = eval(model, test_loader)
    # print(f'acc: {acc}')
    model_bsam_quan = copy.deepcopy(model)
    # acc, _ = eval(model_bsam_quan, test_loader)
    # print(f'acc: {acc}')
    # model.load_state_dict(torch.load('./trained_models/alexnet_w6a6_no_freezing_no_sampling_199.pt', weights_only=True))
    model.load_state_dict(torch.load('./trained_models/cube/alexnet_cubesam_timm_w6a6_clean.pt', weights_only=True))
    model = model.cuda()
    model_quan = copy.deepcopy(model)




    start_epoch = 0
    # for k,_ in model.state_dict().items():
    #     print(k)
    # for n,w in model.named_parameters():
    #     print(f'name: {n}, weight shape: {w.shape}')
    # for n,l in model.named_modules():
    #     print(f'Layer: {n}, Layer Type: {type(l)}')

    accs = []
    nans = []
    accs_bsam_quan = [0.0, 0.0, 0.0, 0.0, 0.0]
    accs_quan = [0.0, 0.0, 0.0, 0.0, 0.0]
    logging.basicConfig(
        # filename='logs/fi_compare_quan_and_bsam.log',
        filename='logs/cube.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logging.info(f'robust model {filename}\r\n')
    for i in range(2):
        # np.random.seed(int(time.time()))
        for j,p in enumerate([2e-2,3e-2,4e-2,1e-1]):
            model_bkp = copy.deepcopy(model_bsam_quan)  # 每次从原始模型创建新副本
            q_alexnet_fi(model_bkp, p, g_cnt)
            acc = eval(model_bkp, test_loader)
            accs_bsam_quan[j] += acc

            model_bkp = copy.deepcopy(model_quan)  # 每次从原始模型创建新副本
            q_alexnet_fi(model_bkp, p, g_cnt)
            acc = eval(model_bkp, test_loader)
            accs_quan[j] += acc
            g_cnt += 1

        print(f'Epoch {i + 1} done!')
        logging.info(
            # f'FI Campaign Epoch [{i + 1}/100] | acc_bsam_quan: {[f"{x / (i + 1):.2f}" for x in accs_bsam_quan]} | acc_quan: {[f"{x / (i + 1):.2f}" for x in accs_quan]}%')
            f'FI Campaign Epoch [{i + 1}/100] | acc_bsam_quan: {", ".join(f"{x / (i + 1):.2f}" for x in accs_bsam_quan)} | acc_quan: {", ".join(f"{x / (i + 1):.2f}" for x in accs_quan)}%')
    print(accs_bsam_quan)
    print(accs_quan)
    logging.info(f'\r\n')
    # print(accs_bsam_quan_mean)
    # print(accs_quan_mean)



if __name__ == "__main__":
    main()
