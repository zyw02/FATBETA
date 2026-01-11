import logging

from .func import *
from .quantizer import *
import torch
from torchvision.ops.misc import Conv2dNormActivation
from timm.models._efficientnet_blocks import InvertedResidual as InvertedResidual_EffNet
from model import InvertedResidual as InvertedResidual_MBb2

def quantizer(default_cfg, this_cfg=None, skip_quantization=False, remap_act=False, scale_grad=False):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v
    
    assert target_cfg['mode'] == 'lsq'

    q = IdentityQuan if skip_quantization else LsqQuan
    
    if remap_act:
        assert target_cfg['all_positive']
        target_cfg['all_positive'] = False

    target_cfg['scale_grad'] = scale_grad
    target_cfg.pop('mode')
    return q(**target_cfg)


def find_modules_to_quantize(model, configs):
    replaced_modules = dict()
    conv = None
    remap_act = False
    target_bits = getattr(configs, 'target_bits', [8])
    
    mb_blocks = (Conv2dNormActivation, InvertedResidual_MBb2, InvertedResidual_EffNet)

    for name, module in model.named_modules():
        if isinstance(module, mb_blocks): #handle MobileNetv2 Block
            idx = 0
            for subname, submodule in module.named_modules():
                if isinstance(submodule, torch.nn.Conv2d) and (submodule.groups == 1) and submodule.in_channels > 3:
                    # if idx == 0:
                    remap_act = True 
                    print(name, subname)

        if 'conv_head' in name or 'features.18' in name: #handle last quantized layer in efficientnet
            remap_act = True

        if type(module) in ops.keys():
            if name in configs.quan.excepts:
                if isinstance(module, torch.nn.BatchNorm2d):
                    replaced_modules[name] = SwithableBatchNorm(module, module.num_features, None)
                else:
                    # 获取该层特定的配置
                    layer_cfg = configs.quan.excepts[name]
                    w_cfg = layer_cfg.get('weight', {})
                    a_cfg = layer_cfg.get('act', {})
                    
                    # --- 智能约定逻辑 ---
                    # 1. 权重位宽：没写则默认为 8
                    w_bit = w_cfg.get('bit')
                    if w_bit is None:
                        w_bit = 8
                    
                    # 2. 激活位宽：没写则 Linear 默认为 8, Conv 默认为 None (32-bit)
                    a_bit = a_cfg.get('bit')
                    if a_bit is None:
                        a_bit = 8 if isinstance(module, torch.nn.Linear) else None
                    
                    # 3. 决定是否跳过量化器
                    skip_w = (w_bit is None)
                    skip_a = (a_bit is None)

                    replaced_modules[name] = ops[type(module)](
                        module,
                        bits_list=[w_bit if w_bit is not None else 32],
                        quan_w_fn=quantizer(configs.quan.weight, this_cfg=w_cfg, skip_quantization=skip_w, scale_grad=getattr(configs, "scale_gradient", True)),
                        quan_a_fn=quantizer(configs.quan.act, this_cfg=a_cfg, skip_quantization=skip_a, scale_grad=getattr(configs, "scale_gradient", True)),
                        fixed_bits=(w_bit, a_bit)
                    )
            else:
                if isinstance(module, torch.nn.BatchNorm2d):
                    # If this BN follows an excepted layer (conv is None), it should not be switchable
                    actual_bits = target_bits if conv is not None else None
                    replaced_modules[name] = SwithableBatchNorm(module, module.num_features, actual_bits, conv)
                else:
                    target_bits = configs.target_bits
                    target_bits = configs.target_bits[:-1] if (min(configs.target_bits) == 2 and module.weight.shape[1] == 1) else configs.target_bits
                    
                    replaced_modules[name] = ops[type(module)](
                        module, 
                        bits_list=target_bits, 
                        quan_w_fn=quantizer(configs.quan.weight, scale_grad=getattr(configs, "scale_gradient", True)),
                        quan_a_fn=quantizer(configs.quan.act, remap_act=remap_act if module.weight.shape[1] != 1 else False, scale_grad=getattr(configs, "scale_gradient", True)),
                        fixed_bits = None,
                        split_aw_cands=configs.split_aw_cands
                    )
                    if remap_act:
                        remap_act = False
                    conv = replaced_modules[name]
        elif name in configs.quan.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: torch.nn.Module):
        for n, c in child.named_children():
            if type(c) in ops.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model
