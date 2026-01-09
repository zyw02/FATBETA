import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from quan.func import QuanConv2d, QuanLinear, SwithableBatchNorm
from quan.quantizer.lsq import LsqQuan

class QDQWrapper(nn.Module):
    def __init__(self, scale, thd_neg, thd_pos, axis=None):
        super().__init__()
        # Ensure scale is a tensor and on CPU initially, 
        # model.to(device) will move it later if needed.
        if isinstance(scale, torch.Tensor):
            self.register_buffer('scale', scale.detach().clone().cpu().view(-1))
            self.register_buffer('zero_point', torch.zeros_like(self.scale, dtype=torch.int32).cpu())
        else:
            self.register_buffer('scale', torch.tensor([scale], dtype=torch.float32))
            self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int32))
        
        self.thd_neg = int(thd_neg)
        self.thd_pos = int(thd_pos)
        self.axis = axis

    def forward(self, x):
        if self.axis is not None and self.scale.numel() > 1:
            return torch.fake_quantize_per_channel_affine(
                x, 
                self.scale, 
                self.zero_point, 
                self.axis,
                self.thd_neg, 
                self.thd_pos
            )
        else:
            return torch.fake_quantize_per_tensor_affine(
                x, 
                float(self.scale[0]), 
                int(self.zero_point[0]), 
                self.thd_neg, 
                self.thd_pos
            )

class TRTQuantizedConv2d(nn.Module):
    def __init__(self, m: QuanConv2d, bits=8):
        super().__init__()
        self.stride = m.stride
        self.padding = m.padding
        self.dilation = m.dilation
        self.groups = m.groups
        self.weight = nn.Parameter(m.weight.detach())
        self.bias = nn.Parameter(m.bias.detach()) if m.bias is not None else None
        
        from quan.quantizer.lsq import compute_thd
        
        # Determine actual bits for this layer
        if getattr(m, 'fixed_bits', None) is not None:
            wbits, abits = m.fixed_bits
        else:
            wbits, abits = bits, bits

        # Weight scale
        if wbits < 32:
            w_scale = m.quan_w_fn.get_scale(wbits)
            w_thd_neg, w_thd_pos = compute_thd(m.quan_w_fn, wbits)
            axis = 0 if m.quan_w_fn.per_channel else None
            self.weight_qdq = QDQWrapper(w_scale, w_thd_neg, w_thd_pos, axis=axis)
        else:
            self.weight_qdq = nn.Identity()
        
        # Activation scale
        if abits < 32:
            a_scale = m.quan_a_fn.get_scale(abits)
            a_thd_neg, a_thd_pos = compute_thd(m.quan_a_fn, abits)
            self.act_qdq = QDQWrapper(a_scale, a_thd_neg, a_thd_pos)
        else:
            self.act_qdq = nn.Identity()

    def forward(self, x):
        x = self.act_qdq(x)
        w = self.weight_qdq(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class TRTQuantizedLinear(nn.Module):
    def __init__(self, m: QuanLinear, bits=8):
        super().__init__()
        self.weight = nn.Parameter(m.weight.detach())
        self.bias = nn.Parameter(m.bias.detach()) if m.bias is not None else None
        
        from quan.quantizer.lsq import compute_thd
        
        # Determine actual bits for this layer
        if getattr(m, 'fixed_bits', None) is not None:
            wbits, abits = m.fixed_bits
        else:
            wbits, abits = bits, bits

        # Weight scale
        if wbits < 32:
            w_scale = m.quan_w_fn.get_scale(wbits)
            w_thd_neg, w_thd_pos = compute_thd(m.quan_w_fn, wbits)
            axis = 0 if m.quan_w_fn.per_channel else None
            self.weight_qdq = QDQWrapper(w_scale, w_thd_neg, w_thd_pos, axis=axis)
        else:
            self.weight_qdq = nn.Identity()
        
        # Activation scale
        if abits < 32:
            a_scale = m.quan_a_fn.get_scale(abits)
            a_thd_neg, a_thd_pos = compute_thd(m.quan_a_fn, abits)
            self.act_qdq = QDQWrapper(a_scale, a_thd_neg, a_thd_pos)
        else:
            self.act_qdq = nn.Identity()

    def forward(self, x):
        x = self.act_qdq(x)
        w = self.weight_qdq(self.weight)
        return F.linear(x, w, self.bias)
        x = self.act_qdq(x)
        w = self.weight_qdq(self.weight)
        return F.linear(x, w, self.bias)

def convert_to_trt_model(model, bits=8):
    """
    Recursively replaces Quan layers with TRT-compatible QDQ layers
    and SwithableBatchNorm with standard BatchNorm2d (8-bit branch)
    """
    for name, module in model.named_children():
        if isinstance(module, QuanConv2d):
            setattr(model, name, TRTQuantizedConv2d(module, bits))
        elif isinstance(module, QuanLinear):
            setattr(model, name, TRTQuantizedLinear(module, bits))
        elif isinstance(module, SwithableBatchNorm):
            # Extract the 8-bit BN
            if module.bits_list is not None:
                try:
                    idx_w = module.bits_list.index(bits)
                    idx_a = module.bits_list.index(bits)
                    new_bn = module.bn_list[idx_w][idx_a]
                except ValueError:
                    # If 8-bit not in list, fallback to the first one or standard bn
                    print(f"Warning: {bits}-bit not found in SwithableBatchNorm {name}, using first available.")
                    new_bn = module.bn_list[0][0]
            else:
                new_bn = module.bn
            setattr(model, name, new_bn)
        else:
            convert_to_trt_model(module, bits)

def main():
    parser = argparse.ArgumentParser(description='Export Model to ONNX with QDQ nodes for TensorRT')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--out', type=str, default='model_qdq.onnx', help='Output ONNX path')
    parser.add_argument('--bits', type=int, default=8, help='Target bit-width for export')
    parser.add_argument('--input_size', type=int, default=32, help='Input image size')
    args = parser.parse_args()

    device = torch.device('cpu') # Exporting on CPU is safer for ONNX
    
    # 1. Load Config & Model
    # Spoof sys.argv for get_config
    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    config = get_config(default_file=args.config)
    sys.argv = orig_argv

    print(f"Creating model: {config.arch}")
    model = create_model(config.arch, dataset=config.dataloader.dataset)
    
    # 2. Setup Quantization (to match checkpoint structure)
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    
    # 3. Load Weights
    print(f"Loading checkpoint: {args.ckpt}")
    load_checkpoint(model, args.ckpt, model_device=device, use_ema=True)
    model.to(device) # Ensure everything is on CPU/Target device
    
    # 4. Convert to TRT-compatible QDQ morphology
    print(f"Converting model to QDQ morphology (Target: W{args.bits}A{args.bits})...")
    model.eval()
    convert_to_trt_model(model, bits=args.bits)
    model.to(device) # Final sync after conversion
    
    # 5. Export to ONNX
    print(f"Exporting to ONNX: {args.out}")
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        args.out,
        export_params=True,
        opset_version=13, # Required for QDQ support
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Success! You can now use this ONNX file with trtexec to build an INT8 engine.")
    print(f"Example: trtexec --onnx={args.out} --int8 --saveEngine=model_int8.engine")

if __name__ == '__main__':
    main()

