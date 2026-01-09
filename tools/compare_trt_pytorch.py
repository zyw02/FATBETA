import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--engine', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    
    # 1. Load PyTorch Model
    print("--- Setting up PyTorch Model ---")
    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    config = get_config(default_file=args.config)
    sys.argv = orig_argv

    model = create_model(config.arch, dataset=config.dataloader.dataset)
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    load_checkpoint(model, args.ckpt, model_device=device, use_ema=True)
    model.to(device).eval()

    # 2. Load TensorRT Engine
    print("--- Setting up TensorRT Engine ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # 3. Prepare Common Input (Real CIFAR10 Image)
    print("--- Loading real CIFAR10 data ---")
    from util.data_loader import init_dataloader
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)
    
    # Get one batch and pick the first image
    images, targets = next(iter(test_loader))
    input_torch = images[0:1].to(device) # Shape: (1, 3, 32, 32)
    input_np = input_torch.cpu().numpy()
    input_shape = input_np.shape
    target_class = targets[0].item()
    print(f"Testing with image from class: {target_class}")

    # 4. Run PyTorch Inference with full Bit-width activation
    from util.qat import set_bit_width
    from util.mpq import switch_bit_width
    from quan.func import QuanConv2d, QuanLinear
    
    model.eval()
    with torch.no_grad():
        # A. Initial Warm-up (populate output_size etc.)
        model(input_torch)
        
        # B. Set Bit-width (W8A8)
        bits = 8
        dynamic_layers = [n for n, m in model.named_modules() if isinstance(m, (QuanConv2d, QuanLinear)) and not (hasattr(m, 'fixed_bits') and m.fixed_bits)]
        set_bit_width(model, [bits]*len(dynamic_layers), [bits]*len(dynamic_layers))
        
        for m in model.modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                m.bits = (bits, bits)
                if hasattr(m, 'quan_w_fn') and m.quan_w_fn: 
                    m.quan_w_fn.bits = bits
        
        # C. Switch BN branches (Important for SwithableBatchNorm)
        switch_bit_width(model, quan_scheduler=config.quan, wbit=bits, abits=bits)
        
        # D. Final Forward for comparison
        output_pytorch = model(input_torch).cpu().numpy()

    # 5. Run TensorRT Inference
    d_input = cuda.mem_alloc(input_np.nbytes)
    d_output = cuda.mem_alloc(output_pytorch.nbytes)
    cuda.memcpy_htod(d_input, input_np)
    
    context.set_input_shape(input_name, input_shape)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    context.execute_async_v3(stream_handle=0) # sync execution
    
    output_trt = np.empty_like(output_pytorch)
    cuda.memcpy_dtoh(output_trt, d_output)

    # 6. Compare Results
    print("\n" + "="*50)
    print(f"{'Category':<15} | {'PyTorch (FP32)':<15} | {'TensorRT (INT8)':<15}")
    print("-" * 50)
    for i in range(min(10, output_pytorch.shape[1])):
        print(f"Class {i:<8} | {output_pytorch[0][i]:<15.6f} | {output_trt[0][i]:<15.6f}")
    
    mse = np.mean((output_pytorch - output_trt)**2)
    max_err = np.max(np.abs(output_pytorch - output_trt))
    
    top1_pytorch = np.argmax(output_pytorch, axis=1)[0]
    top1_trt = np.argmax(output_trt, axis=1)[0]
    
    print("="*50)
    print(f"PyTorch Top-1 Class: {top1_pytorch}")
    print(f"TensorRT Top-1 Class: {top1_trt}")
    print(f"Target Class:       {target_class}")
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Max Absolute Error: {max_err:.8f}")
    
    if top1_pytorch == top1_trt:
        print("\n[Result] Success! Top-1 predictions match.")
    else:
        print("\n[Result] Warning: Top-1 predictions MISMATCH!")

if __name__ == '__main__':
    main()

