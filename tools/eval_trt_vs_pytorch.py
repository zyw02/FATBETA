import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import create_model
from quan import find_modules_to_quantize, replace_module_by_names
from util.checkpoint import load_checkpoint
from util.config import get_config
from util.data_loader import init_dataloader
from util.qat import set_bit_width
from util.mpq import switch_bit_width
from quan.func import QuanConv2d, QuanLinear

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of batches for quick test')
    args = parser.parse_args()

    device = torch.device('cuda')
    
    # 1. Load PyTorch Model (W8A8 Mode)
    print("--- Loading PyTorch Model ---")
    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    config = get_config(default_file=args.config)
    sys.argv = orig_argv

    model = create_model(config.arch, dataset=config.dataloader.dataset)
    modules_to_replace = find_modules_to_quantize(model, config)
    replace_module_by_names(model, modules_to_replace)
    load_checkpoint(model, args.ckpt, model_device=device, use_ema=True)
    model.to(device).eval()
    
    # Warm-up to populate output_size (Required for model_profiling in set_bit_width)
    with torch.no_grad():
        dummy_in = torch.randn(1, 3, 32, 32).to(device)
        model(dummy_in)

    # Set bit-width logic (same as fast_sweep)
    bits = 8
    dynamic_layers = [n for n, m in model.named_modules() if isinstance(m, (QuanConv2d, QuanLinear)) and not (hasattr(m, 'fixed_bits') and m.fixed_bits)]
    set_bit_width(model, [bits]*len(dynamic_layers), [bits]*len(dynamic_layers))
    for m in model.modules():
        if isinstance(m, (QuanConv2d, QuanLinear)):
            m.bits = (bits, bits)
            if hasattr(m, 'quan_w_fn') and m.quan_w_fn: m.quan_w_fn.bits = bits
    switch_bit_width(model, quan_scheduler=config.quan, wbit=bits, abits=bits)

    # 2. Load TensorRT Engine
    print("--- Loading TensorRT Engine ---")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # 3. Data Loader (Force batch_size=1 for single image testing)
    config.dataloader.batch_size = 1
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)

    # 4. Evaluation Loop
    pytorch_correct = 0
    trt_correct = 0
    total = 0
    
    # Alloc TRT buffers for batch size 1
    batch_size = 1
    input_shape = (batch_size, 3, 32, 32)
    output_shape = (batch_size, 10)
    
    d_input = cuda.mem_alloc(1 * 3 * 32 * 32 * 4)
    d_output = cuda.mem_alloc(1 * 10 * 4)
    
    print(f"Starting evaluation on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            if args.limit and i >= args.limit:
                break
            
            actual_batch_size = images.size(0)
            images_cuda = images.to(device)
            targets_np = targets.numpy()
            
            # PyTorch Inference
            outputs_py = model(images_cuda)
            _, pred_py = outputs_py.max(1)
            pytorch_correct += pred_py.cpu().eq(targets).sum().item()
            
            # TensorRT Inference
            # Handle potentially smaller last batch
            curr_input_shape = (actual_batch_size, 3, 32, 32)
            input_np = images.numpy().astype(np.float32)
            
            cuda.memcpy_htod(d_input, input_np)
            context.set_input_shape(input_name, curr_input_shape)
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
            context.execute_async_v3(stream_handle=0)
            
            output_trt = np.empty((actual_batch_size, 10), dtype=np.float32)
            cuda.memcpy_dtoh(output_trt, d_output)
            
            pred_trt = np.argmax(output_trt, axis=1)
            trt_correct += np.sum(pred_trt == targets_np)
            
            total += actual_batch_size

    # 5. Summary
    py_acc = 100. * pytorch_correct / total
    trt_acc = 100. * trt_correct / total
    
    print("\n" + "="*40)
    print(f"Final Results (Total Images: {total})")
    print("-" * 40)
    print(f"PyTorch (W8A8 Fake) Acc: {py_acc:.2f}%")
    print(f"TensorRT (INT8 Real) Acc: {trt_acc:.2f}%")
    print(f"Accuracy Drop:           {py_acc - trt_acc:.4f}%")
    print("="*40)

if __name__ == '__main__':
    main()

