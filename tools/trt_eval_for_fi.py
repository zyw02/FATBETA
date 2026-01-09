import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from util.data_loader import init_dataloader
from util.config import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=1000, help='Total images to evaluate in one run')
    args = parser.parse_args()

    # 1. Load Config & Data (Force Batch 1)
    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    config = get_config(default_file=args.config)
    sys.argv = orig_argv
    
    config.dataloader.batch_size = 1
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)

    # 2. Load Engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # 3. Alloc Buffers (Batch 1)
    d_input = cuda.mem_alloc(1 * 3 * 32 * 32 * 4)
    d_output = cuda.mem_alloc(1 * 10 * 4)
    
    context.set_input_shape(input_name, (1, 3, 32, 32))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # 4. Evaluation Loop
    correct = 0
    total = 0
    
    print(f"Starting evaluation on {args.num_images} images for Fault Injection...")
    
    for i, (image, target) in enumerate(test_loader):
        if i >= args.num_images:
            break
            
        input_np = image.numpy().astype(np.float32)
        target_np = target.numpy()
        
        # Copy to device
        cuda.memcpy_htod(d_input, input_np)
        
        # Execute (This is where NVBitFI will inject!)
        context.execute_async_v3(stream_handle=0)
        
        # Copy back
        output_trt = np.empty((1, 10), dtype=np.float32)
        cuda.memcpy_dtoh(output_trt, d_output)
        
        # Stats
        pred = np.argmax(output_trt, axis=1)
        correct += np.sum(pred == target_np)
        total += 1

    # 5. Output Result
    acc = 100. * correct / total
    print(f"\nFinal_Accuracy: {acc:.2f}%")
    print(f"Total_Correct: {correct}")
    print(f"Total_Evaluated: {total}")

if __name__ == '__main__':
    main()




