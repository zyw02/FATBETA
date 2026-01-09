import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import sys
import hashlib
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
    parser.add_argument('--num_images', type=int, default=100, help='Size of the golden subset')
    args = parser.parse_args()

    # 1. Setup
    orig_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], args.config]
    config = get_config(default_file=args.config)
    sys.argv = orig_argv
    config.dataloader.batch_size = 1
    _, _, test_loader, _, _ = init_dataloader(config.dataloader, config.arch)

    # 2. Load TRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    d_input = cuda.mem_alloc(1 * 3 * 32 * 32 * 4)
    d_output = cuda.mem_alloc(1 * 10 * 4)
    context.set_input_shape(input_name, (1, 3, 32, 32))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # 3. Evaluation
    predictions = []
    for i, (image, target) in enumerate(test_loader):
        if i >= args.num_images: break
        
        cuda.memcpy_htod(d_input, image.numpy().astype(np.float32))
        context.execute_async_v3(stream_handle=0)
        
        output_trt = np.empty((1, 10), dtype=np.float32)
        cuda.memcpy_dtoh(output_trt, d_output)
        predictions.append(int(np.argmax(output_trt, axis=1)[0]))

    # 4. Generate Robustness Checksum
    pred_array = np.array(predictions, dtype=np.int32)
    # 计算准确率作为辅助参考
    # (这里需要额外的逻辑获取子集标签，暂时略过)
    
    # 核心：计算预测向量的哈希值
    result_hash = hashlib.md5(pred_array.tobytes()).hexdigest()
    
    print(f"\n--- INJECTION_RESULT_START ---")
    print(f"Golden_Subset_Size: {args.num_images}")
    print(f"Prediction_Checksum: {result_hash}")
    print(f"--- INJECTION_RESULT_END ---")

if __name__ == '__main__':
    main()




