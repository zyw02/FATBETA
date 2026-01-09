import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Path to TRT engine')
    parser.add_argument('--iters', type=int, default=1, help='Number of inference iterations')
    args = parser.parse_args()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 1. Load Engine
    print(f"Loading engine: {args.engine}")
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    # 2. Create Context
    context = engine.create_execution_context()
    
    # Identify input and output tensor names
    # In TRT 10, we use names instead of binding indices
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    print(f"Input tensor name: {input_name}")
    print(f"Output tensor name: {output_name}")

    # 3. Alloc Buffers
    # Simple buffer allocation for CIFAR10 (1, 3, 32, 32)
    input_shape = (1, 3, 32, 32)
    input_data = np.random.random(input_shape).astype(np.float32)
    output_data = np.empty((1, 10), dtype=np.float32)

    # Device memory allocation
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    
    # Set tensor addresses (Required for execute_async_v3)
    context.set_input_shape(input_name, input_shape)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    stream = cuda.Stream()

    # 4. Inference Loop
    print(f"Starting {args.iters} iterations of inference...")
    for i in range(args.iters):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        # Execute (New API for TRT 10)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()
    
    print("Inference finished successfully.")
    print(f"Output logits (Top 5): {output_data[0][:5]}")

if __name__ == '__main__':
    main()
