
import torch
import torch.nn as nn
import copy
import sys
import os

# Add local directory to path to import modules
sys.path.append(os.getcwd())

from quan.func import QuanConv2d, SwithableBatchNorm
from util.fault_injector import FaultInjector
from util.qat import link_conv_bn

# Mock Quantizer for testing
class MockQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.01)) # Initial arbitrary scale
        self.grad_scale = 1.0
    
    def get_scale(self, bits, detach=False):
        # Return a fixed scale for testing, or the learnable parameter
        # In real LSQ, this depends on bits. Here we simplify.
        if detach:
            return self.scale.detach()
        return self.scale

    def init_from(self, x, *args, **kwargs):
        # Dummy initialization
        pass

    def forward(self, x, bits, is_activation=False, scale=None, **kwargs):
        # Simplified quantization logic matching LSQ
        # If scale is provided (explicitly passed), use it
        s = scale if scale is not None else self.scale
        
        # Calculate range
        thd_neg = -(1 << (bits - 1))
        thd_pos = (1 << (bits - 1)) - 1
        
        # Quantize
        x_s = x / s
        x_int = torch.round(x_s)
        x_q = torch.clamp(x_int, thd_neg, thd_pos)
        x_deq = x_q * s
        return x_deq

# Setup simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        # Use random weights
        nn.init.kaiming_normal_(self.conv.weight)
        
        self.bn = nn.BatchNorm2d(16)
        # Set BN params to something non-trivial
        nn.init.uniform_(self.bn.weight, 0.5, 2.0) # Gamma
        nn.init.uniform_(self.bn.bias, -1.0, 1.0) # Beta
        nn.init.constant_(self.bn.running_mean, 0.1)
        nn.init.constant_(self.bn.running_var, 1.5)
        
        # Wrap in QuanConv2d
        # Mocking the initialization of QuanConv2d
        self.qconv = QuanConv2d(self.conv, quan_w_fn=MockQuantizer(), quan_a_fn=MockQuantizer())
        # Set bits manually
        self.qconv.bits = (8, 8) 
        
        # BN linkage happens later

    def forward(self, x):
        x = self.qconv(x)
        x = self.bn(x)
        return x

def test_bn_folding_consistency():
    print("=== Testing BN-Folding Aware Fault Injection Consistency ===")
    
    # 1. Setup Model
    model = SimpleModel()
    model.eval() # Important: Fault injection (usually) works in eval or training
    # For this test, we assume training mode is False for BN (using running stats)
    # But FaultInjector might require enabling.
    
    # Link BN
    model.qconv.bn_layer = model.bn
    # IMPORTANT: Attach parent layer reference to quantizer (mimicking link_conv_bn logic)
    model.qconv.quan_w_fn.parent_layer = model.qconv
    
    # 2. Setup FaultInjector
    # Use a large BER to ensure some faults happen, or a specific seed
    injector = FaultInjector(model, mode="ber", ber=0.1, enable_in_inference=True, seed=42)
    injector.enable()
    
    # 3. Prepare Input
    x = torch.randn(1, 3, 32, 32)
    
    # 4. Run Forward Pass with Injection (Proposed Method)
    # This will trigger the 'virtual fusion' logic in FaultInjector
    y_proposed = model(x)
    
    # 5. Extract Internals to Verify Manually
    # Get original weights and BN params
    w_orig = model.qconv.weight.detach() # Note: QuanConv2d forward uses self.weight
    gamma = model.bn.weight.detach()
    beta = model.bn.bias.detach()
    running_mean = model.bn.running_mean.detach()
    running_var = model.bn.running_var.detach()
    eps = model.bn.eps
    
    # Calculate BN folding scale
    std = torch.sqrt(running_var + eps)
    s_bn = (gamma / std).view(-1, 1, 1, 1)
    bn_bias_fold = beta - (running_mean * gamma / std)
    
    # 6. Simulate "Ghost Path" (Theoretical Ground Truth)
    # Fused Weight
    w_fused = w_orig * s_bn
    
    # Quantize Fused Weight
    # We need the scale used by the quantizer.
    # In our implementation, we use s_target = s_original * s_bn
    q_fn = model.qconv.quan_w_fn
    s_original = q_fn.get_scale(8, detach=True).view(-1, 1, 1, 1)
    s_target = s_original * s_bn.abs()
    
    print(f"Stats: s_bn range [{s_bn.min():.4f}, {s_bn.max():.4f}]")
    print(f"Stats: s_original range [{s_original.min():.4f}, {s_original.max():.4f}]")
    print(f"Stats: s_target range [{s_target.min():.4f}, {s_target.max():.4f}]")
    
    print(f"Stats: s_target range [{s_target.min():.4f}, {s_target.max():.4f}]")
    
    # Check NO FAULT Consistency first
    # IMPORTANT: Disable fault injection temporarily to verify BN folding math alone
    injector.enable_in_inference = False
    
    # Proposed (No Fault): BN(Conv(Q_a(x), Q(W)))
    # Note: wrapper will skip injection but still execute orig_fn
    # IMPORTANT: Must quantize activation to match what model.forward() does
    x_a = model.qconv.quan_a_fn(x, 8, is_activation=True)
    y_no_fault_proposed = model.bn(torch.nn.functional.conv2d(x_a, model.qconv.quan_w_fn(w_orig, 8, scale=s_original), padding=1))
    
    # Ghost (No Fault): Conv(x, Q(W_fused)) + bias_folded
    code_fused = torch.round(w_fused / s_target)
    thd_neg = -(1 << 7)
    thd_pos = (1 << 7) - 1
    code_fused = torch.clamp(code_fused, thd_neg, thd_pos)
    code_fused = torch.clamp(code_fused, thd_neg, thd_pos)
    q_fused = code_fused * s_target
    
    print(f"[DEBUG TEST] w_fused sum={w_fused.sum().item():.4f}", file=sys.stderr, flush=True)
    print(f"[DEBUG TEST] q_fused sum={q_fused.sum().item():.4f}", file=sys.stderr, flush=True)

    y_no_fault_ghost = torch.nn.functional.conv2d(x_a, q_fused, padding=1) + bn_bias_fold.view(1, -1, 1, 1)
    
    diff_no_fault = (y_no_fault_proposed - y_no_fault_ghost).abs()
    print(f"NO FAULT Max Diff: {diff_no_fault.max():.8f}")
    if diff_no_fault.max() > 1e-5:
        print("CRITICAL WARNING: Basic BN Folding logic has discrepancy even without faults!")
        print("This likely means Q(W)*s_bn != Q(W*s_bn) or definition of s_target is wrong.")
    
    # Re-enable for Fault Injection Test
    injector.enable_in_inference = True
    
    # Replicate Fault Injection on Fused Weight
    # We use the SAME seed logic to ensure same bit flips
    # The injector combines base seed + layer name hash.
    # We need to call the internal _inject method to exactly replicate it or relies on determinism.
    
    # Let's use the public injection method if possible, or replicate the call
    # Q(W_fused)
    # code_fused already computed above
    
    # Inject Faults
    # To get exact same mask, we need to know the seed used by injector for this layer
    # FaultInjector uses: seed + hash(layer_name)
    # We can't easily access the internal seed loop state if it was random, 
    # but we set a fixed seed=42 and enabled inference mode, so it should be deterministic.
    
    import hashlib
    layer_name = "qconv" # We need the exact name used in named_modules...
    # In our script, the model structure is simple. 
    # Let's find the name
    for name, m in model.named_modules():
        if m is model.qconv:
            layer_name = name
            break
            
    layer_hash = int(hashlib.md5(layer_name.encode()).hexdigest()[:8], 16) % (2**31)
    seed = 42 + layer_hash
    print(f"[DEBUG] Test Script using seed {seed} for layer {layer_name}", flush=True)
    
    # Manually inject
    # We'll use the injector's static method to ensure identical logic
    q_fused_faulted = injector._inject_on_quantized_tensor(q_fused, 8, s_target, layer_name=layer_name, forward_seed=seed)
    
    print(f"[DEBUG TEST] q_fused_faulted sum={q_fused_faulted.sum().item():.4f}", file=sys.stderr, flush=True)
    
    delta_fused = q_fused_faulted - q_fused
    delta_unfused = delta_fused / (s_bn + 1e-8 * torch.sign(s_bn))
    print(f"[DEBUG TEST] delta_fused sum={delta_fused.sum().item():.4f}", file=sys.stderr, flush=True)
    print(f"[DEBUG TEST] delta_unfused sum={delta_unfused.sum().item():.4f}", file=sys.stderr, flush=True)

    # 7. Compute Ghost Output
    
    # 7. Compute Ghost Output
    # Conv with faulted fused weights
    # y_ghost_conv = Conv(x, w_eff) = x * w_eff
    # Since we folded BN into W, the only thing remaining from BN is the bias part (and mean subtraction part which is folded into bias)
    # Actually: y = (Conv(x, w) - mean) / std * gamma + beta
    # y = Conv(x, w * gamma/std) - mean * gamma/std + beta
    # y = Conv(x, w_fused) + bias_fused
    
    y_ghost = torch.nn.functional.conv2d(x_a, q_fused_faulted, padding=1) + bn_bias_fold.view(1, -1, 1, 1)
    
    # 8. Compare
    print(f"Proposed Output Shape: {y_proposed.shape}")
    print(f"Ghost Path Output Shape: {y_ghost.shape}")
    
    diff = (y_proposed - y_ghost).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Difference: {max_diff:.8f}")
    print(f"Mean Difference: {mean_diff:.8f}")
    
    # DEBUG: Check if Proposed path behaves linearly as expected
    prop_diff = (y_proposed - y_no_fault_proposed)
    
    # Expected prop_diff = Conv(x, delta_unfused) scaled by BN
    # = Conv(x, delta_unfused) * s_bn + 0 (bias cancels)
    # = Conv(x, delta_unfused * s_bn)
    # = Conv(x, delta_fused)
    
    expected_prop_diff = torch.nn.functional.conv2d(x_a, delta_fused, padding=1)
    
    linearity_check = (prop_diff - expected_prop_diff).abs().max().item()
    print(f"[DEBUG Check] Linearity check (Proposed - NoFault vs Conv(DeltaFused)): {linearity_check:.8f}")
    if linearity_check > 1e-4:
        print("CRITICAL: Proposed path does not follow linear BN logic!")
        print(f"Prop Diff Sum: {prop_diff.sum().item()}")
        print(f"Expected Diff Sum: {expected_prop_diff.sum().item()}")
    
    if max_diff < 1e-5:
        print("SUCCESS: The proposed BN-folding aware injection matches the theoretical fused injection!")
    else:
        print("FAILURE: There is a significant discrepancy.")
        # Debugging
        # Check standard BN calculation
        # y_orig = model.qconv.quan_w_fn(w_orig, 8)
        # out_orig = torch.nn.functional.conv2d(x, y_orig, padding=1)
        # out_bn = (out_orig - running_mean.view(1,-1,1,1)) / std.view(1,-1,1,1) * gamma.view(1,-1,1,1) + beta.view(1,-1,1,1)
        # out_fold = torch.nn.functional.conv2d(x, y_orig * s_bn, padding=1) + bn_bias_fold.view(1,-1,1,1)
        # print("Check BN Folding Math (No Fault):", (out_bn - out_fold).abs().max().item())

if __name__ == "__main__":
    test_bn_folding_consistency()
