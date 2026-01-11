"""
Fault Injector V2 - High Performance Bitwise Parallel Implementation
完全兼容原版功能，但通过整数空间位运算实现更高的并行加速。
"""
import torch
from .fault_injector import FaultInjector

class FaultInjectorV2(FaultInjector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 只要有一张卡打印即可
        import os
        if os.environ.get('LOCAL_RANK', '0') == '0':
            print("[FaultInjectorV2] Initialized with Bitwise Parallel optimization.")

    def _inject_on_quantized_tensor(
        self, x_q: torch.Tensor, k: int, scale: torch.Tensor, 
        layer_name: str = None, forward_seed: int = None, layer_name_for_stats: str = None
    ) -> torch.Tensor:
        """
        优化后的注入逻辑：直接在整数码空间执行位异或，避免张量展开。
        """
        device = x_q.device if self.device is None else self.device
        
        # 判断是否使用格雷码或OLM编码
        layer_name_str = layer_name if layer_name else ""
        use_gray_code = (len(self.gray_code_layers) > 0 and 
                        layer_name_str in self.gray_code_layers)
        use_olm = (len(self.olm_layers) > 0 and 
                  layer_name_str in self.olm_layers)
        
        # 比例系数准备
        if isinstance(scale, torch.Tensor):
            s = scale.to(device)
            if s.dim() > 0 and s.numel() > 1:
                while s.dim() < x_q.dim(): s = s.unsqueeze(-1)
        else:
            s = torch.tensor(float(scale), device=device, dtype=x_q.dtype)
        
        thd_neg = -(1 << (k - 1))
        thd_pos = (1 << (k - 1)) - 1
        
        # Step 1: 映射到整数码空间 [0, 2^k-1]
        code_f = torch.round(x_q.to(device) / s).clamp(thd_neg, thd_pos)
        code_shifted = (code_f - thd_neg).to(torch.int64)
        
        # Step 2: 编码转换 (Gray/OLM)
        if use_gray_code:
            code_encoded = code_shifted ^ (code_shifted >> 1)
        elif use_olm:
            value_to_code = self.olm_layers[layer_name_str]
            lookup_table = torch.arange((1 << k), dtype=torch.int64, device=device)
            for val, enc in value_to_code.items():
                lookup_table[val - thd_neg] = enc
            code_encoded = lookup_table[code_shifted]
        else:
            code_encoded = code_shifted

        # Step 3: 生成位并行掩码 (mask_integer 的每一位代表该 bit 是否翻转)
        mask_seed = forward_seed if forward_seed is not None else self.seed
        mask_integer = self._generate_bitwise_parallel_mask(
            code_encoded.shape, k, device, layer_name_str, mask_seed
        )

        # Step 4: 核心翻转操作 (Bitwise XOR)
        code_faulted_encoded = code_encoded ^ mask_integer

        # Step 5: 统计 (保持兼容)
        if self.enable_statistics:
            stats_key = layer_name_for_stats or layer_name_str or "unknown"
            # 统计翻转的 bit 数
            flipped_bits_sum = self._count_set_bits(mask_integer, k)
            affected_params_sum = (mask_integer > 0).sum()
            total_bits = mask_integer.numel() * k
            total_params = mask_integer.numel()
            self._pending_stats.append((flipped_bits_sum, total_bits, total_params, stats_key, affected_params_sum))

        # Step 6: 解码回二进制码
        if use_gray_code:
            binary = code_faulted_encoded
            for i in range(1, k):
                binary ^= (code_faulted_encoded >> i)
            code_faulted_shifted = binary
        elif use_olm:
            code_to_value = self.olm_code_to_value[layer_name_str]
            max_code = int(max(code_to_value.keys())) if code_to_value else (1 << k) - 1
            reverse_lookup = torch.arange(max_code + 1, dtype=torch.int64, device=device)
            for enc, val in code_to_value.items():
                reverse_lookup[enc] = val - thd_neg
            code_faulted_shifted = reverse_lookup[code_faulted_encoded.clamp(0, max_code)]
        else:
            code_faulted_shifted = code_faulted_encoded

        # Step 7: 反量化
        x_faulted = (code_faulted_shifted.to(x_q.dtype) + thd_neg) * s
        return x_faulted

    def _generate_bitwise_parallel_mask(self, shape, k, device, layer_name, mask_seed):
        """生成高效的整数位掩码"""
        p = float(self.ber or 0.0)
        
        # 如果是 position_based，回退到旧逻辑
        if self.use_position_based_mask:
            N = 1
            for s in shape: N *= s
            # 获取旧逻辑生成的 [N, k] 掩码并转为整数
            old_mask = super()._generate_flip_mask(N, k, device, layer_name, mask_seed)
            bit_weights = (1 << torch.arange(k, device=device, dtype=torch.int64))
            return (old_mask.to(torch.int64) * bit_weights).sum(-1).view(shape)
        
        generator = torch.Generator(device=device)
        if mask_seed is not None:
            generator.manual_seed(mask_seed)
        
        mask_integer = torch.zeros(shape, dtype=torch.int64, device=device)
        
        # 针对不同模式生成位掩码
        if self.bfat_dual_bit:
            p_msb = self.ber_msb if self.ber_msb is not None else p
            p_s_msb = self.ber_secondary_msb if self.ber_secondary_msb is not None else p
            mask_integer |= (torch.rand(shape, generator=generator, device=device) < p_msb).to(torch.int64) << (k - 1)
            mask_integer |= (torch.rand(shape, generator=generator, device=device) < p_s_msb).to(torch.int64) << (k - 2)
        elif self.bfat_bit_index is not None:
            idx = min(self.bfat_bit_index, k - 1)
            mask_integer |= (torch.rand(shape, generator=generator, device=device) < p).to(torch.int64) << idx
        elif self.only_msb:
            mask_integer |= (torch.rand(shape, generator=generator, device=device) < p).to(torch.int64) << (k - 1)
        else:
            # 通用 BER 模式
            for i in range(k):
                if self.skip_msb and i == k - 1: continue
                if self.skip_msbn and i >= k - 2: continue
                
                prob = p
                if i == k - 1 and self.ber_msb is not None: prob = self.ber_msb
                
                bit_mask = (torch.rand(shape, generator=generator, device=device) < prob).to(torch.int64)
                mask_integer |= (bit_mask << i)
        
        return mask_integer

    def _count_set_bits(self, mask, k):
        """高效统计整数中为1的比特数"""
        if hasattr(mask, 'bit_count'): # Torch 2.1+
            return mask.bit_count().sum()
        # 兼容老版本
        cnt = torch.zeros((), device=mask.device)
        temp = mask
        for _ in range(k):
            cnt += (temp & 1).sum()
            temp = temp >> 1
        return cnt

