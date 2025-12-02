"""
V9 输出修正器：流形门控 + Logits统计 + 方向/幅度分解 + BER自适应尺度。

核心思想：
1. 使用能量/概率流形（B+A）建立 WM-ID 门控，动态匹配不同BER下的异常阈值；
2. 提取 logits 的排序/统计/熵/z-score 等特征，捕获“异常形态”；
3. 通过方向二值监督 + 相对幅度回归预测 Δlogits，结合BER分桶统计做尺度重整；
4. 训练/推理完全一致，仅依赖故障logits+激活能量，不依赖target即可执行纠错。
"""

from typing import Dict, List, Optional
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_layer_energy(activations: torch.Tensor) -> torch.Tensor:
    """提取单层激活值的能量（L2 范数）。"""
    batch_size = activations.shape[0]
    activations_flat = activations.view(batch_size, -1)
    return torch.norm(activations_flat, p=2, dim=1, keepdim=True)


def compute_energy_features(activations_list: List[torch.Tensor]) -> torch.Tensor:
    """计算所有层的能量特征，返回 shape=[batch_size, num_layers]。"""
    if len(activations_list) == 0:
        raise ValueError("activations_list不能为空")
    energies = [extract_layer_energy(act) for act in activations_list]
    return torch.cat(energies, dim=1)


def compute_probability_features(logits: torch.Tensor) -> torch.Tensor:
    """返回 softmax 概率。"""
    return F.softmax(logits, dim=1)


class OutputCorrector(nn.Module):
    """V9 输出修正器：自监督相对修正 + BER 自适应尺度。"""

    def __init__(
        self,
        num_classes: int = 10,
        num_layers: int = 8,
        wmid_tau: float = 2.0,
        wmid_beta: float = 2.0,
        ema_decay: float = 0.99,
        logits_feature_dim: int = 32,
        fusion_hidden_dim: int = 128,
        max_correction: float = 3.0,
        gap_stats_momentum: float = 0.95,
        direction_deadzone: float = 0.05,
        anchor_ber: float = 2e-2,
        ber_bucket_centers: Optional[List[float]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.wmid_tau = float(wmid_tau)
        self.wmid_beta = float(wmid_beta)
        self.ema_decay = float(ema_decay)
        self.direction_deadzone = float(direction_deadzone)
        self.gap_stats_momentum = float(gap_stats_momentum)
        self.anchor_ber = float(anchor_ber)

        if ber_bucket_centers is not None:
            if isinstance(ber_bucket_centers, (list, tuple)):
                bucket_centers = [float(x) for x in ber_bucket_centers]
            else:
                bucket_centers = [float(ber_bucket_centers)]
        else:
            bucket_centers = [0.0, 2e-2, 3.5e-2, 5e-2, 8e-2]
        bucket_tensor = torch.tensor(bucket_centers, dtype=torch.float)
        self.num_ber_buckets = len(bucket_centers)
        self.register_buffer('ber_bucket_centers', bucket_tensor)

        self.logits_stats_dim = 10
        self.logits_feature_dim = logits_feature_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.max_correction = float(max_correction)

        self.register_buffer('e_c', torch.zeros(num_classes, num_layers))
        self.register_buffer('Var_e', torch.ones(num_layers))
        self.register_buffer('p_c', torch.zeros(num_classes, num_classes))
        self.register_buffer('Var_p', torch.ones(num_classes))
        self.register_buffer('baseline_initialized', torch.tensor(False))

        self.register_buffer('gap_mean', torch.zeros(self.num_ber_buckets))
        self.register_buffer('gap_var', torch.ones(self.num_ber_buckets))
        self.register_buffer('gap_count', torch.zeros(self.num_ber_buckets))
        self.register_buffer('wmid_mean', torch.zeros(self.num_ber_buckets))
        self.register_buffer('wmid_var', torch.ones(self.num_ber_buckets))
        self.register_buffer('wmid_count', torch.zeros(self.num_ber_buckets))

        self.runtime_context: Dict[str, float] = {'ber': 0.0, 'stage': 'eval', 'update_stats': False}

        self.logits_feature_extractor = nn.Sequential(
            nn.Linear(self.logits_stats_dim, logits_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(logits_feature_dim, logits_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.manifold_feature_extractor = nn.Sequential(
            nn.Linear(num_layers + num_classes, logits_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.direction_head = nn.Sequential(
            nn.Linear(logits_feature_dim * 2, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, num_classes),
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(logits_feature_dim * 2, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------ #
    # Baseline 更新 / 校准
    # ------------------------------------------------------------------ #
    def calibrate_from_samples(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 500,
        device: torch.device = torch.device('cuda'),
    ):
        """从测试集采样少量样本以初始化流形基准。"""
        logger = logging.getLogger(__name__)
        logger.info(f"🔧 Calibrating corrector from {num_samples} test samples...")

        model.eval()
        samples_collected = 0
        all_logits = []
        all_activations = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                if samples_collected >= num_samples:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)

                from util.qat import set_forward_hook_for_conv_linear_layers, remove_hook_for_quantized_layers
                current_activations = []
                hooks = set_forward_hook_for_conv_linear_layers(model, current_activations)
                logits = model(inputs)
                remove_hook_for_quantized_layers(hooks)

                remaining = num_samples - samples_collected
                batch_size = min(inputs.size(0), remaining)

                all_logits.append(logits[:batch_size])
                all_activations.append([act[:batch_size].clone() for act in current_activations])
                all_targets.append(targets[:batch_size])
                samples_collected += batch_size

        if samples_collected == 0:
            logger.warning("Calibration samples empty, skip.")
            return

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        num_layers = len(all_activations[0])
        merged_acts = []
        for layer_idx in range(num_layers):
            layer_acts = [batch[layer_idx] for batch in all_activations]
            merged_acts.append(torch.cat(layer_acts, dim=0))

        p = compute_probability_features(all_logits)
        e = compute_energy_features(merged_acts)

        self.baseline_initialized = torch.tensor(False, device=device)
        for c in range(self.num_classes):
            mask = (all_targets == c)
            if mask.any():
                self.p_c[c] = p[mask].mean(dim=0)
                self.e_c[c] = e[mask].mean(dim=0)

        self.Var_p = p.var(dim=0, unbiased=False).clamp(min=1e-6)
        self.Var_e = e.var(dim=0, unbiased=False).clamp(min=1e-6)
        self.baseline_initialized = torch.tensor(True, device=device)
        logger.info("✓ Corrector calibration completed")

    def update_baseline(
        self,
        logits: torch.Tensor,
        activations: List[torch.Tensor],
        targets: torch.Tensor,
    ):
        """在训练阶段用 EMA 更新类条件原型。"""
        with torch.no_grad():
            p = compute_probability_features(logits)
            if len(activations) > 0:
                e = compute_energy_features(activations)
            else:
                e = torch.zeros(logits.size(0), self.num_layers, device=logits.device, dtype=logits.dtype)

            if not self.baseline_initialized:
                for c in targets.unique().tolist():
                    mask = (targets == c)
                    if mask.any():
                        self.p_c[c] = p[mask].mean(dim=0)
                        self.e_c[c] = e[mask].mean(dim=0)
                self.Var_p = p.var(dim=0, unbiased=False).clamp(min=1e-6)
                self.Var_e = e.var(dim=0, unbiased=False).clamp(min=1e-6)
                self.baseline_initialized = torch.tensor(True, device=logits.device)
            else:
                for c in targets.unique().tolist():
                    mask = (targets == c)
                    if mask.any():
                        self.p_c[c].mul_(self.ema_decay).add_(p[mask].mean(dim=0) * (1 - self.ema_decay))
                        self.e_c[c].mul_(self.ema_decay).add_(e[mask].mean(dim=0) * (1 - self.ema_decay))

                self.Var_p.mul_(self.ema_decay).add_(p.var(dim=0, unbiased=False).clamp(min=1e-6) * (1 - self.ema_decay))
                self.Var_e.mul_(self.ema_decay).add_(e.var(dim=0, unbiased=False).clamp(min=1e-6) * (1 - self.ema_decay))

    # ------------------------------------------------------------------ #
    # 特征提取
    # ------------------------------------------------------------------ #
    def _extract_logits_statistics(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)

        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        centered = logits - mean
        z_scores = centered / std

        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        topk = sorted_logits[:, :3]
        top1 = topk[:, 0:1]
        top2 = topk[:, 1:2]

        features = torch.cat(
            [
                topk,
                top1 - top2,
                top1 - mean,
                top2 - mean,
                std,
                entropy,
                z_scores.max(dim=1, keepdim=True)[0],
                z_scores.min(dim=1, keepdim=True)[0],
            ],
            dim=1,
        )
        return features

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        logits: torch.Tensor,
        activations: Optional[List[torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        intermediate_features: Optional[torch.Tensor] = None,
        normal_intermediate_activations: Optional[List[torch.Tensor]] = None,
        faulted_intermediate_activations: Optional[List[torch.Tensor]] = None,
        fault_context: Optional[Dict[str, float]] = None,
        return_details: bool = False,
    ):
        if not self.baseline_initialized:
            logging.getLogger(__name__).info("[CORRECTOR DEBUG] Baseline not initialized, returning original logits")
            return logits

        if activations is None and faulted_intermediate_activations is not None:
            activations = faulted_intermediate_activations

        if activations is None or len(activations) == 0:
            logging.getLogger(__name__).info("[CORRECTOR DEBUG] No activations provided, returning original logits")
            return logits

        p = compute_probability_features(logits)
        e = compute_energy_features(activations)

        class_indices = targets if targets is not None else torch.argmax(logits, dim=1)
        p_c_selected = self.p_c[class_indices]
        e_c_selected = self.e_c[class_indices]

        z_p = (p - p_c_selected) / torch.sqrt(self.Var_p.unsqueeze(0) + 1e-6)
        z_e = (e - e_c_selected) / torch.sqrt(self.Var_e.unsqueeze(0) + 1e-6)

        context = self._get_effective_context(fault_context)
        ber_value = float(context.get('ber', 0.0))
        update_stats = bool(context.get('update_stats', False))
        bucket_idx = self._get_ber_bucket_index(ber_value)

        logits_stats = self._extract_logits_statistics(logits)
        logits_embed = self.logits_feature_extractor(logits_stats)
        manifold_embed = self.manifold_feature_extractor(torch.cat([z_e, z_p], dim=1))

        combined = torch.cat([logits_embed, manifold_embed], dim=1)
        direction_logits = self.direction_head(combined)
        direction = torch.tanh(direction_logits)
        magnitude_logits = self.magnitude_head(combined)
        magnitude_unit = torch.sigmoid(magnitude_logits)
        magnitude = magnitude_unit * self.max_correction
        delta_logits = direction * magnitude

        wmid = torch.norm(z_e, p=2, dim=1, keepdim=True)
        if update_stats:
            self._update_wmid_statistics(wmid.detach(), ber_value)
        wmid_scale = self._compute_wmid_gate(wmid, ber_value)
        gap_scale = self._compute_gap_scale(ber_value, device=logits.device, dtype=logits.dtype)
        final_delta = wmid_scale * gap_scale * delta_logits
        corrected_logits = logits + final_delta

        logger = logging.getLogger(__name__)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count <= 2:
            with torch.no_grad():
                pred_before = logits.argmax(dim=1)
                pred_after = corrected_logits.argmax(dim=1)
                changed = (pred_before != pred_after).sum().item()
                logger.info(
                    f"[CORRECTOR V9 DEBUG Call {self._debug_count}] "
                    f"WM-ID_e: {wmid[0].item():.2f}, "
                    f"wmid_scale: {wmid_scale[0].item():.4f}, "
                    f"gap_scale: {gap_scale.item():.4f}, "
                    f"correction norm: {torch.norm(delta_logits[0]).item():.4f}, "
                    f"logits diff: {torch.norm(final_delta[0]).item():.4f}, "
                    f"predictions changed: {changed}/{logits.size(0)}"
                )
                logger.info(f"  Sample 0 correction vector: {final_delta[0].cpu().numpy()}")
        if return_details:
            details = {
                'direction_logits': direction_logits,
                'direction': direction,
                'magnitude': magnitude,
                'magnitude_unit': magnitude_unit,
                'magnitude_logits': magnitude_logits,
                'wmid': wmid,
                'wmid_scale': wmid_scale,
                'gap_scale': gap_scale,
                'delta_logits': final_delta,
                'ber_bucket': bucket_idx,
                'ber_value': ber_value,
            }
            return corrected_logits, details
        return corrected_logits

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        return self.get_num_parameters()


    def set_runtime_context(self, **kwargs):
        """在推理阶段设置默认上下文（例如BER）。"""
        for key, value in kwargs.items():
            if key == 'ber':
                self.runtime_context[key] = float(value)
            else:
                self.runtime_context[key] = value

    def update_gap_statistics(self, gap_values: torch.Tensor, ber: float):
        """记录 logit_target - logit_pred 的统计量用于尺度补偿。"""
        if gap_values is None or gap_values.numel() == 0:
            return
        with torch.no_grad():
            idx = self._get_ber_bucket_index(ber)
            gap_values = gap_values.detach()
            mean = gap_values.mean()
            var = gap_values.var(unbiased=False)
            momentum = self.gap_stats_momentum
            self.gap_mean[idx] = self.gap_mean[idx] * momentum + mean * (1 - momentum)
            self.gap_var[idx] = self.gap_var[idx] * momentum + var.clamp_min(1e-6) * (1 - momentum)
            self.gap_count[idx] = self.gap_count[idx] + 1

    def _get_effective_context(self, fault_context: Optional[Dict[str, float]]) -> Dict[str, float]:
        context = dict(self.runtime_context)
        if fault_context:
            context.update(fault_context)
        context.setdefault('ber', 0.0)
        return context

    def _get_ber_bucket_index(self, ber: float) -> int:
        if self.ber_bucket_centers.numel() == 0:
            return 0
        distances = torch.abs(self.ber_bucket_centers - float(ber))
        idx = int(torch.argmin(distances).item())
        return idx

    def _update_wmid_statistics(self, values: torch.Tensor, ber: float):
        idx = self._get_ber_bucket_index(ber)
        momentum = self.ema_decay
        with torch.no_grad():
            mean = values.mean()
            var = values.var(unbiased=False)
            self.wmid_mean[idx] = self.wmid_mean[idx] * momentum + mean * (1 - momentum)
            self.wmid_var[idx] = self.wmid_var[idx] * momentum + var.clamp_min(1e-6) * (1 - momentum)
            self.wmid_count[idx] = self.wmid_count[idx] + 1

    def _compute_wmid_gate(self, wmid: torch.Tensor, ber: float) -> torch.Tensor:
        idx = self._get_ber_bucket_index(ber)
        if self.wmid_count[idx] < 5:
            return torch.sigmoid(self.wmid_beta * (wmid - self.wmid_tau))
        mean = self.wmid_mean[idx].to(wmid.device)
        var = self.wmid_var[idx].to(wmid.device)
        std = torch.sqrt(var + 1e-6)
        normalized = (wmid - mean) / (std + 1e-6)
        return torch.sigmoid(self.wmid_beta * normalized)

    def _compute_gap_scale(self, ber: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.gap_var.numel() == 0:
            return torch.ones(1, 1, device=device, dtype=dtype)
        bucket_idx = self._get_ber_bucket_index(ber)
        anchor_idx = self._get_ber_bucket_index(self.anchor_ber)
        if self.gap_count[bucket_idx] < 1 or self.gap_count[anchor_idx] < 1:
            return torch.ones(1, 1, device=device, dtype=dtype)
        bucket_std = torch.sqrt(self.gap_var[bucket_idx] + 1e-6)
        anchor_std = torch.sqrt(self.gap_var[anchor_idx] + 1e-6)
        scale = (bucket_std / anchor_std).clamp(0.5, 3.0)
        return scale.view(1, 1).to(device=device, dtype=dtype)


def create_output_corrector(
    num_classes: int = 10,
    num_layers: int = 8,
    device: Optional[torch.device] = None,
    wmid_tau: float = 2.0,
    wmid_beta: float = 2.0,
    logits_feature_dim: int = 32,
    fusion_hidden_dim: int = 128,
    max_correction: float = 3.0,
    gap_stats_momentum: float = 0.95,
    direction_deadzone: float = 0.05,
    anchor_ber: float = 2e-2,
    ber_bucket_centers: Optional[List[float]] = None,
    **kwargs,
) -> OutputCorrector:
    """创建输出修正器（兼容V6-V9）；忽略旧版本残留参数。"""
    if kwargs:
        logging.getLogger(__name__).debug(f"Ignoring unsupported corrector args: {list(kwargs.keys())}")

    corrector = OutputCorrector(
        num_classes=num_classes,
        num_layers=num_layers,
        wmid_tau=wmid_tau,
        wmid_beta=wmid_beta,
        logits_feature_dim=logits_feature_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        max_correction=max_correction,
        gap_stats_momentum=gap_stats_momentum,
        direction_deadzone=direction_deadzone,
        anchor_ber=anchor_ber,
        ber_bucket_centers=ber_bucket_centers,
    )
    if device is not None:
        corrector = corrector.to(device)
    return corrector

