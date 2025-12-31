import torch
import torch.nn as nn
from typing import Dict, List
from quan.func import QuanConv2d, QuanLinear


class SensitiveActivationCollector:
    def __init__(self, model, sensitive_info: Dict[str, Dict[str, List[int]]]):
        self.model = model.module if hasattr(model, "module") else model
        self.sensitive_info = sensitive_info
        self.handles = []
        self.buffers = {}
        self._register_hooks()

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        for name in self.sensitive_info.keys():
            if name not in modules:
                continue

            def make_hook(key):
                def hook(module, input, output):
                    self.buffers[key] = output.detach()
                return hook

            handle = modules[name].register_forward_hook(make_hook(name))
            self.handles.append(handle)

    def clear(self):
        self.buffers.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def build_feature_vector(self, device):
        feats = []
        for name, info in self.sensitive_info.items():
            if name not in self.buffers:
                continue
            activations = self.buffers[name]
            idx = info["indices"]
            if len(idx) == 0:
                continue
            act_sel = activations[:, idx]
            if act_sel.dim() == 4:
                energy = act_sel.pow(2).mean(dim=[2, 3])
            else:
                energy = act_sel.pow(2)
            mean = info["mean"].to(device).unsqueeze(0)
            std = info["std"].to(device).unsqueeze(0)
            normalized = (energy - mean) / (std + 1e-6)
            feats.append(normalized)
        if not feats:
            return None
        feature_vec = torch.cat(feats, dim=1)
        return feature_vec


class SensitiveChannelRestorer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.restorer = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, logits, features):
        embed = self.feature_proj(features)
        gate = self.detector(embed)
        augmented = torch.cat([embed, logits], dim=1)
        delta = self.restorer(augmented)
        restored = logits + gate * delta
        return restored, gate

