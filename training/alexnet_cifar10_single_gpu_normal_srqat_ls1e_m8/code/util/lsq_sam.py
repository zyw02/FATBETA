"""
LSQ_SAM: Sharpness-Aware Minimization for Quantized Models
基于BSAM，适配量化模型的SAM实现
"""
import torch
from collections import defaultdict
from quan.func import QuanConv2d, QuanLinear


class LSQ_SAM(torch.optim.Optimizer):
    """
    LSQ_SAM: SAM for Quantized Models with Discrete Perturbation
    
    关键特性：
    1. 只对量化权重添加离散扰动，不扰动scale
    2. Scale的梯度自然累积
    3. 支持双优化器（optimizer和optimizer_q）
    """
    def __init__(
        self,
        optimizer,      # 主优化器（权重）
        optimizer_q,    # 量化参数优化器（scale）
        model,          # 模型（用于访问量化权重和scale）
        rho_max_sharp=0.5,
        rho_min_sharp=0.5,
        rho_scheduler=None,
        adaptive=False,
        discrete_perturbation=True,
        **kwargs
    ):
        # 初始化父类（使用optimizer的参数）
        defaults = dict(
            rho_max_sharp=rho_max_sharp,
            rho_min_sharp=rho_min_sharp,
            adaptive=adaptive,
            discrete_perturbation=discrete_perturbation,
            **kwargs
        )
        super(LSQ_SAM, self).__init__(optimizer.param_groups, defaults)
        
        self.base_optimizer = optimizer
        self.optimizer_q = optimizer_q
        self.model = model
        self.rho_scheduler = rho_scheduler
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho_max_sharp = rho_max_sharp
        self.rho_min_sharp = rho_min_sharp
        self.adaptive = adaptive
        self.discrete_perturbation = discrete_perturbation
        
        # 状态变量
        self.max_norm = 0
        self.sgd_norm = 0
        self.min_norm = 0
        
        # 保存固定的scale（用于计算离散扰动）
        self.frozen_scales = {}
        
        # 更新rho（如果提供了scheduler）
        if self.rho_scheduler is not None:
            self.update_rho_t()
    
    @torch.no_grad()
    def update_rho_t(self):
        """更新rho_min_sharp（如果提供了scheduler）"""
        if self.rho_scheduler is not None:
            self.rho_min_sharp = self.rho_scheduler.step()
        return self.rho_min_sharp
    
    def _get_quantized_weight_grad_norm(self):
        """
        计算量化权重的梯度范数（不包括scale）
        """
        grads = []
        for module in self.model.modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if hasattr(module, 'quantized_weight') and module.quantized_weight is not None:
                    if module.quantized_weight.grad is not None:
                        grad = module.quantized_weight.grad
                        if self.adaptive and hasattr(module, 'quantized_weight'):
                            # 自适应权重
                            weight_adaptive = torch.abs(module.quantized_weight.data)
                            grad = weight_adaptive * grad
                        grads.append(grad.norm(p=2))
        
        if len(grads) == 0:
            return torch.tensor(1.0, device=next(self.model.parameters()).device)
        
        return torch.norm(torch.stack(grads), p=2)
    
    @torch.no_grad()
    def to_max_point(self, zero_grad=False):
        """
        SAM的ascent step：对量化权重添加扰动
        
        关键：
        1. 保存scale（用于计算离散扰动）
        2. 只对量化权重添加扰动，不扰动scale
        3. 只清零权重梯度，不清零scale梯度（让scale梯度累积）
        """
        # 保存scale（用于计算离散扰动）
        self.frozen_scales = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                # 获取bit-width
                if hasattr(module, 'bits') and module.bits is not None:
                    wbits = module.bits[0] if isinstance(module.bits, (list, tuple)) else module.bits
                elif hasattr(module, 'fixed_bits') and module.fixed_bits is not None:
                    wbits = module.fixed_bits[0] if isinstance(module.fixed_bits, (list, tuple)) else module.fixed_bits
                else:
                    continue
                
                if isinstance(wbits, torch.Tensor):
                    wbits = int(wbits.item())
                else:
                    wbits = int(wbits)
                
                # 获取scale（detach，避免梯度影响）
                try:
                    current_w_scale = module.quan_w_fn.get_scale(wbits, detach=True)
                    self.frozen_scales[module] = {
                        'w_scale': current_w_scale.clone(),
                        'wbits': wbits,
                    }
                except Exception as e:
                    # 如果获取scale失败，跳过这个模块
                    continue
        
        # 计算梯度范数（只计算量化权重的梯度）
        self.sgd_norm = self._get_quantized_weight_grad_norm()
        scale_factor = self.rho_max_sharp / (self.sgd_norm + 1e-12)
        
        # 对量化权重添加扰动
        for name, module in self.model.named_modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if module not in self.frozen_scales:
                    continue
                
                if not hasattr(module, 'quantized_weight') or module.quantized_weight is None:
                    continue
                
                qw = module.quantized_weight
                if qw.grad is None:
                    continue
                
                # 保存原始量化权重
                if qw not in self.state:
                    self.state[qw] = {}
                self.state[qw]["old_p"] = qw.data.clone()
                self.state[qw]["sgd_g"] = qw.grad.clone()
                
                # 计算扰动
                frozen_info = self.frozen_scales[module]
                frozen_w_scale = frozen_info['w_scale']
                grad = qw.grad
                
                if self.discrete_perturbation:
                    # 离散扰动：基于量化步长
                    if frozen_w_scale.dim() > 0:
                        # per-channel scale，需要broadcast
                        frozen_w_scale = frozen_w_scale.view(-1, *([1] * (grad.dim() - 1)))
                    
                    # 计算离散扰动
                    # rho_discrete = rho / scale（转换为量化步数）
                    rho_discrete = self.rho_max_sharp / frozen_w_scale
                    scale_factor_discrete = rho_discrete / (self.sgd_norm + 1e-12)
                    
                    # 计算需要多少个量化步
                    num_steps = grad * scale_factor_discrete / frozen_w_scale
                    # 向上取整，确保至少一个步长
                    num_steps_discrete = torch.ceil(torch.abs(num_steps)) * torch.sign(num_steps)
                    
                    # 离散扰动 = 步数 * 量化步长
                    epsilon = num_steps_discrete * frozen_w_scale
                    
                    if self.adaptive:
                        epsilon *= torch.pow(torch.abs(qw.data), 2)
                else:
                    # 连续扰动（标准SAM）
                    epsilon = grad * scale_factor.to(grad.device)
                    if self.adaptive:
                        epsilon *= torch.pow(torch.abs(qw.data), 2)
                
                # 保存epsilon，在forward中使用
                module.epsilon = epsilon
        
        # ⭐ 只清零权重梯度，不清零scale梯度（让scale梯度累积）
        self.base_optimizer.zero_grad()
        # 不清零optimizer_q的梯度！
        # if self.optimizer_q is not None:
        #     self.optimizer_q.zero_grad()  # 不清零！
    
    @torch.no_grad()
    def to_min_point(self, zero_grad=False):
        """
        SAM的descent step：恢复量化权重，使用第二次梯度更新
        
        关键：
        1. 恢复量化权重（移除扰动）
        2. 执行optimizer.step()和optimizer_q.step()
        3. Scale使用累积的梯度更新
        """
        # 计算第二次forward的梯度范数
        self.max_norm = self._get_quantized_weight_grad_norm()
        
        # 恢复量化权重
        for module in self.model.modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if hasattr(module, 'quantized_weight') and module.quantized_weight is not None:
                    qw = module.quantized_weight
                    if qw in self.state and "old_p" in self.state[qw]:
                        qw.data = self.state[qw]["old_p"]
                    # 清除epsilon
                    if hasattr(module, 'epsilon'):
                        module.epsilon = None
        
        # 执行优化器step
        self.base_optimizer.step()
        if self.optimizer_q is not None:
            # ⭐ Scale使用累积的梯度更新（grad_scale_1 + grad_scale_2）
            self.optimizer_q.step()
        
        # 清零所有梯度
        if zero_grad:
            self.base_optimizer.zero_grad()
            if self.optimizer_q is not None:
                self.optimizer_q.zero_grad()
        
        # 清除保存的scale
        self.frozen_scales.clear()
    
    @torch.no_grad()
    def opt_max_min_step(self, epoch, zero_grad=False):
        """
        BSAM的opt_max_min_step：组合两次梯度
        
        注意：这个函数可能不需要，因为我们已经使用标准的SAM流程
        但为了兼容BSAM的接口，保留这个函数
        """
        self.min_norm = self._get_quantized_weight_grad_norm()
        
        # 组合梯度（BSAM的方式）
        for module in self.model.modules():
            if isinstance(module, (QuanConv2d, QuanLinear)):
                if hasattr(module, 'quantized_weight') and module.quantized_weight is not None:
                    qw = module.quantized_weight
                    if qw.grad is None:
                        continue
                    
                    if qw in self.state:
                        # 恢复原始权重
                        qw.data = self.state[qw]["old_p"]
                        
                        # 组合梯度（BSAM的方式）
                        if "max_g" in self.state[qw] and "sgd_g" in self.state[qw]:
                            max_g = self.state[qw]["max_g"]
                            sgd_g = self.state[qw]["sgd_g"]
                            qw.grad = max_g + (sgd_g - (self.max_norm / self.min_norm) * qw.grad)
        
        # 执行优化器step
        self.base_optimizer.step()
        if self.optimizer_q is not None:
            self.optimizer_q.step()
        
        if zero_grad:
            self.base_optimizer.zero_grad()
            if self.optimizer_q is not None:
                self.optimizer_q.zero_grad()
        
        # 清除状态
        self.frozen_scales.clear()
    
    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

