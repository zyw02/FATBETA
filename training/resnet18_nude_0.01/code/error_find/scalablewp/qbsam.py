import torch
from collections import defaultdict
from quan.func import QuanConv2d, QuanLinear


class QBSAM(torch.optim.Optimizer):
    def __init__(self, params, model, base_optimizer, rho_max_sharp, rho_min_sharp, rho_scheduler, adaptive=False, **kwargs):

        defaults = dict(rho_max_sharp=rho_max_sharp, adaptive=adaptive, **kwargs)
        super(QBSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.rho_scheduler = rho_scheduler # 动态调整扰动半径的调度器
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho_max_sharp = rho_max_sharp # rho_max_sharp设定扰动半径的上限
        self.rho_min_sharp = rho_min_sharp # rho_max_sharp设定扰动半径的下限
        self.update_rho_t()
        self.adaptive = False
        self.max_norm = 0
        self.sgd_norm = 0
        self.min_norm = 0
        self.model = model
        self.nstate = defaultdict(dict)

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_min_sharp = self.rho_scheduler.step()
        return self.rho_min_sharp

    '''to_max_point方法负责将模型参数沿着梯度方向扰动到一个"最大损失点" '''
    @torch.no_grad()
    def to_max_point(self, zero_grad=False):
        # 计算当前梯度的范数(norm)
        self.sgd_norm = self._grad_norm(weight_adaptive=self.adaptive)
        # self.sgd_norm = self._grad_norm()
        # 扰动步长与梯度范数成反比，归一化扰动以确保其大小由rho_max_sharp控制
        scale = self.rho_max_sharp / (self.sgd_norm + 1e-12)
        for n, m in self.model.named_modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                p = m.quan_w_fn.qw
                self.nstate[m]["sgd_g"] = p.grad.clone()
                # self.nstate[m]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p,2)
                m.epsilon = e_w
                # 使能epsilon
                m.add_epsilon = True

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def to_min_point(self, zero_grad=False):
        self.max_norm = self._grad_norm(weight_adaptive=self.adaptive)# self._grad_norm()
        scale = self.rho_min_sharp / (self.sgd_norm + 1e-12)
        for n, m in self.model.named_modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                p = m.quan_w_fn.qw
                self.nstate[m]["max_g"] = p.grad.clone()
                # p.data = self.nstate[m]["old_p"] 这里的p是Q(w)，本身只是模型前向传播过程中留下的中间变量，因此不用保存old_p
                e_w = -self.nstate[m]["sgd_g"] * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p,2)
                m.epsilon = e_w
                # 使能epsilon
                m.add_epsilon = True

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def opt_max_min_step(self, epoch, zero_grad=False):
        # 此时的norm是在完成to_min_point后的梯度范数
        self.min_norm = self._grad_norm(weight_adaptive=self.adaptive)# self._grad_norm()
        for n, m in self.model.named_modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                w = m.weight
                p = m.quan_w_fn.qw
                w.grad = (self.nstate[m]["max_g"]) + (self.nstate[m]["sgd_g"] - (self.max_norm / self.min_norm) * p.grad)
                m.add_epsilon = False
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive=False):
        # put everything on the same device, in case of model parallelism
        shared_device = self.base_optimizer.param_groups[0]["params"][0].device
        wgrads = []
        # for n, m in self.model.named_modules():
        #     if isinstance(m, (QuanConv2d, QuanLinear)):
        #         wgrads.append(torch.norm(m.qw.grad, p=2).to(shared_device))
        for n, m in self.model.named_modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                grad = m.quan_w_fn.qw.grad
                if grad is None:
                    continue

                grad_norm = torch.norm(grad, p=2)

                if weight_adaptive:
                    weight = m.weight
                    weight_norm = torch.norm(weight, p=2).clamp(min=1e-12)
                    adjusted_grad_norm = grad_norm / weight_norm
                    wgrads.append(adjusted_grad_norm.to(shared_device))
                else:
                    wgrads.append(grad_norm.to(shared_device))

        if not wgrads:
            return torch.tensor(0.0, device=shared_device)
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)
        return wgrad_norm
    # @torch.no_grad()
    # def _grad_norm(self, by=None, weight_adaptive=False):
    #     # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    #     if not by:
    #         norm = torch.norm(
    #             torch.stack([
    #                 ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
    #                 for group in self.param_groups for p in group["params"]
    #                 if p.grad is not None
    #             ]),
    #             p=2
    #         )
    #     else:
    #         norm = torch.norm(
    #             torch.stack([
    #                 ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
    #                 for group in self.param_groups for p in group["params"]
    #                 if p.grad is not None
    #             ]),
    #             p=2
    #         )
    #     return norm

    @torch.no_grad()
    def _grad_norm_by(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups