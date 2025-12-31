import torch



class BSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho_max_sharp, rho_min_sharp, rho_scheduler, adaptive=False, **kwargs):

        defaults = dict(rho_max_sharp=rho_max_sharp, adaptive=adaptive, **kwargs)
        super(BSAM, self).__init__(params, defaults)

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
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["sgd_g"] = p.grad.clone()
                self.state[p]["old_p"] = p.data.clone()
                # 计算扰动量e_w，等于梯度p.grad乘以缩放因子scale, scale.to(p)确保scale的设备和数据类型与参数p一致
                # e_w表示沿着梯度方向的扰动，扰动大小由rho_max_sharp和梯度范数决定
                e_w = p.grad * scale.to(p)

                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                # 将扰动e_w加到参数p上(p.add_是就地加法操作)
                # 即p = p+e_w
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def to_min_point(self, zero_grad=False):
        self.max_norm = self._grad_norm(weight_adaptive=self.adaptive)# self._grad_norm()
        scale = self.rho_min_sharp / (self.sgd_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # 此时的p.grad是在最大损失点(to_max_point)扰动后计算的梯度。保存它可能用于后续基于最大损失点梯度的参数更新
                self.state[p]["max_g"] = p.grad.clone()
                # 将参数p的值恢复为扰动前的原始参数值(self.state[p]["old_p"])，由to_max_point保存
                # SAM需要在最大损失点计算梯度后，恢复到原始参数，以便基于最大损失点的梯度更新原始参数
                p.data = self.state[p]["old_p"]
                # 计算最小扰动
                e_w = self.state[p]["sgd_g"] * scale.to(p)

                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                # 从参数p中减去扰动e_w
                p.sub_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def opt_max_min_step(self, epoch, zero_grad=False):
        # 此时的norm是在完成to_min_point后的梯度范数
        self.min_norm = self._grad_norm(weight_adaptive=self.adaptive)# self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # 将参数恢复为扰动前的初始值，SAM的最终更新基于原始参数，而不是to_max_point或to_min_point，因此需要恢复到原始点
                p.data = self.state[p]["old_p"]
                # 构造新的梯度 p.grad为执行完to_min_point后的梯度
                p.grad = (self.state[p]["max_g"]) + (self.state[p]["sgd_g"] - (self.max_norm / self.min_norm) * p.grad)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
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