import torch
from collections import defaultdict
from quan.func import QuanConv2d, QuanLinear


class CubeSAM(torch.optim.Optimizer):
    def __init__(self, params, model, base_optimizer, rho, adaptive=False, **kwargs):

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(CubeSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho = rho # rho_max_sharp设定扰动半径的上限
        self.adaptive = False
        self.model = model
        self.nstate = defaultdict(dict)


    '''to_max_point方法负责将模型参数沿着梯度方向扰动到一个"最大损失点" '''
    @torch.no_grad()
    def ascent(self, zero_grad=True):
        for n, m in self.model.named_modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                p = m.quan_w_fn.qw
                sc = m.quan_w_fn.s[0]
                e_w = p.grad.sign()*sc*self.rho.to(p)
                m.epsilon = e_w
                # 使能epsilon
                m.add_epsilon = True

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def descent(self, epoch, zero_grad=True):
        for n, m in self.model.named_modules():
            if isinstance(m, (QuanConv2d, QuanLinear)):
                m.add_epsilon = False
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups