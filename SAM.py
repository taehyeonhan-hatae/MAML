import torch
import contextlib
from torch.distributed import ReduceOp


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho_scheduler, alpha, adaptive, rho=0.0005, perturb_eps=1e-12, **kwargs):

        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer#(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho_scheduler = rho_scheduler
        self.rho_t = rho
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.alpha = alpha

        # initialize self.rho_t
        self.update_rho_t()

    @torch.no_grad()
    def update_rho_t(self):

        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()
        for group in self.param_groups:

            scale = (self.rho_t / (grad_norm + self.perturb_eps) - self.alpha)

            for p in group["params"]:
                if p.grad is None: continue

                self.state[p]["old_p"] = p.data.clone()
                # w에 대한 gradient를 저장
                self.state[p]["old_g"] = p.grad.data.clone()

                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)

                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, balance, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                ## self.state[p]['old_g']가 w에 대한 gradient이고
                ## p.grad는 w + e(w)에서의 gradient이다 이다

                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

                ## Penalizing Gradient Norm for Efficiently Improving Generalization in Deep Learning
                p.grad = (1 - balance) * self.state[p]["old_g"] + balance * p.grad

                ## Sharpness-Aware Gradient Matching for Domain Generalization (SAGM)
                # sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                # p.grad.data.add_(sam_grad)


        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups