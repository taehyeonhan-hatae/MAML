import torch
import contextlib
from torch.distributed import ReduceOp


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho_scheduler, rho=0.0005, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):

        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer#(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho_scheduler = rho_scheduler
        self.rho_t = rho
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, zero_grad=False):

        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:

            scale = self.rho_t / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue

                # w에 대한 gradient를 저장
                self.state[p]["old_g"] = p.grad.data.clone()

                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)

                # perturb_weights
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def unperturb(self):
        # get back to "w" from "w + e(w)"
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, balance, zero_grad=False):

        self.unperturb() # get back to "w" from "w + e(w)"

        # # calculate inner product
        # inner_prod = 0.0
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None: continue
        #         inner_prod += torch.sum(
        #             self.state[p]['old_g'] * p.grad.data
        #         )
        #
        # # get norm
        # new_grad_norm = self._grad_norm()
        # old_grad_norm = self._grad_norm(by='old_g')
        #
        # # get cosine
        # cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)
        #
        # # gradient decomposition
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None: continue
        #         vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (
        #                     new_grad_norm + self.perturb_eps)
        #         p.grad.data.add_(vertical, alpha=-alpha)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                ## self.state[p]['old_g']가 w에 대한 gradient이고
                ## p.grad는 w + e(w)에서의 gradient이다 이다

                # 어느게 맞을까?
                p.grad = (1 - balance) * self.state[p]["old_g"] + balance * p.grad
                # p.grad = balance * self.state[p]["old_p_grad"] + (1 - balance) * p.grad

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.perturb_weights(zero_grad=True)
        closure()
        self.gradient_decompose()

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

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()