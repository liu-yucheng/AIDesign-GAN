"""Module of the optims (optimizers).

==== References ====
PyTorch Adam source code. https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
Yadav, et al., 2018. Stabilizing Adversarial Nets With Prediction Methods. https://openreview.net/pdf?id=Skj8Kag0Z
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from torch import optim
import math
import torch


class PredAdam(optim.Optimizer):
    """Predictive Adam optimizer.

    An Adam optimizer with the predict and restore extra functions.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, pred_factor=1):
        """ Inits self with the given args.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay (L2 penalty)
            amsgrad: whether to use the AMSGrad variant of this algorithm
            pred_factor: prediction factor
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

        self.pred_factor = float(pred_factor)
        """Prediction factor."""

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: a closure that reevaluates the model and returns the loss
        """
        # print("step called")  # Debug

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                grad = param.grad.data
                state = self.state[param]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Add a restore point
                    state["restore_point"] = param.data.clone()

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                beta1, beta2 = group["betas"]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(param.data, group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                if group["amsgrad"]:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1
                param.data.addcdiv_(exp_avg, denom, value=-step_size)
            # end for
        # end for
        return loss

    def predict(self, closure=None):
        """Performs a predictive optimization step.

        Let the previous state be P1, the current state be P2, and the next state be P3. This function will update the
        state to _P3_, where _P3_ = P2 + (P2 - P1). This _P3_ can be seen as the prediction of P3. This function will
        save P2 to a restore point for later restoration.

        Args:
            closure: a closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                # Find the pred_incr (predictive incrementation)
                pred_incr = param.data.sub(state["restore_point"])
                pred_incr.mul_(self.pred_factor)
                # Save the current state as the restore point
                state["restore_point"].copy_(param.data)
                # Apply the pred_incr to complete the prediction
                param.data.add_(pred_incr)
            # end for
        # end for
        return loss

    def restore(self, closure=None):
        """Performs a prediction restoration optimization step.

        Let the current state be _P3_, and the state at the restore point be P2. This function will restore the state
        to P2.

        Args:
            closure: a closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                param.data.copy_(state["restore_point"])
            # end for
        # end for
        return loss
