"""Optimizers.

Based on the "prediction" concept in [2] and the code in [5].

NOTE: The [*] reference list is in AIDesign-GAN's main README.
"""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import math
import torch
from torch import optim

_Optimizer = optim.Optimizer
_sqrt = math.sqrt
_Tensor = torch.Tensor
_torch_maximum = torch.maximum
_zeros_like = torch.zeros_like


class PredAdam(_Optimizer):
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

        Returns:
            loss: the loss, if a non-None closure argument gets passed
        """
        # print("step called")  # Debug
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                param: _Tensor = param

                if param.grad is None:
                    continue

                if param.grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                grad = param.grad.data
                state = self.state[param]

                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = _zeros_like(param, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = _zeros_like(param, memory_format=torch.preserve_format)

                    if group["amsgrad"]:
                        # Maintain the maximum of all exponential moving average of square gradient values
                        state["max_exp_avg_sq"] = _zeros_like(param, memory_format=torch.preserve_format)

                    # Add a restore point
                    state["restore_point"] = param.data.clone()
                # end if

                state["step"] += 1

                exp_avg: _Tensor = state["exp_avg"]
                exp_avg_sq: _Tensor = state["exp_avg_sq"]

                if group["amsgrad"]:
                    max_exp_avg_sq: _Tensor = state["max_exp_avg_sq"]

                beta1, beta2 = group["betas"]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(param.data, group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                if group["amsgrad"]:
                    # Maintain the maximum of all second moment running average until now
                    _torch_maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the maximum for normalizing running average of gradient
                    denom = (max_exp_avg_sq.sqrt() / _sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / _sqrt(bias_correction2)).add_(group["eps"])
                # end if

                step_size = group["lr"] / bias_correction1
                param.data.addcdiv_(exp_avg, denom, value=-step_size)
            # end for
        # end for

        return loss

    def predict(self, closure=None):
        """Performs a predictive optimization step.

        Let the previous state be S1, the current state be S2, and the next state be S3.
        This function will update the state to S3*, where S3* = S2 + (S2 - S1) = 2 * S2 - S1.
        This S3* can be seen as a predictor of S3.
        This function will save S2 to a restore point for later restoration.

        Args:
            closure: a closure that reevaluates the model and returns the loss

        Returns:
            loss: the loss, if a non-None closure argument gets passed
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                param: _Tensor = param

                if param.grad is None:
                    continue

                state = self.state[param]
                # Find the pred_incr (predictive incrementation)
                restore_point: _Tensor = state["restore_point"]
                pred_incr = param.data.sub(restore_point)
                pred_incr.mul_(self.pred_factor)
                # Save the current state as the restore point
                restore_point.copy_(param.data)
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

        Returns:
            loss: the loss, if a non-None closure argument gets passed
        """
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                param: _Tensor = param

                if param.grad is None:
                    continue

                state = self.state[param]
                param.data.copy_(state["restore_point"])
            # end for
        # end for

        return loss
