"""Helpers.

Helper classes and functions.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import torch
import typing
from torch import nn
from torch import optim
from torch.nn import init as nn_init

from aidesign_gan.libs import optims as libs_optims

_Adam = optim.Adam
_BatchNorm = nn.BatchNorm2d
_Conv = nn.Conv2d
_DataParallel = nn.DataParallel
_Module = nn.Module
_normal_inplace = nn_init.normal_
_Optimizer = optim.Optimizer
_PredAdam = libs_optims.PredAdam
_Tensor = torch.Tensor
_torch_device = torch.device
_torch_full = torch.full
_torch_load = torch.load
_torch_save = torch.save
_Union = typing.Union


def load_model(loc, model):
    """Loads the state dict from a location to a model.

    Args:
        loc: a location
        model: a model
    """
    loc = str(loc)
    model: _Module = model

    model.load_state_dict(_torch_load(loc))


def save_model(model, loc):
    """Saves the state dict from a model to a location.

    Args:
        model: a model
        loc: a location
    """
    model: _Module = model
    loc = str(loc)

    file = open(loc, "w+")
    _torch_save(model.state_dict(), loc)
    file.close()


def load_optim(loc, optim):
    """Loads the state dict from a location to an optimizer.

    Args:
        loc: a location
        optim: an optimizer
    """
    loc = str(loc)
    optim: _Optimizer = optim

    optim.load_state_dict(_torch_load(loc))


def save_optim(optim, loc):
    """Saves the state dict from an optimizer to a location.

    Args:
        optim: an optimizer
        loc: a location
    """
    optim: _Optimizer = optim
    loc = str(loc)

    file = open(loc, "w+")
    _torch_save(optim.state_dict(), loc)
    file.close()


def find_model_sizes(model):
    """Finds the total size and training size of a model.

    Args:
        model: a model

    Returns:
        result: a tuple that contains the following items
        size, : total size
        training_size: training size
    """
    model: _Module = model

    training = model.training
    size = 0
    training_size = 0

    for param in model.parameters():
        param_size = param.numel()
        size += param_size

        if training and param.requires_grad:
            training_size += param_size
    # end for

    result = size, training_size
    return result


def paral_model(model, device, gpu_count):
    """Finds the parallelized model with the given args.

    If the GPU count is 0 or 1, or the GPUs do not support CUDA, this function returns the original model.

    Args:
        model: a model
        device: a device to use
        gpu_count: number of GPUs to use

    Returns:
        model: the parallelized/original model
    """
    model: _Module = model
    device: _torch_device = device
    gpu_count = int(gpu_count)

    if device.type == "cuda" and gpu_count > 1:
        model = _DataParallel(model, list(range(gpu_count)))

    return model


def setup_adam(model, config):
    """Sets up an Adam optimizer with the given args.

    Args:
        model: a model
        config: an adam_optimizer config dict

    Returns:
        adam: the Adam optimizer
    """
    model: _Module = model
    config = dict(config)

    params = model.parameters()

    lr = config["learning_rate"]
    lr = float(lr)

    beta1 = config["beta1"]
    beta1 = float(beta1)

    beta2 = config["beta2"]
    beta2 = float(beta2)

    adam = _Adam(params, lr=lr, betas=(beta1, beta2))
    return adam


def setup_pred_adam(model, config):
    """Sets up a predictive Adam optimizer with the given args.

    Args:
        model: a model
        config: an adam_optimizer config dict

    Returns:
        pred_adam: the predictive Adam optimizer
    """
    model: _Module = model
    config = dict(config)

    params = model.parameters()

    lr = config["learning_rate"]
    lr = float(lr)

    beta1 = config["beta1"]
    beta1 = float(beta1)

    beta2 = config["beta2"]
    beta2 = float(beta2)

    pred_factor_key = "pred_factor"

    if pred_factor_key in config:
        pred_factor = config[pred_factor_key]
        pred_factor = float(pred_factor)

        pred_adam = _PredAdam(params, lr=lr, betas=(beta1, beta2), pred_factor=pred_factor)
    else:  # elif pred_factor_key not in config:
        pred_adam = _PredAdam(params, lr=lr, betas=(beta1, beta2))
    # end if

    return pred_adam


def find_params_init_func(config=None):
    """Finds the parameters initialization function with the given args.

    Args:
        config: a params_init config or None

    Returns:
        result_func: the resulting parameters initialization function
    """
    cw_mean = float(0)
    cw_std = 0.02

    bnw_mean = float(1)
    bnw_std = 0.02
    bnb_mean = float(0)
    bnb_std = 0.0002

    if config is not None:
        config = dict(config)

        cw_mean = float(config["conv"]["weight_mean"])
        cw_std = float(config["conv"]["weight_std"])

        bnw_mean = float(config["batch_norm"]["weight_mean"])
        bnw_std = float(config["batch_norm"]["weight_std"])
        bnb_mean = float(config["batch_norm"]["bias_mean"])
        bnb_std = float(config["batch_norm"]["bias_std"])
    # end if

    def result_func(model):
        """Initializes model parameters.

        Args:
            model: the model
        """
        model: _Union[_Module, _Conv, _BatchNorm] = model

        class_name = str(model.__class__.__name__)

        if class_name.find("Conv") != -1:
            _normal_inplace(model.weight.data, cw_mean, cw_std)
        elif class_name.find("BatchNorm") != -1:
            _normal_inplace(model.weight.data, bnw_mean, bnw_std)
            _normal_inplace(model.bias.data, bnb_mean, bnb_std)
        # end if
    # end def

    return result_func


def prep_batch_and_labels(batch, label, device):
    """Prepares batch and labels with the given args.

    Args:
        batch: a batch
        label: a target label
        device: a device to use

    Returns:
        result: a tuple that contains the following items
        batch, : the prepared batch
        labels : the prepared labels
    """
    batch: _Tensor = batch
    label = float(label)
    device: _torch_device = device

    batch = batch.to(device)
    label_size = (batch.size(0),)
    labels = _torch_full(label_size, label, dtype=torch.float, device=device)

    result = batch, labels
    return result
