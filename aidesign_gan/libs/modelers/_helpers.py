"""Helpers.

Helper classes and functions.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import os
import torch
import typing
from os import path as ospath
from torch import jit
from torch import nn
from torch import onnx
from torch import optim
from torch.nn import init as nn_init

from aidesign_gan.libs import optims as libs_optims

_Adam = optim.Adam
_BatchNorm = nn.BatchNorm2d
_Conv = nn.Conv2d
_DataParallel = nn.DataParallel
_dirname = ospath.dirname
_exists = ospath.exists
_jit_load = jit.load
_jit_save = jit.save
_jit_script = jit.script
_join = ospath.join
_makedirs = os.makedirs
_Module = nn.Module
_normal_inplace = nn_init.normal_
_onnx_export = onnx.export
_Optimizer = optim.Optimizer
_PredAdam = libs_optims.PredAdam
_relpath = ospath.relpath
_ScriptModule = torch.ScriptModule
_Tensor = torch.Tensor
_torch_device = torch.device
_torch_full = torch.full
_torch_load = torch.load
_torch_randn = torch.randn
_torch_save = torch.save
_torch_zeros_like = torch.zeros_like
_Union = typing.Union


def load_state_dict(loc, obj, save_dev=None, run_dev=None, optim=False):
    """Loads the state dict from a location to an object.

    This function works as the following.
    If not optim, moves the given object to the saving device.
    Loads the object state dict from the location.
    If not optim, moves the given object to the running device.

    Args:
        loc: a location
        obj: an object that implements the obj.load_state_dict method
        save_dev: an optional saving device
        run_dev: an optional running device
        optim: an optional optim flag marking if the obj is an Optimizer

    Raises:
        ValueError: when optim is False, but either save_device or run_device is None
    """
    loc = str(loc)
    obj: _Union[_Module, _Optimizer] = obj
    save_dev: _torch_device = save_dev
    run_dev: _torch_device = run_dev
    optim = bool(optim)

    if optim is False:
        if (save_dev is None) or (run_dev is None):
            raise ValueError("Both save_dev and run_dev need to be non-None when optim is False")
        # end if
    # end if

    if not optim:
        obj.to(save_dev)

    obj.load_state_dict(_torch_load(loc))

    if not optim:
        obj.to(run_dev)
    # end if


def save_state_dict(obj, loc, save_dev=None, run_dev=None, optim=False):
    """Saves the state dict from an object to a location.

    This function works as the following.
    If not optim, moves the given object to the saving device.
    Saves the object state dict to the location.
    If not optim, moves the given object to the running device.

    Args:
        obj: an object that implements the obj.state_dict method
        loc: a location
        save_dev: an optional saving device
        run_dev: an optional running device
        optim: an optional optim flag marking if the obj is an Optimizer

    Raises:
        ValueError: when optim is False, but either save_device or run_device is None
    """
    obj: _Union[_Module, _Optimizer] = obj
    loc = str(loc)
    save_dev: _torch_device = save_dev
    run_dev: _torch_device = run_dev
    optim = bool(optim)

    if optim is False:
        if (save_dev is None) or (run_dev is None):
            raise ValueError("Both save_dev and run_dev need to be non-None when optim is False")
        # end if
    # end if

    loc_parent = _dirname(loc)

    if not _exists(loc_parent):
        _makedirs(loc_parent)

    if not optim:
        obj.to(save_dev)

    file = open(loc, "w+")
    _torch_save(obj.state_dict(), loc)
    file.close()

    if not optim:
        obj.to(run_dev)
    # end if


def load_torch_script(loc, save_dev, run_dev):
    """Loads the TorchScript from a location.

    This function works as the following.
    Loads the result from the location.
    Moves the result to the saving device.
    Moves the result to the running device.
    Returns the result.

    Args:
        loc: a location
        save_dev: a saving device
        run_dev: a running device

    Returns:
        result: a ScriptModule or ScriptFunction
    """
    loc = str(loc)
    save_dev: _torch_device = save_dev
    run_dev: _torch_device = run_dev

    result: _Union[_ScriptModule, _Module] = _jit_load(loc)
    result.to(save_dev)
    result.to(run_dev)
    return result


def save_torch_script(obj, loc, input_shape, save_dev, run_dev):
    """Saves the TorchScript compiled from an object to a location.

    This function works as the following.
    Moves the given object to the saving device.
    Compiles the given object to JIT TorchScript.
    Saves the compiled script to the location.
    Moves the given object to the running device.

    Args:
        obj: an object that is TorchScript compilable
        loc: a location
        input_shape: an input shape
        save_dev: a saving device
        run_dev: a running device
    """
    obj: _Module = obj
    loc = str(loc)
    input_shape: list[int] = input_shape
    save_dev: _torch_device = save_dev
    run_dev: _torch_device = run_dev

    loc_parent = _dirname(loc)

    if not _exists(loc_parent):
        _makedirs(loc_parent)

    obj.to(save_dev)
    dummy_input = _torch_randn(*input_shape, device=save_dev)
    script_inputs = [(dummy_input, )]
    script: _ScriptModule = _jit_script(obj=obj, example_inputs=script_inputs)
    file = open(loc, "w+")
    _jit_save(script, loc)
    file.close()
    obj.to(run_dev)


def save_onnx(obj, loc, input_shape, save_dev, run_dev):
    """Saves an object in the ONNX format to a location.

    This function works as the following.
    Moves the given object to the saving device.
    Compiles the given object to JIT TorchScript.
    Exports the compiled script to ONNX and saves the exportation to the location.
    Moves the given object to the running device.

    Args:
        obj: an object that is TorchScript and ONNX compilable
        loc: a location
        input_shape: an input shape
        save_dev: a saving device
        run_dev: a running device
    """
    obj: _Module = obj
    loc = str(loc)
    input_shape: list[int] = input_shape
    save_dev: _torch_device = save_dev
    run_dev: _torch_device = run_dev

    loc_parent = _dirname(loc)

    if not _exists(loc_parent):
        _makedirs(loc_parent)

    obj.to(save_dev)
    dummy_input = _torch_randn(*input_shape, device=save_dev)
    script_inputs = [(dummy_input, )]
    script: _ScriptModule = _jit_script(obj=obj, example_inputs=script_inputs)
    onnx_args = (dummy_input, )
    file = open(loc, "w+")

    # NOTE: 12 is the highest ONNX opset version that produces no information, warning, or error at compile time.
    # NOTE: The above situations might change when we receive future PyTorch patches and updates.
    _onnx_export(model=script, args=onnx_args, f=loc, opset_version=12)

    file.close()
    obj.to(run_dev)


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
    config: dict = config

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
    config: dict = config

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
        config: dict = config

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


def find_params_noising_func(config=None):
    """Finds the parameters noising function with the given args.

    Args:
        config: a params_noising config or None

    Returns:
        result_func: the resulting parameters noising function
    """
    cdw_mean = float(0)
    cdw_std = 0.0002

    bndw_mean = float(0)
    bndw_std = 0.0002
    bndb_mean = float(0)
    bndb_std = 2e-6

    if config is not None:
        config: dict = config

        cdw_mean = float(config["conv"]["delta_weight_mean"])
        cdw_std = float(config["conv"]["delta_weight_std"])

        bndw_mean = float(config["batch_norm"]["delta_weight_mean"])
        bndw_std = float(config["batch_norm"]["delta_weight_std"])
        bndb_mean = float(config["batch_norm"]["delta_bias_mean"])
        bndb_std = float(config["batch_norm"]["delta_bias_std"])
    # end if

    def result_func(model):
        """Noises model parameters.

        Args:
            model: the model
        """
        model: _Union[_Module, _Conv, _BatchNorm] = model

        class_name = str(model.__class__.__name__)

        if class_name.find("Conv") != -1:
            dws = _torch_zeros_like(model.weight.data)
            _normal_inplace(dws, cdw_mean, cdw_std)
            model.weight.data.add_(dws)
        elif class_name.find("BatchNorm") != -1:
            dws = _torch_zeros_like(model.weight.data)
            _normal_inplace(dws, bndw_mean, bndw_std)
            model.weight.data.add_(dws)

            dbs = _torch_zeros_like(model.bias.data)
            _normal_inplace(dbs, bndb_mean, bndb_std)
            model.bias.data.add_(dbs)
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
    label_size = (batch.size()[0],)
    labels = _torch_full(label_size, label, dtype=torch.float, device=device)

    result = batch, labels
    return result


def find_fairness_factors(config=None):
    """Finds the fairness factors with the given args.

    Args:
        config: a fairness config or None.

    Returns:
        results: a tuple that contains the following items
        dx_fac, : D(X) factor
        dgz_fac, : D(G(Z)) factor
        clust_dx_fac: cluster D(X) factor
        clust_dgz_fac: cluster D(G(Z)) factor
        clust_dx_oa_slope: cluster D(X) overact slope
        clust_dgz_oa_slope: cluster D(G(Z)) overact slope
    """
    if config is not None:
        dx_fac = float(config["dx_factor"])
        dgz_fac = float(config["dgz_factor"])
        clust_dx_fac = float(config["cluster_dx_factor"])
        clust_dgz_fac = float(config["cluster_dgz_factor"])

        clust_dx_oa_slope_key = "cluster_dx_overact_slope"

        if clust_dx_oa_slope_key in config:
            clust_dx_oa_slope = float(config[clust_dx_oa_slope_key])
        else:
            clust_dx_oa_slope = float(1)
        # end if

        clust_dgz_oa_slope_key = "cluster_dgz_overact_slope"

        if clust_dgz_oa_slope_key in config:
            clust_dgz_oa_slope = float(config[clust_dgz_oa_slope_key])
        else:
            clust_dgz_oa_slope = float(1)
        # end if
    else:
        dx_fac = 0.5
        dgz_fac = 0.5
        clust_dx_fac = float(0)
        clust_dgz_fac = float(0)
        clust_dx_oa_slope = float(1)
        clust_dgz_oa_slope = float(1)
    # end if

    results = (
        dx_fac, dgz_fac,
        clust_dx_fac, clust_dgz_fac,
        clust_dx_oa_slope, clust_dgz_oa_slope
    )
    return results


def find_save_locs(model_path, config):
    """Finds the model save locs.

    Args:
        model_path: a model path
        config: a modelers subconfig

    Returns:
        results: a tuple of the following items
        state_loc, : the state location
        optim_loc, : the optimizer location
    """
    model_path = str(model_path)
    config: dict = config

    state_name = config["state_name"]
    optim_name = config["optim_name"]

    state_name = _relpath(state_name)
    optim_name = _relpath(optim_name)

    state_loc = _join(model_path, state_name)
    optim_loc = _join(model_path, optim_name)

    results = state_loc, optim_loc
    return results


def find_batch_means(batch):
    """Finds the means for the elements in a batch.

    Args:
        batch: a batch

    Returns
        batch: the batch means
    """
    batch: _Tensor = batch
    batch_dims = [index for index in range(len(batch.shape))]
    reduce_dims = batch_dims[1:]
    batch = batch.mean(dim=reduce_dims)
    return batch
