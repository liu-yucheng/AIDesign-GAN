"""Modeler."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import torch
import typing
from os import path as ospath
from torch import nn
from torch import optim

from aidesign_gan.libs import optims as libs_optims
from aidesign_gan.libs.modelers import _helpers

# Aliases

_Adam = optim.Adam
_BCELoss = nn.BCELoss
_Callable = typing.Callable
_exists = ospath.exists
_find_model_sizes = _helpers.find_model_sizes
_find_params_init_func = _helpers.find_params_init_func
_find_params_noising_func = _helpers.find_params_noising_func
_find_save_locs = _helpers.find_save_locs
_join = ospath.join
_load_state_dict = _helpers.load_state_dict
_Module = nn.Module
_PredAdam = libs_optims.PredAdam
_save_state_dict = _helpers.save_state_dict
_save_torch_script = _helpers.save_torch_script
_save_onnx = _helpers.save_onnx
_setup_pred_adam = _helpers.setup_pred_adam
_Softsign = nn.Softsign
_torch_device = torch.device
_Union = typing.Union

# End


class Modeler:
    """Modeler base class."""

    def __init__(self, model_path, config, device, gpu_count):
        """Inits self with the given args.

        Args:
            model_path: a model path
            config: a modelers config
            device: Device to use.
                Will be the GPUs if they are available.
            gpu_count: Number of GPUs to use.
                0 means no GPU available.
                >= 1 means some GPUs available.
        """
        self.model_path = model_path
        """Model path."""
        self.config = config
        """Discriminator / Generator modelers config subconfig."""
        self.device = device
        """Device to use, will be the GPUs if they are available."""
        self.gpu_count = gpu_count
        """Number of GPUs to use, >= 1 if GPUs are available."""

        self.bce_loss = _BCELoss()
        """Binary cross entropy loss function."""
        self.softsign = _Softsign()
        """Softsign function."""
        self.torch_cpu = _torch_device("cpu")
        """Torch cpu."""
        self.eps = 1e-6
        """Epsilon, a small dummy value used to avoid NaNs in computation.

        Typically used to replace the result of 1 / inf.
        """
        self.wmm_factor = float(1)
        """Wasserstein metric mean factor.

        Used to control the slopes of the cluster losses.
        Possible values:
            When eps is 1e-6, the factor is 1.0.
                tensor(1.0) * softsign(logit(tensor(0.0), eps=1e-6)) == tensor(-0.9325).
                tensor(1.0) * softsign(logit(tensor(0.49), eps=1e-6)) == tensor(-0.0385).
        """

        self.model: _Union[_Module, None] = None
        """Model, a pytorch nn module, definitely runs on GPUs if they are available."""
        self.input_shape = None
        """Model input shape."""
        self.output_shape = None
        """Model output shape."""
        self.optim: _Union[_PredAdam, _Adam, None] = None
        """Optimizer, can run on GPUs if they are available."""
        self.noise_func: _Union[_Callable, None] = None
        """Model noise function."""
        self.size = None
        """Total size of the model."""
        self.training_size = None
        """Training size of the model, 0 if the model is not initialized to the training mode."""
        self.has_fairness = None
        """Whether the modeler has a fairness config."""

    def _init_after_model_setup(self, train=True):
        """Inits self with the given args after self.model is setup.

        Args:
            train: Training mode switch.
                Controls whether to setup self.optim
        """
        # Setup model parameters
        params_init_key = "params_init"

        if params_init_key in self.config:
            params_init_func = _find_params_init_func(self.config[params_init_key])
        else:
            params_init_func = _find_params_init_func()
        # end if

        self.model.apply(params_init_func)

        # Setup self.optim
        if train:
            self.optim = _setup_pred_adam(self.model, self.config["adam_optimizer"])

        self.model.train(train)

        # Setup self._noise_func
        params_noising_key = "params_noising"

        if params_noising_key in self.config:
            self.noise_func = _find_params_noising_func(self.config[params_noising_key])
        else:
            self.noise_func = _find_params_noising_func()
        # end if

        # Setup the self.*_size attributes
        size, training_size = _find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

        # Setup the self.has_* attributes
        self.has_fairness = "fairness" in self.config

    def load(self):
        """Loads the model and optimizer state dict."""
        locs = _find_save_locs(self.model_path, self.config)
        state_loc, optim_loc = locs

        if _exists(state_loc):
            _load_state_dict(state_loc, self.model, self.torch_cpu, self.device)
        else:
            _save_state_dict(self.model, state_loc, self.torch_cpu, self.device)
        # end if

        if self.optim is not None:
            if _exists(optim_loc):
                _load_state_dict(optim_loc, self.optim, self.torch_cpu, self.device, optim=True)
            else:
                _save_state_dict(self.optim, optim_loc, self.torch_cpu, self.device, optim=True)
            # end if
        # end if

    def save(self):
        """Saves the model and optimizer state dict."""
        locs = _find_save_locs(self.model_path, self.config)
        state_loc, optim_loc = locs
        _save_state_dict(self.model, state_loc, self.torch_cpu, self.device)

        if self.optim is not None:
            _save_state_dict(self.optim, optim_loc, self.torch_cpu, self.device, optim=True)
        # end if

    def export_model(self, to_path, script_name, onnx_name):
        """Exports the model TorchScript and ONNX with the give args.

        Args:
            to_path: a path
            script_name: a script name
            onnx_name: an ONNX name
        """
        to_path = str(to_path)
        script_name = str(script_name)
        onnx_name = str(onnx_name)

        script_loc = _join(to_path, script_name)
        onnx_loc = _join(to_path, onnx_name)

        _save_torch_script(self.model, script_loc, self.input_shape, self.torch_cpu, self.device)
        _save_onnx(self.model, onnx_loc, self.input_shape, self.torch_cpu, self.device)

    def clear_grads(self):
        """Clears the gradients by calling the zero_grad function of self.model and self.optim."""
        self.model.zero_grad()
        self.optim.zero_grad()

    def predict(self):
        """Updates self.model to the predicted state by calling the predict function of self.optim."""
        self.optim.predict()

    def step_optim(self):
        """Updates self.model by calling the step function of self.optim."""
        self.optim.step()

    def restore(self):
        """Restores self.model to the before prediction state by calling the restore function of self.optim."""
        self.optim.restore()

    def apply_noise(self):
        """Applies the noise function self._noise_func to self.model.

        Raises:
            ValueError: if self._noise_func is None
        """
        if self.noise_func is None:
            raise ValueError("self.noise_func cannot be None")

        self.model.apply(self.noise_func)
