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
_Callable = typing.Callable
_find_model_sizes = _helpers.find_model_sizes
_find_params_init_func = _helpers.find_params_init_func
_find_params_noising_func = _helpers.find_params_noising_func
_join = ospath.join
_load_model = _helpers.load_model
_load_optim = _helpers.load_optim
_Module = nn.Module
_PredAdam = libs_optims.PredAdam
_save_model = _helpers.save_model
_save_optim = _helpers.save_optim
_setup_pred_adam = _helpers.setup_pred_adam
_Union = typing.Union

# End


class Modeler:
    """Modeler base class."""

    def __init__(self, model_path, config, device, gpu_count, loss_func):
        """Inits self with the given args.

        Args:
            model_path: a model path
            config: a modelers config
            device: Device to use.
                Will be the GPUs if they are available.
            gpu_count: Number of GPUs to use.
                0 means no GPU available.
                >= 1 means some GPUs available.
            loss_func: a loss function
        """
        self.model_path = model_path
        """Model path."""
        self.config = config
        """Discriminator / Generator modelers config subconfig."""
        self.device = device
        """Device to use, will be the GPUs if they are available."""
        self.gpu_count = gpu_count
        """Number of GPUs to use, >= 1 if GPUs are available."""
        self.loss_func = loss_func
        """Loss function used to find the classic losses."""
        self.model: _Union[_Module, None] = None
        """Model, a pytorch nn module, definitely runs on GPUs if they are available."""
        self.optim: _Union[_PredAdam, _Adam, None] = None
        """Optimizer, can run on GPUs if they are available."""
        self.size = None
        """Total size of the model."""
        self.training_size = None
        """Training size of the model, 0 if the model is not initialized to the training mode."""
        self.has_fairness = None
        """Whether the modeler has a fairness config."""
        self.eps = 1e-5
        """Epsilon, a small dummy value used to avoid NaNs in computation.

        Typically used to replace the result of 1 / inf.
        """
        self.wmm_factor = 0.231
        """Wasserstein metric mean factor.

        Used to time the w_metric_mean before feeding it to the tanh function.
        Makes the w_metric_mean value more sensible to the tanh function.
        Possible values:
            The target is to estimate the minimum wmm_factor for an given eps_value, so that:
                tanh(tensor(wmm_factor) * logit(tensor(), eps=eps_value)) > tensor(0.9900)
            When eps is 1e-5, this factor, rounded to its 3rd digit, is 0.231.
                tanh(tensor(0.231) * logit(tensor(1), eps=1e-5)) == tensor(0.9902).
        """

        self._noise_func: _Union[_Callable, None] = None
        """Model noise function."""

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
            self.model.train(True)

        self.model.train(train)

        # Setup self._noise_func
        params_noising_key = "params_noising"

        if params_noising_key in self.config:
            self._noise_func = _find_params_noising_func(self.config[params_noising_key])
        else:
            self._noise_func = _find_params_noising_func()
        # end if

        # Setup the self.*_size attributes
        size, training_size = _find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

        # Setup the self.has_* attributes
        self.has_fairness = "fairness" in self.config

    def load(self):
        """Loads the model and optimizer states."""
        state_location = _join(self.model_path, self.config["state_name"])
        optim_location = _join(self.model_path, self.config["optim_name"])

        try:
            _load_model(state_location, self.model)
        except FileNotFoundError:
            _save_model(self.model, state_location)

        if self.optim is not None:
            try:
                _load_optim(optim_location, self.optim)
            except FileNotFoundError:
                _save_optim(self.optim, optim_location)
        # end if

    def save(self):
        """Saves the model and optimizer states."""
        state_location = _join(self.model_path, self.config["state_name"])
        optim_location = _join(self.model_path, self.config["optim_name"])
        _save_model(self.model, state_location)

        if self.optim is not None:
            _save_optim(self.optim, optim_location)

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
        if self._noise_func is None:
            raise ValueError("self._noise_func cannot be None")

        self.model.apply(self._noise_func)
