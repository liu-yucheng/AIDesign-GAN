"""Modeler."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import typing
from os import path as ospath
from torch import nn
from torch import optim

from aidesign_gan.libs import optims as libs_optims
from aidesign_gan.libs.modelers import helpers

_Adam = optim.Adam
_join = ospath.join
_load_model = helpers.load_model
_load_optim = helpers.load_optim
_Module = nn.Module
_PredAdam = libs_optims.PredAdam
_save_model = helpers.save_model
_save_optim = helpers.save_optim
_Union = typing.Union


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
        """Loss function."""
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
            When eps is 1e-5: 0.231
        """

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
