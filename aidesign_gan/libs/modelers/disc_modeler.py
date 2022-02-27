"""Discriminator modeler."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import torch
from os import path as ospath
from torch import nn

from aidesign_gan.libs import structs
from aidesign_gan.libs.modelers import _helpers
from aidesign_gan.libs.modelers import modeler

_DiscStruct = structs.DiscStruct
_find_model_sizes = _helpers.find_model_sizes
_find_params_init_func = _helpers.find_params_init_func
_join = ospath.join
_logit = torch.logit
_Modeler = modeler.Modeler
_Module = nn.Module
_no_grad = torch.no_grad
_paral_model = _helpers.paral_model
_prep_batch_and_labels = _helpers.prep_batch_and_labels
_setup_pred_adam = _helpers.setup_pred_adam
_tanh = torch.tanh
_Tensor = torch.Tensor


class DiscModeler(_Modeler):
    """Discriminator modeler."""

    def __init__(self, model_path, config, device, gpu_count, loss_func, train=True):
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
            train: Training mode flag.
                Controls whether to setup the optimizer.
        """
        super().__init__(model_path, config, device, gpu_count, loss_func)

        # Setup self.model
        struct = _DiscStruct(self.model_path)
        struct.location = _join(self.model_path, self.config["struct_name"])
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = _paral_model(self.model, self.device, self.gpu_count)

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

        # Setup the self.*_size attributes
        size, training_size = _find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

        # Setup the self.has_* attributes
        self.has_fairness = "fairness" in self.config

    def train(self, batch, label):
        """Trains the model with a batch of data and a target label.

        Set the model to training mode.
        Forward pass the batch.
        Find the loss.
        Backward the loss to find the gradients.
        Return the average output and loss value.
        NOTE: The caller of this function needs to manually call the clear_grads and step_optim functions of self to
            ensure the functioning of the training algorithm.

        Args:
            batch: A batch of data.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            label: A target label.
                Definitely on the CPUs.

        Returns:
            out_mean: Mean(D(batch)).
                The average output of D.
                Definitely on the CPUs.
            loss_val: Loss(D(batch), label).
                The loss of D on the batch.
                Definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        self.model.train(True)

        batch, labels = _prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch)
        output: _Tensor = output.view(-1)
        output = output.float()

        loss: _Tensor = self.loss_func(output, labels)
        loss.backward()

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_and_step(self, batch, label):
        """Trains and steps the model with a batch of data and a target label.

        Set the model to training mode.
        Clear the model gradients.
        Forward pass the batch.
        Find the loss.
        Find the gradients through a backward pass.
        Optimize/Update the model.
        Return the average output and loss value.

        Args:
            batch: A batch of data.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            label: A target label.
                Definitely on the CPUs.

        Returns:
            out_mean: Mean(D(batch)).
                The average output of D.
                Definitely on the CPUs.
            loss_val: Loss(D(batch), label).
                The loss of D on the batch.
                Definitely on the CPUs.

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        self.model.train(True)

        self.model.zero_grad()
        self.optim.zero_grad()

        batch, labels = _prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch)
        output: _Tensor = output.view(-1)
        output = output.float()

        loss: _Tensor = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_fair(self, g_model, real_batch, real_label, fake_noises, fake_label):
        """Fairly trains the model with the given args.

        Set self.model to training mode.
        Set g_model to training mode.
        For the real batch:
            Forward pass the batch to self.model.
            Find the loss on real.
        For the fake batch:
            Forward pass the noises to g_model to get a fake batch.
            Forward pass the fake batch to self.model.
            Find the loss on fake.
        After finding the 2 losses, find the cluster loss.
        Find the total loss.
        Backward the total loss to find the gradients.
        Return the results.
        NOTE: the caller of this function needs to manually call the clear_grads and step_optim functions of self to
            ensure the functioning of the training algorithm.

        Args:
            g_model: A generator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of g_model.
            real_batch: A real batch.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            real_label: A real target label wrt. D.
                Usually be close to 1.
                Definitely on the CPUs.
            fake_noises: Some fake noises to feed g_model.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            fake_label: A fake target label wrt. D.
                Usually be close to 0.
                Definitely on the CPUs.

        Returns:
            results: A tuple that contains the following items.
            dx, : Mean(D(X)).
                The output mean of D on the real batch.
                Definitely on the CPUs.
            dgz, : Mean( D(G(Z)) ).
                The output mean of D on the fake batch.
                Definitely on the CPUs.
            ldr, : Loss(D, X).
                The loss of D on the real batch.
                Definitely on the CPUs.
            ldf, : Loss(D, G(Z)).
                The loss of D on the fake batch.
                Definitely on the CPUs.
            ldcr, : Loss(D, Cluster, X).
                = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dxs) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            ldcf, : Loss(D, Cluster, G(Z)).
                = 50 + 50 * tanh(wmm_factor * Mean( logit(dgzs) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            ld: Loss(D).
                = dx_factor * ldr + dgz_factor * ldf + cluster_dx_factor * ldcr + cluster_dgz_factor * ldcf.
                Clamped to range [0, 100].
                Definitely on the CPUs.

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        g_model: _Module = g_model
        fake_noises: _Tensor = fake_noises

        model_training = self.model.training
        g_model_training = g_model.training
        self.model.train(True)
        g_model.train(True)

        real_batch, real_labels = _prep_batch_and_labels(real_batch, real_label, self.device)
        dxs = self.model(real_batch)
        dxs: _Tensor = dxs.view(-1)
        dxs = dxs.float()
        ldr: _Tensor = self.loss_func(dxs, real_labels)

        fake_noises = fake_noises.to(self.device)
        fake_batch: _Tensor = g_model(fake_noises)
        fake_batch = fake_batch.float()
        fake_batch, fake_labels = _prep_batch_and_labels(fake_batch, fake_label, self.device)
        dgzs = self.model(fake_batch)
        dgzs: _Tensor = dgzs.view(-1)
        dgzs = dgzs.float()
        ldf: _Tensor = self.loss_func(dgzs, fake_labels)

        logit_dxs = _logit(dxs, eps=self.eps)
        logit_dgzs = _logit(dgzs, eps=self.eps)
        ldcr = 50 + 50 * _tanh(self.wmm_factor * -1 * logit_dxs.mean())
        ldcf = 50 + 50 * _tanh(self.wmm_factor * logit_dgzs.mean())

        if self.has_fairness:
            config = self.config["fairness"]
            dx_factor = config["dx_factor"]
            dgz_factor = config["dgz_factor"]
            cluster_dx_factor = config["cluster_dx_factor"]
            cluster_dgz_factor = config["cluster_dgz_factor"]
        else:  # elif not self.has_fairness
            dx_factor = 0.5
            dgz_factor = 0.5
            cluster_dx_factor = float(0)
            cluster_dgz_factor = float(0)
        # end if

        ld: _Tensor = \
            dx_factor * ldr + \
            dgz_factor * ldf + \
            cluster_dx_factor * ldcr + \
            cluster_dgz_factor * ldcf

        ld.clamp_(0, 100)
        ld.backward()

        self.model.train(model_training)
        g_model.train(g_model_training)

        dx = dxs.mean()
        dgz = dgzs.mean()

        dx = dx.item()
        dgz = dgz.item()
        ldr = ldr.item()
        ldf = ldf.item()
        ldcr = ldcr.item()
        ldcf = ldcf.item()
        ld = ld.item()

        result = (
            dx, dgz,
            ldr, ldf,
            ldcr, ldcf,
            ld
        )

        return result

    def valid(self, batch, label):
        """Validates the model with a batch of data and a target label.

        Set the model to evaluation mode.
        Forward pass the batch.
        Find the loss.
        Return the average output and loss.

        Args:
            batch: A batch of data.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            label: A target label.
                Definitely on the CPUs.

        Returns:
            out_mean: Mean(D(batch)).
                The average output of D.
                Definitely on the CPUs.
            loss_val: Loss(D(batch), label).
                The loss of D on the batch.
                Definitely on the CPUs.
        """
        model_training = self.model.training
        self.model.train(False)

        batch, labels = _prep_batch_and_labels(batch, label, self.device)

        with _no_grad():
            output = self.model(batch)
            output: _Tensor = output.detach().view(-1)

        output = output.float()

        loss: _Tensor = self.loss_func(output, labels)

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def valid_fair(self, g_model, real_batch, real_label, fake_noises, fake_label):
        """Fairly validates the model with the given args.

        Set self.model and g_model to evaluation mode.
        For the real batch:
            Forward pass the batch to self.model.
            Find the loss on real.
        For the fake batch:
            Forward pass the noises to g_model to get a fake batch.
            Forward pass the fake batch to self.model.
            Find the loss on fake.
        After finding the 2 losses, find the cluster loss.
        Find the total loss.
        Return the results.

        Args:
            g_model: A generator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of g_model.
            real_batch: A real batch.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            real_label: A real target label wrt. D.
                Usually be close to 1.
                Definitely on the CPUs.
            fake_noises: Some fake noises.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            fake_label: A fake target label wrt. D.
                Usually be close to 0.
                Definitely on the CPUs.

        Returns:
            result: A tuple that contains the following items.
            dx, : Mean(D(X)).
                The output mean of D on real.
                Definitely on the CPUs.
            dgz, : Mean( D(G(Z)) ).
                The output mean of D on fake.
                Definitely on the CPUs.
            ldr, : Loss(D, X).
                The loss of D on real.
                Definitely on the CPUs.
            ldf, : Loss(D, G(Z)).
                The loss of D on fake.
                Definitely on the CPUs.
            ldcr, : Loss(D, Cluster, X).
                = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dxs) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            ldcf, : Loss(D, Cluster, G(Z)).
                = 50 + 50 * tanh(wmm_factor * Mean( logit(dgzs) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            ld: Loss(D).
                = dx_factor * ldr + dgz_factor * ldf + cluster_dx_factor * ldcr + cluster_dgz_factor * ldcf.
                Clamped to range [0, 100].
                Definitely on the CPUs.
        """
        g_model: _Module = g_model
        fake_noises: _Tensor = fake_noises

        model_training = self.model.training
        g_model_training = g_model.training
        self.model.train(False)
        g_model.train(False)

        real_batch, real_labels = _prep_batch_and_labels(real_batch, real_label, self.device)

        with _no_grad():
            dxs = self.model(real_batch)
            dxs: _Tensor = dxs.detach().view(-1)

        dxs = dxs.float()
        ldr: _Tensor = self.loss_func(dxs, real_labels)

        fake_noises = fake_noises.to(self.device)

        with _no_grad():
            fake_batch = g_model(fake_noises)
            fake_batch: _Tensor = fake_batch.detach()

        fake_batch = fake_batch.float()
        fake_batch, fake_labels = _prep_batch_and_labels(fake_batch, fake_label, self.device)

        with _no_grad():
            dgzs = self.model(fake_batch)
            dgzs: _Tensor = dgzs.detach().view(-1)

        dgzs = dgzs.float()
        ldf: _Tensor = self.loss_func(dgzs, fake_labels)

        logit_dxs = _logit(dxs, eps=self.eps)
        logit_dgzs = _logit(dgzs, eps=self.eps)
        ldcr = 50 + 50 * _tanh(self.wmm_factor * -1 * logit_dxs.mean())
        ldcf = 50 + 50 * _tanh(self.wmm_factor * logit_dgzs.mean())

        if self.has_fairness:
            config = self.config["fairness"]
            dx_factor = config["dx_factor"]
            dgz_factor = config["dgz_factor"]
            cluster_dx_factor = config["cluster_dx_factor"]
            cluster_dgz_factor = config["cluster_dgz_factor"]
        else:  # elif not self.has_fairness
            dx_factor = 0.5
            dgz_factor = 0.5
            cluster_dx_factor = float(0)
            cluster_dgz_factor = float(0)
        # end if

        ld: _Tensor = \
            dx_factor * ldr + \
            dgz_factor * ldf + \
            cluster_dx_factor * ldcr + \
            cluster_dgz_factor * ldcf

        ld.clamp_(0, 100)

        self.model.train(model_training)
        g_model.train(g_model_training)

        dx = dxs.mean()
        dgz = dgzs.mean()

        dx = dx.item()
        dgz = dgz.item()
        ldr = ldr.item()
        ldf = ldf.item()
        ldcr = ldcr.item()
        ldcf = ldcf.item()
        ld = ld.item()

        result = (
            dx, dgz,
            ldr, ldf,
            ldcr, ldcf,
            ld
        )

        return result

    def test(self, batch):
        """Tests/Uses the model with a batch of data.

        Set the model to evaluation mode.
        Forward pass the batch.
        Return the output.

        Args:
            batch: A batch of data.
            Can be on either the CPUs or GPUs.

        Returns:
            output: D(batch).
                The output of D (NOT the average output).
                Definitely on the CPUs.
        """
        batch: _Tensor = batch

        model_training = self.model.training
        self.model.train(False)

        batch = batch.to(self.device)

        with _no_grad():
            output = self.model(batch)
            output: _Tensor = output.detach().view(-1)

        output = output.float()

        self.model.train(model_training)

        output = output.cpu()
        return output
