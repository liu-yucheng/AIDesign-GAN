"""Generator modeler."""

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

_find_fairness_factors = _helpers.find_fairness_factors
_GenStruct = structs.GenStruct
_join = ospath.join
_logit = torch.logit
_Modeler = modeler.Modeler
_Module = nn.Module
_no_grad = torch.no_grad
_paral_model = _helpers.paral_model
_prep_batch_and_labels = _helpers.prep_batch_and_labels
_tanh = torch.tanh
_Tensor = torch.Tensor
_torch_randn = torch.randn


class GenModeler(_Modeler):
    """Generator modeler."""

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
        struct_loc = _join(self.model_path, self.config["struct_name"])
        struct_def = _GenStruct.load(struct_loc)

        exec(struct_def)

        self.model = self.model.to(self.device)
        self.model = _paral_model(self.model, self.device, self.gpu_count)

        self._init_after_model_setup(train)

    def gen_noises(self, count):
        """Generates a random set of input noises for the model.

        Args:
            count: Number of input noises.
                Definitely on the CPUs.

        Returns:
            noises: The generated set of noises.
                Definitely on the CPUs.
        """
        count = int(count)

        zr = self.config["noise_resolution"]
        zc = self.config["noise_channel_count"]

        noises = _torch_randn(count, zc, zr, zr)
        return noises

    def train(self, d_model, noises, label):
        """Trains the model with the given args.

        Set self.model to training mode.
        Set d_model to training mode.
        Generate a training batch with the given noises.
        Forward pass the batch to the discriminator model.
        Find the loss.
        Backward the loss to find the gradients.
        Return the average output and the loss value.
        NOTE: The caller of this function needs to manually call the clear_grads and step_optim functions of self to
            ensure the functioning of the training algorithm.

        Args:
            d_model: A discriminator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of d_model.
            noises: Some noises used to generate the training batch.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            label: A target label.
                Definitely on the CPUs.

        Returns:
            out_mean: Mean(D(G(noises))).
                The average output of d_model.
                Definitely on the CPUs.
            loss_val: Loss(D(G(noises)), label).
                The loss of the model.
                Definitely on the CPUs.

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        d_model: _Module = d_model
        noises: _Tensor = noises

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(True)
        d_model.train(True)

        noises = noises.to(self.device)
        batch = self.model(noises)
        batch: _Tensor = batch.float()

        batch, labels = _prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch)
        output: _Tensor = output.view(-1)
        output = output.float()

        loss: _Tensor = self.loss_func(output, labels)
        loss.backward()

        self.model.train(model_training)
        d_model.train(d_model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_and_step(self, d_model, noises, label):
        """Trains and steps the model with the given args.

        Set self.model to training mode.
        Set d_model to training mode.
        Clear the model gradients.
        Generate a training batch with the given noises.
        Forward pass the batch to the discriminator model.
        Find the loss.
        Find the gradients with a backward pass.
        Optimize/Update the model.
        Return the average output and the loss value.

        Args:
            d_model: A discriminator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of d_model.
            noises: Some noises used to generate the training batch.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            label: A target label.
                Definitely on the CPUs.

        Returns:
            out_mean: Mean( D(G(noises)) ).
                The average output of d_model.
                Definitely on the CPUs.
            loss_val: Loss(D(G(noises)), label).
                The loss of the model.
                Definitely on the CPUs.

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        d_model: _Module = d_model
        noises: _Tensor = noises

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(True)
        d_model.train(True)

        self.model.zero_grad()
        self.optim.zero_grad()

        noises = noises.to(self.device)
        batch = self.model(noises)
        batch: _Tensor = batch.float()

        batch, labels = _prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch)
        output: _Tensor = output.view(-1)
        output = output.float()

        loss: _Tensor = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()

        self.model.train(model_training)
        d_model.train(d_model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_fair(self, d_model, real_data, real_label, fake_noises, fake_label):
        """Fairly trains the model with the given args.

        Set self.model to training mode.
        Set d_model to training mode.
        For the real batch:
            Forward pass the batch to d_model.
            Find the loss on real.
        For the fake batch:
            Forward pass the noises to self.model to get a fake batch.
            Forward pass the fake batch to d_model.
            Find the loss on fake.
        After finding the 2 losses, find the cluster loss.
        Find the total loss.
        Backward the total loss to find the gradients.
        Return the results.
        NOTE: The caller of this function needs to manually call the clear_grads and step_optim functions of self to
            ensure the functioning of the training algorithm.

        Args:
            d_model: A discriminator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of d_model.
            real_data: A batch of real data.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            real_label: A real target label wrt. G.
                Usually be close to 0.
                Definitely on the CPUs.
            fake_noises: Some fake noises.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            fake_label: A fake target label wrt. G.
                Usually be close to 1.
                Definitely on the CPUs.

        Returns:
            results: A tuple that contains the following items.
            dx2, : Mean(D(X)).
                The mean output of D on the real batch.
                Definitely on the CPUs.
            dgz2, : Mean( D(G(Z)) ).
                The mean output of D on the fake batch.
                Definitely on the CPUs.
            lgr, : Loss(G, X).
                The loss of G on the real batch.
                Definitely on the CPUs.
            lgf, : Loss(G, G(Z)).
                The loss of G on the fake batch.
                Definitely on the CPUs.
            lgcr, : Loss(G, Cluster, X).
                = 50 + 50 * tanh(wmm_factor * Mean( logit(dxs2) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            lgcf, : Loss(G, Cluster, G(Z)).
                = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dgzs2) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            lg: Loss(G).
                = dx_factor * lgr + dgz_factor * lgf + cluster_dx_factor * lgcr + cluster_dgz_factor * lgcf.
                Clamped to range [0, 100].
                Definitely on the CPUs.

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        d_model: _Module = d_model
        fake_noises: _Tensor = fake_noises

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(True)
        d_model.train(True)

        real_data, real_labels = _prep_batch_and_labels(real_data, real_label, self.device)
        dxs2 = d_model(real_data)
        dxs2: _Tensor = dxs2.view(-1)
        dxs2 = dxs2.float()
        lgr: _Tensor = self.loss_func(dxs2, real_labels)

        fake_noises = fake_noises.to(self.device)
        fake_batch = self.model(fake_noises)
        fake_batch: _Tensor = fake_batch.float()
        fake_batch, fake_labels = _prep_batch_and_labels(fake_batch, fake_label, self.device)
        dgzs2 = d_model(fake_batch)
        dgzs2: _Tensor = dgzs2.view(-1)
        dgzs2 = dgzs2.float()
        lgf: _Tensor = self.loss_func(dgzs2, fake_labels)

        logit_dxs2 = _logit(dxs2, eps=self.eps)
        logit_dgzs2 = _logit(dgzs2, eps=self.eps)
        lgcr = 50 + 50 * _tanh(self.wmm_factor * logit_dxs2.mean())
        lgcf = 50 + 50 * _tanh(self.wmm_factor * -1 * logit_dgzs2.mean())

        if self.has_fairness:
            config = self.config["fairness"]
            fairness_factors = _find_fairness_factors(config)
        else:  # elif not self.has_fairness
            fairness_factors = _find_fairness_factors()
        # end if

        dx_factor, dgz_factor, cluster_dx_factor, cluster_dgz_factor = fairness_factors

        lg: _Tensor = \
            dx_factor * lgr + \
            dgz_factor * lgf + \
            cluster_dx_factor * lgcr + \
            cluster_dgz_factor * lgcf

        lg.clamp_(0, 100)
        lg.backward()

        self.model.train(model_training)
        d_model.train(d_model_training)

        dx2 = dxs2.mean()
        dgz2 = dgzs2.mean()

        dx2 = dx2.item()
        dgz2 = dgz2.item()
        lgr = lgr.item()
        lgf = lgf.item()
        lgcr = lgcr.item()
        lgcf = lgcf.item()
        lg = lg.item()

        result = (
            dx2, dgz2,
            lgr, lgf,
            lgcr, lgcf,
            lg
        )

        return result

    def valid(self, d_model, noises, label):
        """Validates the model with the given args.

        Set the model to evaluation mode.
        Generate a validation batch with the given noises.
        Forward pass the batch to the discriminator model.
        Find the loss.
        Return the average output and the loss.

        Args:
            d_model: A discriminator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of d_model.
            noises: Some noises used to generate the validation batch.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            label: A target label.
                Definitely on the CPUs.

        Returns:
            out_mean: Mean( D(G(noises)) ).
                The average output of d_model.
                Definitely on the CPUs.
            loss_val: Loss(D(G(noises)), label).
                The loss of the model.
                Definitely on the CPUs.
        """
        d_model: _Module = d_model
        noises: _Tensor = noises

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(False)
        d_model.train(False)

        noises = noises.to(self.device)

        with _no_grad():
            batch = self.model(noises)
            batch: _Tensor = batch.detach()

        batch = batch.float()

        batch, labels = _prep_batch_and_labels(batch, label, self.device)

        with _no_grad():
            output = d_model(batch)
            output: _Tensor = output.detach().view(-1)

        output = output.float()

        loss: _Tensor = self.loss_func(output, labels)

        self.model.train(model_training)
        d_model.train(d_model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def valid_fair(self, d_model, real_data, real_label, fake_noises, fake_label):
        """Fairly validates the model with the given args.

        Set self.model and d_model to evaluation mode.
        For the real batch:
            Forward pass the batch to d_model.
            Find the loss on real.
        For the fake batch:
            Forward pass the noises to self.model to get a fake batch.
            Forward pass the fake batch to d_model.
            Find the loss on fake.
        After finding the 2 losses, find the cluster loss.
        Find the total loss.
        Return the results.

        Args:
            d_model: A discriminator model.
                Can be on either the CPUs or the GPUs.
                Preferred to be on the GPUs.
                This function will not change the device of d_model.
            real_data: A batch of real data.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            real_label: A real target label wrt. G.
                Usually be close to 0.
                Definitely on the CPUs.
            fake_noises: Some fake noises.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.
            fake_label: A fake target label wrt. G.
                Usually be close to 1.
                Definitely on the CPUs.

        Returns:
            dx2, : Mean(D(X)).
                The output mean of D on real.
                Definitely on the CPUs.
            dgz2, : Mean( D(G(Z)) ).
                The output mean of D on fake.
                Definitely on the CPUs.
            lgr, : Loss(G, X).
                The loss of G on real.
                Definitely on the CPUs.
            lgf, : Loss(G, G(Z)).
                The loss of G on fake.
                Definitely on the CPUs.
            lgcr, : Loss(G, Cluster, X).
                = 50 + 50 * tanh(wmm_factor * Mean( logit(dxs2) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            lgcf, : Loss(G, Cluster, G(Z)).
                = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dgzs2) )).
                tanh'ed Wasserstein 1 metric mean based on the WGAN paper.
                Definitely on the CPUs.
            lg: Loss(G).
                = dx_factor * lgr + dgz_factor * lgf + cluster_dx_factor * lgcr + cluster_dgz_factor * lgcf
                clamped to range [0, 100].
                Definitely on the CPUs.
        """
        d_model: _Module = d_model
        fake_noises: _Tensor = fake_noises

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(False)
        d_model.train(False)

        real_data, real_labels = _prep_batch_and_labels(real_data, real_label, self.device)

        with _no_grad():
            dxs2 = d_model(real_data)
            dxs2: _Tensor = dxs2.detach().view(-1)

        dxs2 = dxs2.float()
        lgr: _Tensor = self.loss_func(dxs2, real_labels)

        fake_noises = fake_noises.to(self.device)

        with _no_grad():
            fake_batch = self.model(fake_noises)
            fake_batch: _Tensor = fake_batch.detach()

        fake_batch = fake_batch.float()
        fake_batch, fake_labels = _prep_batch_and_labels(fake_batch, fake_label, self.device)

        with _no_grad():
            dgzs2 = d_model(fake_batch)
            dgzs2: _Tensor = dgzs2.detach().view(-1)

        dgzs2 = dgzs2.float()
        lgf: _Tensor = self.loss_func(dgzs2, fake_labels)

        logit_dxs2 = _logit(dxs2, eps=self.eps)
        logit_dgzs2 = _logit(dgzs2, eps=self.eps)

        lgcr = 50 + 50 * _tanh(self.wmm_factor * logit_dxs2.mean())
        lgcf = 50 + 50 * _tanh(self.wmm_factor * -1 * logit_dgzs2.mean())

        if self.has_fairness:
            config = self.config["fairness"]
            fairness_factors = _find_fairness_factors(config)
        else:  # elif not self.has_fairness
            fairness_factors = _find_fairness_factors()
        # end if

        dx_factor, dgz_factor, cluster_dx_factor, cluster_dgz_factor = fairness_factors

        lg: _Tensor = \
            dx_factor * lgr + \
            dgz_factor * lgf + \
            cluster_dx_factor * lgcr + \
            cluster_dgz_factor * lgcf

        lg.clamp_(0, 100)

        self.model.train(model_training)
        d_model.train(d_model_training)

        dx2 = dxs2.mean()
        dgz2 = dgzs2.mean()

        dx2 = dx2.item()
        dgz2 = dgz2.item()
        lgr = lgr.item()
        lgf = lgf.item()
        lgcr = lgcr.item()
        lgcf = lgcf.item()
        lg = lg.item()

        result = (
            dx2, dgz2,
            lgr, lgf,
            lgcr, lgcf,
            lg
        )

        return result

    def test(self, noises):
        """Tests/Uses the model with the given args.

        Set the model to evaluation mode.
        Generate an image batch with the given noises.
        Return the batch.

        Args:
            noises: Some noises used to generate the batch.
                Can be on either the CPUs or GPUs.
                Preferred to be on the CPUs.

        Returns:
            output: G(noises): The output of the model.
                Definitely on the CPUs.
        """
        noises: _Tensor = noises

        model_training = self.model.training
        self.model.train(False)

        noises = noises.to(self.device)

        with _no_grad():
            output = self.model(noises)
            output: _Tensor = output.detach()

        output = output.float()

        self.model.train(model_training)

        output = output.cpu()
        return output
