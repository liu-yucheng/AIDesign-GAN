"""Discriminator modeler.

Contains elements based on [1], [2], and [3].
Contains elements added by liu-yucheng.

NOTE: The [*] reference list is in AIDesign-GAN's main README.
"""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import torch
from os import path as ospath
from torch import nn

from aidesign_gan.libs import structs
from aidesign_gan.libs.modelers import _helpers
from aidesign_gan.libs.modelers import modeler

# Aliases

_DiscStruct = structs.DiscStruct
_find_fairness_factors = _helpers.find_fairness_factors
_find_batch_means = _helpers.find_batch_means
_join = ospath.join
_logit = torch.logit
_Modeler = modeler.Modeler
_Module = nn.Module
_no_grad = torch.no_grad
_paral_model = _helpers.paral_model
_prep_batch_and_labels = _helpers.prep_batch_and_labels
_Tensor = torch.Tensor

# End


class DiscModeler(_Modeler):
    """Discriminator modeler."""

    def __init__(self, model_path, config, device, gpu_count, train=True):
        """Inits self with the given args.

        Args:
            model_path: a model path
            config: a modelers config
            device: Device to use.
                Will be the GPUs if they are available.
            gpu_count: Number of GPUs to use.
                0 means no GPU available.
                >= 1 means some GPUs available.
            train: Training mode flag.
                Controls whether to setup the optimizer.
        """
        super().__init__(model_path, config, device, gpu_count)

        # Setup self.model

        struct_loc = _join(self.model_path, self.config["struct_name"])
        struct_def = _DiscStruct.load(struct_loc)
        exec(struct_def)
        self.model = self.model.to(self.device)
        self.model = _paral_model(self.model, self.device, self.gpu_count)

        self.input_shape = [
            1,
            int(self.config["image_channel_count"]),
            int(self.config["image_resolution"]),
            int(self.config["image_resolution"])
        ]

        self.output_shape = [
            1,
            int(self.config["label_channel_count"]),
            int(self.config["label_resolution"]),
            int(self.config["label_resolution"])
        ]

        # End

        self._init_after_model_setup(train)

    def train(self, data, label):
        """Trains the model with a batch of data and a target label.

        Set the model to training mode.
        Forward pass the batch.
        Find the loss.
        Backward the loss to find the gradients.
        Return the average output and loss value.
        NOTE: The caller of this function needs to manually call the clear_grads and step_optim functions of self to
            ensure the functioning of the training algorithm.

        Args:
            data: A batch of data.
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

        data, labels = _prep_batch_and_labels(data, label, self.device)
        output = self.model(data)
        output = _find_batch_means(output)
        output: _Tensor = output.view(-1)
        output = output.float()

        loss: _Tensor = self.bce_loss(output, labels)
        loss.backward()

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_and_step(self, data, label):
        """Trains and steps the model with a batch of data and a target label.

        Set the model to training mode.
        Clear the model gradients.
        Forward pass the batch.
        Find the loss.
        Find the gradients through a backward pass.
        Optimize/Update the model.
        Return the average output and loss value.

        Args:
            data: A batch of data.
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

        data, labels = _prep_batch_and_labels(data, label, self.device)
        output = self.model(data)
        output = _find_batch_means(output)
        output: _Tensor = output.view(-1)
        output = output.float()

        loss: _Tensor = self.bce_loss(output, labels)
        loss.backward()
        self.optim.step()

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def _find_fair_losses(self, real_labels, fake_labels, dxs, dgzs):
        """Finds the fairness losses.

        For use in the train_fair and valid_fair methods.

        Args:
            real_labels: a batch of real labels
            fake_labels: a batch of fake labels
            dxs: a batch of D(X) results
            dgzs: a batch of D(G(Z)) results

        Returns:
            results: A tuple that contains the following items.
            (ldr, ldf, ldcr, ldcf, ld): The results items.
                Appears as defined in the docstrings of train_fair and valid_fair methods.
        """
        real_labels: _Tensor = real_labels
        fake_labels: _Tensor = fake_labels
        dxs: _Tensor = dxs
        dgzs: _Tensor = dgzs

        # Find the classic losses on real and fake
        ldr: _Tensor = self.bce_loss(dxs, real_labels)
        ldf: _Tensor = self.bce_loss(dgzs, fake_labels)
        # -

        # Find dx_factor, dgz_factor,
        #   cluster_dx_factor, cluster_dgz_factor,
        #   cluster_dx_overact_slope, cluster_dgz_overact_slope

        if self.has_fairness:
            config = self.config["fairness"]
            fair_facs = _find_fairness_factors(config)
        else:  # elif not self.has_fairness
            fair_facs = _find_fairness_factors()
        # end if

        (
            dx_fac, dgz_fac,
            clust_dx_fac, clust_dgz_fac,
            clust_dx_oa_slope, clust_dgz_oa_slope
        ) = fair_facs

        # End

        logit_reals = _logit(real_labels, eps=self.eps)
        logit_fakes = _logit(fake_labels, eps=self.eps)
        logit_dxs = _logit(dxs, eps=self.eps)
        logit_dgzs = _logit(dgzs, eps=self.eps)

        sl_reals: _Tensor = self.softsign(logit_reals)
        sl_fakes: _Tensor = self.softsign(logit_fakes)
        sl_dxs: _Tensor = self.softsign(logit_dxs)
        sl_dgzs: _Tensor = self.softsign(logit_dgzs)

        clust_dx_diffs = sl_reals.sub(sl_dxs)
        clust_dgz_diffs = sl_dgzs.sub(sl_fakes)
        clust_dx_diff = clust_dx_diffs.mean()
        clust_dgz_diff = clust_dgz_diffs.mean()

        # Handle overacting

        real = real_labels.mean().item()
        fake = fake_labels.mean().item()
        dx = dxs.mean().item()
        dgz = dgzs.mean().item()

        if dx >= real:
            clust_dx_slope = clust_dx_oa_slope
        else:
            clust_dx_slope = self.wmm_factor
        # end if

        if dgz <= fake:
            clust_dgz_slope = clust_dgz_oa_slope
        else:
            clust_dgz_slope = self.wmm_factor
        # end if

        # End
        # Find the cluster losses on real and fake

        ldcr = 50 + 25 * clust_dx_slope * clust_dx_diff
        ldcf = 50 + 25 * clust_dgz_slope * clust_dgz_diff
        ldcr.clamp_(0, 100)
        ldcf.clamp_(0, 100)

        # End

        # Find the weighted sum of losses
        ld: _Tensor = \
            dx_fac * ldr \
            + dgz_fac * ldf \
            + clust_dx_fac * ldcr\
            + clust_dgz_fac * ldcf

        ld.clamp_(0, 100)

        results = (
            ldr, ldf,
            ldcr, ldcf,
            ld
        )

        return results

    def train_fair(self, g_model, real_data, real_label, fake_noises, fake_label):
        """Fairly trains the model with the given args.

        For use in the Fair Predictive Alternating SGD algorithm by liu-yucheng.
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
            real_data: A batch of real data.
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
            dx_item, : Mean(D(X)).
                The output mean of D on the real batch.
                Definitely on the CPUs.
            dgz_item, : Mean( D(G(Z)) ).
                The output mean of D on the fake batch.
                Definitely on the CPUs.
            ldr_item, : Loss(D, X).
                The classic loss of D on the real batch.
                Definitely on the CPUs.
            ldf_item, : Loss(D, G(Z)).
                The classic loss of D on the fake batch.
                Definitely on the CPUs.
            ldcr_item, : Loss(D, Cluster, X).
                The cluster loss of D on the real batch.
                = 50 + 25 * (
                    softsign(wmm_factor * Mean( logit(real_labels) )) - softsign(wmm_factor * Mean( logit(dxs) ))
                ).
                Softsigned Wasserstein 1 metric mean based on module note reference [3].
                Clamped to range [0, 100].
                Definitely on the CPUs.
            ldcf_item, : Loss(D, Cluster, G(Z)).
                The cluster loss of D on the fake batch.
                = 50 + 25 * (
                    softsign(wmm_factor * Mean( logit(dgzs) )) - softsign(wmm_factor * Mean( logit(fake_labels) ))
                ).
                Softsigned Wasserstein 1 metric mean based on module note reference [3].
                Clamped to range [0, 100].
                Definitely on the CPUs.
            ld_item: Loss(D).
                The weighted sum loss of D.
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

        # Forward pass the real batch
        real_data, real_labels = _prep_batch_and_labels(real_data, real_label, self.device)
        dxs = self.model(real_data)
        dxs = _find_batch_means(dxs)
        dxs: _Tensor = dxs.view(-1)
        dxs = dxs.float()
        # -

        # Forward pass the fake batch
        fake_noises = fake_noises.to(self.device)
        fake_batch: _Tensor = g_model(fake_noises)
        fake_batch = fake_batch.float()
        fake_batch, fake_labels = _prep_batch_and_labels(fake_batch, fake_label, self.device)
        dgzs = self.model(fake_batch)
        dgzs = _find_batch_means(dgzs)
        dgzs: _Tensor = dgzs.view(-1)
        dgzs = dgzs.float()
        # -

        # Find and backward propagate the classic and cluster losses
        ldr, ldf, ldcr, ldcf, ld = self._find_fair_losses(real_labels, fake_labels, dxs, dgzs)
        ld.backward()
        # -

        self.model.train(model_training)
        g_model.train(g_model_training)

        dx = dxs.mean()
        dgz = dgzs.mean()

        dx_item = dx.item()
        dgz_item = dgz.item()
        ldr_item = ldr.item()
        ldf_item = ldf.item()
        ldcr_item = ldcr.item()
        ldcf_item = ldcf.item()
        ld_item = ld.item()

        results = (
            dx_item, dgz_item,
            ldr_item, ldf_item,
            ldcr_item, ldcf_item,
            ld_item
        )

        return results

    def valid(self, data, label):
        """Validates the model with a batch of data and a target label.

        Set the model to evaluation mode.
        Forward pass the batch.
        Find the loss.
        Return the average output and loss.

        Args:
            data: A batch of data.
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

        data, labels = _prep_batch_and_labels(data, label, self.device)

        with _no_grad():
            output = self.model(data)
            output = _find_batch_means(output)
            output: _Tensor = output.detach().view(-1)
        # end with

        output = output.float()

        loss: _Tensor = self.bce_loss(output, labels)

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def valid_fair(self, g_model, real_data, real_label, fake_noises, fake_label):
        """Fairly validates the model with the given args.

        For use in the Fair Predictive Alternating SGD algorithm by liu-yucheng.
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
            real_data: A batch of real data.
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
            results: A tuple that contains the following items.
            dx_item, : Mean(D(X)).
                The output mean of D on the real batch.
                Definitely on the CPUs.
            dgz_item, : Mean( D(G(Z)) ).
                The output mean of D on the fake batch.
                Definitely on the CPUs.
            ldr_item, : Loss(D, X).
                The classic loss of D on the real batch.
                Definitely on the CPUs.
            ldf_item, : Loss(D, G(Z)).
                The classic loss of D on the fake batch.
                Definitely on the CPUs.
            ldcr_item, : Loss(D, Cluster, X).
                The cluster loss of D on the real batch.
                = 50 + 25 * (
                    softsign(wmm_factor * Mean( logit(real_labels) )) - softsign(wmm_factor * Mean( logit(dxs) ))
                ).
                Softsigned Wasserstein 1 metric mean based on module note reference [3].
                Clamped to range [0, 100].
                Definitely on the CPUs.
            ldcf_item, : Loss(D, Cluster, G(Z)).
                The cluster loss of D on the fake batch.
                = 50 + 25 * (
                    softsign(wmm_factor * Mean( logit(dgzs) )) - softsign(wmm_factor * Mean( logit(fake_labels) ))
                ).
                Softsigned Wasserstein 1 metric mean based on module note reference [3].
                Clamped to range [0, 100].
                Definitely on the CPUs.
            ld_item: Loss(D).
                The weighted sum loss of D.
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

        # Forward pass the real batch

        real_data, real_labels = _prep_batch_and_labels(real_data, real_label, self.device)

        with _no_grad():
            dxs = self.model(real_data)
            dxs = _find_batch_means(dxs)
            dxs: _Tensor = dxs.detach().view(-1)
        # end with

        dxs = dxs.float()

        fake_noises = fake_noises.to(self.device)

        with _no_grad():
            fake_batch = g_model(fake_noises)
            fake_batch: _Tensor = fake_batch.detach()

        # End
        # Forward pass the fake batch

        fake_batch = fake_batch.float()
        fake_batch, fake_labels = _prep_batch_and_labels(fake_batch, fake_label, self.device)

        with _no_grad():
            dgzs = self.model(fake_batch)
            dgzs = _find_batch_means(dgzs)
            dgzs: _Tensor = dgzs.detach().view(-1)
        # end with

        dgzs = dgzs.float()

        # End

        # Find the classic and cluster losses
        ldr, ldf, ldcr, ldcf, ld = self._find_fair_losses(real_labels, fake_labels, dxs, dgzs)

        self.model.train(model_training)
        g_model.train(g_model_training)

        dx = dxs.mean()
        dgz = dgzs.mean()

        dx_item = dx.item()
        dgz_item = dgz.item()
        ldr_item = ldr.item()
        ldf_item = ldf.item()
        ldcr_item = ldcr.item()
        ldcf_item = ldcf.item()
        ld_item = ld.item()

        results = (
            dx_item, dgz_item,
            ldr_item, ldf_item,
            ldcr_item, ldcf_item,
            ld_item
        )

        return results

    def test(self, data):
        """Tests/Uses the model with a batch of data.

        Set the model to evaluation mode.
        Forward pass the batch.
        Return the output.

        Args:
            data: A batch of data.
            Can be on either the CPUs or GPUs.

        Returns:
            output: D(batch).
                The output of D (NOT the average output).
                Definitely on the CPUs.
        """
        data: _Tensor = data

        model_training = self.model.training
        self.model.train(False)

        data = data.to(self.device)

        with _no_grad():
            output = self.model(data)
            output = _find_batch_means(output)
            output: _Tensor = output.detach().view(-1)

        output = output.float()

        self.model.train(model_training)

        output = output.cpu()
        return output
