"""Module of the modeler classes.

==== References ====
Arjovsky, et al., 2017. Wasserstein Generative Adversarial Networks. https://arxiv.org/abs/1701.07875
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import torch

from aidesign_gan.libs import structs
from aidesign_gan.libs import utils

_logit = torch.logit
_no_grad = torch.no_grad
_Tensor = torch.Tensor
_tanh = torch.tanh


class Modeler:
    """Super class of the modeler classes."""

    def __init__(self, model_path, config, device, gpu_count, loss_func):
        """Inits self with the given args.

        Args:
            model_path: the model path
            config: the config
            device: the device to use, will be the GPUs if they are available
            gpu_count: the number of GPUs to use
            loss_func: the loss function
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
        self.model = None
        """Model, a pytorch nn module, definitely runs on GPUs if they are available."""
        self.optim = None
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
        state_location = utils.find_in_path(self.config["state_name"], self.model_path)
        optim_location = utils.find_in_path(self.config["optim_name"], self.model_path)
        try:
            utils.load_model(state_location, self.model)
        except FileNotFoundError:
            utils.save_model(self.model, state_location)
        if self.optim is not None:
            try:
                utils.load_optim(optim_location, self.optim)
            except FileNotFoundError:
                utils.save_optim(self.optim, optim_location)

    def save(self):
        """Saves the model and optimizer states."""
        state_location = utils.find_in_path(self.config["state_name"], self.model_path)
        optim_location = utils.find_in_path(self.config["optim_name"], self.model_path)
        utils.save_model(self.model, state_location)
        if self.optim is not None:
            utils.save_optim(self.optim, optim_location)

    def clear_grads(self):
        """Clears the gradients by calling the zero_grad function of self.model and self.optim."""
        self.model.zero_grad()
        self.optim.zero_grad()

    def predict(self):
        """Updates self.model to the predicted state by calling the predict function of self.optim.

        Let the previous state be S1, the current state be S2, the predicted state be S3. We define the predicted state
        as: S3 = S2 + (S2 - S1) = 2 * S2 - S1.
        """
        self.optim.predict()

    def step_optim(self):
        """Updates self.model by calling the step function of self.optim."""
        self.optim.step()

    def restore(self):
        """Restores self.model to the before prediction state by calling the restore function of self.optim.

        Let the previous state be S1, the current state be S2, the predicted state be S3. This function restores the
        model from S3 to S2.
        """
        self.optim.restore()


class DModeler(Modeler):
    """Discriminator modeler."""

    def __init__(self, model_path, config, device, gpu_count, loss_func, train=True):
        """Inits self with the given args.

        Args:
            model_path: the model path
            config: the config
            device: the device to use, will be the GPUs if they are available
            gpu_count: the number of GPUs to use, >= 1 if GPUs are available
            loss_func: the loss function
            train: training mode, whether to setup the optimizer
        """
        super().__init__(model_path, config, device, gpu_count, loss_func)

        # Init self.model
        struct = structs.DStruct(self.model_path)
        struct.location = utils.find_in_path(self.config["struct_name"], self.model_path)
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.parallelize_model(self.model, self.device, self.gpu_count)

        # Init model parameters
        params_init_key = "params_init"
        if params_init_key in self.config:
            params_init_func = utils.find_params_init_func(self.config[params_init_key])
        else:
            params_init_func = utils.find_params_init_func()
        self.model.apply(params_init_func)

        # Init self.optim
        if train:
            self.optim = utils.setup_pred_adam(self.model, self.config["adam_optimizer"])
            self.model.train(True)
        self.model.train(train)

        # Init self.size, self.training_size
        size, training_size = utils.find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

        # Init self.has_... attributes
        self.has_fairness = "fairness" in self.config

    def train(self, batch, label):
        """Trains the model with a batch of data and a target label.

        Set the model to training mode. Forward pass the batch. Find the loss. Backward the loss to find the gradients.
        Return the average output and loss value. NOTE: The caller of this function needs to manually call the
        clear_grads and step_optim functions of self to ensure the functioning of the training algorithm.

        Args:
            batch: the batch of data, can be on either the CPUs or GPUs, preferred to be on the CPUs
            label: the target label, definitely on the CPUs

        Returns:
            out_mean: Mean(D(batch)), the average output of D, definitely on the CPUs
            loss_val: Loss(D(batch), label), the loss of D on the batch, definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        self.model.train(True)

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        output = output.float()

        loss = self.loss_func(output, labels)
        loss.backward()

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_and_step(self, batch, label):
        """Trains and steps the model with a batch of data and a target label.

        Set the model to training mode. Clear the model gradients. Forward pass the batch. Find the loss. Find the
        gradients through a backward pass. Optimize/Update the model. Return the average output and loss value.

        Args:
            batch: the batch of data, can be on either the CPUs or GPUs, preferred to be on the CPUs
            label: the target label, definitely on the CPUs

        Returns:
            out_mean: Mean(D(batch)), the average output of D, definitely on the CPUs
            loss_val: Loss(D(batch), label), the loss of D on the batch, definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        self.model.train(True)

        self.model.zero_grad()
        self.optim.zero_grad()

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        output = output.float()

        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_fair(self, g_model, real_batch, real_label, fake_noises, fake_label):
        """Fairly trains the model with the given args.

        Set self.model to training mode. Set g_model to training mode. For the real batch: Forward pass the batch to
        self.model; Find the loss. For the fake batch: Forward pass the noises to g_model to get a fake batch; Forward
        pass the fake batch to self.model; Find the loss. After finding the 2 losses, find the cluster loss. Find the
        total loss. Backward the total loss to find the gradients. Return the results. NOTE: the caller of this
        function needs to manually call the clear_grads and step_optim functions of self to ensure the functioning of
        the training algorithm.

        Args:
            g_model: the generator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of g_model
            real_batch: the real batch, can be on either the CPUs or GPUs, preferred to be on the CPUs
            real_label: the real target label wrt. D, usually be 1, definitely on the CPUs
            fake_noises: the fake noises, can be on either the CPUs or GPUs, preferred to be on the CPUs
            fake_label: the fake target label wrt. D, usually be 0, definitely on the CPUs

        Returns:
            dx, : Mean(D(X)), the output mean of D on the real batch, definitely on the CPUs
            dgz, : Mean( D(G(Z)) ), the output mean of D on the fake batch, definitely on the CPUs
            ldr, : Loss(D, X), the loss of D on the real batch, definitely on the CPUs
            ldf, : Loss(D, G(Z)), the loss of D on the fake batch, definitely on the CPUs
            ldcr, : Loss(D, Cluster, X), ldcr = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dxs) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            ldcf, : Loss(D, Cluster, G(Z)), ldcf = 50 + 50 * tanh(wmm_factor * Mean( logit(dgzs) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            ld: Loss(D), ld = dx_factor * ldr + dgz_factor * ldf + cluster_dx_factor * ldcr + cluster_dgz_factor *
                ldcf, clamped to range [0, 100], definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        g_model_training = g_model.training
        self.model.train(True)
        g_model.train(True)

        real_batch, real_labels = utils.prep_batch_and_labels(real_batch, real_label, self.device)
        dxs: _Tensor = self.model(real_batch).view(-1)
        dxs = dxs.float()
        ldr: _Tensor = self.loss_func(dxs, real_labels)

        fake_noises = fake_noises.to(self.device)
        fake_batch = g_model(fake_noises)
        fake_batch = fake_batch.float()
        fake_batch, fake_labels = utils.prep_batch_and_labels(fake_batch, fake_label, self.device)
        dgzs: _Tensor = self.model(fake_batch).view(-1)
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

        Set the model to evaluation mode. Forward pass the batch. Find the loss. Return the average output and loss.

        Args:
            batch: the batch of data, can be on either the CPUs or GPUs, preferred to be on the CPUs
            label: the target label, definitely on the CPUs

        Returns:
            out_mean: Mean(D(batch)), the average output of D, definitely on the CPUs
            loss_val: Loss(D(batch), label), the loss of D on the batch, definitely on the CPUs
        """
        model_training = self.model.training
        self.model.train(False)

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with _no_grad():
            output = self.model(batch).detach().view(-1)
        output = output.float()

        loss = self.loss_func(output, labels)

        self.model.train(model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def valid_fair(self, g_model, real_batch, real_label, fake_noises, fake_label):
        """Fairly validates the model with the given args.

        Set self.model and g_model to evaluation mode. For the real batch: Forward pass the batch to self.model; Find
        the loss. For the fake batch: Forward pass the noises to g_model to get a fake batch; Forward pass the fake
        batch to self.model; Find the loss. After finding the 2 losses, find the cluster loss. Find the total loss.
        Return the results.

        Args:
            g_model: the generator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of g_model
            real_batch: the real batch, can be on either the CPUs or GPUs, preferred to be on the CPUs
            real_label: the real target label wrt. D, usually 1, definitely on the CPUs
            fake_noises: the fake noises, can be on either the CPUs or GPUs, preferred to be on the CPUs
            fake_label: the fake target label wrt. D, usually 0, definitely on the CPUs

        Returns:
            dx, : Mean(D(X)), the output mean of D on real, definitely on the CPUs
            dgz, : Mean( D(G(Z)) ), the output mean of D on fake, definitely on the CPUs
            ldr, : Loss(D, X), the loss of D on real, definitely on the CPUs
            ldf, : Loss(D, G(Z)), the loss of D on fake, definitely on the CPUs
            ldcr, : Loss(D, Cluster, X), ldcr = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dxs) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            ldcf, : Loss(D, Cluster, G(Z)), ldcf = 50 + 50 * tanh(wmm_factor * Mean( logit(dgzs) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            ld: Loss(D), ld = dx_factor * ldr + dgz_factor * ldf + cluster_dx_factor * ldcr + cluster_dgz_factor *
                ldcf, clamped to range [0, 100], definitely on the CPUs
        """
        model_training = self.model.training
        g_model_training = g_model.training
        self.model.train(False)
        g_model.train(False)

        real_batch, real_labels = utils.prep_batch_and_labels(real_batch, real_label, self.device)
        with _no_grad():
            dxs: _Tensor = self.model(real_batch).detach().view(-1)
        dxs = dxs.float()
        ldr: _Tensor = self.loss_func(dxs, real_labels)

        fake_noises = fake_noises.to(self.device)
        with _no_grad():
            fake_batch = g_model(fake_noises).detach()
        fake_batch = fake_batch.float()
        fake_batch, fake_labels = utils.prep_batch_and_labels(fake_batch, fake_label, self.device)
        with _no_grad():
            dgzs: _Tensor = self.model(fake_batch).detach().view(-1)
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

        Set the model to evaluation mode. Forward pass the batch. Return the output.

        Args:
            batch: the batch of data, can be on either the CPUs or GPUs

        Returns:
            output: D(batch), the output of D, definitely on the CPUs
        """
        model_training = self.model.training
        self.model.train(False)

        batch = batch.to(self.device)
        with _no_grad():
            output = self.model(batch).detach().view(-1)
        output = output.float()

        self.model.train(model_training)

        output = output.cpu()
        return output


class GModeler(Modeler):
    """Generator modeler."""

    def __init__(self, model_path, config, device, gpu_count, loss_func, train=True):
        """Inits self with the given args.

        Args:
            model_path: the model path
            config: the config
            device: the device to use, will be the GPUs if they are available
            gpu_count: the number of GPUs to use, >= 1 if GPUs are available
            loss_func: the loss function
            train: training mode, whether to setup the optimizer
        """
        super().__init__(model_path, config, device, gpu_count, loss_func)

        # Init self.model
        struct = structs.GStruct(self.model_path)
        struct.location = utils.find_in_path(self.config["struct_name"], self.model_path)
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.parallelize_model(self.model, self.device, self.gpu_count)

        # Init model parameters
        params_init_key = "params_init"
        if params_init_key in self.config:
            params_init_func = utils.find_params_init_func(self.config[params_init_key])
        else:
            params_init_func = utils.find_params_init_func()
        self.model.apply(params_init_func)

        # Init self.optim
        if train:
            self.optim = utils.setup_pred_adam(self.model, self.config["adam_optimizer"])
        self.model.train(train)

        # Init self.size, self.training_size
        size, training_size = utils.find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

        # Init self.has_... attributes
        self.has_fairness = "fairness" in self.config

    def generate_noises(self, count):
        """Generates a random set of input noises for the model.

        Args:
            count: the number of input noises, definitely on the CPUs

        Returns:
            noises: the generated set of noises, definitely on the CPUs
        """
        zr = self.config["noise_resolution"]
        zc = self.config["noise_channel_count"]
        noises = torch.randn(count, zc, zr, zr)
        return noises

    def train(self, d_model, noises, label):
        """Trains the model with the given args.

        Set self.model to training mode. Set d_model to training mode. Generate a training batch with the given noises.
        Forward pass the batch to the discriminator model. Find the loss. Backward the loss to find the gradients.
        Return the average output and the loss value. NOTE: The caller of this function needs to manually call the
        clear_grads and step_optim functions of self to ensure the functioning of the training algorithm.

        Args:
            d_model: the discriminator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of d_model
            noises: the noises used to generate the training batch, can be on either the CPUs or GPUs, preferred to be
                on the CPUs
            label: the target label, definitely on the CPUs

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model, definitely on the CPUs
            loss_val: Loss(D(G(noises)), label), the loss of the model, definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(True)
        d_model.train(True)

        noises = noises.to(self.device)
        batch = self.model(noises)
        batch = batch.float()

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        output = output.float()

        loss = self.loss_func(output, labels)
        loss.backward()

        self.model.train(model_training)
        d_model.train(d_model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_and_step(self, d_model, noises, label):
        """Trains and steps the model with the given args.

        Set self.model to training mode. Set d_model to training mode. Clear the model gradients. Generate a training
        batch with the given noises. Forward pass the batch to the discriminator model. Find the loss. Find the
        gradients with a backward pass. Optimize/Update the model. Return the average output and the loss value.

        Args:
            d_model: the discriminator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of d_model
            noises: the noises used to generate the training batch, can be on either the CPUs or GPUs, preferred to be
                on the CPUs
            label: the target label, definitely on the CPUs

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model, definitely on the CPUs
            loss_val: Loss(D(G(noises)), label), the loss of the model, definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(True)
        d_model.train(True)

        self.model.zero_grad()
        self.optim.zero_grad()

        noises = noises.to(self.device)
        batch = self.model(noises)
        batch = batch.float()

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        output = output.float()

        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()

        self.model.train(model_training)
        d_model.train(d_model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def train_fair(self, d_model, real_batch, real_label, fake_noises, fake_label):
        """Fairly trains the model with the given args.

        Set self.model to training mode. Set d_model to training mode. For the real batch: Forward pass the batch to
        d_model; Find the loss. For the fake batch: Forward pass the noises to self.model to get a fake batch; Forward
        pass the fake batch to d_model; Find the loss. After finding the 2 losses, find the cluster loss. Find the
        total loss. Backward the total loss to find the gradients. Return the results. NOTE: The caller of this
        function needs to manually call the clear_grads and step_optim functions of self to ensure the functioning of
        the training algorithm.

        Args:
            d_model: the discriminator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of d_model
            real_batch: the real batch, can be on either the CPUs or GPUs, preferred to be on the CPUs
            real_label: the real target label wrt. G, usually be 0, definitely on the CPUs
            fake_noises: the fake noises, can be on either the CPUs or GPUs, preferred to be on the CPUs
            fake_label: the fake target label wrt. G, usually be 1, definitely on the CPUs

        Returns:
            dx2, : Mean(D(X)), the mean output of D on the real batch, definitely on the CPUs
            dgz2, : Mean( D(G(Z)) ), the mean output of D on the fake batch, definitely on the CPUs
            lgr, : Loss(G, X), the loss of G on the real batch, definitely on the CPUs
            lgf, : Loss(G, G(Z)), the loss of G on the fake batch, definitely on the CPUs
            lgcr, : Loss(G, Cluster, X), lgcr = 50 + 50 * tanh(wmm_factor * Mean( logit(dxs2) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            lgcf, : Loss(G, Cluster, G(Z)), lgcf = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dgzs2) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            lg: Loss(G), lg = dx_factor * lgr + dgz_factor * lgf + cluster_dx_factor * lgcr + cluster_dgz_factor *
                lgcf, clamped to range [0, 100], definitely on the CPUs

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(True)
        d_model.train(True)

        real_batch, real_labels = utils.prep_batch_and_labels(real_batch, real_label, self.device)
        dxs2: _Tensor = d_model(real_batch).view(-1)
        dxs2 = dxs2.float()
        lgr: _Tensor = self.loss_func(dxs2, real_labels)

        fake_noises = fake_noises.to(self.device)
        fake_batch = self.model(fake_noises)
        fake_batch = fake_batch.float()
        fake_batch, fake_labels = utils.prep_batch_and_labels(fake_batch, fake_label, self.device)
        dgzs2: _Tensor = d_model(fake_batch).view(-1)
        dgzs2 = dgzs2.float()
        lgf: _Tensor = self.loss_func(dgzs2, fake_labels)

        logit_dxs2 = _logit(dxs2, eps=self.eps)
        logit_dgzs2 = _logit(dgzs2, eps=self.eps)
        lgcr = 50 + 50 * _tanh(self.wmm_factor * logit_dxs2.mean())
        lgcf = 50 + 50 * _tanh(self.wmm_factor * -1 * logit_dgzs2.mean())

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

        Set the model to evaluation mode. Generate a validation batch with the given noises. Forward pass the batch to
        the discriminator model. Find the loss. Return the average output and the loss.

        Args:
            d_model: the discriminator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of d_model
            noises: the noises used to generate the validation batch, can be on either the CPUs or GPUs, preferred to
                be on the CPUs
            label: the target label, definitely on the CPUs

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model, definitely on the CPUs
            loss_val: Loss(D(G(noises)), label), the loss of the model, definitely on the CPUs
        """
        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(False)
        d_model.train(False)

        noises = noises.to(self.device)
        with _no_grad():
            batch = self.model(noises).detach()
        batch = batch.float()

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with _no_grad():
            output = d_model(batch).detach().view(-1)
        output = output.float()

        loss = self.loss_func(output, labels)

        self.model.train(model_training)
        d_model.train(d_model_training)

        out_mean = output.mean().item()
        loss_val = loss.item()
        return out_mean, loss_val

    def valid_fair(self, d_model, real_batch, real_label, fake_noises, fake_label):
        """Fairly validates the model with the given args.

        Set self.model and d_model to evaluation mode. For the real batch: Forward pass the batch to d_model; Find the
        loss. For the fake batch: Forward pass the noises to self.model to get a fake batch; Forward pass the fake
        batch to d_model; Find the loss. After finding the 2 losses, find the cluster loss. Find the total loss. Return
        the results.

        Args:
            d_model: the discriminator model, can be on either the CPUs or the GPUs, preferred to be on the GPUs; this
                function will not change the device of d_model
            real_batch: the real batch, can be on either the CPUs or GPUs, preferred to be on the CPUs
            real_label: the real target label wrt. G, usually 0, definitely on the CPUs
            fake_noises: the fake noises, can be on either the CPUs or GPUs, preferred to be on the CPUs
            fake_label: the fake target label wrt. G, usually 1, definitely on the CPUs

        Returns:
            dx2, : Mean(D(X)), the output mean of D on real, definitely on the CPUs
            dgz2, : Mean( D(G(Z)) ), the output mean of D on fake, definitely on the CPUs
            lgr, : Loss(G, X), the loss of G on real, definitely on the CPUs
            lgf, : Loss(G, G(Z)), the loss of G on fake, definitely on the CPUs
            lgcr, : Loss(G, Cluster, X), lgcr = 50 + 50 * tanh(wmm_factor * Mean( logit(dxs2) )), tanh'ed Wasserstein
                1 metric mean, based on the WGAN paper, definitely on the CPUs
            lgcf, : Loss(G, Cluster, G(Z)), lgcf = 50 + 50 * tanh(wmm_factor * -1 * Mean( logit(dgzs2) )), tanh'ed
                Wasserstein 1 metric mean, based on the WGAN paper, definitely on the CPUs
            lg: Loss(G), lg = dx_factor * lgr + dgz_factor * lgf + cluster_dx_factor * lgcr + cluster_dgz_factor *
                lgcf, clamped to range [0, 100], definitely on the CPUs
        """
        model_training = self.model.training
        d_model_training = d_model.training
        self.model.train(False)
        d_model.train(False)

        real_batch, real_labels = utils.prep_batch_and_labels(real_batch, real_label, self.device)
        with _no_grad():
            dxs2: _Tensor = d_model(real_batch).detach().view(-1)
        dxs2 = dxs2.float()
        lgr: _Tensor = self.loss_func(dxs2, real_labels)

        fake_noises = fake_noises.to(self.device)
        with _no_grad():
            fake_batch = self.model(fake_noises).detach()
        fake_batch = fake_batch.float()
        fake_batch, fake_labels = utils.prep_batch_and_labels(fake_batch, fake_label, self.device)
        with _no_grad():
            dgzs2: _Tensor = d_model(fake_batch).detach().view(-1)
        dgzs2 = dgzs2.float()
        lgf: _Tensor = self.loss_func(dgzs2, fake_labels)

        logit_dxs2 = _logit(dxs2, eps=self.eps)
        logit_dgzs2 = _logit(dgzs2, eps=self.eps)

        lgcr = 50 + 50 * _tanh(self.wmm_factor * logit_dxs2.mean())
        lgcf = 50 + 50 * _tanh(self.wmm_factor * -1 * logit_dgzs2.mean())

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

        Set the model to evaluation mode. Generate an image batch with the given noises. Return the batch.

        Args:
            noises: the noises used to generate the batch, can be on either the CPUs or GPUs, preferred to be on the
                CPUs

        Returns:
            output: G(noises): the output of the model, definitely on the CPUs
        """
        model_training = self.model.training
        self.model.train(False)

        noises = noises.to(self.device)
        with _no_grad():
            output = self.model(noises).detach()
        output = output.float()

        self.model.train(model_training)

        output = output.cpu()
        return output
