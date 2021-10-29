"""Module of the modeler classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

from torch.cuda.amp import autocast_mode
import torch

from aidesign_gan.libs import structs
from aidesign_gan.libs import utils

autocast = autocast_mode.autocast


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
        self.scaler = torch.cuda.amp.GradScaler()
        """CUDA AMP gradient scaler."""

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

    def rollback(self, count):
        """Rollbacks the model.

        Clear the model gradients. Load the previous best model states. Reset the optimizer and halves the optimizer
        learning rate.

        Args:
            count: the rollback count

        Raises:
            ValueError: if self.model or self.optim is None
        """
        if self.model is None:
            raise ValueError("self.model cannot be None")
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.model.zero_grad()
        self.load()


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
        self.model.apply(utils.init_model_weights)
        # Init self.optim
        if train:
            self.optim = utils.setup_pred_adam(self.model, self.config["adam_optimizer"])
            self.model.train(True)
        self.model.train(train)
        # Init self.size, self.trainable_size
        size, training_size = utils.find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

    @autocast()
    def train(self, batch, label):
        """Trains the model with a batch of data and a target label.

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
        self.model.train(True)
        self.model.zero_grad()
        self.optim.zero_grad()

        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        logit_output = torch.logit(output, eps=1e-5)
        loss = self.loss_func(logit_output, labels)
        loss.backward()
        self.optim.step()

        # self.scaler.scale(loss).backward()
        # self.model.apply(utils.nan_to_num_model)
        # utils.nan_to_num_optim(self.optim)
        # self.scaler.unscale_(self.optim)
        # print("D step?")  # Debug
        # self.scaler.step(self.optim)
        # self.optim.step()  # Debug
        # self.scaler.update()

        out_mean = output.float().mean().item()
        loss_val = loss.float().item()
        return out_mean, loss_val

    @autocast()
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
        self.model.train(False)
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
            logit_output = torch.logit(output, eps=1e-5)
            loss = self.loss_func(logit_output, labels)
        out_mean = output.float().mean().item()
        loss_val = loss.float().item()
        return out_mean, loss_val

    @autocast()
    def test(self, batch):
        """Tests/Uses the model with a batch of data.

        Set the model to evaluation mode. Forward pass the batch. Return the output.

        Args:
            batch: the batch of data, can be on either the CPUs or GPUs

        Returns:
            output: D(batch), the output of D, definitely on the CPUs
        """
        self.model.train(False)
        batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch).detach().cpu().view(-1)
        output = output.float()
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
        self.model.apply(utils.init_model_weights)
        # Init self.optim
        if train:
            self.optim = utils.setup_pred_adam(self.model, self.config["adam_optimizer"])
        self.model.train(train)
        # Init self.size, self.trainable_size
        size, training_size = utils.find_model_sizes(self.model)
        self.size = size
        self.training_size = training_size

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

    @autocast()
    def train(self, d_model, noises, label):
        """Trains the model with the given args.

        Set the model to training mode. Clear the model gradients. Generate a training batch with the given noises.
        Forward pass the batch to the discriminator model. Find the loss. Find the gradients with a backward pass.
        Optimize/Update the model. Return the average output and the loss value.

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
        self.model.train(True)
        d_model_training = d_model.training
        d_model.train(True)
        self.model.zero_grad()
        self.optim.zero_grad()

        noises = noises.to(self.device)
        batch = self.model(noises)
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        logit_output = torch.logit(output, eps=1e-5)
        loss = self.loss_func(logit_output, labels)
        loss.backward()
        self.optim.step()

        # self.scaler.scale(loss).backward()
        # self.model.apply(utils.nan_to_num_model)
        # utils.nan_to_num_optim(self.optim)
        # self.scaler.unscale_(self.optim)
        # print("G step?")  # Debug
        # self.scaler.step(self.optim)
        # self.optim.step()  # Debug
        # self.scaler.update()

        if not d_model_training:
            d_model.train(False)
        out_mean = output.float().mean().item()
        loss_val = loss.float().item()
        return out_mean, loss_val

    @autocast()
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
        self.model.train(False)
        d_model_training = d_model.training
        d_model.train(False)
        noises = noises.to(self.device)
        with torch.no_grad():
            batch = self.model(noises).detach()
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = d_model(batch).detach().view(-1)
            logit_output = torch.logit(output, eps=1e-5)
            loss = self.loss_func(logit_output, labels)
        if d_model_training:
            d_model.train(True)
        out_mean = output.float().mean().item()
        loss_val = loss.float().item()
        return out_mean, loss_val

    @autocast()
    def test(self, noises):
        """Tests/Uses the model with the given args.

        Set the model to evaluation mode. Generate an image batch with the given noises. Return the batch.

        Args:
            noises: the noises used to generate the batch, can be on either the CPUs or GPUs, preferred to be on the
                CPUs

        Returns:
            output: G(noises): the output of the model, definitely on the CPUs
        """
        self.model.train(False)
        noises = noises.to(self.device)
        with torch.no_grad():
            output = self.model(noises).detach().cpu()
        output = output.float()
        return output
