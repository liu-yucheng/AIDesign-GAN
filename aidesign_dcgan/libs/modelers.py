"""Module of the modeler classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import torch

from aidesign_dcgan.libs import structs
from aidesign_dcgan.libs import utils


class Modeler:
    """Super class of the modeler classes.

    Attributes:
        model_path: the model path
        config: the config
        device: the device to use
        gpu_count: the number of GPUs to use
        loss_func: the loss function
        model: the model, a pytorch nn module
        size: the total size of the model
        optim: the optimizer
    """

    def __init__(self, model_path, config, device, gpu_count, loss_func):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_count: the number of GPUs to use
            loss_func: the loss function
        """
        self.model_path = model_path
        self.config = config
        self.device = device
        self.gpu_count = gpu_count
        self.loss_func = loss_func
        self.model = None
        self.size = None
        self.optim = None

    def load(self):
        """Loads the states of the model."""
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
        """Saves the states of the model."""
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
        self.optim = utils.setup_adam(self.model, self.config["adam_optimizer"], count)


class DModeler(Modeler):
    """Discriminator modeler."""

    def __init__(self, model_path, config, device, gpu_count, loss_func, train=True):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_count: the number of GPUs to use
            loss_func: the loss function
            train: training mode, whether to setup the optimizer
        """
        super().__init__(model_path, config, device, gpu_count, loss_func)
        # Init self.model
        struct = structs.DStruct()
        struct.location = utils.find_in_path(self.config["struct_name"], self.model_path)
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.parallelize_model(self.model, self.device, self.gpu_count)
        self.model.apply(utils.init_model_weights)
        # Init self.size
        size = 0
        for param in self.model.parameters():
            size_of_param = 1
            for size_of_dim in param.size():
                size_of_param *= size_of_dim
            size += size_of_param
        self.size = size
        # Init self.optim
        if train:
            self.optim = utils.setup_adam(self.model, self.config["adam_optimizer"])

    def train(self, batch, label):
        """Trains the model with a batch of data and a target label.

        Set the model to training mode. Clear the model gradients. Forward pass the batch. Find the loss. Find the
        gradients through a backward pass. Optimize/Update the model. Return the average output and loss value.

        Args:
            batch: the batch of data
            label: the target label

        Returns:
            out_mean: Mean(D(batch)), the average output of D
            loss_val: Loss(D(batch), label), the loss of D on the batch

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.model.train(True)
        self.model.zero_grad()
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        return out_mean, loss_val

    def valid(self, batch, label):
        """Validates the model with a batch of data and a target label.

        Set the model to evaluation mode. Forward pass the batch. Find the loss. Return the average output and loss.

        Args:
            batch: the batch of data
            label: the target label

        Returns:
            out_mean: Mean(D(batch)), the average output of D
            loss_val: Loss(D(batch), label), the loss of D on the batch
        """
        self.model.train(False)
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
            loss = self.loss_func(output, labels)
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        return out_mean, loss_val

    def test(self, batch):
        """Tests/Uses the model with a batch of data.

        Set the model to evaluation mode. Forward pass the batch. Return the output.

        Args:
            batch: the batch of data

        Returns:
            output: D(batch), the output of D
        """
        self.model.train(False)
        batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
        return output


class GModeler(Modeler):
    """Generator modeler."""

    def __init__(self, model_path, config, device, gpu_count, loss_func, train=True):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_count: the number of GPUs to use
            loss_func: the loss function
            train: training mode, whether to setup the optimizer
        """
        super().__init__(model_path, config, device, gpu_count, loss_func)
        # Init self.model
        struct = structs.GStruct()
        struct.location = utils.find_in_path(self.config["struct_name"], self.model_path)
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.parallelize_model(self.model, self.device, self.gpu_count)
        self.model.apply(utils.init_model_weights)
        # Init self.size
        size = 0
        for param in self.model.parameters():
            size_of_param = 1
            for size_of_dim in param.size():
                size_of_param *= size_of_dim
            size += size_of_param
        self.size = size
        # Init self.optim
        if train:
            self.optim = utils.setup_adam(self.model, self.config["adam_optimizer"])

    def generate_noises(self, count):
        """Generates a set of input noises for the model.

        Args:
            count: the number of input noises

        Returns:
            noises: the generated set of noises
        """
        noises = torch.randn(count, self.config["input_size"], 1, 1, device=self.device)
        return noises

    def train(self, d_model, noises, label):
        """Trains the model with the given args.

        Set the model to training mode. Clear the model gradients. Generate a training batch with the given noises.
        Forward pass the batch to the discriminator model. Find the loss. Find the gradients with a backward pass.
        Optimize/Update the model. Return the average output and the loss value.

        Args:
            d_model: the discriminator model
            noises: the noises used to generate the training batch
            label: the target label

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model
            loss_val: Loss(D(G(noises)), label), the loss of the model

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.model.train(True)
        d_model_training = d_model.training
        d_model.train(True)
        self.model.zero_grad()
        batch = self.model(noises)
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        if not d_model_training:
            d_model.train(False)
        return out_mean, loss_val

    def valid(self, d_model, noises, label):
        """Validates the model with the given args.

        Set the model to evaluation mode. Generate a validation batch with the given noises. Forward pass the batch to
        the discriminator model. Find the loss. Return the average output and the loss.

        Args:
            d_model: the discriminator model
            noises: the noises used to generate the validation batch
            label: the target label

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model
            loss_val: Loss(D(G(noises)), label), the loss of the model
        """
        self.model.train(False)
        d_model_training = d_model.training
        d_model.train(False)
        with torch.no_grad():
            batch = self.model(noises).detach()
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = d_model(batch).detach().view(-1)
            loss = self.loss_func(output, labels)
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        if d_model_training:
            d_model.train(True)
        return out_mean, loss_val

    def test(self, noises):
        """Tests/Uses the model with the given args.

        Set the model to evaluation mode. Generate an image batch with the given noises. Return the batch.

        Args:
            noises: the noises used to generate the batch.

        Returns:
            output: G(noises): the output of the model
        """
        self.model.train(False)
        with torch.no_grad():
            output = self.model(noises).detach()
        return output
