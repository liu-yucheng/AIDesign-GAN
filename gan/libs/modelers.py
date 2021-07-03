"""Module of the modeler classes."""

import torch

from gan.libs import nnstructs
from gan.libs import utils


class Modeler:
    """Super class of the modeler classes.

    Attributes:
        config: the config
        device: the device to use
        gpu_count: the number of GPUs to use
        loss_func: the loss function
        model: the model, a pytorch nn module
        optim: the optimizer
        rb_count: the rollback count
    """

    def __init__(self, config, device, gpu_count, loss_func):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_count: the number of GPUs to use
            loss_func: the loss function
        """
        self.config = config
        self.device = device
        self.gpu_count = gpu_count
        self.loss_func = loss_func
        self.model = None
        self.optim = None
        self.rb_count = 0

    def load(self):
        """Loads the states of the model."""
        try:
            utils.load_model(self.config["state_location"], self.model)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the states of the model."""
        utils.save_model(self.model, self.config["state_location"])

    def rollback(self):
        """Rollbacks the model.

        Clear the model gradients. Load the previous best model states. Reset the optimizer and halves the optimizer
        learning rate.

        Raises:
            ValueError: if self.model or self.optim is None
        """
        if self.model is None:
            raise ValueError("self.model cannot be None")
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.rb_count += 1
        self.model.zero_grad()
        self.load()
        self.optim = utils.setup_adam(self.model, self.config["adam_optimizer"], self.rb_count)


class DModeler(Modeler):
    """Discriminator modeler."""

    def __init__(self, config, device, gpu_count, loss_func, training=True):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_count: the number of GPUs to use
            loss_func: the loss function
            training: whether to setup the optimizer
        """
        super().__init__(config, device, gpu_count, loss_func)
        # Init self.model
        struct = nnstructs.DStruct()
        struct.location = self.config["struct_location"]
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.parallelize_model(self.model, self.device, self.gpu_count)
        self.model.apply(utils.init_model_weights)
        # Init self.optim
        if training:
            self.optim = utils.setup_adam(self.model, self.config["adam_optimizer"])

    def train(self, batch, label):
        """Trains the model with a batch of data and a target label.

        Clear the model gradients. Forward pass the batch. Find the loss. Find the gradients through a backward pass.
        Optimize/Update the model. Return the average output and loss value.

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
        self.model.zero_grad()
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        return out_mean, loss_val

    def validate(self, batch, label):
        """Validates the model with a batch of data and a target label.

        Forward pass the batch. Find the loss. Return the average output and loss value.

        Args:
            batch: the batch of data
            label: the target label

        Returns:
            out_mean: Mean(D(batch)), the average output of D
            loss_val: Loss(D(batch), label), the loss of D on the batch
        """
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
        loss = self.loss_func(output, labels)
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        return out_mean, loss_val

    def test(self, batch):
        """Tests/Uses the model with a batch of data.

        Forward pass the batch. Return the output.

        Args:
            batch: the batch of data

        Returns:
            output: D(batch), the output of D
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
        return output


class GModeler(Modeler):
    """Generator modeler."""

    def __init__(self, config, device, gpu_count, loss_func, training=True):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_count: the number of GPUs to use
            loss_func: the loss function
            training: whether to setup the optimizer
        """
        super().__init__(config, device, gpu_count, loss_func)
        # Init self.model
        struct = nnstructs.GStruct()
        struct.location = self.config["struct_location"]
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.parallelize_model(self.model, self.device, self.gpu_count)
        self.model.apply(utils.init_model_weights)
        # Init self.optim
        if training:
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

        Clear the model gradients. Generate a training batch with the given noises. Forward pass the batch to the
        discriminator model. Find the loss. Find the gradients with a backward pass. Optimize/Update the model. Return
        the average output and the loss value.

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
        self.model.zero_grad()
        batch = self.model(noises)
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        return out_mean, loss_val

    def validate(self, d_model, noises, label):
        """Validates the model with the given args.

        Generate a validation batch with the given noises. Forward pass the batch to the discriminator model. Find the
        loss. Return the average output and the loss value.

        Args:
            d_model: the discriminator model
            noises: the noises used to generate the validation batch
            label: the target label

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model
            loss_val: Loss(D(G(noises)), label), the loss of the model
        """
        with torch.no_grad():
            batch = self.model(noises).detach()
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = d_model(batch).detach().view(-1)
        loss = self.loss_func(output, labels)
        out_mean = output.mean().item()
        loss_val = loss.detach().cpu()
        return out_mean, loss_val

    def test(self, noises):
        """Tests/Uses the model with the given args.

        Generate an image batch with the given noises. Return the batch.

        Args:
            noises: the noises used to generate the batch.

        Returns:
            output: G(noises): the output of the model
        """
        with torch.no_grad():
            output = self.model(noises).detach()
        return output
