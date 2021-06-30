"""Module of the modeler classes."""

import torch

from gan_libs import nn_structs
from gan_libs import utils


class Modeler:
    """Super class of the modeler classes.

    Attributes:
        config: the config
        device: the device to use
        gpu_cnt: the number of GPUs to use
        loss_fn: the loss function
        model: the model, a pytorch nn module
        optim: the optimizer
        rb_cnt: the rollback count
    """

    def __init__(self, config, device, gpu_cnt, loss_fn):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_cnt: the number of GPUs to use
            loss_func: the loss function
        """
        self.config = config
        self.device = device
        self.gpu_cnt = gpu_cnt
        self.loss_fn = loss_fn
        self.model = None
        self.optim = None
        self.rb_cnt = 0

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

        Clear the model gradients. Loads the previous best model states. Resets
        the optimizer and halves the optimizer learning rate.

        Raises:
            ValueError: if self.model or self.optim is None
        """
        if self.model is None:
            raise ValueError("self.model cannot be None")
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.rb_cnt += 1
        self.model.zero_grad()
        self.load()
        self.optim = utils.\
            setup_adam(self.model, self.config["adam_optimizer"], self.rb_cnt)


class DModeler(Modeler):
    """Discriminator modeler."""

    def __init__(self, config, device, gpu_cnt, loss_fn, training=True):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_cnt: the number of GPUs to use
            loss_fn: the loss function
            training: whether to setup the optimizer
        """
        super().__init__(config, device, gpu_cnt, loss_fn)
        # Init self.model
        struct = nn_structs.DStruct()
        struct.location = self.config["struct_location"]
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.\
            parallelize_model(self.model, self.device, self.gpu_cnt)
        self.model.apply(utils.init_model_weights)
        # Init self.optim
        if training:
            self.optim = utils.\
                setup_adam(self.model, self.config["adam_optimizer"])

    def train(self, batch, label):
        """Trains the model with a batch of data and a target label.

        Clear the model gradients. Forward passes the batch. Finds the loss.
        Finds the gradients through a backward pass. Optimizes/Updates the
        model.

        Args:
            batch: the batch of data
            label: the target label

        Returns:
            out_mean: Mean(D(batch)), the average output of D
            loss: Loss(D(batch), label), the loss of D on the batch

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.model.zero_grad()
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        out_mean = output.mean().item()
        loss = self.loss_fn(output, labels)
        loss.backward()
        self.optim.step()
        return out_mean, loss

    def validate(self, batch, label):
        """Validates the model with a batch of data and a target label.

        Forward passes the batch. Finds the loss.

        Args:
            batch: the batch of data
            label: the target label

        Returns:
            out_mean: Mean(D(batch)), the average output of D
            loss: Loss(D(batch), label), the loss of D on the batch
        """
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
        out_mean = output.mean().item()
        loss = self.loss_fn(output, labels)
        return out_mean, loss

    def test(self, batch):
        """Test/Use the model with a batch of data.

        Forward passes the batch.

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

    def __init__(self, config, device, gpu_cnt, loss_fn, training=True):
        """Inits self with the given args.

        Args:
            config: the config
            device: the device to use
            gpu_cnt: the number of GPUs to use
            loss_fn: the loss function
            training: whether to setup the optimizer
        """
        super().__init__(config, device, gpu_cnt, loss_fn)
        # Init self.model
        struct = nn_structs.GStruct()
        struct.location = self.config["struct_location"]
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = utils.\
            parallelize_model(self.model, self.device, self.gpu_cnt)
        self.model.apply(utils.init_model_weights)
        # Init self.optim
        if training:
            self.optim = utils.\
                setup_adam(self.model, self.config["adam_optimizer"])

    def gen_noises(self, cnt):
        """Generates a set of input noises for the model.

        Args:
            cnt: the number of input noises

        Returns:
            noises: the generated set of noises
        """
        noises = torch.\
            randn(cnt, self.config["input_size"], 1, 1, device=self.device)
        return noises

    def train(self, d_model, batch_size, label):
        """Trains the model with the given args.

        Clear the model gradients. Generates a training batch of the specified
        size. Forward passes the batch to the discriminator model. Finds the
        loss. Finds the gradients with a backward pass. Optimize/Update the
        model.

        Args:
            d_model: the discriminator model
            batch_size: size of the generated training batch
            label: the target label

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model
            loss: Loss(D(G(noises)), label), the loss of the model

        Raises:
            ValueError: if self.optim is None
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")
        self.model.zero_grad()
        noises = self.gen_noises(batch_size)
        batch = self.model(noises)
        batch, labels = utils.prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        out_mean = output.mean().item()
        loss = self.loss_fn(output, labels)
        loss.backward()
        self.optim.step()
        return out_mean, loss

    def validate(self, d_model, batch_size, label, noises=None):
        """Validates the model with the given parameters.

        Generates a validation batch of the specified size. Forward passes the
        batch to the discriminator model. Finds the loss. If there is a
        specified set of noises, the size of noises will override the batch
        size.

        Args:
            d_model: the discriminator model
            batch_size: size of the generated validation batch
            label: the target label

        Returns:
            out_mean: Mean(D(G(noises))), the average output of d_model
            loss: Loss(D(G(noises)), label), the loss of the model
        """
        if noises is None:
            noises = self.gen_noises(batch_size)
        with torch.no_grad():
            batch = self.model(noises).detach()
        batch, labels = utils.\
            prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = d_model(batch).detach().view(-1)
        out_mean = output.mean().item()
        loss = self.loss_fn(output, labels)
        return out_mean, loss

    def test(self, batch_size, noises=None):
        """Test/Use the model with the specified batch size.

        Generates a batch of images with the specified batch size. If there is
        a specified set of noises, the size of noises will override the batch
        size.

        Args:
            batch_size: the number of images to generate
            noises: the specific set of input

        Returns:
            output: G(noises): the output of the model
        """
        if noises is None:
            noises = self.gen_noises(batch_size)
        with torch.no_grad():
            output = self.model(noises).detach()
        return output
