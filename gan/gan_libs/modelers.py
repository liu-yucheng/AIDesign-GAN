"""Module of the modeler classes."""

import torch
import torch.nn as nn
import torch.optim as optim

import gan_libs.nn_structs as nn_structs


class _Helpers:
    """Helpers for classes in the module."""

    @classmethod
    def save_model(cls, model, location):
        """Saves the states of a model to a location."""
        file = open(location, "w+")
        torch.save(model.state_dict(), location)
        file.close()

    @classmethod
    def load_model(cls, location, model):
        """Loads the states from a location into a model."""
        model.load_state_dict(torch.load(location))
        model.eval()

    @classmethod
    def parallelize_model(cls, model, device, gpu_count):
        """Finds the parallelized model if the device and gpu_count allow.

        Returns: the parallelized/original model
        """
        if device.type == "cuda" and gpu_count > 1:
            model = nn.DataParallel(model, list(range(gpu_count)))
        return model

    @classmethod
    def prep_batch_and_labels(cls, batch, label, device):
        """Prepares batch and labels with the given device.

        Args:
            batch:  the batch to prepare
            label:  the targeted label
            device: the device to use

        Returns: the prepared batch and labels
        """
        batch = batch.to(device)
        labels = torch.full(
            (batch.size(0),),
            label,
            dtype=torch.float,
            device=device
        )
        return batch, labels


class _Modeler:
    """Super class of the modeler classes."""

    def __init__(self, config, device, gpu_count, loss_func):
        """Initializes a modeler with the given args.

        Args:
            config:     the config of the neural network model
            device:     the device used to run the model
            gpu_count:  the number of GPUs used to run the model
            loss_func:  the loss function of the model
        """
        self.config = config
        self.device = device
        self.gpu_count = gpu_count
        self.loss_func = loss_func

        self.model = None
        self.optim = None
        self.rollback_count = 0

    def load(self):
        """Loads the states of the model."""
        try:
            _Helpers.load_model(self.config["state_location"], self.model)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the states of the model."""
        _Helpers.save_model(self.model, self.config["state_location"])

    def rollback(self):
        """Rollbacks the model.

        Clear the model gradients. Loads the previous best model states. Resets
        the optimizer and halves the optimizer learning rate.
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        self.rollback_count += 1

        self.model.zero_grad()
        self.load()

        adam_config = self.config["adam_optimizer"]
        self.optim = optim.Adam(
            self.model.parameters(),
            lr=adam_config["learning_rate"] / (2 ** self.rollback_count),
            betas=(adam_config["beta1"], adam_config["beta2"])
        )


class DModeler(_Modeler):
    """Discriminator modeler."""

    def __init__(self, config, device, gpu_count, loss_func, training=True):
        """Initializes an object with the given args.

        Args:
            config:     the config of the neural network model
            device:     the device used to run the model
            gpu_count:  the number of GPUs used to run the model
            loss_func:  the loss function of the model
            training:   whether to setup the optimizer
        """
        super().__init__(config, device, gpu_count, loss_func)

        # Init self.model
        struct = nn_structs.DStruct()
        struct.location = self.config["struct_location"]
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = _Helpers.\
            parallelize_model(self.model, self.device, self.gpu_count)
        self.model.apply(Utils.init_weights)

        if training:
            # Init self.optim
            adam_config = self.config["adam_optimizer"]
            self.optim = optim.Adam(
                self.model.parameters(),
                lr=adam_config["learning_rate"],
                betas=(adam_config["beta1"], adam_config["beta2"])
            )

    def train(self, batch, label):
        """Trains the model with a batch of data and a targeted label.

        Clear the model gradients. Forward passes the batch. Finds the loss.
        Finds the gradients through a backward pass. Optimizes/Updates the
        model.

        Args:
            batch:  the batch of data
            label:  the targeted label

        Returns: output, loss
            output: D(batch), the output of D
            loss:   Loss(D(batch), labels), the loss of D on the batch
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        self.model.zero_grad()
        batch, labels = _Helpers.\
            prep_batch_and_labels(batch, label, self.device)
        output = self.model(batch).view(-1)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()
        return output, loss

    def validate(self, batch, label):
        """Validates the model with a batch of data and a targeted label.

        Forward passes the batch. Finds the loss.

        Args:
            batch:  the batch of data
            label:  the targeted label

        Returns: output, loss
            output: D(batch), the output of D
            loss:   Loss(D(batch), labels), the loss of D on the batch
        """
        batch, labels = _Helpers.\
            prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
        loss = self.loss_func(output, labels)
        return output, loss

    def test(self, batch):
        """Test/Use the model with a batch of data.

        Forward passes the batch.

        Args: batch: the batch of data

        Returns: output: D(batch), the output of D
        """
        batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch).detach().view(-1)
        return output


class GModeler(_Modeler):
    """Generator modeler."""

    def __init__(self, config, device, gpu_count, loss_func, training=True):
        """Initializes an object with the given args.

        Args:
            config:     the config of the neural network model
            device:     the device used to run the model
            gpu_count:  the number of GPUs used to run the model
            loss_func:  the loss function of the model
            training:   whether to setup the optimizer
        """
        super().__init__(config, device, gpu_count, loss_func)

        # Init self.model
        struct = nn_structs.GStruct()
        struct.location = self.config["struct_location"]
        struct.load()
        exec(struct.definition)
        self.model = self.model.to(self.device)
        self.model = _Helpers.\
            parallelize_model(self.model, self.device, self.gpu_count)
        self.model.apply(Utils.init_weights)

        if training:
            # Init self.optim
            adam_config = self.config["adam_optimizer"]
            self.optim = optim.Adam(
                self.model.parameters(),
                lr=adam_config["learning_rate"],
                betas=(adam_config["beta1"], adam_config["beta2"])
            )

    def gen_noises(self, count):
        """Generates a input noise set for the model.

        Args: count: input noise count

        Returns: the generated noise
        """
        noises = torch.\
            randn(count, self.config["input_size"], 1, 1, device=self.device)
        return noises

    def train(self, d_model, batch_size, label):
        """Trains the model with the given parameters.

        Clear the model gradients. Generates a training batch of the specified
        size. Forward passes the batch to the discriminator model. Finds the
        loss. Finds the gradients with a backward pass. Optimize/Update the
        model.

        Args:
            d_model:    the discriminator model
            batch_size: size of the generated training batch
            label:      the targeted label

        Returns: output, loss
            output: D(G(Z)), the output of d_model on the batch
            loss:   Loss(D(G(Z)), label), the loss of the model on the batch
        """
        if self.optim is None:
            raise ValueError("self.optim cannot be None")

        self.model.zero_grad()
        noises = self.gen_noises(batch_size)
        batch = self.model(noises)
        batch, labels = _Helpers.\
            prep_batch_and_labels(batch, label, self.device)
        output = d_model(batch).view(-1)
        loss = self.loss_func(output, labels)
        loss.backward()
        self.optim.step()
        return output, loss

    def validate(self, d_model, batch_size, label):
        """Validates the model with the given parameters.

        Generates a validation batch of the specified size. Forward passes the
        batch to the discriminator model. Finds the loss.

        Args:
            d_model:    the discriminator model
            batch_size: size of the generated validation batch
            label:      the targeted label

        Returns: output, loss
            output: D(G(Z)), the output of d_model on the batch
            loss:   Loss(D(G(Z)), label), the loss of the model on the batch
        """
        noises = self.gen_noises(batch_size)
        with torch.no_grad():
            batch = self.model(noises).detach()
        batch, labels = _Helpers.\
            prep_batch_and_labels(batch, label, self.device)
        with torch.no_grad():
            output = d_model(batch).detach().view(-1)
        loss = self.loss_func(output, labels)
        return output, loss

    def test(self, batch_size):
        """Test/Use the model with the specified batch size.

        Generates a batch of the specified size.

        Args: batch_size: size of the generated batch

        Returns: output: G(Z): the output of the model
        """
        noises = self.gen_noises(batch_size)
        with torch.no_grad():
            output = self.model(noises).detach()
        return output


class Utils:
    """Utilities for classes in the module."""

    @classmethod
    def init_weights(cls, model):
        """Initializes the weights inside a model.

        Args: model: the model to initialize
        """
        class_name = model.__class__.__name__
        if class_name.find("Conv") != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
