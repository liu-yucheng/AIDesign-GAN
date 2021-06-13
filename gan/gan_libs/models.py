"""A module that includes the model classes."""

# import system and third party modules
import numpy
import pathlib
import torch
# import custom modules
import gan_libs.configs as configs


class _Helpers:
    """Helpers for classes in the models module."""

    default_model_path = configs.TrainConfig().items["model_path"]

    @classmethod
    def save_model(cls, model, loc):
        """Saves the state dict of a model to a json file.

        Params:
        model: the model to save
        loc: the location of the json file
        """
        file = open(loc, "w+")
        torch.save(model.state_dict(), loc)
        file.close()

    @classmethod
    def load_model(cls, loc, model):
        """Loads the state dict from a json file to a model.

        Params:
        loc: the location of the json file
        model: the model to load
        """
        model.load_state_dict(torch.load(loc))
        model.eval()


class TrainingModel:
    """Training model.

    A coordinator class that handles the actions involving the components
    needed to train a GAN model.
    """

    def __init__(self, model_path=None):
        """Initializes a training model.

        Params: model_path
        """
        if model_path is None:
            self.model_path = _Helpers.default_model_path
        else:
            self.model_path = model_path

    def train_d(self):
        """Trains the discriminator with the training set."""
        self.t_batch_index = 0
        while self.t_batch_index < self.t_batch_count:
            self.train_d_with_curr_batch()
            self.print_train_d_stats_if_needed()
            self.t_batch_index += 1

    def train_g(self):
        """Trains the generator with the training set."""
        self.t_batch_index = 0
        while self.t_batch_index < self.t_batch_count:
            self.train_g_with_curr_batch()
            self.print_train_g_stats_if_needed()
            self.t_batch_index += 1

    def validate_d(self):
        """Validate the discriminator with the validation set."""
        self.v_batch_index = 0
        while self.v_batch_index < self.v_batch_count:
            self.d_v_batch_errs = numpy.array()
            self.validate_d_with_curr_batch()
            self.print_validate_d_stats_if_needed()
            self.v_batch_index += 1
        self.d_v_errs.append(self.d_v_batch_errs.mean())

    def validate_g(self):
        """Validate the generator with the validation set."""
        self.v_batch_index = 0
        while self.v_batch_index < self.v_batch_count:
            self.g_v_batch_errs = numpy.array()
            self.validate_g_with_curr_batch()
            self.print_validate_g_stats_if_needed()
            self.v_batch_index += 1
        self.g_v_errs.append(self.g_v_batch_errs.mean())

    def run_epoch(self):
        """Runs an epoch."""
        self.print_epoch_header()

        self.train_d()
        self.train_g()
        self.save_training_error_plot()

        self.validate_d()
        self.validate_g()
        self.save_validation_error_plot()

        self.save_generated_images()
        self.save_real_vs_fake_images()

        self.print_epoch_trailer()

    def run_iter(self):
        """Runs an iteration."""
        self.print_iter_header()

        self.epoch = 0
        while self.epoch < self.epochs_per_iter:
            self.run_epoch()
            self.epoch += 1

        self.rollback_or_save_d()
        self.rollback_or_save_g()

        self.print_iter_trailer()

    def start_training(self):
        """Starts the training."""
        self.print_training_header()

        self.save_training_images()

        self.iter = 0
        while self.iter < self.iter_count:
            self.run_iter()
            self.iter += 1

        self.print_training_trailer()
