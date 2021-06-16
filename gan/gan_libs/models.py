"""A module that includes the model classes."""

# import system and third party modules
import numpy
import pathlib
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import custom modules
import gan_libs.configs as configs


class _Helpers:
    """Helpers for the model classes."""

    @classmethod
    def make_model_config(cls, path):
        """Makes a model config from a path."""
        model_config = configs.ModelConfig()
        model_config.location = str(
            pathlib.Path(path + "/model_config.json").resolve()
        )
        model_config.load()
        print("Loaded model_config from {}".format(model_config.location))
        return model_config

    @classmethod
    def set_random_seeds(cls, value):
        """Sets the random seeds with the given value.

        Sets the seeds for the numpy, random, and torch modules.
        """
        if value is not None:
            print("Random seed (manual): {}".format(value))
        else:
            random.seed(None)
            value = random.randint(0, 2 ** 32 - 1)
            print("Random seed (auto): {}".format(value))
        numpy.random.seed(value)
        random.seed(value)
        torch.manual_seed(value)

    @classmethod
    def make_data_loaders(cls, path, config):
        """Makes the data loaders given a path and a data sets config.

        Makes the training set and validation set loaders based on the weights
        provided in the data sets config (a subset of a model config).

        Returns: the training set loader, and the validation set loader
        """
        image_res = config["image_resolution"]
        data_set = datasets.ImageFolder(
            root=path,
            transform=transforms.Compose([
                transforms.Resize(image_res),
                transforms.CenterCrop(image_res),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        total_size = len(data_set)
        print("Data set total size: {}; ".format(total_size), end="")

        subset_ratio = numpy.array([
            config["training_set_weight"], config["validation_set_weight"]
        ])
        subset_ratio = subset_ratio / subset_ratio.sum()
        print("Data subset ratio: {}".format(subset_ratio))

        t_start, t_end = 0, int(subset_ratio[0] * total_size)
        v_start, v_end = t_end, total_size
        t_indices = list(range(t_start, t_end))
        v_indices = list(range(v_start, v_end))
        t_set = data.Subset(data_set, t_indices)
        v_set = data.Subset(data_set, v_indices)
        print("Training set size: {}; ".format(len(t_set)), end="")
        print("Validation set size: {}".format(len(v_set)))

        worker_count = config["loader_worker_count"]
        if worker_count is None:
            worker_count = 0
        images_per_batch = config["images_per_batch"]
        t_loader = data.DataLoader(
            t_set,
            batch_size=images_per_batch,
            shuffle=True,
            num_workers=worker_count
        )
        v_loader = data.DataLoader(
            v_set,
            batch_size=images_per_batch,
            shuffle=True,
            num_workers=worker_count
        )
        print("Training batch count: {}; ".format(len(t_loader)), end="")
        print("Validation batch count: {}".format(len(v_loader)))

        return t_loader, v_loader


class TrainingModel:
    """Training model.

    A coordinator class that handles the actions involving the components
    needed to train a GAN model.
    """

    def __init__(self, data_path, model_path):
        """Initializes a training model.

        Params: data_path, model_path
        """
        if data_path is None:
            raise ValueError("data_path cannot be None")
        if model_path is None:
            raise ValueError("model_path cannot be None")

        self.data_path = data_path
        self.model_path = model_path
        self.completed_context_setup = False

    def setup_context(self):
        """Sets up the internal states of the model.

        Gets ready to start the training.
        """
        print("[ Started training model context setup ]")
        self.model_config = _Helpers.make_model_config(self.model_path)
        _Helpers.set_random_seeds(self.model_config["training"]["manual_seed"])
        self.t_loader, self.v_loader = _Helpers.make_data_loaders(
            self.data_path, self.model_config["data_sets"]
        )
        self.completed_context_setup = True
        print("[ Completed training model context setup ]")
        print()

    def train_d(self):
        """Trains the discriminator with the training set."""
        self.t_batch = 0
        while self.t_batch < self.t_batch_count:
            self.train_d_with_curr_batch()
            self.print_train_d_stats_if_needed()
            self.t_batch += 1

    def train_g(self):
        """Trains the generator with the training set."""
        self.t_batch = 0
        while self.t_batch < self.t_batch_count:
            self.train_g_with_curr_batch()
            self.print_train_g_stats_if_needed()
            self.t_batch += 1

    def validate_d(self):
        """Validate the discriminator with the validation set."""
        self.v_batch = 0
        while self.v_batch < self.v_batch_count:
            self.d_v_batch_errs = numpy.array()
            self.validate_d_with_curr_batch()
            self.print_validate_d_stats_if_needed()
            self.v_batch += 1
        self.d_v_errs.append(self.d_v_batch_errs.mean())

    def validate_g(self):
        """Validate the generator with the validation set."""
        self.v_batch = 0
        while self.v_batch < self.v_batch_count:
            self.g_v_batch_errs = numpy.array()
            self.validate_g_with_curr_batch()
            self.print_validate_g_stats_if_needed()
            self.v_batch += 1
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
