"""Module of the coordinator classes."""

# import system and third party modules
import numpy
import os
import pathlib
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import custom modules
import gan_libs.configs as configs
import gan_libs.modelers as modelers
import gan_libs.results as results


class _Helpers:
    """Helpers for classes in the module."""

    @classmethod
    def find_coords_config(cls, path):
        """Finds the coords config in a given path.

        Returns: the coords config
        """
        config = configs.CoordsConfig()
        config.location = configs.CoordsConfig.find_in_path(path)
        config.load()
        print("Coords config: {}".format(config.location))
        return config

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
    def prep_data_loaders(cls, path, config):
        """Prepares the data loaders given a path and a data sets config.

        Prepares the training set and validation set loaders based on the
        weights and percentage to use in the config.

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
        print("Subset ratio: {}; ".format(subset_ratio), end="")

        prop_to_use = config["percentage_to_use"] / 100
        if prop_to_use < 0:
            prop_to_use = 0
        if prop_to_use > 1:
            prop_to_use = 1
        print("Proportion to use: {}/1".format(prop_to_use))

        total_to_use = int(prop_to_use * total_size)
        t_start, t_end = 0, int(subset_ratio[0] * total_to_use)
        v_start, v_end = t_end, total_to_use
        t_indices = list(range(t_start, t_end))
        v_indices = list(range(v_start, v_end))
        t_set = data.Subset(data_set, t_indices)
        v_set = data.Subset(data_set, v_indices)
        print("Training set size: {}; ".format(len(t_set)), end="")
        print("Validation set size: {}".format(len(v_set)))

        worker_count = config["loader_worker_count"]
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
        # print("Training batch count: {}; ".format(len(t_loader)), end="")
        # print("Validation batch count: {}".format(len(v_loader)))

        return t_loader, v_loader

    @classmethod
    def find_device(cls, gpu_count):
        """Finds the pytorch device to use given a GPU count.

        Returns: the pytorch device
        """
        device_name = None
        if torch.cuda.is_available() and gpu_count > 0:
            device_name = "cuda:0"
        else:
            device_name = "cpu"
        device = torch.device(device_name)
        print("Device name: {}".format(device_name))
        return device

    @classmethod
    def init_folder(cls, path, clean=False):
        """
        Initializes a folder given a path.

        Args:
            path:   the path to the folder
            clean:  whether to clean up the folder
        """
        if clean and os.path.exists(path):
            shutil.rmtree(path)
        pathlib.Path(path).mkdir(exist_ok=True)
        if clean:
            print("Init'd folder (clean): {}".format(path))
        else:
            print("Init'd folder: {}".format(path))

    @classmethod
    def find_modelers_config(cls, path):
        """Finds the modelers config in a given path.

        Returns: the modelers config
        """
        config = configs.ModelersConfig()
        config.location = configs.ModelersConfig.find_in_path(path)
        config.load()
        print("Modelers config: {}".format(config.location))
        return config

    @classmethod
    def prep_modelers(cls, config, device, gpu_count):
        """Prepares the modelers with the given arguments.

        Returns: the discriminator and generateor modelers.
        """
        loss_func = nn.BCELoss()
        d = modelers.DModeler(
            config["discriminator"], device, gpu_count, loss_func,
            training=True
        )
        print("==== D modeler's model ====")
        print(d.model)
        print()
        g = modelers.GModeler(
            config["generator"], device, gpu_count, loss_func, training=True
        )
        print("==== G modeler's model ====")
        print(g.model)
        print()
        return d, g


class TrainingCoord:
    """Training coordinator.

    A coordinator that handles the actions involving the components needed for
    GAN training.
    """

    def __init__(self, data_path, model_path):
        """Initializes a trainer given the data and model paths."""
        if data_path is None:
            raise ValueError("data_path cannot be None")
        if model_path is None:
            raise ValueError("model_path cannot be None")

        self.data_path = data_path
        self.model_path = model_path
        self.completed_context_setup = False

    def setup_context(self):
        """Sets up the internal states of the trainer.

        Gets ready to start the training.
        """
        print("____ Training context setup ____")

        coords_config = _Helpers.find_coords_config(self.model_path)
        self.config = coords_config["training"]

        _Helpers.set_random_seeds(self.config["manual_seed"])

        self.t_loader, self.v_loader = _Helpers.\
            prep_data_loaders(self.data_path, self.config["data_sets"])

        gpu_count = self.config["gpu_count"]
        self.device = _Helpers.find_device(gpu_count)

        self.results_path = str(
            pathlib.Path(self.model_path + "/Training-Results").absolute()
        )
        _Helpers.init_folder(self.results_path)
        self.generated_images_path = str(
            pathlib.Path(self.results_path + "/Generated-Images").absolute()
        )
        _Helpers.init_folder(self.generated_images_path, clean=True)

        modelers_config = _Helpers.find_modelers_config(self.model_path)
        self.d, self.g = _Helpers.\
            prep_modelers(modelers_config, self.device, gpu_count)

        mode = self.config["mode"]
        if mode == "new":
            self.d.save()
            self.g.save()
        elif mode == "resume":
            self.d.load()
            self.g.load()
        else:
            raise ValueError("Unknown training mode: {}".format(mode))
        print("Training mode: {}".format(mode))

        self.iter_count = self.config["iteration_count"]
        self.iter = 0
        self.epoch_count = self.config["epochs_per_iteration"]
        self.epoch = 0
        self.t_batch_count = len(self.t_loader)
        self.t_index = 0
        self.v_batch_count = len(self.v_loader)
        self.v_index = 0
        self.batch_size = self.config["data_sets"]["images_per_batch"]

        self.real = 1.0
        self.fake = 0.0

        self.dt_losses = []
        self.dv_losses = []
        self.gt_losses = []
        self.gv_losses = []

        self.d_best_loss = sys.maxsize
        self.g_best_loss = sys.maxsize
        self.d_iter_best_loss = sys.maxsize
        self.g_iter_best_loss = sys.maxsize

        self.completed_context_setup = True
        print("^^^^ Training context setup ^^^^")
        print()

    def train_d(self):
        """Trains the discriminator with the training set."""
        print("== {}.{}. Train D ==".format(self.iter + 1, self.epoch + 1))
        self.t_index = 0
        for real_batch in self.t_loader:
            real_batch = real_batch[0]
            real_output, real_loss = self.d.train(real_batch, self.real)
            fake_batch = self.g.test(self.batch_size)
            fake_output, fake_loss = self.d.train(fake_batch, self.fake)
            loss = real_loss + fake_loss
            if self.t_index == 0 or (self.t_index + 1) % 50 == 0 or self.t_index == self.t_batch_count - 1:
                print(
                    "Batch {}/{}: Average D(X): {:.4f} Averge D(G(Z)): {:.4f} Loss(D): {:.4f}".format(
                        self.t_index + 1, self.t_batch_count,
                        real_output.mean().item(), fake_output.mean().item(),
                        loss
                    )
                )
            self.t_index += 1

    def train_g(self):
        """Trains the generator with the training set."""
        print("== {}.{}. Train G ==".format(self.iter + 1, self.epoch + 1))
        self.t_index = 0
        while self.t_index < self.t_batch_count:
            output, loss = self.g.train(
                self.d.model, 2 * self.batch_size, self.real)
            if self.t_index == 0 or (self.t_index + 1) % 50 == 0 or self.t_index == self.t_batch_count - 1:
                print(
                    "Batch {}/{}: Averge D(G(Z)): {:.4f} Loss(G): {:.4f}".format(
                        self.t_index + 1, self.t_batch_count, output.mean().item(), loss
                    )
                )
            self.t_index += 1
    
    def train_d_and_g(self):
        print("== {}.{}. Train D and G ==".format(self.iter + 1, self.epoch + 1))
        self.t_index = 0
        for real_batch in self.t_loader:
            real_batch = real_batch[0]
            real_output, real_loss = self.d.train(real_batch, self.real)
            fake_batch = self.g.test(self.batch_size)
            fake_output, fake_loss = self.d.train(fake_batch, self.fake)
            d_loss = real_loss + fake_loss

            g_output, g_loss = self.g.train(self.d.model, 2 * self.batch_size, self.real)

            if self.t_index == 0 or (self.t_index + 1) % 50 == 0 or self.t_index == self.t_batch_count - 1:
                print(
                    "Batch {}/{}:\n"
                    "\tTrain D: Average D(X): {:.4f} Averge D(G(Z)): {:.4f} Loss(D): {:.4f}\n"
                    "\tTrain G: Average D(G(Z)): {:.4f} Loss(G): {:.4f}".format(
                        self.t_index + 1, self.t_batch_count,
                        real_output.mean().item(), fake_output.mean().item(), d_loss,
                        g_output.mean().item(), g_loss
                    )
                )
            self.t_index += 1


    def validate_d(self):
        """Validate the discriminator with the validation set."""
        print("== {}.{}. Validate D ==".format(self.iter + 1, self.epoch + 1))
        self.v_index = 0
        losses = numpy.array([])
        for real_batch in self.v_loader:
            real_batch = real_batch[0]
            real_output, real_loss = self.d.validate(real_batch, self.real)
            fake_batch = self.g.test(self.batch_size)
            fake_output, fake_loss = self.d.validate(fake_batch, self.fake)
            loss = real_loss + fake_loss
            losses = numpy.append(losses, [loss.cpu()])
            if self.v_index == 0 or (self.v_index + 1) % 50 == 0 or self.v_index == self.v_batch_count - 1:
                print(
                    "Batch {}/{}: Average D(X): {:.4f} Averge D(G(Z)): {:.4f} Loss(D): {:.4f}".format(
                        self.v_index + 1, self.v_batch_count,
                        real_output.mean().item(), fake_output.mean().item(),
                        loss
                    )
                )
            self.v_index += 1
        average_loss = losses.mean()
        if average_loss < self.d_iter_best_loss:
            self.d_iter_best_loss = average_loss

    def validate_g(self):
        """Validate the generator with the validation set."""
        print("== {}.{}. Validate G ==".format(self.iter + 1, self.epoch + 1))
        self.v_index = 0
        losses = numpy.array([])
        while self.v_index < self.v_batch_count:
            output, loss = self.g.validate(
                self.d.model, 2 * self.batch_size, self.real)
            losses = numpy.append(losses, [loss.cpu()])
            if self.v_index == 0 or (self.v_index + 1) % 50 == 1 or self.v_index == self.v_batch_count - 1:
                print(
                    "Batch {}/{}: Averge D(G(Z)): {:.4f} Loss(G): {:.4f}".format(
                        self.v_index + 1, self.v_batch_count, output.mean().item(), loss
                    )
                )
            self.v_index += 1
        average_loss = losses.mean()
        if average_loss < self.g_iter_best_loss:
            self.g_iter_best_loss = average_loss

    def run_epoch(self):
        """Runs an epoch."""
        print("==== {}. Epoch {}/{} ====".format(
            self.iter + 1, self.epoch + 1, self.epoch_count
        ))

        self.dt_losses[-1].append([])
        self.dv_losses[-1].append([])
        self.gt_losses[-1].append([])
        self.gv_losses[-1].append([])

        # self.train_d()
        # self.train_g()
        self.train_d_and_g()

        results.save_generated_images(self.g.model, self.g.gen_noises(
            64), self.iter, self.epoch, self.generated_images_path)

        self.validate_d()
        self.validate_g()

        print()

    def run_iter(self):
        """Runs an iteration."""
        print("==== ==== Iter {}/{} ==== ====".format(self.iter + 1, self.iter_count))

        self.dt_losses.append([])
        self.dv_losses.append([])
        self.gt_losses.append([])
        self.gv_losses.append([])
        self.d_iter_best_loss = sys.maxsize
        self.g_iter_best_loss = sys.maxsize

        self.epoch = 0
        while self.epoch < self.epoch_count:
            self.run_epoch()
            self.epoch += 1

        if self.d_iter_best_loss < self.d_best_loss:
            self.d.save()
            self.d_best_loss = self.d_iter_best_loss
            print("Saved D")
        else:
            self.d.rollback()
            print("Rollbacked (count: {}) D".format(self.d.rollback_count))

        if self.g_iter_best_loss < self.g_best_loss:
            self.g.save()
            self.g_best_loss = self.g_iter_best_loss
            print("Saved G")
        else:
            self.g.rollback()
            print("Rollbacked (count: {}) G".format(self.g.rollback_count))

        print()

    def start_training(self):
        """Starts the training."""
        print("____ Training ____")

        if not self.completed_context_setup:
            self.setup_context()

        results.save_training_images(
            next(iter(self.t_loader)), self.results_path, self.device
        )
        results.save_validation_images(
            next(iter(self.v_loader)), self.results_path, self.device
        )

        self.iter = 0
        while self.iter < self.iter_count:
            self.run_iter()
            self.iter += 1

        print("^^^^ Training ^^^^")
        print()
