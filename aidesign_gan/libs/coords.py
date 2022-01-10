"""Module of the coord (coordinator) classes."""

# First added by liu-yucheng
# Last updated by liu-yucheng

from torchvision import transforms
from torchvision import utils as vutils

import math
import torch

from aidesign_gan.libs import algos
from aidesign_gan.libs import configs
from aidesign_gan.libs import contexts
from aidesign_gan.libs import results
from aidesign_gan.libs import utils


class Coord:
    """Super class of the coord classes."""

    def __init__(self, model_path, logs):
        """Inits self with the given args.

        Args:
            model_path: the model path
            logs: the log file objects
        """
        self.model_path = model_path
        """Model path."""
        self.logs = logs
        """Log file objects"""

        self.coords_config = None
        """Coordinators configuration."""
        self.modelers_config = None
        """Modelers configuration."""
        self.results = None
        """Results."""
        self.context = None
        """Context."""
        self.results_ready = False
        """Whether self.results is ready."""
        self.context_ready = False
        """Whether self.context is ready."""

    def setup_results(self):
        """Sets up self.result.

        Raises:
            NotImplementedError: always
        """
        raise NotImplementedError("setup_results not implemented")

    def setup_context(self):
        """Sets up self.context.

        Raises:
            NotImplementedError: always
        """
        raise NotImplementedError("setup_context not implemented")


class TrainingCoord(Coord):
    """Training coordinator."""

    def __init__(self, data_path, model_path, logs):
        """Inits self with the given args.

        Args:
            data_path: the data path
            model_path: the model path
            logs: the log file objects
        """
        super().__init__(model_path, logs)
        self.data_path = data_path
        """Data path."""
        self.algo = None
        """Algorithm."""
        self.algo_ready = False
        """Whether self.algo is ready."""

    def setup_results(self):
        """Sets up self.result."""
        path = utils.concat_paths(self.model_path, "Training-Results")
        self.results = results.TrainingResults(path, self.logs)
        self.results.init_folders()
        self.results_ready = True

        self.results.logln("Completed results setup")

    def setup_context(self):
        """Sets up self.context."""
        if not self.results_ready:
            self.setup_results()

        self.coords_config = configs.CoordsConfig(self.model_path)
        self.modelers_config = configs.ModelersConfig(self.model_path)

        self.coords_config.load()
        self.modelers_config.load()

        self.results.log_configs(self.coords_config, self.modelers_config)

        self.context = contexts.TrainingContext()
        self.results.bind_context(self.context)

        training_key = "training"
        config = self.coords_config[training_key]
        self.context.setup_rand(config)
        self.results.log_rand()

        config = self.coords_config[training_key]
        self.context.setup_hw(config)
        self.results.log_hw()

        config = self.coords_config[training_key]
        self.context.setup_data(self.data_path, config)
        self.results.log_data()

        config = self.modelers_config
        self.context.setup_mods(config)
        self.results.log_mods()

        config = self.coords_config[training_key]
        self.context.setup_mode(config)
        self.results.log_mode()

        labels_key = "labels"
        if labels_key in self.coords_config[training_key]:
            config = self.coords_config[training_key][labels_key]
            self.context.setup_labels(config=config)
        else:  # elif "labels" not in self.coords_config[training_key]:
            self.context.setup_labels()
        self.results.log_labels()

        config = self.coords_config[training_key]
        self.context.setup_loops(config)

        self.context.setup_stats()

        self.context.setup_noises()

        self.context_ready = True
        self.results.logln("Completed context setup")

    def setup_algo(self):
        """Sets up self.algo.

        Raises:
            ValueError: if the algo's name is unknown
        """
        if not self.results_ready:
            self.setup_results()

        if not self.context_ready:
            self.setup_context()

        algo_name = self.coords_config["training"]["algorithm"]
        if algo_name == "alt_sgd_algo":
            self.algo = algos.AltSGDAlgo()
        elif algo_name == "pred_alt_sgd_algo":
            self.algo = algos.PredAltSGDAlgo()
        elif algo_name == "fair_pred_alt_sgd_algo":
            self.algo = algos.FairPredAltSGDAlgo()
        else:
            raise ValueError(f"Unknown algo: {algo_name}")

        self.results.log_algo(algo_name)
        self.algo.bind_context_and_results(self.context, self.results)
        self.algo_ready = True

        self.results.logln("Completed algo setup")

    def start_training(self):
        """Starts the training."""
        if not self.results_ready:
            self.setup_results()

        if not self.context_ready:
            self.setup_context()

        if not self.algo_ready:
            self.setup_algo()

        r = self.results
        r.logln("Started training")
        r.logln("-")
        self.algo.start()
        r.logln("Completed training")


class GenerationCoord(Coord):
    """Generation coordinator."""

    def __init__(self, model_path, log):
        """Inits self with the given args.

        Args:
            model_path: the model path
            log: the log file object
        """
        super().__init__(model_path, log)

    def setup_results(self):
        """Sets up self.result."""
        path = utils.concat_paths(self.model_path, "Generation-Results")
        self.results = results.GenerationResults(path, self.logs)
        self.results.init_folders()
        self.results_ready = True

        self.results.logln("Completed results setup")

    def setup_context(self):
        """Sets up self.context."""
        if not self.results_ready:
            self.setup_results()

        self.coords_config = configs.CoordsConfig(self.model_path)
        self.coords_config.load()
        self.modelers_config = configs.ModelersConfig(self.model_path)
        self.modelers_config.load()
        self.results.log_configs(self.coords_config, self.modelers_config)
        self.context = contexts.GenerationContext()
        self.results.bind_context(self.context)
        config = self.coords_config["generation"]
        self.context.setup_rand(config)
        self.results.log_rand()
        self.context.setup_hw(config)
        self.results.log_hw()
        self.context.setup_all(self.coords_config, self.modelers_config)
        self.results.log_g()
        self.context_ready = True

        self.results.logln("Completed context setup")

    def normalize_images(self):
        """Normalizes the images."""
        r = self.results
        c = self.context

        for index in range(len(c.images.to_save)):
            c.images.to_save[index] = vutils.make_grid(
                c.images.to_save[index], normalize=True, value_range=(-0.75, 0.75)
            )

        r.logln("Normalized images")

    def convert_images_to_grids(self):
        """Converts the images to grids."""
        r = self.results
        c = self.context

        orig_list = c.images.to_save
        c.images.to_save = []
        start_index = 0
        while start_index < c.images.count:
            end_index = start_index + c.grids.size_each
            grid = vutils.make_grid(
                orig_list[start_index: end_index], nrow=math.ceil(c.grids.size_each ** 0.5), padding=c.grids.padding,
            )
            c.images.to_save.append(grid)
            start_index = end_index

        r.logln("Converted images to grids")

    def start_generation(self):
        """Starts the generation."""
        if not self.results_ready:
            self.setup_results()

        if not self.context_ready:
            self.setup_context()

        r = self.results
        c = self.context

        r.logln("Started generation")
        c.images.to_save = []
        c.batch_prog.index = 0
        while c.batch_prog.index < c.batch_prog.count:
            noise_batch = c.noise_batches[c.batch_prog.index]
            image_batch = c.g.test(noise_batch)
            c.images.to_save.append(image_batch)
            r.log_batch()
            c.batch_prog.index += 1

        c.images.to_save = torch.cat(c.images.to_save)

        self.normalize_images()

        if c.grids.enabled:
            self.convert_images_to_grids()

        r.save_generated_images()

        r.logln("Completed generation")
