"""Module of the coord (coordinator) classes."""

from torchvision import utils as vutils
import math

from dcgan.libs import algos
from dcgan.libs import configs
from dcgan.libs import contexts
from dcgan.libs import results
from dcgan.libs import utils


class Coord:
    """Super class of the coord classes.

    Attributes:
        model_path: the model path
        log: the log file object
        coords_config: the coords config
        modelers_config: the modelers config
        results: the training results
        context: the training context
        results_ready: whether the results are ready
        context_ready: whether the context is ready
    """

    def __init__(self, model_path, log):
        """Inits self with the given args.

        Args:
            model_path: the model path
            log: the log file object
        """
        self.model_path = model_path
        self.log = log
        self.coords_config = None
        self.modelers_config = None
        self.results = None
        self.context = None
        self.results_ready = False
        self.context_ready = False

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
    """Training coordinator.

    Attributes:
        data_path: the data path
        algo: the training algo (algorithm)
        algo_ready: whether the algo is ready
    """

    def __init__(self, data_path, model_path, log):
        """Inits self with the given args.

        Args:
            data_path: the data path
            model_path: the model path
            log: the log file object
        """
        super().__init__(model_path, log)
        self.data_path = data_path
        self.algo = None
        self.algo_ready = False

    def setup_results(self):
        """Sets up self.result."""
        path = utils.concat_paths(self.model_path, "Training-Results")
        self.results = results.TrainingResults(path, self.log)
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
        self.context = contexts.TrainingContext()
        self.results.bind_context(self.context)
        config = self.coords_config["training"]
        self.context.setup_rand(config)
        self.results.log_rand()
        self.context.setup_hw(config)
        self.results.log_hw()
        self.context.setup_data(self.data_path, config)
        self.results.log_data()
        config = self.modelers_config
        self.context.setup_mods(config)
        self.results.log_mods()
        config = self.coords_config["training"]
        self.context.setup_mode(config)
        self.results.log_mode()
        self.context.setup_labels()
        self.context.setup_loops(config)
        self.context.setup_stats()
        self.context.setup_noises()
        self.context_ready = True
        self.results.logln("Completed context setup")

    def setup_algo(self):
        """Sets up self.algo."""
        if not self.results_ready:
            self.setup_results()
        if not self.context_ready:
            self.setup_context()
        self.algo = algos.IterLevelAlgo()
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
        self.algo.start_training()


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
        self.results = results.GenerationResults(path, self.log)
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
        for index in range(len(c.images.list)):
            c.images.list[index] = vutils.make_grid(c.images.list[index], normalize=True)
        r.logln("Normalized images")

    def convert_images_to_grids(self):
        """Converts the images to grids."""
        r = self.results
        c = self.context
        orig_list = c.images.list
        c.images.list = []
        start_index = 0
        while start_index < c.images.count:
            end_index = start_index + c.grids.each_size
            grid = vutils.make_grid(
                orig_list[start_index: end_index],
                nrow=math.ceil(c.grids.each_size ** 0.5),
                padding=c.grids.padding
            )
            c.images.list.append(grid)
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
        c.images.list = c.g.test(c.noises).cpu()
        self.normalize_images()
        if c.grids.enabled:
            self.convert_images_to_grids()
        r.save_generated_images()
        r.logln("Completed generation")
