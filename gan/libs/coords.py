"""Module of the coord (coordinator) classes."""

from torchvision import transforms
from torchvision import utils as visionutils
import numpy
import math

from torchvision.transforms.transforms import ToTensor

from gan.libs import configs
from gan.libs import contexts
from gan.libs import results
from gan.libs import utils


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

    def train_d(self):
        """Trains D with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training D")
        lds_on_real = []
        c.loops.train_index = 0
        for real_batch in c.data.train.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.train(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            lds_on_real.append(ld)
            r.log_batch("d", "tr")
            c.loops.train_index += 1
        lds_on_fake = []
        c.loops.train_index = 0
        while c.loops.train_index < c.data.train.batch_count:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.train(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            lds_on_fake.append(ld)
            r.log_batch("d", "tf")
            c.loops.train_index += 1
        lds = []
        for index in range(c.data.train.batch_count):
            lds.append(lds_on_real[index] + lds_on_fake[index])
        epoch_ld = numpy.array(lds).mean()
        c.losses.train.d.append(epoch_ld)
        r.log_epoch_loss("td")

    def valid_d(self):
        """Validates D with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating D")
        lds_on_real = []
        c.loops.valid_index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            lds_on_real.append(ld)
            r.log_batch("d", "vr")
            c.loops.valid_index += 1
        lds_on_fake = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            lds_on_fake.append(ld)
            r.log_batch("d", "vf")
            c.loops.valid_index += 1
        lds = []
        for index in range(c.data.valid.batch_count):
            lds.append(lds_on_real[index] + lds_on_fake[index])
        epoch_ld = numpy.array(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")

    def save_best_d(self):
        """Saves the D model that performs the best."""
        r = self.results
        c = self.context
        r.log_best_losses("d")
        curr_ld = c.losses.valid.d[-1]
        if c.bests.d is None or curr_ld <= c.bests.d:
            c.bests.d = curr_ld
            c.mods.d.save()
            r.log_model_action("save", "d")
        elif c.loops.es.d < c.loops.es.max:
            c.loops.es.d += 1
            r.log_model_action("es", "d")
        elif c.loops.rb.d < c.loops.rb.max:
            c.loops.rb.d += 1
            c.rbs.d.append((c.loops.iter, c.loops.epoch))
            c.loops.es.d = 0
            c.mods.d.rollback(c.loops.rb.d)
            r.log_model_action("rb", "d")
        else:
            c.mods.d.load()
            r.log_model_action("load", "d")

    def train_g(self):
        """Trains G with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training G")
        lgs = []
        c.loops.train_index = 0
        while c.loops.train_index < c.data.train.batch_count:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            dgz, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz, c.latest.lg = dgz, lg
            lgs.append(lg)
            r.log_batch("g", "t")
            c.loops.train_index += 1
        epoch_lg = numpy.array(lgs).mean()
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("tg")

    def valid_g(self):
        """Validates G with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating G")
        lgs = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            dgz, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz, c.latest.lg = dgz, lg
            lgs.append(lg)
            r.log_batch("g", "v")
            c.loops.valid_index += 1
        epoch_lg = numpy.array(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")

    def save_best_g(self):
        """Saves the G model that performs the best."""
        r = self.results
        c = self.context
        r.log_best_losses("g")
        curr_lg = c.losses.valid.g[-1]
        if c.bests.g is None or curr_lg <= c.bests.g:
            c.bests.g = curr_lg
            c.mods.g.save()
            r.log_model_action("save", "g")
        elif c.loops.es.g < c.loops.es.max:
            c.loops.es.g += 1
            r.log_model_action("es", "g")
        elif c.loops.rb.g < c.loops.rb.max:
            c.loops.rb.g += 1
            c.rbs.g.append((c.loops.iter, c.loops.epoch))
            c.loops.es.g = 0
            c.mods.g.rollback(c.loops.rb.g)
            r.log_model_action("rb", "g")
        else:
            c.mods.g.load()
            r.log_model_action("load", "g")

    def run_d_iter(self):
        """Runs a iter of training, validating, and saving D."""
        r = self.results
        c = self.context
        c.loops.epoch = 0
        while c.loops.epoch < c.loops.epoch_count:
            r.log_epoch("Started", "d")
            self.train_d()
            self.valid_d()
            self.save_best_d()
            r.save_d_losses()
            r.log_epoch("Completed", "d")
            c.loops.epoch += 1

    def run_g_iter(self):
        """Runs a iter of training, validating, and saving G."""
        r = self.results
        c = self.context
        c.loops.epoch = 0
        while c.loops.epoch < c.loops.epoch_count:
            r.log_epoch("Started", "g")
            self.train_g()
            self.valid_g()
            self.save_best_g()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.log_epoch("Completed", "g")
            c.loops.epoch += 1

    def start_training(self):
        """Starts the training."""
        if not self.results_ready:
            self.setup_results()
        if not self.context_ready:
            self.setup_context()
        r = self.results
        c = self.context
        r.logln("Started training")
        r.save_training_images()
        r.save_validation_images()
        c.loops.iter = 0
        while c.loops.iter < c.loops.iter_count:
            r.log_iter("Started")
            self.run_d_iter()
            self.run_g_iter()
            r.log_iter("Completed")
            c.loops.iter += 1
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
            c.images.list[index] = visionutils.make_grid(c.images.list[index], normalize=True)
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
            grid = visionutils.make_grid(
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
