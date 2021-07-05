"""Module of the coord (coordinator) classes."""

import numpy

from gan.libs import configs
from gan.libs import contexts
from gan.libs import results
from gan.libs import utils


class TrainingCoord:
    """Training coordinator.

    Attributes:
        data_path: the data path
        model_path: the model path
        log: the log file object
        coords_config: the coords config
        modelers_config: the modelers config
        results: the training results
        context: the training context
        results_ready: whether the results are ready
        context_ready: whether the context is ready
    """

    def __init__(self, data_path, model_path, log):
        """Inits self with the given args.

        Args:
            data_path: the data path
            model_path: the model path
            log: the log file object
        """
        self.data_path = data_path
        self.model_path = model_path
        self.log = log
        self.coords_config = None
        self.modelers_config = None
        self.results = None
        self.context = None
        self.results_ready = False
        self.context_ready = False

    def setup_results(self):
        """Sets up the results, self.rst."""
        path = utils.concat_paths(self.model_path, "Training-Results")
        self.results = results.TrainingResults(path, self.log)
        self.results.init_folders()
        self.results_ready = True
        self.results.logln("Completed results setup")

    def setup_context(self):
        """Sets up the context, self.ctx."""
        if not self.results_ready:
            self.setup_results()
        self.coords_config = configs.CoordsConfig(self.model_path)
        self.coords_config.load()
        self.modelers_config = configs.ModelersConfig(self.model_path)
        self.modelers_config.load()
        self.results.log_configs(self.coords_config, self.modelers_config)
        self.context = contexts.TContext()
        self.results.bind_context(self.context)
        config = self.coords_config["training"]
        self.context.set_rand_seeds(config)
        self.results.log_rand_seeds()
        self.context.setup_device(config)
        self.results.log_device()
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

    def train_d_epoch(self):
        """Trains D for an epoch with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training D")
        losses = []
        r.logln("Started training D on real")
        c.loops.train_index = 0
        for batch in c.data.train.loader:
            batch = batch[0]
            out_mean, loss = c.mods.d.train(batch, c.labels.real)
            losses.append(loss)
            c.latest.dx = out_mean
            c.latest.ld = loss
            r.log_train_dr()
            c.loops.train_index += 1
        c.loops.train_index = 0
        r.logln("Started training D on fake")
        while c.loops.train_index < c.data.train.batch_count:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            batch = c.mods.g.test(noises)
            out_mean, loss = c.mods.d.train(batch, c.labels.fake)
            losses.append(loss)
            c.latest.dgz = out_mean
            c.latest.ld = loss
            r.log_train_df()
            c.loops.train_index += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.train.d.append(loss_mean)
        r.log_losses_train_d()

    def train_g_epoch(self):
        """Trains G for an epoch with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training G")
        losses = []
        c.loops.train_index = 0
        while c.loops.train_index < c.data.train.batch_count:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            out_mean, loss = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
            losses.append(loss)
            c.latest.dgz = out_mean
            c.latest.lg = loss
            r.log_train_g()
            c.loops.train_index += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.train.g.append(loss_mean)
        r.log_losses_train_g()

    def valid_d_epoch(self):
        """Validate D for an epoch with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating D")
        losses = []
        r.logln("Started validating D on real")
        c.loops.valid_index = 0
        for batch in c.data.valid.loader:
            batch = batch[0]
            out_mean, loss = c.mods.d.valid(batch, c.labels.real)
            losses.append(loss)
            c.latest.dx = out_mean
            c.latest.ld = loss
            r.log_validate_dr()
            c.loops.valid_index += 1
        r.logln("Started validating D on fake")
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            batch = c.mods.g.test(noises)
            out_mean, loss = c.mods.d.valid(batch, c.labels.fake)
            losses.append(loss)
            c.latest.dgz = out_mean
            c.latest.ld = loss
            r.log_validate_df()
            c.loops.valid_index += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.valid.d.append(loss_mean)
        r.log_losses_valid_d()

    def valid_g_epoch(self):
        """Validate G for an epoch with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating G")
        losses = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            out_mean, loss = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
            losses.append(loss)
            c.latest.dgz = out_mean
            c.latest.lg = loss
            r.log_validate_g()
            c.loops.valid_index += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.valid.g.append(loss_mean)
        r.log_losses_valid_g()

    def run_iter(self):
        """Runs an iter."""
        r = self.results
        c = self.context
        r.log_iter("Started")
        c.loops.epoch = 0
        while c.loops.epoch < c.loops.epoch_count:
            self.run_epoch()
            c.loops.epoch += 1
        r.log_iter("Completed")

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
            self.run_iter()
            c.loops.iter += 1
        r.logln("Completed training")
