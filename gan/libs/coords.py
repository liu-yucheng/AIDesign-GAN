"""Module of the coord (coordinator) classes."""

import numpy
import sys

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
        self.context.setup_data_loaders(self.data_path, config)
        self.results.log_data_loaders()
        config = self.modelers_config
        self.context.setup_modelers(config)
        self.results.log_modelers()
        config = self.coords_config["training"]
        self.context.setup_mode(config)
        self.results.log_mode()
        self.context.setup_labels()
        self.context.setup_loops(config)
        self.context.setup_stats()
        self.context.setup_noises()
        self.context_ready = True
        self.results.logln("Completed context setup")

    def save_mods(self):
        """Saves the models if needed."""
        r = self.results
        c = self.context
        if c.losses.vd[-1] < c.bests.iter_d:
            c.bests.iter_d = c.losses.vd[-1]
            r.log_save_d()
        if c.losses.vg[-1] < c.bests.iter_g:
            c.bests.iter_g = c.losses.vg[-1]
            r.log_save_g()

    def save_or_rollback_mods(self):
        """Saves or rollbacks the models if needed."""
        r = self.results
        c = self.context
        r.log_bests()
        if True or c.bests.iter_d <= c.bests.d:
            c.bests.d = c.bests.iter_d
            c.mods.d.save()
            r.log_save_d()
        else:
            c.mods.d.rollback()
            r.log_rollback_d()
        if True or c.bests.iter_g <= c.bests.g:
            c.bests.g = c.bests.iter_g
            c.mods.g.save()
            r.log_save_g()
        else:
            c.mods.g.rollback()
            r.log_rollback_g()

    def train_d(self):
        """Trains the discriminator with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training D")
        losses = []
        r.logln("Started training D on real")
        c.loops.t_idx = 0
        for batch in c.data.tdl:
            batch = batch[0]
            out_mean, loss = c.mods.d.train(batch, c.labels.r)
            losses.append(loss)
            c.outs.dx = out_mean
            c.outs.ld = loss
            r.log_train_dr()
            c.loops.t_idx += 1
        c.loops.t_idx = 0
        r.logln("Started training D on fake")
        while c.loops.t_idx < c.data.t_batch_cnt:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            batch = c.mods.g.test(noises)
            out_mean, loss = c.mods.d.train(batch, c.labels.f)
            losses.append(loss)
            c.outs.dgz = out_mean
            c.outs.ld = loss
            r.log_train_df()
            c.loops.t_idx += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.td.append(loss_mean)
        r.log_td_loss()

    def train_g(self):
        """Trains the generator with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training G")
        losses = []
        c.loops.t_idx = 0
        while c.loops.t_idx < c.data.t_batch_cnt:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            out_mean, loss = c.mods.g.train(c.mods.d.model, noises, c.labels.r)
            losses.append(loss)
            c.outs.dgz = out_mean
            c.outs.lg = loss
            r.log_train_g()
            c.loops.t_idx += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.tg.append(loss_mean)
        r.log_tg_loss()

    def train_both(self):
        """Trains d and g togehter at batch level with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training D and G")
        lds, lgs = [], []
        c.loops.t_idx = 0
        for r_batch in c.data.tdl:
            r_batch = r_batch[0]
            dx, ldr = c.mods.d.train(r_batch, c.labels.r)
            noises = c.mods.g.generate_noises(c.data.batch_size)
            f_batch = c.mods.g.test(noises)
            dgz, ldf = c.mods.d.train(f_batch, c.labels.f)
            noises = c.mods.g.generate_noises(c.data.batch_size)
            dgz2, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.r)
            ld = ldr + ldf
            lds.append(ld)
            lgs.append(lg)
            c.outs.dx = dx
            c.outs.dgz = dgz
            c.outs.dgz2 = dgz2
            c.outs.ld = ld
            c.outs.lg = lg
            r.log_train_both()
            c.loops.t_idx += 1
        ld_mean = numpy.array(lds).mean()
        lg_mean = numpy.array(lgs).mean()
        c.losses.td.append(ld_mean)
        c.losses.tg.append(lg_mean)
        r.log_td_loss()
        r.log_tg_loss()

    def validate_d(self):
        """Validate the discriminator with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating D")
        losses = []
        r.logln("Started validating D on real")
        c.loops.v_idx = 0
        for batch in c.data.vdl:
            batch = batch[0]
            out_mean, loss = c.mods.d.validate(batch, c.labels.r)
            losses.append(loss)
            c.outs.dx = out_mean
            c.outs.ld = loss
            r.log_validate_dr()
            c.loops.v_idx += 1
        r.logln("Started validating D on fake")
        c.loops.v_idx = 0
        for noises in c.noises.vbs:
            batch = c.mods.g.test(noises)
            out_mean, loss = c.mods.d.validate(batch, c.labels.f)
            losses.append(loss)
            c.outs.dgz = out_mean
            c.outs.ld = loss
            r.log_validate_df()
            c.loops.v_idx += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.vd.append(loss_mean)
        r.log_vd_loss()

    def validate_g(self):
        """Validate the generator with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating G")
        losses = []
        c.loops.v_idx = 0
        for noises in c.noises.vbs:
            out_mean, loss = c.mods.g.validate(c.mods.d.model, noises, c.labels.r)
            losses.append(loss)
            c.outs.dgz = out_mean
            c.outs.lg = loss
            r.log_validate_g()
            c.loops.v_idx += 1
        loss_mean = numpy.array(losses).mean()
        c.losses.vg.append(loss_mean)
        r.log_vg_loss()

    def run_epoch(self):
        """Runs an epoch."""
        r = self.results
        c = self.context
        r.log_epoch("Started")
        # self.train_d()
        # self.train_g()
        self.train_both()
        self.validate_d()
        self.validate_g()
        r.save_generated_images()
        r.save_d_losses()
        r.save_g_losses()
        r.save_tvg()
        self.save_mods()
        r.log_epoch("Completed")

    def run_iter(self):
        """Runs an iter."""
        r = self.results
        c = self.context
        r.log_iter("Started")
        c.bests.iter_d = sys.maxsize
        c.bests.iter_g = sys.maxsize
        c.loops.epoch_num = 0
        while c.loops.epoch_num < c.loops.epoch_cnt:
            self.run_epoch()
            self.save_or_rollback_mods()
            c.loops.epoch_num += 1
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
        c.loops.iter_num = 0
        while c.loops.iter_num < c.loops.iter_cnt:
            self.run_iter()
            c.loops.iter_num += 1
        r.logln("Completed training")
