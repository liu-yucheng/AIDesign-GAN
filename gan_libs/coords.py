"""Module of the coord (coordinator) classes."""

from gan_libs import configs
from gan_libs import contexts
from gan_libs import results
from gan_libs import utils
import numpy
import sys


class TCoord:
    """Training coordinator.

    Attributes:
        d_path: the data path
        m_path: the model path
        c_config: the coords config
        m_config: the modelers config
        log: the log file object
        res: the result
        res_ready: whether the results are ready
        ctx: the context
        ctx_ready: whether the context is ready
    """

    def __init__(self, d_path, m_path, log):
        """Inits self with the given args.

        Args:
            d_path: the data path
            m_path: the model path
            log: the log file object
        """
        self.d_path = d_path
        self.m_path = m_path
        self.c_config = None
        self.m_config = None
        self.log = log
        self.res = None
        self.res_ready = False
        self.ctx = None
        self.ctx_ready = False

    def setup_results(self):
        """Sets up the results, self.rst."""
        path = utils.concat_paths(self.m_path, "Training-Results")
        self.res = results.TResults(path, self.log)
        self.res.init_folders()
        self.res_ready = True
        self.res.logln("Completed results setup")

    def setup_context(self):
        """Sets up the context, self.ctx."""
        if not self.res_ready:
            self.setup_results()
        self.c_config = configs.CoordsConfig(self.m_path)
        self.c_config.load()
        self.m_config = configs.ModelersConfig(self.m_path)
        self.m_config.load()
        self.res.log_configs(self.c_config, self.m_config)
        self.ctx = contexts.TContext()
        config = self.c_config["training"]
        self.ctx.set_rand_seeds(config)
        self.res.log_rand_seeds(self.ctx)
        self.ctx.setup_device(config)
        self.res.log_device(self.ctx)
        self.ctx.setup_data_loaders(self.d_path, config)
        self.res.log_data_loaders(self.ctx)
        config = self.m_config
        self.ctx.setup_modelers(config)
        self.res.log_modelers(self.ctx)
        config = self.c_config["training"]
        self.ctx.setup_mode(config)
        self.res.log_mode(self.ctx)
        self.ctx.setup_labels()
        self.ctx.setup_loops(config)
        self.ctx.setup_stats()
        self.ctx.setup_noises()
        self.ctx_ready = True
        self.res.logln("Completed context setup")

    def save_mods(self):
        """Saves the models if needed."""
        res = self.res
        ctx = self.ctx
        if ctx.losses.vd[-1] < ctx.bests.iter_d:
            ctx.bests.iter_d = ctx.losses.vd[-1]
            res.log_save_d()
        if ctx.losses.vg[-1] < ctx.bests.iter_g:
            ctx.bests.iter_g = ctx.losses.vg[-1]
            res.log_save_g()

    def save_or_rollback_mods(self):
        """Saves or rollbacks the models if needed."""
        res = self.res
        ctx = self.ctx
        res.log_bests(ctx)
        if True or ctx.bests.iter_d <= ctx.bests.d:
            ctx.bests.d = ctx.bests.iter_d
            ctx.mods.d.save()
            res.log_save_d()
        else:
            ctx.mods.d.rollback()
            res.log_rollback_d(ctx)
        if True or ctx.bests.iter_g <= ctx.bests.g:
            ctx.bests.g = ctx.bests.iter_g
            ctx.mods.g.save()
            res.log_save_g()
        else:
            ctx.mods.g.rollback()
            res.log_rollback_g(ctx)

    def train_d(self):
        """Trains the discriminator with the training set."""
        res = self.res
        ctx = self.ctx
        res.logln("Started training D")
        losses = []
        res.logln("Started training D on real")
        ctx.loops.t_idx = 0
        for batch in ctx.data.tdl:
            batch = batch[0]
            out_mean, loss = ctx.mods.d.train(batch, ctx.labels.r)
            losses.append(loss)
            ctx.outs.dx = out_mean
            ctx.outs.ld = loss
            res.log_train_dr(ctx)
            ctx.loops.t_idx += 1
        ctx.loops.t_idx = 0
        res.logln("Started training D on fake")
        while ctx.loops.t_idx < ctx.data.t_batch_cnt:
            batch = ctx.mods.g.test(ctx.data.batch_size)
            out_mean, loss = ctx.mods.d.train(batch, ctx.labels.f)
            losses.append(loss)
            ctx.outs.dgz = out_mean
            ctx.outs.ld = loss
            res.log_train_df(ctx)
            ctx.loops.t_idx += 1
        losses = [x.detach().cpu() for x in losses]
        loss = numpy.array(losses).mean()
        ctx.losses.td.append(loss)
        res.log_td_loss(ctx)

    def train_g(self):
        """Trains the generator with the training set."""
        res = self.res
        ctx = self.ctx
        res.logln("Started training G")
        losses = []
        ctx.loops.t_idx = 0
        while ctx.loops.t_idx < ctx.data.t_batch_cnt:
            out_mean, loss = ctx.mods.g.train(
                ctx.mods.d.model, ctx.data.batch_size, ctx.labels.r)
            losses.append(loss)
            ctx.outs.dgz = out_mean
            ctx.outs.lg = loss
            res.log_train_g(ctx)
            ctx.loops.t_idx += 1
        losses = [x.detach().cpu() for x in losses]
        loss = numpy.array(losses).mean()
        ctx.losses.tg.append(loss)
        res.log_tg_loss(ctx)

    def train_both(self):
        """Trains d and g togehter at batch level with the training set."""
        res = self.res
        ctx = self.ctx
        res.logln("Started training D and G")
        lds, lgs = [], []
        ctx.loops.t_idx = 0
        for r_batch in ctx.data.tdl:
            r_batch = r_batch[0]
            dx, ldr = ctx.mods.d.train(r_batch, ctx.labels.r)
            f_batch = ctx.mods.g.test(ctx.data.batch_size)
            dgz, ldf = ctx.mods.d.train(f_batch, ctx.labels.f)
            dgz2, lg = ctx.mods.g.train(
                ctx.mods.d.model, ctx.data.batch_size, ctx.labels.r)
            ld = ldr + ldf
            lds.append(ld)
            lgs.append(lg)
            ctx.outs.dx = dx
            ctx.outs.dgz = dgz
            ctx.outs.dgz2 = dgz2
            ctx.outs.ld = ld
            ctx.outs.lg = lg
            res.log_train_both(ctx)
            ctx.loops.t_idx += 1
        lds = [x.detach().cpu() for x in lds]
        lgs = [x.detach().cpu() for x in lgs]
        ld = numpy.array(lds).mean()
        lg = numpy.array(lgs).mean()
        ctx.losses.td.append(ld)
        ctx.losses.tg.append(lg)
        res.log_td_loss(ctx)
        res.log_tg_loss(ctx)

    def validate_d(self):
        """Validate the discriminator with the validation set."""
        res = self.res
        ctx = self.ctx
        res.logln("Started validating D")
        losses = []
        res.logln("Started validating D on real")
        ctx.loops.v_idx = 0
        for noises in ctx.data.vdl:
            noises = noises[0]
            out_mean, loss = ctx.mods.d.validate(noises, ctx.labels.r)
            losses.append(loss)
            ctx.outs.dx = out_mean
            ctx.outs.ld = loss
            res.log_validate_dr(ctx)
            ctx.loops.v_idx += 1
        ctx.loops.v_idx = 0
        res.logln("Started validating D on fake")
        for noises in ctx.noises.vbs:
            batch = ctx.mods.g.test(None, noises=noises)
            out_mean, loss = ctx.mods.d.validate(batch, ctx.labels.f)
            losses.append(loss)
            ctx.outs.dgz = out_mean
            ctx.outs.ld = loss
            res.log_validate_df(ctx)
            ctx.loops.v_idx += 1
        losses = [x.detach().cpu() for x in losses]
        loss = numpy.array(losses).mean()
        ctx.losses.vd.append(loss)
        res.log_vd_loss(ctx)

    def validate_g(self):
        """Validate the generator with the validation set."""
        res = self.res
        ctx = self.ctx
        res.logln("Started validating G")
        losses = []
        ctx.loops.v_idx = 0
        for noises in ctx.noises.vbs:
            out_mean, loss = ctx.mods.g.validate(
                ctx.mods.d.model, None, ctx.labels.r, noises=noises)
            losses.append(loss)
            ctx.outs.dgz = out_mean
            ctx.outs.lg = loss
            res.log_validate_g(ctx)
            ctx.loops.v_idx += 1
        losses = [x.detach().cpu() for x in losses]
        loss = numpy.array(losses).mean()
        ctx.losses.vg.append(loss)
        res.log_vg_loss(ctx)

    def run_epoch(self):
        """Runs an epoch."""
        res = self.res
        ctx = self.ctx
        res.log_epoch(ctx, "Started")
        # self.train_d()
        # self.train_g()
        self.train_both()
        self.validate_d()
        self.validate_g()
        res.save_generated_images(ctx)
        res.save_d_losses(ctx)
        res.save_g_losses(ctx)
        res.save_tvg(ctx)
        self.save_mods()
        res.log_epoch(ctx, "Completed")

    def run_iter(self):
        """Runs an iter."""
        res = self.res
        ctx = self.ctx
        res.log_iter(ctx, "Started")
        ctx.bests.iter_d = sys.maxsize
        ctx.bests.iter_g = sys.maxsize
        ctx.loops.epoch_num = 0
        while ctx.loops.epoch_num < ctx.loops.epoch_cnt:
            self.run_epoch()
            self.save_or_rollback_mods()
            ctx.loops.epoch_num += 1
        res.log_iter(ctx, "Completed")

    def start_training(self):
        """Starts the training."""
        if not self.res_ready:
            self.setup_results()
        if not self.ctx_ready:
            self.setup_context()
        res = self.res
        ctx = self.ctx
        res.logln("Started training")
        res.save_training_images(ctx)
        res.save_validation_images(ctx)
        ctx.loops.iter_num = 0
        while ctx.loops.iter_num < ctx.loops.iter_cnt:
            self.run_iter()
            ctx.loops.iter_num += 1
        res.logln("Completed training")
