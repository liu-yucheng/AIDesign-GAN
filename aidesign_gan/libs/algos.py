"""Module of the training algos (algorithms)."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import numpy

from aidesign_gan.libs import contexts
from aidesign_gan.libs import results
from aidesign_gan.libs import utils

_TrainingResults = results.TrainingResults
_TrainingContext = contexts.TrainingContext


class Algo:
    """Super class of the algo classes.

    Attributes:
        context: the binded training context
        results: the binded training results
    """

    def __init__(self):
        """Inits self."""
        self.context = None
        self.results = None

    def bind_context_and_results(self, context, results):
        """Binds the context and results.

        Args:
            context: the context to bind
            results: the results to bind
        """
        self.context = context
        self.results = results

    def check_context_and_results(self):
        """Checks if the context and results are binded.

        Raises:
            ValueError: if the context or results are None
        """
        if self.context is None:
            raise ValueError("self.context cannot be None")
        if self.results is None:
            raise ValueError("self.results cannot be None")

    def start_training(self):
        """Starts the training algorithm.

        Raises:
            NotImplementedError: always
        """
        raise NotImplementedError("start_training not implemented")


class AltSGDAlgo(Algo):
    """Algorithm that trains D and G alternatively with SGD."""

    def __init__(self):
        """Inits self."""
        super().__init__()

    def train_and_step_d(self, real_batch, fake_batch):
        """Trains and steps D.

        Args:
            real_batch: the real batch
            fake_batch: the fake batch

        Returns:
            dx, : the output mean of D on the real batch
            ldr, : the loss of D on the real batch
            dgz, : the output mean of D on the fake batch
            ldf: the loss of D on the fake batch
        """
        c: _TrainingContext = self.context

        c.mods.d.clear_grads()
        dx, ldr = c.mods.d.train(real_batch, c.labels.real)
        c.mods.d.step_optim()

        c.mods.d.clear_grads()
        dgz, ldf = c.mods.d.train(fake_batch, c.labels.fake)
        c.mods.d.step_optim()

        return dx, ldr, dgz, ldf

    def train_and_step_g(self, noises):
        """Trains and steps G.

        Args:
            noises: the G input batch of noises

        Returns:
            dgz2, : the output mean of D on G(noises)
            lg: the loss of G on the batch
        """
        c: _TrainingContext = self.context

        c.mods.g.clear_grads()
        dgz2, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
        c.mods.g.step_optim()

        return dgz2, lg

    def train_both(self):
        """Trains D and G together."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started training both D and G")
        lds = []
        lgs = []
        c.collapses.batch_count = 0
        c.loops.train.index = 0
        for real_batch in c.data.train.loader:
            # Prepare D training materials
            d_real_batch = real_batch[0]
            d_noises = c.mods.g.generate_noises(c.data.batch_size)
            d_fake_batch = c.mods.g.test(d_noises)

            # Prepare G training materials
            g_noises = c.mods.g.generate_noises(c.data.batch_size)

            # Filp a coin to determine whether to reverse the training order
            reverse_order = utils.rand_bool()
            if reverse_order:
                # Train G, and then train D
                g_results = self.train_and_step_g(g_noises)
                d_results = self.train_and_step_d(d_real_batch, d_fake_batch)
            else:  # elif not reverse_order:
                # Train D, and then train G (original order)
                d_results = self.train_and_step_d(d_real_batch, d_fake_batch)
                g_results = self.train_and_step_g(g_noises)

            # Parse the training results
            dx, ldr, dgz, ldf = d_results
            dgz2, lg = g_results

            # Detect training collapse
            collapsed = bool(ldr >= c.collapses.max_loss)
            collapsed = collapsed or bool(ldf >= c.collapses.max_loss)
            collapsed = collapsed or bool(lg >= c.collapses.max_loss)
            if collapsed:
                c.collapses.batch_count += 1

            # Update the statistics
            ld = ldr + ldf
            c.latest.dx, c.latest.dgz, c.latest.ld = dx, dgz, ld
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lds.append(ld)
            lgs.append(lg)
            r.log_batch_2("t")

            c.loops.train.index += 1

        # end for

        epoch_collapsed = c.collapses.batch_count >= c.collapses.max_batch_count
        if epoch_collapsed:
            c.collapses.epochs.append((c.loops.iteration.index, c.loops.epoch.index))
            r.logln("Epoch training collapsed")

        epoch_ld = numpy.array(lds).mean()
        epoch_lg = numpy.array(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")

    def valid_d(self):
        """Validates D with the validation set."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started validating D")
        ldrs = []
        c.loops.valid.index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            ldrs.append(ld)
            r.log_batch_2("vdr")
            c.loops.valid.index += 1

        ldfs = []
        c.loops.valid.index = 0
        for noises in c.noises.valid:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            ldfs.append(ld)
            r.log_batch_2("vdf")
            c.loops.valid.index += 1

        lds = []
        for index in range(c.loops.valid.count):
            lds.append(ldrs[index] + ldfs[index])

        epoch_ld = numpy.array(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")

    def valid_g(self):
        """Validates G with the validation set."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started validating G")
        lgs = []
        c.loops.valid.index = 0
        for noises in c.noises.valid:
            dgz2, lg = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lgs.append(lg)
            r.log_batch_2("vg")
            c.loops.valid.index += 1

        epoch_lg = numpy.array(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")

    def save_best_d(self):
        """Saves the best D."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.log_best_losses("d")
        curr_ld = c.losses.valid.d[-1]
        if c.bests.d is None or curr_ld <= c.bests.d:
            c.bests.d = curr_ld

        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iteration.index, c.loops.epoch.index) == c.collapses.epochs[-1]

        if epoch_collapsed:
            c.mods.d.load()
            r.log_model_action("load", "d")
        else:
            c.mods.d.save()
            r.log_model_action("save", "d")

    def save_best_g(self):
        """Saves the best G."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.log_best_losses("g")
        curr_lg = c.losses.valid.g[-1]
        if c.bests.g is None or curr_lg <= c.bests.g:
            c.bests.g = curr_lg

        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iteration.index, c.loops.epoch.index) == c.collapses.epochs[-1]

        if epoch_collapsed:
            c.mods.g.load()
            r.log_model_action("load", "g")
        else:
            c.mods.g.save()
            r.log_model_action("save", "g")

    def run_iter(self):
        """Runs a iter of multiple epochs."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        c.loops.epoch.index = 0
        while c.loops.epoch.index < c.loops.epoch.count:
            r.log_epoch("Started", "")
            self.train_both()
            self.valid_d()
            self.save_best_d()
            self.valid_g()
            self.save_best_g()
            r.save_d_losses()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln("-")
            c.loops.epoch.index += 1

    def start_training(self):
        """Starts the training algorithm."""
        self.check_context_and_results()
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started batch level algorithm")
        r.logln("-")
        r.save_training_images()
        r.save_validation_images()
        r.save_images_before_training()
        c.loops.iteration.index = 0
        while c.loops.iteration.index < c.loops.iteration.count:
            r.log_iter("Started")
            self.run_iter()
            r.logln("-")
            c.loops.iteration.index += 1


class PredAltSGDAlgo(Algo):
    """Predictive alternating SGD, which is a "alt SGD algo" with prediction steps."""

    def __init__(self):
        """Inits self."""
        super().__init__()

    def train_and_step_d(self, real_batch, fake_batch):
        """Trains and steps D with predicted G.

        Args:
            real_batch: the real batch
            fake_batch: the fake batch

        Returns:
            dx, : the output mean of D on the real batch
            ldr, : the loss of D on the real batch
            dgz, : the output mean of D on the fake batch
            ldf: the loss of D on the fake batch
        """
        c: _TrainingContext = self.context

        not_1st_batch = bool(c.loops.train.index > 0)
        if not_1st_batch:
            c.mods.g.predict()

        c.mods.d.clear_grads()
        dx, ldr = c.mods.d.train(real_batch, c.labels.real)
        c.mods.d.step_optim()

        c.mods.d.clear_grads()
        dgz, ldf = c.mods.d.train(fake_batch, c.labels.fake)
        c.mods.d.step_optim()

        if not_1st_batch:
            c.mods.g.restore()

        return dx, ldr, dgz, ldf

    def train_and_step_g(self, noises):
        """Trains and steps G with predicted D.

        Args:
            noises: the G input batch of noises

        Returns:
            dgz2, : the output mean of D on G(noises)
            lg: the loss of G on the batch
        """
        c: _TrainingContext = self.context

        not_1st_batch = bool(c.loops.train.index > 0)
        if not_1st_batch:
            c.mods.d.predict()

        c.mods.g.clear_grads()
        dgz2, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
        c.mods.g.step_optim()

        if not_1st_batch:
            c.mods.d.restore()

        return dgz2, lg

    def train_both(self):
        """Trains both D and G together."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started training both D and G")
        lds = []
        lgs = []
        c.collapses.batch_count = 0
        c.loops.train.index = 0
        for real_batch in c.data.train.loader:
            # Prepare D training materials
            d_real_batch = real_batch[0]
            d_noises = c.mods.g.generate_noises(c.data.batch_size)
            d_fake_batch = c.mods.g.test(d_noises)

            # Prepare G training materials
            g_noises = c.mods.g.generate_noises(c.data.batch_size)

            # Filp a coin to determine whether to reverse the training order
            reverse_order = utils.rand_bool()
            if reverse_order:
                # Train G, and then train D
                g_results = self.train_and_step_g(g_noises)
                d_results = self.train_and_step_d(d_real_batch, d_fake_batch)
            else:  # elif not reverse order:
                # Train D, and then train G (original order)
                d_results = self.train_and_step_d(d_real_batch, d_fake_batch)
                g_results = self.train_and_step_g(g_noises)

            # Parse the training results
            dx, ldr, dgz, ldf = d_results
            dgz2, lg = g_results

            # Detect training collapse
            collapsed = bool(ldr >= c.collapses.max_loss)
            collapsed = collapsed or bool(ldf >= c.collapses.max_loss)
            collapsed = collapsed or bool(lg >= c.collapses.max_loss)
            if collapsed:
                c.collapses.batch_count += 1

            # Update the statistics
            ld = ldr + ldf
            c.latest.dx, c.latest.dgz, c.latest.ld = dx, dgz, ld
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lds.append(ld)
            lgs.append(lg)
            r.log_batch_2("t")

            c.loops.train.index += 1

        # end for

        epoch_collapsed = c.collapses.batch_count >= c.collapses.max_batch_count
        if epoch_collapsed:
            c.collapses.epochs.append((c.loops.iteration.index, c.loops.epoch.index))
            r.logln("Epoch training collapsed")

        epoch_ld = numpy.array(lds).mean()
        epoch_lg = numpy.array(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")

    def valid_d(self):
        """Validates D with the validation set."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started validating D")
        ldrs = []
        c.loops.valid.index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            ldrs.append(ld)
            r.log_batch_2("vdr")
            c.loops.valid.index += 1

        ldfs = []
        c.loops.valid.index = 0
        for noises in c.noises.valid:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            ldfs.append(ld)
            r.log_batch_2("vdf")
            c.loops.valid.index += 1

        lds = []
        for index in range(c.loops.valid.count):
            lds.append(ldrs[index] + ldfs[index])

        epoch_ld = numpy.array(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")

    def valid_g(self):
        """Validates G with the validation set."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started validating G")
        lgs = []
        c.loops.valid.index = 0
        for noises in c.noises.valid:
            dgz2, lg = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lgs.append(lg)
            r.log_batch_2("vg")
            c.loops.valid.index += 1

        epoch_lg = numpy.array(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")

    def save_best_d(self):
        """Saves the best D."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.log_best_losses("d")
        curr_ld = c.losses.valid.d[-1]
        if c.bests.d is None or curr_ld <= c.bests.d:
            c.bests.d = curr_ld

        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iteration.index, c.loops.epoch.index) == c.collapses.epochs[-1]

        if epoch_collapsed:
            c.mods.d.load()
            r.log_model_action("load", "d")
        else:
            c.mods.d.save()
            r.log_model_action("save", "d")

    def save_best_g(self):
        """Saves the best G."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.log_best_losses("g")
        curr_lg = c.losses.valid.g[-1]
        if c.bests.g is None or curr_lg <= c.bests.g:
            c.bests.g = curr_lg

        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iteration.index, c.loops.epoch.index) == c.collapses.epochs[-1]

        if epoch_collapsed:
            c.mods.g.load()
            r.log_model_action("load", "g")
        else:
            c.mods.g.save()
            r.log_model_action("save", "g")

    def run_iter(self):
        """Runs a iter of multiple epochs."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        c.loops.epoch.index = 0
        while c.loops.epoch.index < c.loops.epoch.count:
            r.log_epoch("Started", "")
            self.train_both()
            self.valid_d()
            self.save_best_d()
            self.valid_g()
            self.save_best_g()
            r.save_d_losses()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln("-")
            c.loops.epoch.index += 1

    def start_training(self):
        """Starts the training algorithm."""
        self.check_context_and_results()
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started predictive alternating SGD algorithm")
        r.logln("-")
        r.save_training_images()
        r.save_validation_images()
        r.save_images_before_training()
        c.loops.iteration.index = 0
        while c.loops.iteration.index < c.loops.iteration.count:
            r.log_iter("Started")
            self.run_iter()
            r.logln("-")
            c.loops.iteration.index += 1


class FairPredAltSGDAlgo(Algo):
    """Fair predictive alternating SGD, a fairer predictive alternating SGD.

    Created by liu-yucheng based on the predictive alternating SGD algorithm.
    """

    def __init__(self):
        """Inits self."""
        super().__init__()

    def train_and_step_d(self, real_batch, fake_noises):
        """Trains and steps D with predicted G.

        Args:
            real_batch: the real batch
            fake_noises: the fake noises

        Returns:
            dx, : the output mean of D on the real batch
            ldr, : the loss of D on the real batch
            dgz, : the output mean of D on the fake batch
            ldf, : the loss of D on the fake batch
            ldc, : D cluster loss
            ld: the loss of D
        """
        c: _TrainingContext = self.context

        can_predict = bool(c.loops.train.index > 0)
        if can_predict:
            c.mods.g.predict()

        c.mods.d.clear_grads()
        train_results = c.mods.d.train_fair(c.mods.g.model, real_batch, c.labels.real, fake_noises, c.labels.fake)
        c.mods.d.step_optim()

        if can_predict:
            c.mods.g.restore()

        dx, ldr, dgz, ldf, ldc, ld = train_results
        return dx, ldr, dgz, ldf, ldc, ld

    def train_and_step_g(self, real_batch, fake_noises):
        """Trains and steps G with predicted D.

        Args:
            real_batch: the real batch
            fake_noises: the fake noises

        Returns:
            dx2, : the output mean of D on the real batch
            lgr, : the loss of G on the real batch
            dgz2, : the output mean of D on the fake batch
            lgf, : the loss of G on the fake batch
            lgc, : G cluster loss
            lg: the loss of G
        """
        c: _TrainingContext = self.context

        can_predict = bool(c.loops.train.index > 0)
        if can_predict:
            c.mods.d.predict()

        c.mods.g.clear_grads()
        train_results = c.mods.g.train_fair(c.mods.d.model, real_batch, c.labels.fake, fake_noises, c.labels.real)
        c.mods.g.step_optim()

        if can_predict:
            c.mods.d.restore()

        dx2, lgr, dgz2, lgf, lgc, lg = train_results
        return dx2, lgr, dgz2, lgf, lgc, lg

    def train_dg(self):
        """Trains D and G together."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started training D and G")
        lds = []
        lgs = []
        c.collapses.batch_count = 0
        c.loops.train.index = 0
        for real_batch in c.data.train.loader:
            # Prepare training materials
            real_batch = real_batch[0]
            fake_noises = c.mods.g.generate_noises(c.data.batch_size)

            # Filp a coin to determine whether to reverse the training order
            reverse_order = utils.rand_bool()
            if reverse_order:
                # Train G, and then train D
                g_results = self.train_and_step_g(real_batch, fake_noises)
                d_results = self.train_and_step_d(real_batch, fake_noises)
            else:  # elif not reverse order:
                # Train D, and then train G (original order)
                d_results = self.train_and_step_d(real_batch, fake_noises)
                g_results = self.train_and_step_g(real_batch, fake_noises)

            # Parse the training results
            dx, ldr, dgz, ldf, ldc, ld = d_results
            dx2, lgr, dgz2, lgf, lgc, lg = g_results

            # Detect training collapse
            collapsed = bool(ldr >= c.collapses.max_loss)
            collapsed = collapsed or bool(ldf >= c.collapses.max_loss)
            collapsed = collapsed or bool(lg >= c.collapses.max_loss)
            if collapsed:
                c.collapses.batch_count += 1

            # Update the statistics
            c.latest.dx = dx
            c.latest.ldr = ldr
            c.latest.dgz = dgz
            c.latest.ldf = ldf
            c.latest.ldc = ldc
            c.latest.ld = ld

            c.latest.dx2 = dx2
            c.latest.lgr = lgr
            c.latest.dgz2 = dgz2
            c.latest.lgf = lgf
            c.latest.lgc = lgc
            c.latest.lg = lg

            lds.append(ld)
            lgs.append(lg)
            r.log_batch_3("t")

            c.loops.train.index += 1

        # end for

        epoch_collapsed = c.collapses.batch_count >= c.collapses.max_batch_count
        if epoch_collapsed:
            c.collapses.epochs.append((c.loops.iteration.index, c.loops.epoch.index))
            r.logln("Epoch training collapsed")

        epoch_ld = numpy.array(lds).mean()
        epoch_lg = numpy.array(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")

    def valid_dg(self):
        """Validates D and G together with the validation set."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started validating D and G")

        lds = []
        lgs = []
        c.loops.valid.index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            fake_noises = c.noises.valid[c.loops.valid.index]

            d_valid_results = c.mods.d.valid_fair(
                c.mods.g.model, real_batch, c.labels.real, fake_noises, c.labels.fake
            )
            g_valid_results = c.mods.g.valid_fair(
                c.mods.d.model, real_batch, c.labels.fake, fake_noises, c.labels.real
            )

            dx, ldr, dgz, ldf, ldc, ld = d_valid_results
            dx2, lgr, dgz2, lgf, lgc, lg = g_valid_results

            lds.append(ld)
            lgs.append(lg)

            c.latest.dx = dx
            c.latest.ldr = ldr
            c.latest.dgz = dgz
            c.latest.ldf = ldf
            c.latest.ldc = ldc
            c.latest.ld = ld

            c.latest.dx2 = dx2
            c.latest.lgr = lgr
            c.latest.dgz2 = dgz2
            c.latest.lgf = lgf
            c.latest.lgc = lgc
            c.latest.lg = lg

            r.log_batch_3("v")

            c.loops.valid.index += 1

        # end for

        epoch_ld = numpy.array(lds).mean()
        epoch_lg = numpy.array(lgs).mean()

        c.losses.valid.d.append(epoch_ld)
        c.losses.valid.g.append(epoch_lg)

        r.log_epoch_loss("vd")
        r.log_epoch_loss("vg")

    def save_best_dg(self):
        """Saves the best D and G."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.log_best_losses("d")
        r.log_best_losses("g")

        curr_ld = c.losses.valid.d[-1]
        curr_lg = c.losses.valid.g[-1]

        if c.bests.d is None or curr_ld <= c.bests.d:
            c.bests.d = curr_ld
        if c.bests.g is None or curr_lg <= c.bests.g:
            c.bests.g = curr_lg

        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iteration.index, c.loops.epoch.index) == c.collapses.epochs[-1]

        if epoch_collapsed:
            c.mods.d.load()
            r.log_model_action("load", "d")

            c.mods.g.load()
            r.log_model_action("load", "g")

        else:
            c.mods.d.save()
            r.log_model_action("save", "d")

            c.mods.g.save()
            r.log_model_action("save", "g")

        # end if

    def run_iter(self):
        """Runs a iter of multiple epochs."""
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        c.loops.epoch.index = 0
        while c.loops.epoch.index < c.loops.epoch.count:
            r.log_epoch("Started", "")
            self.train_dg()
            self.valid_dg()
            self.save_best_dg()
            r.save_d_losses()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln("-")
            c.loops.epoch.index += 1

    def start_training(self):
        """Starts the training algorithm."""
        self.check_context_and_results()
        r: _TrainingResults = self.results
        c: _TrainingContext = self.context

        r.logln("Started fair predictive alternating SGD algorithm")

        has_fairness = c.mods.d.has_fairness
        has_fairness = has_fairness and c.mods.g.has_fairness
        if has_fairness:
            r.log_fairness()

        r.logln("-")
        r.save_training_images()
        r.save_validation_images()
        r.save_images_before_training()
        c.loops.iteration.index = 0
        while c.loops.iteration.index < c.loops.iteration.count:
            r.log_iter("Started")
            self.run_iter()
            r.logln("-")
            c.loops.iteration.index += 1
