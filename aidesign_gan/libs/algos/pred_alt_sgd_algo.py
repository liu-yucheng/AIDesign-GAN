"""Predictive alternating SGD algorithm."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import numpy

from aidesign_gan.libs import contexts
from aidesign_gan.libs import results
from aidesign_gan.libs import utils
from aidesign_gan.libs.algos import algo

# Aliases

_Algo = algo.Algo
_nparray = numpy.array
_rand_bool = utils.rand_bool
_TrainContext = contexts.TrainContext
_TrainResults = results.TrainResults

# End of aliases


class PredAltSGDAlgo(_Algo):
    """Predictive alternating SGD algorithm.

    A variant of alternating SGD algo.
    Features optimizer steps with predictions.
    """

    def __init__(self):
        """Inits self."""
        super().__init__()

    def _train_and_step_d(self, real_batch, fake_batch, context=None, results=None):
        """Returns (dx, ldr, dgz, ldf)."""
        c: _TrainContext = self.find_context(context)
        _ = results

        not_1st_batch = c.loops.train.index > 0

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

    def _train_and_step_g(self, noises, context=None, results=None):
        """Returns (dgz2, lg)."""
        c: _TrainContext = self.find_context(context)
        _ = results

        not_1st_batch = c.loops.train.index > 0

        if not_1st_batch:
            c.mods.d.predict()

        c.mods.g.clear_grads()
        dgz2, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
        c.mods.g.step_optim()

        if not_1st_batch:
            c.mods.d.restore()

        return dgz2, lg

    def _train_d_and_g(self, context=None, results=None):
        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

        r.logln("Started training discriminator and generator")

        lds = []
        lgs = []
        c.collapses.batch_count = 0
        c.loops.train.index = 0

        for real_batch in c.data.train.loader:
            # Prepare D training materials
            d_real_batch = real_batch[0]
            batch_size = d_real_batch.size()[0]
            # print(f"[Debug] Batch size: {batch_size}")

            d_noises = c.mods.g.generate_noises(batch_size)
            d_fake_batch = c.mods.g.test(d_noises)

            # Prepare G training materials
            g_noises = c.mods.g.generate_noises(batch_size)

            # Filp a coin to determine whether to reverse the training order
            reverse_order = _rand_bool()

            if reverse_order:
                # Train G, and then train D
                g_results = self._train_and_step_g(g_noises, context, results)
                d_results = self._train_and_step_d(d_real_batch, d_fake_batch, context, results)
            else:  # elif not reverse order:
                # Train D, and then train G (original order)
                d_results = self._train_and_step_d(d_real_batch, d_fake_batch, context, results)
                g_results = self._train_and_step_g(g_noises, context, results)

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
            r.log_batch_v2("t")

            c.loops.train.index += 1
        # end for

        epoch_collapsed = c.collapses.batch_count >= c.collapses.max_batch_count

        if epoch_collapsed:
            c.collapses.epochs.append((c.loops.iteration.index, c.loops.epoch.index))
            r.logln("Epoch training collapsed")

        epoch_ld = _nparray(lds).mean()
        epoch_lg = _nparray(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")

    def _valid_d(self, context=None, results=None):
        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

        r.logln("Started validating discriminator")

        ldrs = []
        c.loops.valid.index = 0

        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            ldrs.append(ld)
            r.log_batch_v2("vdr")
            c.loops.valid.index += 1

        ldfs = []
        c.loops.valid.index = 0

        for noises in c.noises.valid:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            ldfs.append(ld)
            r.log_batch_v2("vdf")
            c.loops.valid.index += 1

        lds = []

        for index in range(c.loops.valid.count):
            lds.append(ldrs[index] + ldfs[index])

        epoch_ld = _nparray(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")

    def _valid_g(self, context=None, results=None):
        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

        r.logln("Started validating generator")

        lgs = []
        c.loops.valid.index = 0

        for noises in c.noises.valid:
            dgz2, lg = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lgs.append(lg)
            r.log_batch_v2("vg")
            c.loops.valid.index += 1

        epoch_lg = _nparray(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")

    def _save_best_d(self, context=None, results=None):
        """Saves the best D."""
        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

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
        # end if

    def _save_best_g(self, context=None, results=None):
        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

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
        # end if

    def _run_iter(self, context=None, results=None):
        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

        c.loops.epoch.index = 0

        while c.loops.epoch.index < c.loops.epoch.count:
            r.log_epoch("Started", "")
            self._train_d_and_g(context, results)
            self._valid_d(context, results)
            self._save_best_d(context, results)
            self._valid_g(context, results)
            self._save_best_g(context, results)
            r.save_disc_losses()
            r.save_gen_losses()
            r.save_gen_imgs()
            r.save_tvg_fig()
            r.logln("--")
            r.flushlogs()
            c.loops.epoch.index += 1
        # end while

    def start(self, context=None, results=None):
        """Starts the algorithm.

        Args:
            context: optional context
            results: optional results
        """
        super().start(context, results)

        c: _TrainContext = self.find_context(context)
        r: _TrainResults = self.find_results(results)

        info = str(
            "Started predictive alternating SGD algorithm\n"
            "-"
        )

        r.logln(info)
        r.log_pred_factor()
        r.logln("-")
        r.save_train_imgs()
        r.save_valid_imgs()
        r.save_imgs_before_train()

        c.loops.iteration.index = 0

        while c.loops.iteration.index < c.loops.iteration.count:
            r.log_iter("Started")
            self._run_iter(context, results)
            r.logln("-")
            r.flushlogs()
            c.loops.iteration.index += 1
        # ene while

        info = str(
            "-\n"
            "Completed predictive alternating SGD algorithm"
        )

        r.logln(info)
