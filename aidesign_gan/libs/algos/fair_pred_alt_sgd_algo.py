"""Fair predictive alternative SGD algorithm."""

# Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.
# GNU GPL Version 3 copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by: liu-yucheng
# Last updated by: liu-yucheng

import numpy

from aidesign_gan.libs import contexts
from aidesign_gan.libs import results
from aidesign_gan.libs import utils
from aidesign_gan.libs.algos import algo

# Aliases

_Algo = algo.Algo
_nparray = numpy.array
_rand_bool = utils.rand_bool
_TrainContext = contexts.TrainingContext
_TrainResults = results.TrainingResults

# End of aliases


class FairPredAltSGDAlgo(_Algo):
    """Fair predictive alternative SGD algorithm.

    A variant of predictive alternating SGD algo.
    Features a fairer training and loss calculation process.
    First proposed by liu-yucheng.

    NOTE:
        This algorithm is not yet mathematically validated to be effective.
        However, it yields visually promising results in most AIDesign-GAN project training cases.
    """

    def __init__(self):
        """Inits self."""
        super().__init__()

    def _train_and_step_d(self, real_batch, fake_noises):
        """Returns (dx, dgz, ldr, ldf, ldcr, ldcf, ld)."""
        c: _TrainContext = self.context

        can_predict = c.loops.train.index > 0

        if can_predict:
            c.mods.g.predict()

        c.mods.d.clear_grads()
        train_results = c.mods.d.train_fair(c.mods.g.model, real_batch, c.labels.real, fake_noises, c.labels.fake)
        c.mods.d.step_optim()

        if can_predict:
            c.mods.g.restore()

        return train_results
    # end def

    def _train_and_step_g(self, real_batch, fake_noises):
        """Returns (dx2, dgz2, lgr, lgf, lgcr, lgcf, lg)."""
        c: _TrainContext = self.context

        can_predict = c.loops.train.index > 0

        if can_predict:
            c.mods.d.predict()

        c.mods.g.clear_grads()
        train_results = c.mods.g.train_fair(c.mods.d.model, real_batch, c.labels.fake, fake_noises, c.labels.real)
        c.mods.g.step_optim()

        if can_predict:
            c.mods.d.restore()

        return train_results
    # end def

    def _train_dg(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        r.logln("Started training D and G")

        lds = []
        lgs = []
        c.collapses.batch_count = 0
        c.loops.train.index = 0

        for real_batch in c.data.train.loader:
            # Prepare training materials
            real_batch = real_batch[0]
            batch_size = real_batch.size()[0]
            # print(f"[Debug] Batch size: {batch_size}")

            fake_noises = c.mods.g.generate_noises(batch_size)

            # Filp a coin to determine whether to reverse the training order
            reverse_order = _rand_bool()

            if reverse_order:
                # Train G, and then train D
                g_results = self._train_and_step_g(real_batch, fake_noises)
                d_results = self._train_and_step_d(real_batch, fake_noises)
            else:  # elif not reverse order:
                # Train D, and then train G (original order)
                d_results = self._train_and_step_d(real_batch, fake_noises)
                g_results = self._train_and_step_g(real_batch, fake_noises)

            # Parse the training results
            dx, dgz, ldr, ldf, ldcr, ldcf, ld = d_results
            dx2, dgz2, lgr, lgf, lgcr, lgcf, lg = g_results

            # Detect training collapse
            collapsed = bool(ldr >= c.collapses.max_loss)
            collapsed = collapsed or bool(ldf >= c.collapses.max_loss)
            collapsed = collapsed or bool(lg >= c.collapses.max_loss)

            if collapsed:
                c.collapses.batch_count += 1

            # Update the statistics
            c.latest.dx = dx
            c.latest.dgz = dgz
            c.latest.ldr = ldr
            c.latest.ldf = ldf
            c.latest.ldcr = ldcr
            c.latest.ldcf = ldcf
            c.latest.ld = ld

            c.latest.dx2 = dx2
            c.latest.dgz2 = dgz2
            c.latest.lgr = lgr
            c.latest.lgf = lgf
            c.latest.lgcr = lgcr
            c.latest.lgcf = lgcf
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

        epoch_ld = _nparray(lds).mean()
        epoch_lg = _nparray(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")
    # end def

    def _valid_dg(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

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

            dx, dgz, ldr, ldf, ldcr, ldcf, ld = d_valid_results
            dx2, dgz2, lgr, lgf, lgcr, lgcf, lg = g_valid_results

            lds.append(ld)
            lgs.append(lg)

            c.latest.dx = dx
            c.latest.ldr = ldr
            c.latest.dgz = dgz
            c.latest.ldf = ldf
            c.latest.ldcr = ldcr
            c.latest.ldcf = ldcf
            c.latest.ld = ld

            c.latest.dx2 = dx2
            c.latest.lgr = lgr
            c.latest.dgz2 = dgz2
            c.latest.lgf = lgf
            c.latest.lgcr = lgcr
            c.latest.lgcf = lgcf
            c.latest.lg = lg

            r.log_batch_3("v")

            c.loops.valid.index += 1
        # end for

        epoch_ld = _nparray(lds).mean()
        epoch_lg = _nparray(lgs).mean()

        c.losses.valid.d.append(epoch_ld)
        c.losses.valid.g.append(epoch_lg)

        r.log_epoch_loss("vd")
        r.log_epoch_loss("vg")
    # end def

    def _save_best_dg(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

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
    # end def

    def _run_iter(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        c.loops.epoch.index = 0

        while c.loops.epoch.index < c.loops.epoch.count:
            r.log_epoch("Started", "")
            self._train_dg()
            self._valid_dg()
            self._save_best_dg()
            r.save_d_losses()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln("-")
            c.loops.epoch.index += 1

    # end def

    def start(self):
        """Starts the algorithm."""
        super().start()

        self.check_context_and_results()
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        r.logln("Started fair predictive alternating SGD algorithm")
        r.log_pred_factor()

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
            self._run_iter()
            r.logln("-")
            c.loops.iteration.index += 1

    # end def
# end class