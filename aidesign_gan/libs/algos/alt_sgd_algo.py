"""Alternating SGD algorithm."""

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


class AltSGDAlgo(_Algo):
    """Algorithm that trains D and G alternatively with SGD."""

    def __init__(self):
        """Inits self."""
        super().__init__()

    def _train_and_step_d(self, real_batch, fake_batch):
        """Returns (dx, ldr, dgz, ldf)."""
        c: _TrainContext = self.context

        c.mods.d.clear_grads()
        dx, ldr = c.mods.d.train(real_batch, c.labels.real)
        c.mods.d.step_optim()

        c.mods.d.clear_grads()
        dgz, ldf = c.mods.d.train(fake_batch, c.labels.fake)
        c.mods.d.step_optim()

        return dx, ldr, dgz, ldf
    # end def

    def _train_and_step_g(self, noises):
        """Returns (dgz2, lg)."""
        c: _TrainContext = self.context

        c.mods.g.clear_grads()
        dgz2, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
        c.mods.g.step_optim()

        return dgz2, lg

    def _train_dg(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        r.logln("Started training both D and G")

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
                g_results = self._train_and_step_g(g_noises)
                d_results = self._train_and_step_d(d_real_batch, d_fake_batch)
            else:  # elif not reverse_order:
                # Train D, and then train G (original order)
                d_results = self._train_and_step_d(d_real_batch, d_fake_batch)
                g_results = self._train_and_step_g(g_noises)

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

        epoch_ld = _nparray(lds).mean()
        epoch_lg = _nparray(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")
    # end def

    def _valid_d(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

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
        # end for

        ldfs = []
        c.loops.valid.index = 0

        for noises in c.noises.valid:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            ldfs.append(ld)
            r.log_batch_2("vdf")
            c.loops.valid.index += 1
        # end for

        lds = []

        for index in range(c.loops.valid.count):
            lds.append(ldrs[index] + ldfs[index])

        epoch_ld = _nparray(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")
    # end def

    def _valid_g(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        r.logln("Started validating G")

        lgs = []
        c.loops.valid.index = 0

        for noises in c.noises.valid:
            dgz2, lg = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lgs.append(lg)
            r.log_batch_2("vg")
            c.loops.valid.index += 1
        # end for

        epoch_lg = _nparray(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")
    # end def

    def _save_best_d(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

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

    # end def

    def _save_best_g(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

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

    # end def

    def _run_iter(self):
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        c.loops.epoch.index = 0

        while c.loops.epoch.index < c.loops.epoch.count:
            r.log_epoch("Started", "")
            self._train_dg()
            self._valid_d()
            self._save_best_d()
            self._valid_g()
            self._save_best_g()
            r.save_d_losses()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln("-")
            c.loops.epoch.index += 1
        # end while
    # end def

    def start(self):
        """Starts the algorithm."""
        super().start()

        self.check_context_and_results()
        r: _TrainResults = self.results
        c: _TrainContext = self.context

        r.logln("Started alternating SGD algorithm")
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
        # end while
    # end def
# end class