"""Training results."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import datetime
import math
import numpy
import os
from matplotlib import lines
from matplotlib import pyplot
from os import path as ospath
from torchvision import utils as vutils

from aidesign_gan.libs import contexts
from aidesign_gan.libs.results import results

_ceil = math.ceil
_join = ospath.join
_Line2D = lines.Line2D
_makedirs = os.makedirs
_make_grid = vutils.make_grid
_now = datetime.datetime.now
_np_transpose = numpy.transpose
_plt_axis = pyplot.axis
_plt_axvline = pyplot.axvline
_plt_close = pyplot.close
_plt_figure = pyplot.figure
_plt_gca = pyplot.gca
_plt_imshow = pyplot.imshow
_plt_legend = pyplot.legend
_plt_plot = pyplot.plot
_plt_savefig = pyplot.savefig
_plt_subplots = pyplot.subplots
_plt_title = pyplot.title
_plt_xlabel = pyplot.xlabel
_plt_xticks = pyplot.xticks
_plt_ylabel = pyplot.ylabel
_TrainContext = contexts.TrainContext
_Results = results.Results


class TrainResults(_Results):
    """Training results."""

    def __init__(self, path, logs):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            logs: the log file objects
        """
        super().__init__(path, logs)

        self.gen_img_path = _join(self.path, "Generated-Images")
        """Generated images path."""

    def ensure_folders(self):
        """Ensures the result folders."""
        super().ensure_folders()

        _makedirs(self.gen_img_path, exist_ok=True)
        self.logln(f"Ensured folder: {self.gen_img_path}")

    def log_data(self, context=None):
        """Logs the data loaders info.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        self.logln(f"Data total size: {c.data.size}; Data size to use: {c.data.size_to_use}")
        self.logln(f"Training set size: {c.data.train.size}; Validation set size: {c.data.valid.size}")

    def log_mods(self, context=None):
        """Logs the modelers info.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        d_config = c.mods.d.config["adam_optimizer"]
        d_lr = d_config["learning_rate"]
        d_beta1 = d_config["beta1"]
        d_beta2 = d_config["beta2"]
        self.logstr(
            str(
                "D's modeler:\n"
                "  Model:  Size: {}  Training size: {}  Struct: (See below)\n"
                "  Adam optimizer:  Learning rate: {}  Beta 1: {}  Beta 2: {}\n"
                "==== D's model struct ====\n"
                "{}\n"
            ).format(
                c.mods.d.size,
                c.mods.d.training_size,
                d_lr, d_beta1, d_beta2,
                str(c.mods.d.model)
            )
        )

        g_config = c.mods.g.config["adam_optimizer"]
        g_lr = g_config["learning_rate"]
        g_beta1 = g_config["beta1"]
        g_beta2 = g_config["beta2"]
        self.logstr(
            str(
                "G's modeler:\n"
                "  Model:  Size: {}  Training size: {}  Struct: (See below)\n"
                "  Adam optimizer:  Learning rate: {}  Beta 1: {}  Beta 2: {}\n"
                "==== G's model struct ====\n"
                "{}\n"
            ).format(
                c.mods.g.size,
                c.mods.g.training_size,
                g_lr, g_beta1, g_beta2,
                str(c.mods.g.model)
            )
        )

    def log_mode(self, context=None):
        """Logs the training mode info.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        self.logln(f"Training mode: {c.mode}")

    def log_labels(self, context=None):
        """Logs the labels info.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        self.logln("Labels:  Real: {}  Fake: {}".format(f"{c.labels.real:.6f}", f"{c.labels.fake:.6f}"))

    def log_algo(self, algo_name):
        """Logs the training algorithm.

        Args:
            algo_name: the name of the algorithm
        """
        algo_name = str(algo_name)
        self.logln(f"Algo: {algo_name}")

    def log_fairness(self, context=None):
        """Logs the fairness config.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        d_dx_factor = c.mods.d.config["fairness"]["dx_factor"]
        d_dgz_factor = c.mods.d.config["fairness"]["dgz_factor"]
        d_cluster_dx_factor = c.mods.d.config["fairness"]["cluster_dx_factor"]
        d_cluster_dgz_factor = c.mods.d.config["fairness"]["cluster_dgz_factor"]

        g_dx_factor = c.mods.g.config["fairness"]["dx_factor"]
        g_dgz_factor = c.mods.g.config["fairness"]["dgz_factor"]
        g_cluster_dx_factor = c.mods.g.config["fairness"]["cluster_dx_factor"]
        g_cluster_dgz_factor = c.mods.g.config["fairness"]["cluster_dgz_factor"]

        self.logstr(
            str(
                "Fairness:\n"
                "  D:  D(X) factor: {}  D(G(Z)) factor: {}\n"
                "      Cluster D(X) factor: {}  Cluster D(G(Z)) factor: {}\n"
                "  G:  D(X) factor: {}  D(G(Z)) factor: {}\n"
                "      Cluster D(X) factor: {}  Cluster D(G(Z)) factor: {}\n"
            ).format(
                f"{d_dx_factor:.6f}", f"{d_dgz_factor:.6f}",
                f"{d_cluster_dx_factor:.6f}", f"{d_cluster_dgz_factor:.6f}",
                f"{g_dx_factor:.6f}", f"{g_dgz_factor:.6f}",
                f"{g_cluster_dx_factor:.6f}", f"{g_cluster_dgz_factor:.6f}"
            )
        )

    def log_pred_factor(self, context=None):
        """Logs the prediction factor.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        d_pred_factor = c.mods.d.optim.pred_factor
        g_pred_factor = c.mods.g.optim.pred_factor

        self.logstr(
            str(
                "Prediction factor:  D: {}  G: {}\n"
            ).format(
                f"{d_pred_factor:.6f}", f"{g_pred_factor:.6f}"
            )
        )

    def log_iter(self, prefix, context=None):
        """Logs the iter info.

        Args:
            prefix: the contents to log before the info
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        self.logstr("==== ==== ")

        self.logstr(f"{prefix} iter {c.loops.iteration.index + 1} / {c.loops.iteration.count}")

        self.logstr(" ==== ====\n")

    def log_epoch(self, prefix, epoch_type, context=None):
        """Logs the epoch info.

        Args:
            prefix: the contents to log before the info
            epoch_type: epoch type (d/g/(empty string))
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        self.logstr("==== ")

        self.logstr(f"{prefix} epoch {c.loops.iteration.index + 1}.{epoch_type}{c.loops.epoch.index + 1} / ")
        self.logstr(f"{c.loops.iteration.count}.{epoch_type}{c.loops.epoch.count}")

        self.logstr(" ====\n")

    def find_train_needs_log(self, context=None):
        """Finds if the current training batch needs to be logged.

        Returns:
            result: whether the batch needs to be logged
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        result = c.loops.train.index == 0
        result = result or (c.loops.train.index + 1) % 30 == 0
        result = result or c.loops.train.index == c.data.train.batch_count - 1
        return result

    def find_valid_needs_log(self, context=None):
        """Finds if the current validation batch needs to be logged.

        Returns:
            result: whether the batch needs to be logged
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        result = c.loops.valid.index == 0
        result = result or (c.loops.valid.index + 1) % 15 == 0
        result = result or c.loops.valid.index == c.data.valid.batch_count - 1
        return result

    def log_batch_v2(self, batch_type, context=None):
        """Logs the batch info.

        Version 2.
        Designed to log the results of AltSGDAlgo.
        Designed to log the results of PredAltSGDAlgo.

        Args:
            batch_type: batch type (t/vdr/vdf/vg)
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        needs_log = False
        if "t" in batch_type:
            needs_log = self.find_train_needs_log()
        elif "v" in batch_type:
            needs_log = self.find_valid_needs_log()
        if not needs_log:
            return

        batch_index = None
        batch_count = None
        if "t" in batch_type:
            batch_index = c.loops.train.index
            batch_count = c.loops.train.count
        elif "v" in batch_type:
            batch_index = c.loops.valid.index
            batch_count = c.loops.valid.count

        self.logstr("Batch {}.{}.{}{} /".format(
            c.loops.iteration.index + 1,
            c.loops.epoch.index + 1,
            batch_type, batch_index + 1
        ))
        self.logstr(" {}.{}.{}{}:".format(
            c.loops.iteration.count,
            c.loops.epoch.count,
            batch_type, batch_count
        ))

        if "t" in batch_type:
            self.logstr("\n")
            self.logstr(f"  D:  D(X): {c.latest.dx:.6f}  D(G(Z)): {c.latest.dgz:.6f}  L(D): {c.latest.ld:.6f}\n")
            self.logstr(f"  G:  D(G(Z)): {c.latest.dgz2:.6f}  L(G): {c.latest.lg:.6f}")
        elif "v" in batch_type:
            if "d" in batch_type:
                if "r" in batch_type:
                    self.logstr(f"  D(X): {c.latest.dx:.6f}  L(D): {c.latest.ld:.6f}")
                elif "f" in batch_type:
                    self.logstr(f"  D(G(Z)): {c.latest.dgz:.6f}  L(D): {c.latest.ld:.6f}")
            elif "g" in batch_type:
                self.logstr(f"  D(G(Z)): {c.latest.dgz2:.6f}  L(G): {c.latest.lg:.6f}")

        self.logstr("\n")

    def log_batch_v3(self, batch_type, context=None):
        """Logs the batch info.

        Version 3.
        Designed to log the results of FairPredAltSGDAlgo.

        Args:
            batch_type: batch type (t/v)
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        needs_log = False
        if "t" in batch_type:
            needs_log = self.find_train_needs_log()
        elif "v" in batch_type:
            needs_log = self.find_valid_needs_log()
        if not needs_log:
            return

        batch_index = None
        batch_count = None
        if "t" in batch_type:
            batch_index = c.loops.train.index
            batch_count = c.loops.train.count
        elif "v" in batch_type:
            batch_index = c.loops.valid.index
            batch_count = c.loops.valid.count

        self.logstr("Batch {}.{}.{}{} /".format(
            c.loops.iteration.index + 1,
            c.loops.epoch.index + 1,
            batch_type, batch_index + 1
        ))
        self.logstr(" {}.{}.{}{}:".format(
            c.loops.iteration.count,
            c.loops.epoch.count,
            batch_type, batch_count
        ))

        self.logstr("\n")
        self.logstr(
            str(
                "     D:  D(X): {}  D(G(Z)): {}\n"
                "  L(D):  L(D,X): {}  L(D,G(Z)): {}\n"
                "         L(D,Cluster,X): {}  L(D,Cluster,G(Z)): {}\n"
                "         L(D): {}\n"
            ).format(
                f"{c.latest.dx:.6f}", f"{c.latest.dgz:.6f}",
                f"{c.latest.ldr:.6f}", f"{c.latest.ldf:.6f}",
                f"{c.latest.ldcr:.6f}", f"{c.latest.ldcf:.6f}",
                f"{c.latest.ld:.6f}"
            )
        )
        self.logstr(
            str(
                "     G:  D(X): {}  D(G(Z)): {}\n"
                "  L(G):  L(G,X): {}  L(G,G(Z)): {}\n"
                "         L(G,Cluster,X): {}  L(G,Cluster,G(Z)): {}\n"
                "         L(G): {}\n"
            ).format(
                f"{c.latest.dx2:.6f}", f"{c.latest.dgz2:.6f}",
                f"{c.latest.lgr:.6f}", f"{c.latest.lgf:.6f}",
                f"{c.latest.lgcr:.6f}", f"{c.latest.lgcf:.6f}",
                f"{c.latest.lg:.6f}"
            )
        )

    def log_epoch_loss(self, loss_type, context=None):
        """Logs the epoch loss info.

        Args:
            loss_type: loss type (td/vd/tg/vg)
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        loss = None
        if loss_type == "td":
            loss = c.losses.train.d[-1]
        elif loss_type == "vd":
            loss = c.losses.valid.d[-1]
        elif loss_type == "tg":
            loss = c.losses.train.g[-1]
        elif loss_type == "vg":
            loss = c.losses.valid.g[-1]

        self.logstr("Epoch")

        if "d" in loss_type:
            self.logstr(" D")
        elif "g" in loss_type:
            self.logstr(" G")

        if "t" in loss_type:
            self.logstr(" training")
        elif "v" in loss_type:
            self.logstr(" validation")

        self.logstr(f" loss: {loss:.6f}\n")

    def log_best_losses(self, model_type, context=None):
        """Logs the best loss info.

        Args:
            model_type: model type (d/g)
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        if model_type == "d":
            curr_loss = c.losses.valid.d[-1]
            prev_best = c.bests.d
            self.logstr("D:")
        elif model_type == "g":
            curr_loss = c.losses.valid.g[-1]
            prev_best = c.bests.g
            self.logstr("G:")

        self.logstr(f"  Curr loss: {curr_loss:.6f}")

        if prev_best is None:
            self.logstr("  Prev best loss: None")
        else:
            self.logstr(f"  Prev best loss: {prev_best:.6f}")

        self.logstr("\n")

    def log_model_action(self, action_type, model_type, context=None):
        """Logs the model action info.

        Args:
            action_type: action type (save/es/rb/load)
            model_type: model type (d/g)
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        model_name = None
        es_count = None
        rb_count = None
        if model_type == "d":
            model_name = "D"
            es_count = c.loops.es.d
            rb_count = c.loops.rb.d
        elif model_type == "g":
            model_name = "G"
            es_count = c.loops.es.g
            rb_count = c.loops.rb.g

        if action_type == "save":
            self.logln(f"Saved {model_name}")
        elif action_type == "es":
            self.logln(f"Early stopped {model_name}, count {es_count} / {c.loops.es.max}")
        elif action_type == "rb":
            self.logln(f"Rollbacked {model_name}, count {rb_count} / {c.loops.rb.max}")
        elif action_type == "load":
            self.logln(f"Loaded {model_name}")

    def save_training_images(self, context=None):
        """Saves the first batch of the training images.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        batch = next(iter(c.data.train.loader))
        batch = batch[0].cpu()
        batch = batch[:c.data.batch_size]

        grid = _make_grid(
            batch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = _np_transpose(grid, (1, 2, 0))

        location = _join(self.path, "Training-Images.jpg")
        figure = _plt_figure(figsize=(8, 8))
        _plt_axis("off")
        _plt_title("Training Images")
        _plt_imshow(grid)
        _plt_savefig(location, dpi=160)
        _plt_close(figure)

        self.logln("Saved training images")

    def save_validation_images(self, context=None):
        """Saves the first batch of the validation images.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        batch = next(iter(c.data.valid.loader))
        batch = batch[0].cpu()
        batch = batch[:c.data.batch_size]

        grid = _make_grid(
            batch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = _np_transpose(grid, (1, 2, 0))

        location = _join(self.path, "Validation-Images.jpg")
        figure = _plt_figure(figsize=(8, 8))
        _plt_axis("off")
        _plt_title("Validation Images")
        _plt_imshow(grid)
        _plt_savefig(location, dpi=160)
        _plt_close(figure)

        self.logln("Saved validation images")

    def save_images_before_training(self, context=None):
        """Saves a batch of the generated images grid before any training.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        batch = c.mods.g.test(c.noises.ref_batch)

        grid = _make_grid(
            batch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = _np_transpose(grid, (1, 2, 0))

        now = _now()
        timestamp = f"Time-{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-"\
            f"{now.microsecond:06}"
        file_name = f"Before-Training-{timestamp}.jpg"

        location = _join(self.gen_img_path, file_name)
        figure = _plt_figure(figsize=(8, 8))
        _plt_axis("off")
        _plt_title(f"Generated Images Before Any Training")
        _plt_imshow(grid)
        _plt_savefig(location, dpi=120)
        _plt_close(figure)

        self.logln("Saved images before training")

    def save_generated_images(self, context=None):
        """Saves a batch of the generated images.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        batch = c.mods.g.test(c.noises.ref_batch)

        grid = _make_grid(
            batch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = _np_transpose(grid, (1, 2, 0))

        now = _now()
        timestamp = f"Time-{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-"\
            f"{now.microsecond:06}"
        file_name = f"Iter-{c.loops.iteration.index + 1}-Epoch-{c.loops.epoch.index + 1}-{timestamp}.jpg"
        location = _join(self.gen_img_path, file_name)
        figure = _plt_figure(figsize=(8, 8))
        _plt_axis("off")
        _plt_title(f"Iter {c.loops.iteration.index + 1} Epoch {c.loops.epoch.index + 1} Generated Images")
        _plt_imshow(grid)
        _plt_savefig(location, dpi=120)
        _plt_close(figure)

        self.logln("Saved generated images")

    def save_disc_losses(self, context=None):
        """Saves the discriminator training/validation losses plot.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        iteration = c.loops.iteration.index
        epoch_count, epoch = c.loops.epoch.count, c.loops.epoch.index

        epoch_list = list(range(1, epoch_count * iteration + epoch + 2))
        iter_x_list = [epoch_count * x + 0.5 for x in range(iteration + 1)]
        rb_x_list = [epoch_count * x[0] + x[1] + 1 + 0.5 for x in c.rbs.d]
        collapse_x_list = [epoch_count * x[0] + x[1] + 1 for x in c.collapses.epochs]

        location = _join(self.path, "Discriminator-Losses.jpg")
        figure = _plt_figure(figsize=(10, 5))
        _plt_title("Discriminator Losses")

        _plt_plot(epoch_list, c.losses.train.d, alpha=0.8, color="b", label="Training")
        _plt_plot(epoch_list, c.losses.valid.d, alpha=0.8, color="r", label="Validation")
        for x in iter_x_list:
            _plt_axvline(x, alpha=0.6, color="gray")
        for x in rb_x_list:
            _plt_axvline(x, alpha=0.6, color="purple")
        for x in collapse_x_list:
            _plt_axvline(x, alpha=0.6, color="orange")

        _plt_xlabel("Epoch No.")
        _plt_xticks(epoch_list)
        _plt_ylabel("Loss")

        box = _plt_gca().get_position()
        _plt_gca().set_position([box.x0 * 0.825, box.y0, box.width * 0.9, box.height])
        handles, labels = _plt_gca().get_legend_handles_labels()
        handles.append(_Line2D([0], [0], alpha=0.6, color="gray"))
        labels.append("Iteration")
        handles.append(_Line2D([0], [0], alpha=0.6, color="purple"))
        labels.append("Rollback")
        handles.append(_Line2D([0], [0], alpha=0.6, color="orange"))
        labels.append("Training Collapse")
        _plt_legend(handles=handles, labels=labels, bbox_to_anchor=(1.125, 0.5), loc="center", fontsize="small")

        _plt_savefig(location, dpi=160)
        _plt_close(figure)

        self.logln("Saved D losses plot")

    def save_gen_losses(self, context=None):
        """Saves the generator training/validation losses plot.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        iteration = c.loops.iteration.index
        epoch_count, epoch = c.loops.epoch.count, c.loops.epoch.index

        epoch_list = list(range(1, epoch_count * iteration + epoch + 2))
        iter_x_list = [epoch_count * x + 0.5 for x in range(iteration + 1)]
        rb_x_list = [epoch_count * x[0] + x[1] + 1 + 0.5 for x in c.rbs.g]
        collapse_x_list = [epoch_count * x[0] + x[1] + 1 for x in c.collapses.epochs]

        location = _join(self.path, "Generator-Losses.jpg")
        figure = _plt_figure(figsize=(10, 5))
        _plt_title("Generator Losses")

        _plt_plot(epoch_list, c.losses.train.g, alpha=0.8, color="b", label="Training")
        _plt_plot(epoch_list, c.losses.valid.g, alpha=0.8, color="r", label="Validation")
        for x in iter_x_list:
            _plt_axvline(x, alpha=0.6, color="gray")
        for x in rb_x_list:
            _plt_axvline(x, alpha=0.6, color="purple")
        for x in collapse_x_list:
            _plt_axvline(x, alpha=0.6, color="orange")

        _plt_xlabel("Epoch No.")
        _plt_xticks(epoch_list)
        _plt_ylabel("Loss")

        box = _plt_gca().get_position()
        _plt_gca().set_position([box.x0 * 0.825, box.y0, box.width * 0.9, box.height])
        handles, labels = _plt_gca().get_legend_handles_labels()
        handles.append(_Line2D([0], [0], alpha=0.6, color="gray"))
        labels.append("Iteration")
        handles.append(_Line2D([0], [0], alpha=0.6, color="purple"))
        labels.append("Rollback")
        handles.append(_Line2D([0], [0], alpha=0.6, color="orange"))
        labels.append("Training Collapse")
        _plt_legend(handles=handles, labels=labels, bbox_to_anchor=(1.125, 0.5), loc="center", fontsize="small")

        _plt_savefig(location, dpi=160)
        _plt_close(figure)

        self.logln("Saved G losses plot")

    def save_tvg(self, context=None):
        """Saves the TVG (training-validation-generated) figure.

        Args:
            context: optional context
        """
        c: _TrainContext = self.find_context(context)

        tbatch = next(iter(c.data.train.loader))
        tbatch = tbatch[0].cpu()
        tbatch = tbatch[:c.data.batch_size]

        vbatch = next(iter(c.data.valid.loader))
        vbatch = vbatch[0].cpu()
        vbatch = vbatch[:c.data.batch_size]

        gbatch = c.mods.g.test(c.noises.ref_batch)

        tgrid = _make_grid(
            tbatch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        vgrid = _make_grid(
            vbatch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        ggrid = _make_grid(
            gbatch, nrow=_ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )

        tgrid = tgrid.cpu()
        vgrid = vgrid.cpu()
        ggrid = ggrid.cpu()

        tgrid = _np_transpose(tgrid, (1, 2, 0))
        vgrid = _np_transpose(vgrid, (1, 2, 0))
        ggrid = _np_transpose(ggrid, (1, 2, 0))

        location = _join(self.path, "Training-Validation-Generated.jpg")
        figure = _plt_figure(figsize=(24, 24))
        sp_figure, axes = _plt_subplots(1, 3)
        sp1, sp2, sp3 = axes[0], axes[1], axes[2]
        sp1.axis("off")
        sp1.set_title("Training Images")
        sp1.imshow(tgrid)
        sp2.axis("off")
        sp2.set_title("Validation Images")
        sp2.imshow(vgrid)
        sp3.axis("off")
        sp3.set_title("Generated Images")
        sp3.imshow(ggrid)
        sp_figure.tight_layout()
        _plt_savefig(location, dpi=240)
        _plt_close(sp_figure)
        _plt_close(figure)

        self.logln("Saved TVG figure")
