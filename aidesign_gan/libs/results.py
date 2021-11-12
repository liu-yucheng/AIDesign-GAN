"""Module of the results classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

from matplotlib import lines
from matplotlib import pyplot
from torchvision import utils as vutils

import datetime
import math
import numpy

from aidesign_gan.libs import contexts
from aidesign_gan.libs import utils

_Context = contexts.Context
_TrainingContext = contexts.TrainingContext
_GenerationContext = contexts.GenerationContext


class Results:
    """The super class of the results classes."""

    def __init__(self, path, logs):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            logs: the log file objects
        """
        self.path = path
        """Root path of the results."""
        self.logs = logs
        """Log file objects."""
        self.context = None
        """Context to be bind."""

    def init_folders(self):
        """Inits the result folders.

        Raises:
            NotImplementedError: always
        """
        raise NotImplementedError("init_folders not implemented")

    def bind_context(self, context):
        """Binds a context to self.

        Args:
            context: the context to bind
        """
        self.context = context

    def check_context(self):
        """Check if self.context is not None.

        Raises:
            ValueError: if self.context is None
        """
        if self.context is None:
            raise ValueError("self.context cannot be None")

    def logstr(self, string=""):
        """Logs a string.

        Args:
            string: the string to log
        """
        utils.logstr(self.logs, string)

    def logln(self, line=""):
        """Logs a line.

        Args:
            line: the line to log
        """
        utils.logln(self.logs, line)

    def log_configs(self, coords_config, modelers_config):
        """Logs the coords and modelers config info.

        Args:
            coords_config: the coords config
            modelers_config: the modelers config
        """
        self.logln(f"Coords config: {coords_config.location}")
        self.logln(f"Modelers config: {modelers_config.location}")

    def log_rand(self):
        """Logs the random info."""
        self.check_context()
        c: _Context = self.context

        self.logln(f"Random seed ({c.rand.mode}): {c.rand.seed}")

    def log_hw(self):
        """Logs the torch hardware info."""
        self.check_context()
        c: _Context = self.context

        self.logln(f"Torch device: {c.hw.device}; GPU count: {c.hw.gpu_count}")


class TrainingResults(Results):
    """Training results."""

    def __init__(self, path, logs):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            logs: the log file objects
        """
        super().__init__(path, logs)
        self.generated_images_path = utils.concat_paths(path, "Generated-Images")
        """Generated images path."""

    def init_folders(self):
        """Inits the result folders."""
        utils.init_folder(self.path)
        self.logln(f"Initialized folder: {self.path}")
        utils.init_folder(self.generated_images_path)
        self.logln(f"Initialized folder: {self.generated_images_path}")

    def log_data(self):
        """Logs the data loaders info."""
        self.check_context()
        c: _TrainingContext = self.context

        self.logln(f"Data total size: {c.data.size}; Data size to use: {c.data.size_to_use}")
        self.logln(f"Training set size: {c.data.train.size}; Validation set size: {c.data.valid.size}")

    def log_mods(self):
        """Logs the modelers info."""
        self.check_context()
        c: _TrainingContext = self.context

        self.logstr(
            str(
                "D's modeler:\n"
                "  Model:  Size: {}  Training size: {}\n"
                "  Adam optimizer:  Learning rate: {}\n"
                "==== D's model struct ====\n"
                "{}\n"
            ).format(
                c.mods.d.size,
                c.mods.d.training_size,
                c.mods.d.config["adam_optimizer"]["learning_rate"],
                str(c.mods.d.model)
            )
        )
        self.logstr(
            str(
                "G's modeler:\n"
                "  Model:  Size: {}  Training size: {}\n"
                "  Adam optimizer:  Learning rate: {}\n"
                "==== G's model struct ====\n"
                "{}\n"
            ).format(
                c.mods.g.size,
                c.mods.g.training_size,
                c.mods.g.config["adam_optimizer"]["learning_rate"],
                str(c.mods.g.model)
            )
        )

    def log_mode(self):
        """Logs the training mode info."""
        self.check_context()
        c: _TrainingContext = self.context

        self.logln(f"Training mode: {c.mode}")

    def log_labels(self):
        """Logs the labels info."""
        self.check_context()
        c: _TrainingContext = self.context

        self.logln("Labels:  Real = {}  Fake = {}".format(f"{c.labels.real:.3f}", f"{c.labels.fake:.3f}"))

    def log_algo(self, algo_name):
        """Logs the training algorithm.

        Args:
            algo_name: the name of the algorithm
        """
        algo_name = str(algo_name)
        self.logln(f"Algo: {algo_name}")

    def log_iter(self, prefix):
        """Logs the iter info.

        Args:
            prefix: the contents to log before the info
        """
        self.check_context()
        c: _TrainingContext = self.context

        self.logstr("==== ==== ")

        self.logstr(f"{prefix} iter {c.loops.iteration.index + 1} / {c.loops.iteration.count}")

        self.logstr(" ==== ====\n")

    def log_epoch(self, prefix, epoch_type):
        """Logs the epoch info.

        Args:
            prefix: the contents to log before the info
            epoch_type: epoch type (d/g/(empty string))
        """
        self.check_context()
        c: _TrainingContext = self.context

        self.logstr("==== ")

        self.logstr(f"{prefix} epoch {c.loops.iteration.index + 1}.{epoch_type}{c.loops.epoch.index + 1} / ")
        self.logstr(f"{c.loops.iteration.count}.{epoch_type}{c.loops.epoch.count}")

        self.logstr(" ====\n")

    def find_train_needs_log(self):
        """Finds if the current training batch needs to be logged.

        Returns:
            result: whether the batch needs to be logged
        """
        self.check_context()
        c: _TrainingContext = self.context

        result = c.loops.train.index == 0
        result = result or (c.loops.train.index + 1) % 30 == 0
        result = result or c.loops.train.index == c.data.train.batch_count - 1
        return result

    def find_valid_needs_log(self):
        """Finds if the current validation batch needs to be logged.

        Returns:
            result: whether the batch needs to be logged
        """
        self.check_context()
        c: _TrainingContext = self.context

        result = c.loops.valid.index == 0
        result = result or (c.loops.valid.index + 1) % 15 == 0
        result = result or c.loops.valid.index == c.data.valid.batch_count - 1
        return result

    def log_batch(self, epoch_type, batch_type):
        """Logs the batch info for the iter level algo.

        Args:
            epoch_type: epoch type (d/g)
            batch_type: batch type (tr/tf/vr/vf/t/v)
        """
        self.check_context()
        c: _TrainingContext = self.context

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

        self.logstr("Batch {}.{}{}.{}{} / ".format(
            c.loops.iteration.index + 1,
            epoch_type, c.loops.epoch.index + 1,
            batch_type, batch_index + 1
        ))
        self.logstr("{}.{}{}.{}{}: ".format(
            c.loops.iteration.count,
            epoch_type, c.loops.epoch.count,
            batch_type, batch_count
        ))

        if "d" in epoch_type:
            if "t" in batch_type:
                self.logstr(f"D(X) = {c.latest.dx:.6f} D(G(Z)) = {c.latest.dgz:.6f} ")
            elif "v" in batch_type:
                if "r" in batch_type:
                    self.logstr(f"D(X) = {c.latest.dx:.6f} ")
                elif "f" in batch_type:
                    self.logstr(f"D(G(Z)) = {c.latest.dgz:.6f} ")
            self.logstr(f"L(D) = {c.latest.ld:.6f}")
        elif "g" in epoch_type:
            self.logstr(f"D(G(Z)) = {c.latest.dgz:.6f} L(G) = {c.latest.lg:.6f}")

        self.logstr("\n")

    def log_batch_2(self, batch_type):
        """Logs the batch info for the batch level algo.

        Args:
            batch_type: batch type (t/vdr/vdf/vg)
        """
        self.check_context()
        c: _TrainingContext = self.context

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

        self.logstr("Batch {}.{}.{}{} / ".format(
            c.loops.iteration.index + 1,
            c.loops.epoch.index + 1,
            batch_type, batch_index + 1
        ))
        self.logstr("{}.{}.{}{}: ".format(
            c.loops.iteration.count,
            c.loops.epoch.count,
            batch_type, batch_count
        ))

        if "t" in batch_type:
            self.logstr("\n")
            self.logstr(f"  D: D(X) = {c.latest.dx:.6f} D(G(Z)) = {c.latest.dgz:.6f} L(D) = {c.latest.ld:.6f}\n")
            self.logstr(f"  G: D(G(Z)) = {c.latest.dgz2:.6f} L(G) = {c.latest.lg:.6f}")
        elif "v" in batch_type:
            if "d" in batch_type:
                if "r" in batch_type:
                    self.logstr(f"D(X) = {c.latest.dx:.6f} L(D) = {c.latest.ld:.6f}")
                elif "f" in batch_type:
                    self.logstr(f"D(G(Z)) = {c.latest.dgz:.6f} L(D) = {c.latest.ld:.6f}")
            elif "g" in batch_type:
                self.logstr(f"D(G(Z)): {c.latest.dgz2:.6f} L(G) = {c.latest.lg:.6f}")

        self.logstr("\n")

    def log_batch_3(self, batch_type):
        """Logs the batch info for the batch level algo.

        Args:
            batch_type: batch type (t/v)
        """
        self.check_context()
        c: _TrainingContext = self.context

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

        self.logstr("Batch {}.{}.{}{} / ".format(
            c.loops.iteration.index + 1,
            c.loops.epoch.index + 1,
            batch_type, batch_index + 1
        ))
        self.logstr("{}.{}.{}{}: ".format(
            c.loops.iteration.count,
            c.loops.epoch.count,
            batch_type, batch_count
        ))

        self.logstr("\n")
        self.logstr(
            str(
                "     D:  D(X) = {}  D(G(Z)) = {}\n"
                "  L(D):  L(D,X) = {}  L(D,G(Z)) = {}  L(D) = {}\n"
            ).format(
                f"{c.latest.dx:.6f}", f"{c.latest.dgz:.6f}",
                f"{c.latest.ldr:.6f}", f"{c.latest.ldf:.6f}", f"{c.latest.ld:.6f}"
            )
        )
        self.logstr(
            str(
                "     G:  D(X) = {}  D(G(Z)) = {}\n"
                "  L(G):  L(G,X) = {}  L(G,G(Z)) = {}  L(G) = {}\n"
            ).format(
                f"{c.latest.dx2:.6f}", f"{c.latest.dgz2:.6f}",
                f"{c.latest.lgr:.6f}", f"{c.latest.lgf:.6f}", f"{c.latest.lg:.6f}"
            )
        )

    def log_epoch_loss(self, loss_type):
        """Logs the epoch loss info.

        Args:
            loss_type: loss type (td/vd/tg/vg)
        """
        self.check_context()
        c: _TrainingContext = self.context

        loss = None
        if loss_type == "td":
            loss = c.losses.train.d[-1]
        elif loss_type == "vd":
            loss = c.losses.valid.d[-1]
        elif loss_type == "tg":
            loss = c.losses.train.g[-1]
        elif loss_type == "vg":
            loss = c.losses.valid.g[-1]

        self.logstr("Epoch ")

        if "d" in loss_type:
            self.logstr("D ")
        elif "g" in loss_type:
            self.logstr("G ")

        if "t" in loss_type:
            self.logstr("training ")
        elif "v" in loss_type:
            self.logstr("validation ")

        self.logstr(f"loss = {loss:.6f}\n")

    def log_best_losses(self, model_type):
        """Logs the best loss info.

        Args:
            model_type: model type (d/g)
        """
        self.check_context()
        c: _TrainingContext = self.context

        if model_type == "d":
            curr_loss = c.losses.valid.d[-1]
            prev_best = c.bests.d
            self.logstr("D: ")
        elif model_type == "g":
            curr_loss = c.losses.valid.g[-1]
            prev_best = c.bests.g
            self.logstr("G: ")

        self.logstr(f"curr loss = {curr_loss:.6f} ")

        if prev_best is None:
            self.logstr("prev best loss = None")
        else:
            self.logstr(f"prev best loss = {prev_best:.6f}")

        self.logstr("\n")

    def log_model_action(self, action_type, model_type):
        """Logs the model action info.

        Args:
            action_type: action type (save/es/rb/load)
            model_type: model type (d/g)
        """
        self.check_context()
        c: _TrainingContext = self.context

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

    def save_training_images(self):
        """Saves the first 64 training images."""
        self.check_context()
        c: _TrainingContext = self.context

        batch = next(iter(c.data.train.loader))
        batch = batch[0].cpu()
        batch = batch[:c.data.batch_size]

        grid = vutils.make_grid(
            batch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = numpy.transpose(grid, (1, 2, 0))

        location = utils.find_in_path("Training-Images.jpg", self.path)
        figure = pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Training Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=160)
        pyplot.close(figure)

        self.logln("Saved training images")

    def save_validation_images(self):
        """Saves the first 64 validation images."""
        self.check_context()
        c: _TrainingContext = self.context

        batch = next(iter(c.data.valid.loader))
        batch = batch[0].cpu()
        batch = batch[:c.data.batch_size]

        grid = vutils.make_grid(
            batch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = numpy.transpose(grid, (1, 2, 0))

        location = utils.find_in_path("Validation-Images.jpg", self.path)
        figure = pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Validation Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=160)
        pyplot.close(figure)

        self.logln("Saved validation images")

    def save_images_before_training(self):
        """Saves the generated images grid before any training."""
        self.check_context()
        c: _TrainingContext = self.context

        batch = c.mods.g.test(c.noises.ref_batch)

        grid = vutils.make_grid(
            batch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = numpy.transpose(grid, (1, 2, 0))

        now = datetime.datetime.now()
        timestamp = f"Time-{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-"\
            f"{now.microsecond:06}"
        file_name = f"Before-Training-{timestamp}.jpg"

        location = utils.find_in_path(file_name, self.generated_images_path)
        figure = pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title(f"Generated Images Before Any Training")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=120)
        pyplot.close(figure)

        self.logln("Saved images before training")

    def save_generated_images(self):
        """Saves the generated images grid."""
        self.check_context()
        c: _TrainingContext = self.context

        batch = c.mods.g.test(c.noises.ref_batch)

        grid = vutils.make_grid(
            batch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        grid = grid.cpu()
        grid = numpy.transpose(grid, (1, 2, 0))

        now = datetime.datetime.now()
        timestamp = f"Time-{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-"\
            f"{now.microsecond:06}"
        file_name = f"Iter-{c.loops.iteration.index + 1}-Epoch-{c.loops.epoch.index + 1}-{timestamp}.jpg"
        location = utils.find_in_path(file_name, self.generated_images_path)
        figure = pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title(f"Iter {c.loops.iteration.index + 1} Epoch {c.loops.epoch.index + 1} Generated Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=120)
        pyplot.close(figure)

        self.logln("Saved generated images")

    def save_d_losses(self):
        """Saves the discriminator training/validation losses plot."""
        self.check_context()
        c: _TrainingContext = self.context

        iteration = c.loops.iteration.index
        epoch_count, epoch = c.loops.epoch.count, c.loops.epoch.index

        epoch_list = list(range(1, epoch_count * iteration + epoch + 2))
        iter_x_list = [epoch_count * x + 0.5 for x in range(iteration + 1)]
        rb_x_list = [epoch_count * x[0] + x[1] + 1 + 0.5 for x in c.rbs.d]
        collapse_x_list = [epoch_count * x[0] + x[1] + 1 for x in c.collapses.epochs]

        location = utils.find_in_path("Discriminator-Losses.jpg", self.path)
        figure = pyplot.figure(figsize=(10, 5))
        pyplot.title("Discriminator Losses")

        pyplot.plot(epoch_list, c.losses.train.d, alpha=0.8, color="b", label="Training")
        pyplot.plot(epoch_list, c.losses.valid.d, alpha=0.8, color="r", label="Validation")
        for x in iter_x_list:
            pyplot.axvline(x, alpha=0.6, color="gray")
        for x in rb_x_list:
            pyplot.axvline(x, alpha=0.6, color="purple")
        for x in collapse_x_list:
            pyplot.axvline(x, alpha=0.6, color="orange")

        pyplot.xlabel("Epoch No.")
        pyplot.xticks(epoch_list)
        pyplot.ylabel("Loss")

        box = pyplot.gca().get_position()
        pyplot.gca().set_position([box.x0 * 0.825, box.y0, box.width * 0.9, box.height])
        handles, labels = pyplot.gca().get_legend_handles_labels()
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="gray"))
        labels.append("Iteration")
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="purple"))
        labels.append("Rollback")
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="orange"))
        labels.append("Training Collapse")
        pyplot.legend(handles=handles, labels=labels, bbox_to_anchor=(1.125, 0.5), loc="center", fontsize="small")

        pyplot.savefig(location, dpi=160)
        pyplot.close(figure)

        self.logln("Saved D losses plot")

    def save_g_losses(self):
        """Saves the generator training/validation losses plot."""
        self.check_context()
        c: _TrainingContext = self.context

        iteration = c.loops.iteration.index
        epoch_count, epoch = c.loops.epoch.count, c.loops.epoch.index

        epoch_list = list(range(1, epoch_count * iteration + epoch + 2))
        iter_x_list = [epoch_count * x + 0.5 for x in range(iteration + 1)]
        rb_x_list = [epoch_count * x[0] + x[1] + 1 + 0.5 for x in c.rbs.g]
        collapse_x_list = [epoch_count * x[0] + x[1] + 1 for x in c.collapses.epochs]

        location = utils.find_in_path("Generator-Losses.jpg", self.path)
        figure = pyplot.figure(figsize=(10, 5))
        pyplot.title("Generator Losses")

        pyplot.plot(epoch_list, c.losses.train.g, alpha=0.8, color="b", label="Training")
        pyplot.plot(epoch_list, c.losses.valid.g, alpha=0.8, color="r", label="Validation")
        for x in iter_x_list:
            pyplot.axvline(x, alpha=0.6, color="gray")
        for x in rb_x_list:
            pyplot.axvline(x, alpha=0.6, color="purple")
        for x in collapse_x_list:
            pyplot.axvline(x, alpha=0.6, color="orange")

        pyplot.xlabel("Epoch No.")
        pyplot.xticks(epoch_list)
        pyplot.ylabel("Loss")

        box = pyplot.gca().get_position()
        pyplot.gca().set_position([box.x0 * 0.825, box.y0, box.width * 0.9, box.height])
        handles, labels = pyplot.gca().get_legend_handles_labels()
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="gray"))
        labels.append("Iteration")
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="purple"))
        labels.append("Rollback")
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="orange"))
        labels.append("Training Collapse")
        pyplot.legend(handles=handles, labels=labels, bbox_to_anchor=(1.125, 0.5), loc="center", fontsize="small")

        pyplot.savefig(location, dpi=160)
        pyplot.close(figure)

        self.logln("Saved G losses plot")

    def save_tvg(self):
        """Saves the TVG (training-validation-generated) figure."""
        self.check_context()
        c: _TrainingContext = self.context

        tbatch = next(iter(c.data.train.loader))
        tbatch = tbatch[0].cpu()
        tbatch = tbatch[:c.data.batch_size]

        vbatch = next(iter(c.data.valid.loader))
        vbatch = vbatch[0].cpu()
        vbatch = vbatch[:c.data.batch_size]

        gbatch = c.mods.g.test(c.noises.ref_batch)

        tgrid = vutils.make_grid(
            tbatch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        vgrid = vutils.make_grid(
            vbatch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )
        ggrid = vutils.make_grid(
            gbatch, nrow=math.ceil(c.data.batch_size ** 0.5), padding=2, normalize=True, value_range=(-0.75, 0.75)
        )

        tgrid = tgrid.cpu()
        vgrid = vgrid.cpu()
        ggrid = ggrid.cpu()

        tgrid = numpy.transpose(tgrid, (1, 2, 0))
        vgrid = numpy.transpose(vgrid, (1, 2, 0))
        ggrid = numpy.transpose(ggrid, (1, 2, 0))

        location = utils.find_in_path("Training-Validation-Generated.jpg", self.path)
        figure = pyplot.figure(figsize=(24, 24))
        sp_figure, axes = pyplot.subplots(1, 3)
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
        pyplot.savefig(location, dpi=240)
        pyplot.close(sp_figure)
        pyplot.close(figure)

        self.logln("Saved TVG figure")


class GenerationResults(Results):
    """Generation results."""

    def __init__(self, path, logs):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            logs: the log file objects
        """
        super().__init__(path, logs)

    def init_folders(self):
        """Inits the result folders."""
        utils.init_folder(self.path)
        self.logln(f"Initialized folder: {self.path}")

    def log_g(self):
        """Logs the G modelers info."""
        self.check_context()
        c: _GenerationContext = self.context

        self.logln(f"G's size: {c.g.size}")
        self.logln(f"G's training size: {c.g.training_size}")
        self.logln(f"==== G's struct ====")
        self.logln(str(c.g.model))

    def log_batch(self):
        """Logs the batch info."""
        self.check_context()
        c: _GenerationContext = self.context

        needs_log = c.batch_prog.index == 0
        needs_log = needs_log or (c.batch_prog.index + 1) % 15 == 0
        needs_log = needs_log or c.batch_prog.index == c.batch_prog.count - 1
        if not needs_log:
            return

        self.logln(f"Generated image batch {c.batch_prog.index + 1} / {c.batch_prog.count}")

    def save_generated_images(self):
        """Saves the generated images."""
        self.check_context()
        c: _GenerationContext = self.context

        for index, image in enumerate(c.images.to_save):
            now = datetime.datetime.now()
            timestamp = f"Time-{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-"\
                f"{now.microsecond:06}"

            name = ""
            if c.grids.enabled:
                name += "Grid-"
            else:
                name += "Image-"
            name += f"{index + 1}-{timestamp}.jpg"

            location = utils.find_in_path(name, self.path)
            vutils.save_image(image, location, "JPEG")

        count = len(c.images.to_save)
        if c.grids.enabled:
            self.logln(f"Generated {count} grids, each has {c.grids.size_each} images")
        else:
            self.logln(f"Generated {count} images")
