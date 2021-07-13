"""Module of the results classes."""
from matplotlib import lines
from matplotlib import pyplot
from torchvision import utils as vutils
import numpy

from dcgan.libs import utils


class Results:
    """The super class of the result classes.

    Attributes:
        path: the root path of the results
        log: the log file
        context: the binded context
    """

    def __init__(self, path, log):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            log: the log file
        """
        self.path = path
        self.log = log
        self.context = None

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
        self.log.write(string)

    def logln(self, line=""):
        """Logs a line.

        Args:
            line: the line to log
        """
        self.log.write(line + "\n")

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
        c = self.context
        self.logln(f"Random seed ({c.rand.mode}): {c.rand.seed}")

    def log_hw(self):
        """Logs the torch hardware info."""
        self.check_context()
        c = self.context
        self.logln(f"Torch device: {c.hw.device}; GPU count: {c.hw.gpu_count}")


class TrainingResults(Results):
    """Training results.

    Attributes:
        generated_images_path: the generated images path
    """

    def __init__(self, path, log):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            log: the log file
        """
        super().__init__(path, log)
        self.generated_images_path = utils.concat_paths(path, "Generated-Images")

    def init_folders(self):
        """Inits the result folders."""
        utils.init_folder(self.path)
        self.logln(f"Init'd folder: {self.path}")
        utils.init_folder(self.generated_images_path, clean=True)
        self.logln(f"Init'd folder (clean): {self.generated_images_path}")

    def log_data(self):
        """Logs the data loaders info."""
        self.check_context()
        c = self.context
        self.logln(f"Data total size: {c.data.size}; Data size to use: {c.data.size_to_use}")
        self.logln(f"Training set size: {c.data.train.size}; Validation set size: {c.data.valid.size}")

    def log_mods(self):
        """Logs the modelers info."""
        self.check_context()
        c = self.context
        self.logln(f"D's size: {c.mods.d.size}")
        self.logln(f"==== D's struct ====")
        self.logln(str(c.mods.d.model))
        self.logln(f"G's size: {c.mods.g.size}")
        self.logln(f"==== G's struct ====")
        self.logln(str(c.mods.g.model))

    def log_mode(self):
        """Logs the training mode info."""
        self.check_context()
        c = self.context
        self.logln(f"Training mode: {c.mode}")

    def log_iter(self, prefix):
        """Logs the iter info.

        Args:
            prefix: the contents to log before the info
        """
        self.check_context()
        c = self.context
        self.logstr("==== ==== ")
        self.logstr(f"{prefix} iter {c.loops.iter + 1} / {c.loops.iter_count}")
        self.logstr(" ==== ====\n")

    def log_epoch(self, prefix, epoch_type):
        """Logs the epoch info.

        Args:
            prefix: the contents to log before the info
            epoch_type: epoch type (d/g/(empty string))
        """
        self.check_context()
        c = self.context
        self.logstr("==== ")
        self.logstr(f"{prefix} epoch {c.loops.iter + 1}.{epoch_type}{c.loops.epoch + 1} / ")
        self.logstr(f"{c.loops.iter_count}.{epoch_type}{c.loops.epoch_count}")
        self.logstr(" ====\n")

    def find_train_needs_log(self):
        """Finds if the current training batch needs to be logged.

        Returns:
            result: whether the batch needs to be logged
        """
        self.check_context()
        c = self.context
        result = c.loops.train_index == 0
        result = result or (c.loops.train_index + 1) % 50 == 0
        result = result or c.loops.train_index == c.data.train.batch_count - 1
        return result

    def find_valid_needs_log(self):
        """Finds if the current validation batch needs to be logged.

        Returns:
            result: whether the batch needs to be logged
        """
        self.check_context()
        c = self.context
        result = c.loops.valid_index == 0
        result = result or (c.loops.valid_index + 1) % 15 == 0
        result = result or c.loops.valid_index == c.data.valid.batch_count - 1
        return result

    def log_batch(self, epoch_type, batch_type):
        """Logs the batch info for the iter level algo.

        Args:
            epoch_type: epoch type (d/g)
            batch_type: batch type (tr/tf/vr/vf/t/v)
        """
        self.check_context()
        c = self.context
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
            batch_index = c.loops.train_index
            batch_count = c.data.train.batch_count
        elif "v" in batch_type:
            batch_index = c.loops.valid_index
            batch_count = c.data.valid.batch_count
        self.logstr(f"Batch {c.loops.iter + 1}.{epoch_type}{c.loops.epoch + 1}.{batch_type}{batch_index + 1} / ")
        self.logstr(f"{c.loops.iter_count}.{epoch_type}{c.loops.epoch_count}.{batch_type}{batch_count}: ")
        if "d" in epoch_type:
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
        c = self.context
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
            batch_index = c.loops.train_index
            batch_count = c.data.train.batch_count
        elif "v" in batch_type:
            batch_index = c.loops.valid_index
            batch_count = c.data.valid.batch_count
        self.logstr(f"Batch {c.loops.iter + 1}.{c.loops.epoch + 1}.{batch_type}{batch_index + 1} / ")
        self.logstr(f"{c.loops.iter_count}.{c.loops.epoch_count}.{batch_type}{batch_count}:")
        if "t" in batch_type:
            self.logstr("\n")
            self.logstr(f"  D: D(X) = {c.latest.dx:.6f} D(G(Z)) = {c.latest.dgz:.6f} L(D) = {c.latest.ld:.6f}\n")
            self.logstr(f"  G: D(G(Z)) = {c.latest.dgz2:.6f} L(G) = {c.latest.lg:.6f}")
        elif "v" in batch_type:
            self.logstr(" ")
            if "d" in batch_type:
                if "r" in batch_type:
                    self.logstr(f"D(X) = {c.latest.dx:.6f} L(D) = {c.latest.ld:.6f}")
                elif "f" in batch_type:
                    self.logstr(f"D(G(Z)) = {c.latest.dgz:.6f} L(D) = {c.latest.ld:.6f}")
            elif "g" in batch_type:
                self.logstr(f"D(G(Z)): {c.latest.dgz2:.6f} L(G) = {c.latest.lg:.6f}")
        self.logstr("\n")

    def log_epoch_loss(self, loss_type):
        """Logs the epoch loss info.

        Args:
            loss_type: loss type (td/vd/tg/vg)
        """
        self.check_context()
        c = self.context
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
        c = self.context
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
        c = self.context
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
        c = self.context
        batch = next(iter(c.data.train.loader))
        batch = batch[0].to(c.hw.device)[:64]
        grid = vutils.make_grid(batch, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        location = utils.find_in_path("training-images.jpg", self.path)
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Training Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved training images")

    def save_validation_images(self):
        """Saves the first 64 validation images."""
        self.check_context()
        c = self.context
        batch = next(iter(c.data.valid.loader))
        batch = batch[0].to(c.hw.device)[:64]
        grid = vutils.make_grid(batch, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        location = utils.find_in_path("validation-images.jpg", self.path)
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Validation Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved validation images")

    def save_generated_images(self):
        """Saves the generated images grid."""
        self.check_context()
        c = self.context
        batch = c.mods.g.test(c.noises.batch_64)
        grid = vutils.make_grid(batch, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        file_name = f"iter-{c.loops.iter + 1}_epoch-{c.loops.epoch + 1}.jpg"
        location = utils.find_in_path(file_name, self.generated_images_path)
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title(f"Iter {c.loops.iter + 1} Epoch {c.loops.epoch + 1} Generated Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=120)
        pyplot.close()
        self.logln("Saved generated images")

    def save_d_losses(self):
        """Saves the discriminator training/validation losses plot."""
        self.check_context()
        c = self.context
        iter = c.loops.iter
        epoch_count, epoch = c.loops.epoch_count, c.loops.epoch
        epoch_list = list(range(1, epoch_count * iter + epoch + 2))
        iter_x_list = [epoch_count * x + 0.5 for x in range(iter + 1)]
        rb_x_list = [epoch_count * x[0] + x[1] + 1.5 for x in c.rbs.d]
        location = utils.find_in_path("discriminator-losses.jpg", self.path)
        pyplot.figure(figsize=(10, 5))
        pyplot.title("Discriminator Losses")
        pyplot.plot(epoch_list, c.losses.train.d, alpha=0.8, color="b", label="Training")
        pyplot.plot(epoch_list, c.losses.valid.d, alpha=0.8, color="r", label="Validation")
        for x in iter_x_list:
            pyplot.axvline(x, alpha=0.6, color="gray")
        for x in rb_x_list:
            pyplot.axvline(x, alpha=0.6, color="purple")
        pyplot.xlabel("Epoch No.")
        pyplot.xticks(epoch_list)
        pyplot.ylabel("Loss")
        handles, labels = pyplot.gca().get_legend_handles_labels()
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="gray"))
        labels.append("Iteration")
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="purple"))
        labels.append("Rollback")
        pyplot.legend(handles=handles, labels=labels, loc="upper right")
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved D losses plot")

    def save_g_losses(self):
        """Saves the generator training/validation losses plot."""
        self.check_context()
        c = self.context
        iter = c.loops.iter
        epoch_count, epoch = c.loops.epoch_count, c.loops.epoch
        epoch_list = list(range(1, epoch_count * iter + epoch + 2))
        iter_x_list = [epoch_count * x + 0.5 for x in range(iter + 1)]
        rb_x_list = [epoch_count * x[0] + x[1] + 1.5 for x in c.rbs.g]
        location = utils.find_in_path("generator-losses.jpg", self.path)
        pyplot.figure(figsize=(10, 5))
        pyplot.title("Generator Losses")
        pyplot.plot(epoch_list, c.losses.train.g, alpha=0.8, color="b", label="Training")
        pyplot.plot(epoch_list, c.losses.valid.g, alpha=0.8, color="r", label="Validation")
        for x in iter_x_list:
            pyplot.axvline(x, alpha=0.6, color="gray")
        for x in rb_x_list:
            pyplot.axvline(x, alpha=0.6, color="purple")
        pyplot.xlabel("Epoch No.")
        pyplot.xticks(epoch_list)
        pyplot.ylabel("Loss")
        handles, labels = pyplot.gca().get_legend_handles_labels()
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="gray"))
        labels.append("Iteration")
        handles.append(lines.Line2D([0], [0], alpha=0.6, color="purple"))
        labels.append("Rollback")
        pyplot.legend(handles=handles, labels=labels, loc="upper right")
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved G losses plot")

    def save_tvg(self):
        """Saves the TVG (training-validation-generated) figure."""
        self.check_context()
        c = self.context
        tbatch = next(iter(c.data.train.loader))
        tbatch = tbatch[0].to(c.hw.device)[:64]
        vbatch = next(iter(c.data.valid.loader))
        vbatch = vbatch[0].to(c.hw.device)[:64]
        gbatch = c.mods.g.test(c.noises.batch_64)
        tgrid = vutils.make_grid(tbatch, padding=2, normalize=True).cpu()
        vgrid = vutils.make_grid(vbatch, padding=2, normalize=True).cpu()
        ggrid = vutils.make_grid(gbatch, padding=2, normalize=True).cpu()
        tgrid = numpy.transpose(tgrid, (1, 2, 0))
        vgrid = numpy.transpose(vgrid, (1, 2, 0))
        ggrid = numpy.transpose(ggrid, (1, 2, 0))
        location = utils.find_in_path("training-validation-generated.jpg", self.path)
        pyplot.figure(figsize=(24, 24))
        fig, axes = pyplot.subplots(1, 3)
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
        fig.tight_layout()
        pyplot.savefig(location, dpi=240)
        pyplot.close()
        self.logln("Saved TVG figure")


class GenerationResults(Results):
    """Generation results."""

    def __init__(self, path, log):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            log: the log file
        """
        super().__init__(path, log)

    def init_folders(self):
        """Inits the result folders."""
        utils.init_folder(self.path, clean=True)
        self.logln(f"Init'd folder (clean): {self.path}")

    def log_g(self):
        """Logs the G modelers info."""
        self.check_context()
        c = self.context
        self.logln(f"G's size: {c.g.size}")
        self.logln(f"==== G's struct ====")
        self.logln(str(c.g.model))

    def save_generated_images(self):
        """Saves the generated images."""
        self.check_context()
        c = self.context
        for index, image in enumerate(c.images.list):
            location = utils.find_in_path(f"image-{index + 1}.jpg", self.path)
            vutils.save_image(image, location, "JPEG")
        count = len(c.images.list)
        if c.grids.enabled:
            self.logln(f"Generated {count} grids, each has {c.grids.each_size} images")
        else:
            self.logln(f"Generated {count} images")
