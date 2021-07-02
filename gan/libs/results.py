"""Module of the results classes."""

from matplotlib import pyplot
from torchvision import utils as visutils
import numpy

from gan.libs import utils


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
            text: the string to log
        """
        self.log.write(string)

    def logln(self, line=""):
        """Logs a line.

        Args:
            line: the line to log
        """
        self.log.write(line + "\n")


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

    def log_configs(self, coords_config, modelers_config):
        """Logs the coords and modelers config info.

        Args:
            coords_config: the coords config
            modelers_config: the modelers config
        """
        self.logln(f"Coords config: {coords_config.location}")
        self.logln(f"Modelers config: {modelers_config.location}")

    def log_rand_seeds(self):
        """Logs the random seeds info."""
        self.check_context()
        c = self.context
        self.logln(f"Random seed ({c.rand.mode}): {c.rand.seed}")

    def log_device(self):
        """Logs the torch device info."""
        self.check_context()
        c = self.context
        self.logln(f"Torch device: {c.hw.dev}; GPU count: {c.hw.gpu_cnt}")

    def log_data_loaders(self):
        """Logs the data loaders info."""
        self.check_context()
        c = self.context
        self.logln(f"Data total size: {c.data.total_size}; Data size to use: {c.data.size_to_use}")
        self.logln(f"Training set size: {c.data.t_size}; Validation set size: {c.data.v_size}")

    def log_modelers(self):
        """Logs the modelers info."""
        self.check_context()
        c = self.context
        self.logln(f"==== D's struct ====")
        self.logln(c.mods.d_str)
        self.logln(f"==== G's struct ====")
        self.logln(c.mods.g_str)

    def log_mode(self):
        """Logs the training mode info."""
        self.check_context()
        c = self.context
        self.logln(f"Training mode: {c.mode}")

    def log_iter(self, prefix):
        """Logs the iter info.

        Args:
            prefix: the string to log before the info
        """
        self.check_context()
        c = self.context
        self.logln(f"{prefix} iter {c.loops.iter_num + 1} / {c.loops.iter_cnt}")

    def log_epoch(self, prefix):
        """Logs the epoch info.

        Args:
            prefix: the string to log before the info
        """
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        self.logstr(prefix + " ")
        self.logstr(f"epoch {iter_num + 1}.{epoch_num + 1} / ")
        self.logstr(f"{iter_cnt}.{epoch_cnt}\n")

    def log_bests(self):
        """Logs the best losses info"""
        self.check_context()
        c = self.context
        self.logstr(f"Iter best D loss: {c.bests.iter_d:.4f}; ")
        self.logstr(f"Overall best D loss: {c.bests.d:.4f}\n")
        self.logstr(f"Iter best G loss: {c.bests.iter_g:.4f}; ")
        self.logstr(f"Overall best G loss: {c.bests.g:.4f}\n")

    def log_save_d(self):
        """Logs save discriminator info."""
        self.logln("Saved D")

    def log_save_g(self):
        """Logs save generator info."""
        self.logln("Saved G")

    def log_rollback_d(self):
        """Logs rollback discriminator info."""
        self.check_context()
        c = self.context
        self.logln(f"Rollbacked D, count {c.mods.d.rb_cnt}")

    def log_rollback_g(self):
        """Logs rollback generator info."""
        self.check_context()
        c = self.context
        self.logln(f"Rollbacked G, count {c.mods.g.rb_cnt}")

    def log_train_dr(self):
        """Logs train_d on real stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.t_idx
        batch_cnt = c.data.t_batch_cnt
        dx = c.outs.dx
        loss = c.outs.ld
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.r-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.r-{batch_cnt}: ")
            self.logstr(f"D(X) = {dx:.4f} L(D) = {loss:.4f}\n")

    def log_train_df(self):
        """Logs train_d on fake stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.t_idx
        batch_cnt = c.data.t_batch_cnt
        dgz = c.outs.dgz
        loss = c.outs.ld
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.f-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.f-{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(D) = {loss:.4f}\n")

    def log_train_g(self):
        """Logs train_g stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.t_idx
        batch_cnt = c.data.t_batch_cnt
        dgz = c.outs.dgz
        loss = c.outs.lg
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(G) = {loss:.4f}\n")

    def log_train_both(self):
        """Logs train_dg_stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.t_idx
        batch_cnt = c.data.t_batch_cnt
        dx = c.outs.dx
        dgz = c.outs.dgz
        dgz2 = c.outs.dgz2
        ld = c.outs.ld
        lg = c.outs.lg
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.{batch_cnt}:\n")
            self.logstr(f"  D: D(X) = {dx:.4f} D(G(Z)) = {dgz:.4f} ")
            self.logstr(f"L(D) = {ld:.4f}\n")
            self.logstr(f"  G: D(G(Z)) = {dgz2:.4f} L(G) = {lg:.4f}\n")

    def log_validate_dr(self):
        """Logs validate_d on real stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.v_idx
        batch_cnt = c.data.v_batch_cnt
        dx = c.outs.dx
        loss = c.outs.ld
        if idx == 0 or (idx + 1) % 20 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.r-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.r-{batch_cnt}: ")
            self.logstr(f"D(X) = {dx:.4f} L(D) = {loss:.4f}\n")

    def log_validate_df(self):
        """Logs validate_d on fake stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.v_idx
        batch_cnt = c.data.v_batch_cnt
        dgz = c.outs.dgz
        loss = c.outs.ld
        if idx == 0 or (idx + 1) % 20 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.f-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.f-{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(D) = {loss:.4f}\n")

    def log_validate_g(self):
        """Logs validate_g stats."""
        self.check_context()
        c = self.context
        iter_num = c.loops.iter_num
        iter_cnt = c.loops.iter_cnt
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        idx = c.loops.v_idx
        batch_cnt = c.data.v_batch_cnt
        dgz = c.outs.dgz
        loss = c.outs.lg
        if idx == 0 or (idx + 1) % 20 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(G) = {loss:.4f}\n")

    def log_td_loss(self):
        """Logs the latest discriminator training loss."""
        self.check_context()
        c = self.context
        self.logln(f"D training loss: {c.losses.td[-1]:.4f}")

    def log_tg_loss(self):
        """Logs the latest generator training loss."""
        self.check_context()
        c = self.context
        self.logln(f"G training loss: {c.losses.tg[-1]:.4f}")

    def log_vd_loss(self):
        """Logs the latest discriminator validation loss."""
        self.check_context()
        c = self.context
        self.logln(f"D validation loss: {c.losses.vd[-1]:.4f}")

    def log_vg_loss(self):
        """Logs the latest generator validation loss."""
        self.check_context()
        c = self.context
        self.logln(f"G validation loss: {c.losses.vg[-1]:.4f}")

    def save_training_images(self):
        """Saves the first 64 training images."""
        self.check_context()
        c = self.context
        batch0 = next(iter(c.data.tdl))
        batch0 = batch0[0].to(c.hw.dev)[:64]
        grid = visutils.make_grid(batch0, padding=2, normalize=True).cpu()
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
        batch0 = next(iter(c.data.vdl))
        batch0 = batch0[0].to(c.hw.dev)[:64]
        grid = visutils.make_grid(batch0, padding=2, normalize=True).cpu()
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
        iter_num = c.loops.iter_num
        epoch_num = c.loops.epoch_num
        batch = c.mods.g.test(c.noises.b64)
        grid = visutils.make_grid(batch, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        location = utils.find_in_path(f"iter-{iter_num + 1}_epoch-{epoch_num + 1}.jpg", self.generated_images_path)
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title(f"Iter {iter_num + 1} Epoch {epoch_num + 1} Generated Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=120)
        pyplot.close()
        self.logln("Saved generated images")

    def save_d_losses(self):
        """Saves the discriminator training/validation losses plot."""
        self.check_context()
        c = self.context
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        iter_num = c.loops.iter_num
        epochs = list(range(1, iter_num * epoch_cnt + epoch_num + 2))
        iter_lns = [epoch_cnt * x + 1.5 for x in range(iter_num + 1)]
        location = utils.find_in_path("discriminator-losses.jpg", self.path)
        pyplot.figure(figsize=(10, 5))
        pyplot.title("Discriminator Training and Validation Losses")
        pyplot.plot(epochs, c.losses.td, label="Training", color="b")
        pyplot.plot(epochs, c.losses.vd, label="Validation", color="r")
        for ln in iter_lns:
            pyplot.axvline(ln, alpha=0.6, color="gray")
        pyplot.xlabel("Epoch No.")
        pyplot.ylabel("Loss")
        pyplot.legend()
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved D losses")

    def save_g_losses(self):
        """Saves the generator training/validation losses plot."""
        self.check_context()
        c = self.context
        epoch_num = c.loops.epoch_num
        epoch_cnt = c.loops.epoch_cnt
        iter_num = c.loops.iter_num
        epochs = list(range(1, iter_num * epoch_cnt + epoch_num + 2))
        iter_lns = [epoch_cnt * x + 1.5 for x in range(iter_num + 1)]
        location = utils.find_in_path("generator-losses.jpg", self.path)
        pyplot.figure(figsize=(10, 5))
        pyplot.title("Generator Training and Validation Losses")
        pyplot.plot(epochs, c.losses.tg, label="Training", color="b")
        pyplot.plot(epochs, c.losses.vg, label="Validation", color="r")
        for ln in iter_lns:
            pyplot.axvline(ln, alpha=0.6, color="gray")
        pyplot.xlabel("Epoch No.")
        pyplot.ylabel("Loss")
        pyplot.legend()
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved G losses")

    def save_tvg(self):
        """Saves the TVG (training-validation-generated) figure."""
        self.check_context()
        c = self.context
        t_batch0 = next(iter(c.data.tdl))
        t_batch0 = t_batch0[0].to(c.hw.dev)[:64]
        v_batch0 = next(iter(c.data.vdl))
        v_batch0 = v_batch0[0].to(c.hw.dev)[:64]
        g_batch0 = c.mods.g.test(c.noises.b64)
        t_grid = visutils.make_grid(t_batch0, padding=2, normalize=True).cpu()
        v_grid = visutils.make_grid(v_batch0, padding=2, normalize=True).cpu()
        g_grid = visutils.make_grid(g_batch0, padding=2, normalize=True).cpu()
        t_grid = numpy.transpose(t_grid, (1, 2, 0))
        v_grid = numpy.transpose(v_grid, (1, 2, 0))
        g_grid = numpy.transpose(g_grid, (1, 2, 0))
        location = utils.find_in_path("training-validation-generated.jpg", self.path)
        pyplot.figure(figsize=(24, 24))
        fig, axs = pyplot.subplots(1, 3)
        sp1, sp2, sp3 = axs[0], axs[1], axs[2]
        sp1.axis("off")
        sp1.set_title("Training Images")
        sp1.imshow(t_grid)
        sp2.axis("off")
        sp2.set_title("Validation Images")
        sp2.imshow(v_grid)
        sp3.axis("off")
        sp3.set_title("Generated Images")
        sp3.imshow(g_grid)
        fig.tight_layout()
        pyplot.savefig(location, dpi=240)
        pyplot.close()
        print("Saved TVG figure")
