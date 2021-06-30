"""Module of the results classes."""

from gan_libs import utils
from matplotlib import pyplot
from torchvision import utils as visutils
import numpy


class Results:
    """The super class of the result classes.

    Attributes:
        path: the root path of the results
        log: the log file
    """

    def __init__(self, path, log):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            log: the log file
        """
        self.path = path
        self.log = log

    def logstr(self, string=""):
        """Logs a string.

        Args:
            text: the string to be logged
        """
        self.log.write(string)

    def logln(self, line=""):
        """Logs a line.

        Args:
            line: the line to be logged
        """
        self.log.write(line + "\n")


class TResults(Results):
    """Training results.

    Attributes:
        gimg_path: the generated images path
    """

    def __init__(self, path, log):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            log: the log file
        """
        super().__init__(path, log)
        self.gimg_path = utils.concat_paths(path, "Generated-Images")

    def init_folders(self):
        """Inits the result folders."""
        utils.init_folder(self.path)
        self.logln(f"Init'd folder: {self.path}")
        utils.init_folder(self.gimg_path, clean=True)
        self.logln(f"Init'd folder (clean): {self.gimg_path}")

    def log_configs(self, c_config, m_config):
        """Logs the coords and modelers config info.

        Args:
            c_config: the coords config
            m_config: the modelers config
        """
        self.logln(f"Coords config: {c_config.location}")
        self.logln(f"Modelers config: {m_config.location}")

    def log_rand_seeds(self, ctx):
        """Logs the random seeds info.

        Args:
            ctx: the training context
        """
        self.logln(f"Random seed ({ctx.rand.mode}): {ctx.rand.seed}")

    def log_device(self, ctx):
        """Logs the torch device info.

        Args:
            ctx: the training context
        """
        self.logln(f"Torch device: {ctx.hw.dev}; GPU count: {ctx.hw.gpu_cnt}")

    def log_data_loaders(self, ctx):
        """Logs the data loaders info.

        Args:
            ctx: the training context
        """
        self.logstr(f"Data total size: {ctx.data.total_size}; ")
        self.logstr(f"Data size to use: {ctx.data.size_to_use}\n")
        self.logstr(f"Training set size: {ctx.data.t_size}; ")
        self.logstr(f"Validation set size: {ctx.data.v_size}\n")

    def log_modelers(self, ctx):
        """Logs the modelers info.

        Args:
            ctx: the training context
        """
        self.logln(f"==== D's struct ====")
        self.logln(ctx.mods.d_str)
        self.logln(f"==== G's struct ====")
        self.logln(ctx.mods.g_str)

    def log_mode(self, ctx):
        """Logs the training mode info.

        Args:
            ctx: the training context
        """
        self.logln(f"Training mode: {ctx.mode}")

    def log_iter(self, ctx, prefix):
        """Logs the iter info.

        Args:
            ctx: the training context
            prefix: the string to be displayed before the info
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        self.logstr(prefix + " ")
        self.logstr(f"iter {iter_num + 1} / {iter_cnt}\n")

    def log_epoch(self, ctx, prefix):
        """Logs the epoch info.

        Args:
            ctx: the training context
            prefix: the string to be displayed before the info
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        self.logstr(prefix + " ")
        self.logstr(f"epoch {iter_num + 1}.{epoch_num + 1} / ")
        self.logstr(f"{iter_cnt}.{epoch_cnt}\n")

    def log_bests(self, ctx):
        """Logs the best losses info.

        Args:
            ctx: the training context
        """
        self.logstr(f"Iter best D loss: {ctx.bests.iter_d:.4f}; ")
        self.logstr(f"Overall best D loss: {ctx.bests.d:.4f}\n")
        self.logstr(f"Iter best G loss: {ctx.bests.iter_g:.4f}; ")
        self.logstr(f"Overall best G loss: {ctx.bests.g:.4f}\n")

    def log_save_d(self):
        """Logs save discriminator info."""
        self.logln("Saved D")

    def log_save_g(self):
        """Logs save generator info."""
        self.logln("Saved G")

    def log_rollback_d(self, ctx):
        """Logs rollback discriminator info.

        Args:
            ctx: the training context
        """
        self.logln(f"Rollbacked D, count {ctx.mods.d.rb_cnt}")

    def log_rollback_g(self, ctx):
        """Logs rollback generator info.

        Args:
            ctx: the training context
        """
        self.logln(f"Rollbacked G, count {ctx.mods.g.rb_cnt}")

    def log_train_dr(self, ctx):
        """Logs train_d on real stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.t_idx
        batch_cnt = ctx.data.t_batch_cnt
        dx = ctx.outs.dx
        loss = ctx.outs.ld
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.r-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.r-{batch_cnt}: ")
            self.logstr(f"D(X) = {dx:.4f} L(D) = {loss:.4f}\n")

    def log_train_df(self, ctx):
        """Logs train_d on fake stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.t_idx
        batch_cnt = ctx.data.t_batch_cnt
        dgz = ctx.outs.dgz
        loss = ctx.outs.ld
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.f-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.f-{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(D) = {loss:.4f}\n")

    def log_train_g(self, ctx):
        """Logs train_g stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.t_idx
        batch_cnt = ctx.data.t_batch_cnt
        dgz = ctx.outs.dgz
        loss = ctx.outs.lg
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(G) = {loss:.4f}\n")

    def log_train_both(self, ctx):
        """Logs train_dg_stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.t_idx
        batch_cnt = ctx.data.t_batch_cnt
        dx = ctx.outs.dx
        dgz = ctx.outs.dgz
        dgz2 = ctx.outs.dgz2
        ld = ctx.outs.ld
        lg = ctx.outs.lg
        if idx == 0 or (idx + 1) % 50 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.{batch_cnt}:\n")
            self.logstr(f"  D: D(X) = {dx:.4f} D(G(Z)) = {dgz:.4f} ")
            self.logstr(f"L(D) = {ld:.4f}\n")
            self.logstr(f"  G: D(G(Z)) = {dgz2:.4f} L(G) = {lg:.4f}\n")

    def log_validate_dr(self, ctx):
        """Logs validate_d on real stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.v_idx
        batch_cnt = ctx.data.v_batch_cnt
        dx = ctx.outs.dx
        loss = ctx.outs.ld
        if idx == 0 or (idx + 1) % 20 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.r-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.r-{batch_cnt}: ")
            self.logstr(f"D(X) = {dx:.4f} L(D) = {loss:.4f}\n")

    def log_validate_df(self, ctx):
        """Logs validate_d on fake stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.v_idx
        batch_cnt = ctx.data.v_batch_cnt
        dgz = ctx.outs.dgz
        loss = ctx.outs.ld
        if idx == 0 or (idx + 1) % 20 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.f-{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.f-{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(D) = {loss:.4f}\n")

    def log_validate_g(self, ctx):
        """Logs validate_g stats.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        iter_cnt = ctx.loops.iter_cnt
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        idx = ctx.loops.v_idx
        batch_cnt = ctx.data.v_batch_cnt
        dgz = ctx.outs.dgz
        loss = ctx.outs.lg
        if idx == 0 or (idx + 1) % 20 == 0 or idx == batch_cnt - 1:
            self.logstr(f"Batch ")
            self.logstr(f"{iter_num + 1}.{epoch_num + 1}.{idx + 1} / ")
            self.logstr(f"{iter_cnt}.{epoch_cnt}.{batch_cnt}: ")
            self.logstr(f"D(G(Z)) = {dgz:.4f} L(G) = {loss:.4f}\n")

    def log_td_loss(self, ctx):
        """Logs the latest discriminator training loss.

        Args:
            ctx: the training context
        """
        self.logln(f"D training loss: {ctx.losses.td[-1]:.4f}")

    def log_tg_loss(self, ctx):
        """Logs the latest generator training loss.

        Args:
            ctx: the training context
        """
        self.logln(f"G training loss: {ctx.losses.tg[-1]:.4f}")

    def log_vd_loss(self, ctx):
        """Logs the latest discriminator validation loss.

        Args:
            ctx: the training context
        """
        self.logln(f"D validation loss: {ctx.losses.vd[-1]:.4f}")

    def log_vg_loss(self, ctx):
        """Logs the latest generator validation loss.

        Args:
            ctx: the training context
        """
        self.logln(f"G validation loss: {ctx.losses.vg[-1]:.4f}")

    def save_training_images(self, ctx):
        """Saves the first 64 training images.

        Args:
            ctx: the training context
        """
        batch0 = next(iter(ctx.data.tdl))
        batch0 = batch0[0].to(ctx.hw.dev)[:64]
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

    def save_validation_images(self, ctx):
        """Saves the first 64 validation images.

        Args:
            ctx: the training context
        """
        batch0 = next(iter(ctx.data.vdl))
        batch0 = batch0[0].to(ctx.hw.dev)[:64]
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

    def save_generated_images(self, ctx):
        """Saves the generated images grid.

        Args:
            ctx: the training context
        """
        iter_num = ctx.loops.iter_num
        epoch_num = ctx.loops.epoch_num
        batch = ctx.mods.g.test(None, noises=ctx.noises.b64)
        grid = visutils.make_grid(batch, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        location = utils.find_in_path(
            f"iter-{iter_num + 1}_epoch-{epoch_num + 1}.jpg", self.gimg_path)
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title(
            f"Iter {iter_num + 1} Epoch {epoch_num + 1} Generated Images")
        pyplot.imshow(grid)
        pyplot.savefig(location, dpi=120)
        pyplot.close()
        self.logln("Saved generated images")

    def save_d_losses(self, ctx):
        """Saves the discriminator training/validation losses plot.

        Args:
            ctx: the training context
        """
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        iter_num = ctx.loops.iter_num
        epochs = list(range(1, iter_num * epoch_cnt + epoch_num + 2))
        iter_lns = [epoch_cnt * x + 1 for x in range(iter_num + 1)]
        location = utils.find_in_path("discriminator-losses.jpg", self.path)
        pyplot.figure(figsize=(10, 5))
        pyplot.title("Discriminator Training and Validation Losses")
        pyplot.plot(epochs, ctx.losses.td, label="Training", color="b")
        pyplot.plot(epochs, ctx.losses.vd, label="Validation", color="r")
        for ln in iter_lns:
            pyplot.axvline(ln, alpha=0.6, color="gray")
        pyplot.xlabel("Epoch No.")
        pyplot.ylabel("Loss")
        pyplot.legend()
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved D losses")

    def save_g_losses(self, ctx):
        """Saves the generator training/validation losses plot.

        Args:
            ctx: the training context
        """
        epoch_num = ctx.loops.epoch_num
        epoch_cnt = ctx.loops.epoch_cnt
        iter_num = ctx.loops.iter_num
        epochs = list(range(1, iter_num * epoch_cnt + epoch_num + 2))
        iter_lns = [epoch_cnt * x + 1 for x in range(iter_num + 1)]
        location = utils.find_in_path("generator-losses.jpg", self.path)
        pyplot.figure(figsize=(10, 5))
        pyplot.title("Generator Training and Validation Losses")
        pyplot.plot(epochs, ctx.losses.tg, label="Training", color="b")
        pyplot.plot(epochs, ctx.losses.vg, label="Validation", color="r")
        for ln in iter_lns:
            pyplot.axvline(ln, alpha=0.6, color="gray")
        pyplot.xlabel("Epoch No.")
        pyplot.ylabel("Loss")
        pyplot.legend()
        pyplot.savefig(location, dpi=160)
        pyplot.close()
        self.logln("Saved G losses")

    def save_tvg(self, ctx):
        """Saves the TVG (training-validation-generated) figure.

        Args:
            ctx: the training context
        """
        t_batch0 = next(iter(ctx.data.tdl))
        t_batch0 = t_batch0[0].to(ctx.hw.dev)[:64]
        v_batch0 = next(iter(ctx.data.vdl))
        v_batch0 = v_batch0[0].to(ctx.hw.dev)[:64]
        g_batch0 = ctx.mods.g.test(None, noises=ctx.noises.b64)
        t_grid = visutils.make_grid(t_batch0, padding=2, normalize=True).cpu()
        v_grid = visutils.make_grid(v_batch0, padding=2, normalize=True).cpu()
        g_grid = visutils.make_grid(g_batch0, padding=2, normalize=True).cpu()
        t_grid = numpy.transpose(t_grid, (1, 2, 0))
        v_grid = numpy.transpose(v_grid, (1, 2, 0))
        g_grid = numpy.transpose(g_grid, (1, 2, 0))
        location = utils.\
            find_in_path("training-validation-generated.jpg", self.path)
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
