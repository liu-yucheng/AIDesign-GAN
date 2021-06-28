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
        gimg_path: generated images path
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
        """Logs the coords and modelers config information.

        Args:
            c_config: the coords config
            m_config: the modelers config
        """
        self.logln(f"Coords config: {c_config.location}")
        self.logln(f"Modelers config: {m_config.location}")

    def log_rand_seeds(self, ctx):
        """Logs the random seeds information.

        Args:
            ctx: the training context
        """
        self.logln(f"Random seed ({ctx.rand.mode}): {ctx.rand.seed}")

    def log_device(self, ctx):
        """Logs the torch device information.

        Args:
            ctx: the training context
        """
        self.logln(f"Torch device: {ctx.hw.dev}; GPU count: {ctx.hw.gpu_cnt}")

    def log_data_loaders(self, ctx):
        """Logs the data loaders information.

        Args:
            ctx: the training context
        """
        self.logstr(f"Data total size: {ctx.data.total_size}; ")
        self.logstr(f"Data size to use: {ctx.data.size_to_use}\n")
        self.logstr(f"Training set size: {ctx.data.t_size}; ")
        self.logstr(f"Validation set size: {ctx.data.v_size}\n")

    def log_modelers(self, ctx):
        """Logs the modelers information.

        Args:
            ctx: the training context
        """
        self.logln(f"==== D's struct ====")
        self.logln(ctx.mods.d_str)
        self.logln(f"==== G's struct ====")
        self.logln(ctx.mods.g_str)

    def log_mode(self, ctx):
        """Logs the training mode information.

        Args:
            ctx: the training context
        """
        self.logln(f"Training mode: {ctx.mode}")

    def save_training_images(self, ctx):
        """Saves the first 64 training images.

        Args:
            ctx: the training context
        """
        batch0 = next(iter(ctx.data.tdl))
        batch0 = batch0[0].to(ctx.hw.dev)[:64]
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Training Images")
        grid = visutils.make_grid(batch0, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        pyplot.imshow(grid)
        location = utils.find_in_path("training-images.jpg", self.path)
        pyplot.savefig(location)
        pyplot.close()
        self.logln("Saved training images")

    def save_validation_images(self, ctx):
        """Saves the first 64 validation images.

        Args:
            ctx: the training context
        """
        batch0 = next(iter(ctx.data.vdl))
        batch0 = batch0[0].to(ctx.hw.dev)[:64]
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Validation Images")
        grid = visutils.make_grid(batch0, padding=2, normalize=True).cpu()
        grid = numpy.transpose(grid, (1, 2, 0))
        pyplot.imshow(grid)
        location = utils.find_in_path("validation-images.jpg", self.path)
        pyplot.savefig(location)
        pyplot.close()
        self.logln("Saved validation images")
