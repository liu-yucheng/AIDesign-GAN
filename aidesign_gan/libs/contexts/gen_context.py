"""Generation context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from torch import nn

from aidesign_gan.libs import modelers
from aidesign_gan.libs import utils
from aidesign_gan.libs.contexts import context

_AttrDict = utils.AttrDict
_BCELoss = nn.BCELoss
_Context = context.Context
_GModeler = modelers.GModeler


class GenContext(_Context):
    """Generation context."""

    class Images(_AttrDict):
        """Images info."""

        count = None
        """Image count."""
        per_batch = None
        """Image count of each batch."""
        to_save = None
        """Images to save."""

    class Grids(_AttrDict):
        """Grid mode info."""

        enabled = None
        """Whether grid mode is enabled."""
        size_each = None
        """Size of each grid."""
        padding = None
        """Grid padding width."""

    class BatchProg(_AttrDict):
        """Generation batch progress info."""

        count = None
        """Count."""
        index = None
        """Current index."""

    def __init__(self):
        """Inits self."""
        super().__init__()

        self.g = None
        """Generator modeler instance."""
        self.images = GenContext.Images()
        """Images info attr dict."""
        self.grids = GenContext.Grids()
        """Grid mode info attr dict."""
        self.noise_batches = None
        """Noise batch list."""
        self.batch_prog = GenContext.BatchProg()
        """Generation batch progress info."""

    def setup_all(self, coords_config, modelers_config):
        """Sets up the entire context.

        Sets up self.g, self.images, self.grids, and self.noises.

        Args:
            coords_config: the coords config
            modelers_config: the modelers config

        Raises:
            ValueError: if self.hw.device is None
        """
        if self.hw.device is None:
            raise ValueError("self.hw.device cannot be None")

        # Setup self.g
        model_path = modelers_config.model_path
        config = modelers_config["generator"]
        loss_func = _BCELoss()
        self.g = _GModeler(model_path, config, self.hw.device, self.hw.gpu_count, loss_func, train=False)
        self.g.load()

        # Setup self.images
        config = coords_config["generation"]
        self.images.count = config["image_count"]
        self.images.per_batch = config["images_per_batch"]
        self.images.to_save = []

        # Setup self.grids
        config = config["grid_mode"]
        self.grids.enabled = config["enabled"]
        self.grids.size_each = config["images_per_grid"]
        self.grids.padding = config["padding"]

        # Setup self.noise_batches
        noises_left = self.images.count
        noise_batches = []

        while noises_left > 0:
            noise_count = min(noises_left, self.images.per_batch)
            noise_batch = self.g.generate_noises(noise_count)
            noise_batches.append(noise_batch)
            noises_left -= noise_count

        self.noise_batches = noise_batches

        # Setup self.batch_prog
        self.batch_prog.count = len(self.noise_batches)
        self.batch_prog.index = 0
