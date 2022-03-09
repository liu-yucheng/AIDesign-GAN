"""Generation context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from torch import nn

from aidesign_gan.libs import modelers
from aidesign_gan.libs import utils
from aidesign_gan.libs.contexts import context

_BCELoss = nn.BCELoss
_Context = context.Context
_DotDict = utils.DotDict
_GenModeler = modelers.GenModeler


class GenContext(_Context):
    """Generation context."""

    class Images(_DotDict):
        """Images info."""

        count = None
        """Image count."""
        per_batch = None
        """Image count of each batch."""
        to_save = None
        """Images to save."""

    class Grids(_DotDict):
        """Grid mode info."""

        enabled = None
        """Whether grid mode is enabled."""
        size_each = None
        """Size of each grid."""
        padding = None
        """Grid padding width."""

    class BatchProg(_DotDict):
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
        self.noises_batches = None
        """Noise batch list."""
        self.batch_prog = GenContext.BatchProg()
        """Generation batch progress info."""

    def setup_rand(self, model_path=None, cconfig=None, mconfig=None):
        """Sets the random seeds with the given args.

        Set up seeds for numpy, random and torch. Set up self.rand and its attributes.

        Args:
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        model_path = self.find_model_path(model_path)
        cconfig = self.find_cconfig(cconfig)
        mconfig = self.find_mconfig(mconfig)

        self._setup_rand("generation", model_path, cconfig, mconfig)

    def setup_hw(self, model_path=None, cconfig=None, mconfig=None):
        """Sets up the torch hardware with the given args.

        Set up self.hw and its attributes.

        Args:
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        model_path = self.find_model_path(model_path)
        cconfig = self.find_cconfig(cconfig)
        mconfig = self.find_mconfig(mconfig)

        self._setup_hw("generation", model_path, cconfig, mconfig)

    def setup_the_rest(self, model_path=None, cconfig=None, mconfig=None):
        """Sets up the rest of the context.

        Sets up self.g, self.images, self.grids, and self.noises.

        Args:
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config

        Raises:
            ValueError: if self.hw.device is None
        """
        if self.hw.device is None:
            raise ValueError("self.hw.device cannot be None")

        model_path = self.find_model_path(model_path)
        cconfig = self.find_cconfig(cconfig)
        mconfig = self.find_mconfig(mconfig)

        # Setup self.g
        config = mconfig["generator"]
        loss_func = _BCELoss()
        self.g = _GenModeler(model_path, config, self.hw.device, self.hw.gpu_count, loss_func, train=False)
        self.g.load()

        # Setup self.images
        config = cconfig["generation"]
        self.images.count = config["image_count"]
        self.images.per_batch = config["images_per_batch"]
        self.images.to_save = []

        # Setup self.grids
        config = cconfig["generation"]["grid_mode"]
        self.grids.enabled = config["enabled"]
        self.grids.size_each = config["images_per_grid"]
        self.grids.padding = config["padding"]

        # Setup self.noise_batches
        noises_count_remain = self.images.count
        noises_batches = []

        while noises_count_remain > 0:
            noises_count = min(noises_count_remain, self.images.per_batch)
            noises_batch = self.g.gen_noises(noises_count)
            noises_batches.append(noises_batch)
            noises_count_remain -= noises_count
        # end while

        self.noises_batches = noises_batches

        # Setup self.batch_prog
        self.batch_prog.count = len(self.noises_batches)
        self.batch_prog.index = 0
