"""Exportation context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from aidesign_gan.libs import configs
from aidesign_gan.libs import modelers
from aidesign_gan.libs import utils
from aidesign_gan.libs.contexts import context

_Context = context.Context
_CoordsConfig = configs.CoordsConfig
_DiscConfig = configs.DiscConfig
_DotDict = utils.DotDict
_GenConfig = configs.GenConfig
_GenModeler = modelers.GenModeler


class ExportContext(_Context):
    """Exportation context."""

    class Images(_DotDict):
        """Images."""

        count = None
        """Image count."""
        per_batch = None
        """Image count of each batch."""
        to_save = None
        """Images to save."""

    class Previews(_DotDict):
        """Preview grids."""

        image_count = None
        """Image count."""
        padding = None
        """Grid padding width."""

    class BatchProg(_DotDict):
        """Batch progress."""

        count = None
        """Count."""
        index = None
        """Current index."""

    class Configs:
        """Export configs."""

        def __init__(self):
            """Inits self with the given args."""
            self.d = None
            """Discriminator config."""
            self.g = None
            """Generator config."""

    def __init__(self):
        """Inits self."""
        super().__init__()

        self.g = None
        """Generator modeler instance."""
        self.images = type(self).Images()
        """Images."""
        self.previews = type(self).Previews()
        """Preview grids."""
        self.noises_batches = None
        """Batches of noises."""
        self.batch_prog = type(self).BatchProg()
        """Batch progress."""
        self.configs = type(self).Configs()
        """Export configs."""

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

        self._setup_rand("exportation", model_path, cconfig, mconfig)

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

        self._setup_hw("exportation", model_path, cconfig, mconfig)

    def setup_the_rest(self, model_path=None, cconfig=None, mconfig=None):
        """Sets up the rest of the context.

        Sets up self.g, self.images, self.previews, self.noises, and self.configs.

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
        gmod = mconfig["generator"]
        self.g = _GenModeler(model_path, gmod, self.hw.device, self.hw.gpu_count, train=False)
        self.g.load()

        # Setup self.images and self.grids

        if "exportation" in cconfig:
            export = cconfig["exportation"]
        else:
            dc_config = _CoordsConfig.load_default()
            export = dc_config["exportation"]
        # end if

        previews = export["preview_grids"]

        self.images.count = previews["images_per_grid"]
        self.images.per_batch = export["images_per_batch"]
        self.images.to_save = []

        self.previews.image_count = previews["images_per_grid"]
        self.previews.padding = previews["padding"]

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

        # Setup self.configs

        dmod = mconfig["discriminator"]
        gmod = mconfig["generator"]

        dkeys = [
            "image_resolution",
            "image_channel_count",
            "feature_map_size",
            "struct_name",
            "state_name"
        ]

        gkeys = [
            "noise_resolution",
            "noise_channel_count",
            "image_resolution",
            "image_channel_count",
            "feature_map_size",
            "struct_name",
            "state_name",
            # "preview_name"  # gmod does not have "preview_name"
        ]

        dconfig = _DiscConfig.load_default()
        gconfig = _GenConfig.load_default()

        for key in dkeys:
            dconfig[key] = dmod[key]

        for key in gkeys:
            gconfig[key] = gmod[key]

        dconfig = _DiscConfig.verify(dconfig)
        gconfig = _GenConfig.verify(gconfig)

        self.configs.d = dconfig
        self.configs.g = gconfig
