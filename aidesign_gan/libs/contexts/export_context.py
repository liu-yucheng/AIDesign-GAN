"""Exportation context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import typing

from aidesign_gan.libs import configs
from aidesign_gan.libs import modelers
from aidesign_gan.libs import utils
from aidesign_gan.libs.contexts import context

_Context = context.Context
_CoordsConfig = configs.CoordsConfig
_DiscConfig = configs.DiscConfig
_DiscModeler = modelers.DiscModeler
_DotDict = utils.DotDict
_GenConfig = configs.GenConfig
_GenModeler = modelers.GenModeler
_Union = typing.Union


class ExportContext(_Context):
    """Exportation context."""

    class Mods(_DotDict):
        """Modelers."""

        d: _Union[_DiscModeler, None] = None
        """Discriminator."""
        g: _Union[_GenModeler, None] = None
        """Generator."""

    class GenImages(_DotDict):
        """Generated images."""

        count = None
        """Count."""
        per_batch = None
        """Count of each batch."""
        to_save = None
        """The generated images to save."""

    class GenPreviews(_DotDict):
        """Generator preview grids."""

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

    class Configs(_DotDict):
        """Configs."""

        d = None
        """Discriminator."""
        g = None
        """Generator."""

    def __init__(self):
        """Inits self."""
        super().__init__()

        self.mods = type(self).Mods()
        """Modelers."""
        self.gen_images = type(self).GenImages()
        """Generated images."""
        self.gen_previews = type(self).GenPreviews()
        """Generator preview grids."""
        self.noises_batches = None
        """Batches of noises."""
        self.batch_prog = type(self).BatchProg()
        """Batch progress."""
        self.configs = type(self).Configs()
        """Configs."""

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

        # Setup self.mods

        disc_mconfig = mconfig["discriminator"]
        self.mods.d = _DiscModeler(model_path, disc_mconfig, self.hw.device, self.hw.gpu_count, train=False)
        self.mods.d.load()

        gen_mconfig = mconfig["generator"]
        self.mods.g = _GenModeler(model_path, gen_mconfig, self.hw.device, self.hw.gpu_count, train=False)
        self.mods.g.load()

        # End
        # Setup self.gen_images and self.gen_previews

        if "exportation" in cconfig:
            export = cconfig["exportation"]
        else:
            dc_config = _CoordsConfig.load_default()
            export = dc_config["exportation"]
        # end if

        gen_previews = export["preview_grids"]

        self.gen_images.count = gen_previews["images_per_grid"]
        self.gen_images.per_batch = export["images_per_batch"]
        self.gen_images.to_save = []

        self.gen_previews.image_count = gen_previews["images_per_grid"]
        self.gen_previews.padding = gen_previews["padding"]

        # End
        # Setup self.noise_batches

        noises_count_remain = self.gen_images.count
        noises_batches = []

        while noises_count_remain > 0:
            noises_count = min(noises_count_remain, self.gen_images.per_batch)
            noises_batch = self.mods.g.gen_noises(noises_count)
            noises_batches.append(noises_batch)
            noises_count_remain -= noises_count
        # end while

        self.noises_batches = noises_batches

        # Setup self.batch_prog
        self.batch_prog.count = len(self.noises_batches)
        self.batch_prog.index = 0

        # Setup self.configs

        disc_mconfig = mconfig["discriminator"]
        gen_mconfig = mconfig["generator"]

        disc_keys = [
            "image_resolution",
            "image_channel_count",
            "label_resolution",
            "label_channel_count",
            "feature_map_count",
            "struct_name",
            # "state_name"
        ]

        gen_keys = [
            "noise_resolution",
            "noise_channel_count",
            "image_resolution",
            "image_channel_count",
            "feature_map_count",
            "struct_name",
            # "state_name"
            # "preview_name"
        ]

        disc_config = _DiscConfig.load_default()
        gen_config = _GenConfig.load_default()

        for key in disc_keys:
            disc_config[key] = disc_mconfig[key]

        for key in gen_keys:
            gen_config[key] = gen_mconfig[key]

        disc_config = _DiscConfig.verify(disc_config)
        gen_config = _GenConfig.verify(gen_config)

        self.configs.d = disc_config
        self.configs.g = gen_config
