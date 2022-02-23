"""Generation Coordinator."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import math
import torch
from os import path as ospath

from torchvision import utils as tv_utils

from aidesign_gan.libs import configs
from aidesign_gan.libs import contexts
from aidesign_gan.libs import results
from aidesign_gan.libs import utils
from aidesign_gan.libs.coords import coord

_ceil = math.ceil
_Coord = coord.Coord
_CoordsConfig = configs.CoordsConfig
_GenContext = contexts.GenContext
_GenResults = results.GenerationResults
_join = ospath.join
_make_grid = tv_utils.make_grid
_ModelersConfig = configs.ModelersConfig
_torch_cat = torch.cat


class GenCoord(_Coord):
    """Generation coordinator."""

    def __init__(self, model_path, log):
        """Inits self with the given args.

        Args:
            model_path: the model path
            log: the log file object
        """
        super().__init__(model_path, log)

    def setup_results(self):
        """Sets up self.result."""
        path = _join(self.model_path, "Generation-Results")
        self.results = _GenResults(path, self.logs)
        self.results.init_folders()
        self.results_ready = True

        self.results.logln("Completed results setup")

    def setup_context(self):
        """Sets up self.context."""
        if not self.results_ready:
            self.setup_results()

        self.coords_config = _CoordsConfig(self.model_path)
        self.coords_config.load()
        self.modelers_config = _ModelersConfig(self.model_path)
        self.modelers_config.load()

        self.results.log_configs(self.coords_config, self.modelers_config)
        self.context = _GenContext()
        self.results.bind_context(self.context)
        config = self.coords_config["generation"]

        self.context.setup_rand(config)
        self.results.log_rand()

        self.context.setup_hw(config)
        self.results.log_hw()

        self.context.setup_all(self.coords_config, self.modelers_config)
        self.results.log_g()

        self.context_ready = True
        self.results.logln("Completed context setup")

    def normalize_images(self):
        """Normalizes the images."""
        r = self.results
        c = self.context

        for index in range(len(c.images.to_save)):
            c.images.to_save[index] = _make_grid(c.images.to_save[index], normalize=True, value_range=(-0.75, 0.75))

        r.logln("Normalized images")

    def convert_images_to_grids(self):
        """Converts the images to grids."""
        r = self.results
        c = self.context

        orig_list = c.images.to_save
        c.images.to_save = []
        start_index = 0

        while start_index < c.images.count:
            end_index = start_index + c.grids.size_each

            grid = _make_grid(
                orig_list[start_index: end_index], nrow=_ceil(c.grids.size_each ** 0.5), padding=c.grids.padding
            )

            c.images.to_save.append(grid)
            start_index = end_index
        # end while

        r.logln("Converted images to grids")

    def start_generation(self):
        """Starts the generation."""
        if not self.results_ready:
            self.setup_results()

        if not self.context_ready:
            self.setup_context()

        r = self.results
        c = self.context

        r.logln("Started generation")
        c.images.to_save = []
        c.batch_prog.index = 0

        while c.batch_prog.index < c.batch_prog.count:
            noise_batch = c.noise_batches[c.batch_prog.index]
            image_batch = c.g.test(noise_batch)
            c.images.to_save.append(image_batch)
            r.log_batch()
            c.batch_prog.index += 1
        # end while

        c.images.to_save = _torch_cat(c.images.to_save)
        self.normalize_images()

        if c.grids.enabled:
            self.convert_images_to_grids()

        r.save_generated_images()
        r.logln("Completed generation")
