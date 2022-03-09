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
from aidesign_gan.libs import defaults
from aidesign_gan.libs import results
from aidesign_gan.libs.coords import coord

_ceil = math.ceil
_Coord = coord.Coord
_CoordsConfig = configs.CoordsConfig
_GenContext = contexts.GenContext
_GenResults = results.GenResults
_join = ospath.join
_make_grid = tv_utils.make_grid
_ModelersConfig = configs.ModelersConfig
_torch_cat = torch.cat


class GenCoord(_Coord):
    """Generation coordinator."""

    def __init__(self, model_path, logs, debug_level=0):
        """Inits self with the given args.

        Args:
            model_path: the model path
            log: the log file objects
            debug_level: an optional debug level
        """
        super().__init__(model_path, logs, debug_level)

    def _prep_results(self):
        """Prepares self.result."""
        super()._prep_results()

        path = _join(self._model_path, "Generation-Results")
        self._results = _GenResults(path, self._logs, self._debug_level)
        self._results.ensure_folders()
        self._results_ready = True

        self._results.logln("Coordinator prepared results")

    def _prep_context(self):
        """Prepares self.context."""
        super()._prep_context()

        if not self._results_ready:
            self._prep_results()

        self._cconfig = _CoordsConfig.load_from_path(self._model_path)
        self._cconfig_loc = _join(self._model_path, defaults.coords_config_name)
        self._cconfig = _CoordsConfig.verify(self._cconfig)

        self._mconfig = _ModelersConfig.load_from_path(self._model_path)
        self._mconfig_loc = _join(self._model_path, defaults.modelers_config_name)
        self._mconfig = _ModelersConfig.verify(self._mconfig)

        self._results.log_config_locs(self._cconfig_loc, self._mconfig_loc)

        self._context = _GenContext()
        self._context.model_path = self._model_path
        self._context.cconfig = self._cconfig
        self._context.mconfig = self._mconfig

        self._results.context = self._context

        self._context.setup_rand()
        self._results.log_rand()

        self._context.setup_hw()
        self._results.log_hw()

        self._context.setup_the_rest()
        self._results.log_g()

        self._context_ready = True
        self._results.logln("Coordinator prepared context")

    def prep(self):
        """Prepares everything that the start method needs."""
        if not self._results_ready:
            self._prep_results()

        if not self._context_ready:
            self._prep_context()

        self._prepared = True
        self._results.logln("Coordinator completed preparation")
        self._results.flushlogs()

    def _normalize_images(self):
        """Normalizes the images."""
        r = self._results
        c = self._context

        images_to_save_len = len(c.images.to_save)

        for index in range(images_to_save_len):
            orig = c.images.to_save[index]
            c.images.to_save[index] = _make_grid(orig, normalize=True, value_range=(-0.75, 0.75))

        r.logln("Normalized images")

    def _convert_images_to_grids(self):
        """Converts the images to grids."""
        r = self._results
        c = self._context

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

    def start(self):
        """Starts the generation process."""
        if not self._prepared:
            self.prep()

        r = self._results
        c = self._context

        info = str(
            "Started generation\n"
            "-"
        )

        r.logln(info)
        c.images.to_save = []
        c.batch_prog.index = 0

        while c.batch_prog.index < c.batch_prog.count:
            noises_batch = c.noises_batches[c.batch_prog.index]
            image_batch = c.g.test(noises_batch)
            c.images.to_save.append(image_batch)
            r.log_batch()
            c.batch_prog.index += 1
        # end while

        c.images.to_save = _torch_cat(c.images.to_save)
        self._normalize_images()

        if c.grids.enabled:
            self._convert_images_to_grids()

        r.save_gen_images()

        info = str(
            "-\n"
            "Completed generation"
        )

        r.logln(info)
        r.flushlogs()
