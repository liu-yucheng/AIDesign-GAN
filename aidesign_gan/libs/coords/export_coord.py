"""Exportation coordinator."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import math
import shutil
import torch
from os import path as ospath

from torchvision import utils as tv_utils

from aidesign_gan.libs import configs
from aidesign_gan.libs import contexts
from aidesign_gan.libs import defaults
from aidesign_gan.libs import results
from aidesign_gan.libs.coords import coord

_copy = shutil.copy
_copytree = shutil.copytree
_ceil = math.ceil
_Coord = coord.Coord
_CoordsConfig = configs.CoordsConfig
_ExportContext = contexts.ExportContext
_ExportResults = results.ExportResults
_join = ospath.join
_make_grid = tv_utils.make_grid
_ModelersConfig = configs.ModelersConfig
_torch_cat = torch.cat


class ExportCoord(_Coord):
    """Exportation coordinator."""

    def __init__(self, model_path, export_path, logs, debug_level=0):
        """Inits self with the given args.

        Args:
            model_path: the model path
            log: the log file objects
            debug_level: an optional debug level
        """
        super().__init__(model_path, logs, debug_level)

        self._export_path = export_path

    def _prep_results(self):
        """Prepares self.result."""
        super()._prep_results()

        self._results = _ExportResults(self._export_path, self._logs, self._debug_level)
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

        self._context = _ExportContext()
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

    def _copy_default_export(self):
        """Copies the default export to the export path."""
        r = self._results
        _ = self._context

        _copytree(defaults.default_gan_export_path, self._export_path, dirs_exist_ok=True)
        r.logln("Copied the default GAN export")

    def _copy_format_config(self):
        """Copies the model format config to the export path."""
        r = self._results
        _ = self._context

        srcloc = _join(self._model_path, defaults.format_config_name)
        dstloc = _join(self._export_path, defaults.format_config_name)
        _copy(srcloc, dstloc)
        r.logln("Copied the model format config")

    def _copy_structs_and_states(self):
        """Copies discriminator and generator structs and states."""
        r = self._results
        _ = self._context

        dmod = self._mconfig["discriminator"]
        gmod = self._mconfig["generator"]

        dstruct_name = dmod["struct_name"]
        gstruct_name = gmod["struct_name"]
        dstate_name = dmod["state_name"]
        gstate_name = gmod["state_name"]

        dstruct_src = _join(self._model_path, dstruct_name)
        gstruct_src = _join(self._model_path, gstruct_name)
        dstate_src = _join(self._model_path, dstate_name)
        gstate_src = _join(self._model_path, gstate_name)

        dstruct_dst = _join(self._export_path, dstruct_name)
        gstruct_dst = _join(self._export_path, gstruct_name)
        dstate_dst = _join(self._export_path, dstate_name)
        gstate_dst = _join(self._export_path, gstate_name)

        _copy(dstruct_src, dstruct_dst)
        _copy(gstruct_src, gstruct_dst)
        r.logln("Copied discriminator and generator structs")
        _copy(dstate_src, dstate_dst)
        _copy(gstate_src, gstate_dst)
        r.logln("Copied discriminator and generator states")

    def _normalize_images(self):
        """Normalizes the images."""
        r = self._results
        c = self._context

        images_to_save_len = len(c.images.to_save)

        for index in range(images_to_save_len):
            orig = c.images.to_save[index]
            c.images.to_save[index] = _make_grid(orig, normalize=True, value_range=(-0.75, 0.75))

        r.logln("Normalized images")

    def _convert_images_to_previews(self):
        """Converts the images to preview grids."""
        r = self._results
        c = self._context

        c.images.to_save = _make_grid(
            c.images.to_save, nrow=_ceil(c.previews.image_count ** 0.5), padding=c.previews.padding
        )

        r.logln("Converted images to preview grids")

    def start(self):
        """Starts the exportation process."""
        if not self._prepared:
            self.prep()

        r = self._results
        c = self._context

        info = str(
            "Started exportation\n"
            "-"
        )

        r.logln(info)

        self._copy_default_export()
        self._copy_format_config()
        self._copy_structs_and_states()

        # print(c.configs.d)  # Debug
        # print(c.configs.g)  # Debug
        r.save_configs()

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
        self._convert_images_to_previews()
        r.save_previews()

        info = str(
            "-\n"
            "Completed exportation"
        )

        r.logln(info)
        r.flushlogs()
