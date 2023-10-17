"""Exportation coordinator."""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
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
_DiscConfig = configs.DiscConfig
_ExportContext = contexts.ExportContext
_ExportResults = results.ExportResults
_GenConfig = configs.GenConfig
_join = ospath.join
_make_grid = tv_utils.make_grid
_ModelersConfig = configs.ModelersConfig
_save_image = tv_utils.save_image
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
        self._results.log_mods()

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

    def _copy_structs(self):
        """Copies discriminator and generator structs and states."""
        r = self._results
        _ = self._context

        disc_mod = self._mconfig["discriminator"]
        gen_mod = self._mconfig["generator"]

        disc_name = disc_mod["struct_name"]
        gen_name = gen_mod["struct_name"]

        disc_src = _join(self._model_path, disc_name)
        gen_src = _join(self._model_path, gen_name)

        disc_dst = _join(self._export_path, disc_name)
        gen_dst = _join(self._export_path, gen_name)

        _copy(disc_src, disc_dst)
        _copy(gen_src, gen_dst)

        r.logln("Copied discriminator and generator structs")

    def _save_configs(self):
        """Saves the discriminator and generator configs."""
        r = self._results
        c = self._context

        disc_loc = _join(self._export_path, defaults.disc_config_name)
        gen_loc = _join(self._export_path, defaults.gen_config_name)

        _DiscConfig.save(c.configs.d, disc_loc)
        _GenConfig.save(c.configs.g, gen_loc)

        r.logln("Saved discriminator and generator configs")

    def _export_models(self):
        """Saves the discriminator and generator configs."""
        r = self._results
        c = self._context

        export_path = _join(self._export_path, defaults.model_saves_name)
        c.mods.d.export_model(export_path, defaults.disc_state_script_name, defaults.disc_state_onnx_name)
        c.mods.g.export_model(export_path, defaults.gen_state_script_name, defaults.gen_state_onnx_name)

        r.logln("Exported discriminator and generator models in TorchScript and ONNX formats")

    def _normalize_images(self):
        """Normalizes the images."""
        r = self._results
        c = self._context

        images_to_save_len = len(c.gen_images.to_save)

        for index in range(images_to_save_len):
            orig = c.gen_images.to_save[index]
            c.gen_images.to_save[index] = _make_grid(orig, normalize=True, value_range=(-0.75, 0.75))

        r.logln("Normalized images")

    def _convert_images_to_previews(self):
        """Converts the images to preview grids."""
        r = self._results
        c = self._context

        c.gen_images.to_save = _make_grid(
            c.gen_images.to_save, nrow=_ceil(c.gen_previews.image_count ** 0.5), padding=c.gen_previews.padding
        )

        r.logln("Converted images to preview grids")

    def _save_gen_previews(self):
        """Saves the preview grids."""
        r = self._results
        c = self._context

        name = "generator_preview.jpg"
        loc = _join(self._export_path, name)

        # The "quality" parameter is no longer supported by PyTorch 2.1.
        # _save_image(c.gen_images.to_save, loc, "JPEG", quality=95)
        _save_image(c.gen_images.to_save, loc, "JPEG")

        image_count = c.gen_previews.image_count
        info = f"Saved the generator preview grid, which contains {image_count} images"
        r.logln(info)

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
        self._copy_structs()
        self._save_configs()
        self._export_models()

        c.gen_images.to_save = []
        c.batch_prog.index = 0

        while c.batch_prog.index < c.batch_prog.count:
            noises_batch = c.noises_batches[c.batch_prog.index]
            image_batch = c.mods.g.test(noises_batch)
            c.gen_images.to_save.append(image_batch)
            r.log_batch()
            c.batch_prog.index += 1
        # end while

        c.gen_images.to_save = _torch_cat(c.gen_images.to_save)
        self._normalize_images()
        self._convert_images_to_previews()
        self._save_gen_previews()

        info = str(
            "-\n"
            "Completed exportation"
        )

        r.logln(info)
        r.flushlogs()
