"""Exportation results."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from os import path as ospath
from torchvision import utils as vutils

from aidesign_gan.libs import configs
from aidesign_gan.libs import contexts
from aidesign_gan.libs import defaults
from aidesign_gan.libs.results import results

_DiscConfig = configs.DiscConfig
_ExportContext = contexts.ExportContext
_GenConfig = configs.GenConfig
_join = ospath.join
_save_image = vutils.save_image
_Results = results.Results


class ExportResults(_Results):
    """Exportation results."""

    def __init__(self, path, logs, debug_level=0):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            logs: the log file objects
            debug_level: an optional debug level
        """
        super().__init__(path, logs, debug_level)

    def log_g(self, context=None, debug_level=0):
        """Logs the generator modelers info.

        Args:
            context: optional context
            debug_level: an optional debug level
        """
        c: _ExportContext = self.find_context(context)

        info = str(
            "- Generator modeler\n"
            "Model:  Size: {}  Training size: {}  Struct: See below\n"
            "- Generator model structure\n"
            "{}\n"
            "-"
        ).format(
            c.g.size, c.g.training_size,
            str(c.g.model)
        )

        self.logln(info, debug_level)

    def log_batch(self, context=None, debug_level=0):
        """Logs the batch info.

        Args:
            context: optional context
            debug_level: an optional debug level
        """
        c: _ExportContext = self.find_context(context)

        needs_log = c.batch_prog.index == 0 or \
            (c.batch_prog.index + 1) % 15 == 0 or \
            c.batch_prog.index == c.batch_prog.count - 1

        if not needs_log:
            return

        self.logln(f"Generated preview batch {c.batch_prog.index + 1} / {c.batch_prog.count}", debug_level)

    def save_configs(self, context=None, debug_level=0):
        """Saves the configs.

        Args:
            context: optional context
            debug_level: an optional debug level
        """
        c: _ExportContext = self.find_context(context)

        dname = defaults.disc_config_name
        gname = defaults.gen_config_name

        dloc = _join(self._path, dname)
        gloc = _join(self._path, gname)

        needs_log = self.find_needs_log(debug_level)

        if needs_log:
            _DiscConfig.save(c.configs.d, dloc)
            _GenConfig.save(c.configs.g, gloc)

        self.logln("Saved discriminator and generator configs", debug_level)

    def save_previews(self, context=None, debug_level=0):
        """Saves the preview grids.

        Args:
            context: optional context
            debug_level: an optional debug level
        """
        c: _ExportContext = self.find_context(context)

        name = "generator_preview.jpg"
        loc = _join(self._path, name)
        needs_log = self.find_needs_log(debug_level)

        if needs_log:
            _save_image(c.images.to_save, loc, "JPEG")

        image_count = c.previews.image_count
        info = f"Saved the generator preview grid, which contains {image_count} images"
        self.logln(info, debug_level)
