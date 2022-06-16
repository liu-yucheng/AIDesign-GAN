"""Exportation results."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from aidesign_gan.libs import contexts
from aidesign_gan.libs.results import results

_ExportContext = contexts.ExportContext
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

    def log_mods(self, context=None, debug_level=0):
        """Logs the modelers info.

        Args:
            context: optional context
            debug_level: an optional debug level
        """
        c: _ExportContext = self.find_context(context)

        info = str(
            "- Discriminator modeler\n"
            "Model:  Size: {}  Training size: {}  Struct: See below\n"
            "- Discriminator model structure\n"
            "{}\n"
            "-\n"
            "- Generator modeler\n"
            "Model:  Size: {}  Training size: {}  Struct: See below\n"
            "- Generator model structure\n"
            "{}\n"
            "-"
        ).format(
            c.mods.d.size, c.mods.d.training_size,
            str(c.mods.d.model),
            c.mods.g.size, c.mods.g.training_size,
            str(c.mods.g.model)
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
