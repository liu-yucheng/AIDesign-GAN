"""Generation results."""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from aidesign_gan.libs import contexts
from aidesign_gan.libs.results import results

_GenContext = contexts.GenContext
_Results = results.Results


class GenResults(_Results):
    """Generation results."""

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
        c: _GenContext = self.find_context(context)

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
        c: _GenContext = self.find_context(context)

        needs_log = c.batch_prog.index == 0 or \
            (c.batch_prog.index + 1) % 15 == 0 or \
            c.batch_prog.index == c.batch_prog.count - 1

        if not needs_log:
            return

        self.logln(f"Generated image batch {c.batch_prog.index + 1} / {c.batch_prog.count}", debug_level)
