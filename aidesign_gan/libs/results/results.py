"""Results."""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import os
import typing

from aidesign_gan.libs import contexts
from aidesign_gan.libs import utils

_Context = contexts.Context
_flushlogs = utils.flushlogs
_logln = utils.logln
_logstr = utils.logstr
_makedirs = os.makedirs
_Union = typing.Union


class Results:
    """Results base class."""

    def __init__(self, path, logs, debug_level=0):
        """Inits self with the given args.

        Args:
            path: a path as the results root path
            logs: a list of the log file objects to use
            debug_level: an optional debug level
        """
        path = str(path)
        logs = list(logs)
        debug_level = int(debug_level)

        self.logs = logs
        """Log file objects."""
        self.debug_level = debug_level
        """Debug level."""
        self.context: _Union[_Context, None] = None
        """Context.

        Used to replace the context parameters of the context-involved results functions.
        """
        self._path = path
        """Results root path."""

    def find_context(self, context_arg):
        """Finds the context to use.

        Ensures that there is at least 1 context to use.
        NOTE: Context usage priorities: context_arg > self.context

        Args:
            context_arg: the context argument

        Returns:
            context: the context to use

        Raises:
            ValueError: if both context_arg and self.context are None
        """
        context_arg: _Union[_Context, None] = context_arg

        if context_arg is None and self.context is None:
            err_info = str(
                f"At least 1 of the following items must be non-None:\n"
                f"  context_arg: {context_arg}\n"
                f"  self.context: {self.context}"
            )

            raise ValueError(err_info)
        # end if

        if context_arg is not None:
            context = context_arg
        elif self.context is not None:
            context = self.context
        # end if

        return context

    def find_needs_log(self, debug_level):
        """Finds whether the given debug_level needs to trigger a log.

        Args:
            debug_level: a debug level

        Returns:
            result: whether the given debug_level needs to trigger a log
        """
        debug_level = int(debug_level)

        if debug_level <= self.debug_level:
            result = True
        else:
            result = False

        return result

    def logstr(self, string="", debug_level=0):
        """Logs a string.

        Args:
            string: the string to log
        """
        debug_level = int(debug_level)

        needs_log = self.find_needs_log(debug_level)

        if needs_log:
            _logstr(self.logs, string)

    def logln(self, line="", debug_level=0):
        """Logs a line.

        Args:
            line: the line to log
        """
        debug_level = int(debug_level)

        needs_log = self.find_needs_log(debug_level)

        if needs_log:
            _logln(self.logs, line)

    def flushlogs(self):
        """Flushes every log in self.logs."""
        _flushlogs(self.logs)

    def ensure_folders(self, debug_level=0):
        """Ensures the result folders.

        Args:
            debug_level: an optional debug level
        """
        _makedirs(self._path, exist_ok=True)
        self.logln(f"Results ensured folder: {self._path}", debug_level)

    def log_config_locs(self, cconfig_loc, mconfig_loc, debug_level=0):
        """Logs the coords and modelers config info.

        Args:
            cconfig_loc: a coords config location
            mconfig_loc: a modelers config location
            debug_level: an optional debug level
        """
        cconfig_loc = str(cconfig_loc)
        mconfig_loc = str(mconfig_loc)

        info = str(
            f"Used the coords config at: {cconfig_loc}\n"
            f"Used the modelers config at: {mconfig_loc}"
        )

        self.logln(info, debug_level)

    def log_rand(self, context=None, debug_level=0):
        """Logs the random info.

        Args:
            context: optional context
            debug_level: an optional debug level
        """
        c: _Context = self.find_context(context)

        info = f"Random seed ({c.rand.mode}): {c.rand.seed}"
        self.logln(info, debug_level)

    def log_hw(self, context=None, debug_level=0):
        """Logs the torch hardware info.

        Args:
            context: optional context
            debug_level: an optional debug level
        """

        c: _Context = self.find_context(context)

        info = f"PyTorch device: \"{c.hw.device}\"  GPU count: {c.hw.gpu_count}"
        self.logln(info, debug_level)
