"""Results."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
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

    def __init__(self, path, logs):
        """Inits self with the given args.

        Args:
            path: a path as the results root path
            logs: a list of the log file objects to use
        """
        path = str(path)
        logs = list(logs)

        self.path = path
        """Results root path."""
        self.logs = logs
        """Log file objects."""
        self.context: _Union[_Context, None] = None
        """Context.

        Used to replace the context parameters of the context-involved results functions.
        """

    def ensure_folders(self):
        """Ensures the result folders."""
        _makedirs(self.path, exist_ok=True)
        self.logln(f"Ensured folder: {self.path}")

    def find_context(self, context_arg):
        """Finds the context to use.

        Ensures that there is at least 1 context to use.
        NOTE: Context usage priorities: context_arg > self.context

        Returns:
            context: the context to use

        Raises:
            ValueError: if both context_arg and self.context are None
        """
        context_arg: _Union[_Context, None] = context_arg

        if context_arg is None and self.context is None:
            err_info = str(
                f"At least 1 of the following must be non-None:\n"
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

    def logstr(self, string=""):
        """Logs a string.

        Args:
            string: the string to log
        """
        _logstr(self.logs, string)

    def logln(self, line=""):
        """Logs a line.

        Args:
            line: the line to log
        """
        _logln(self.logs, line)

    def flushlogs(self):
        """Flushes every log in self.logs."""
        _flushlogs(self.logs)

    def log_config_locs(self, ccfg_loc, mcfg_loc):
        """Logs the coords and modelers config info.

        Args:
            ccfg_loc: coords config location
            mcfg_loc: modelers config location
        """
        ccfg_loc = str(ccfg_loc)
        mcfg_loc = str(mcfg_loc)

        info = str(
            f"Used the coords config at: {ccfg_loc}\n"
            f"Used the modelers config at: {mcfg_loc}"
        )

        self.logln(info)

    def log_rand(self, context=None):
        """Logs the random info.

        Args:
            context: optional context
        """
        c: _Context = self.find_context(context)

        info = f"Random seed ({c.rand.mode}): {c.rand.seed}"
        self.logln(info)

    def log_hw(self, context=None):
        """Logs the torch hardware info.

        Args:
            context: optional context
        """

        c: _Context = self.find_context(context)

        info = f"PyTorch device: \"{c.hw.device}\"  GPU count: {c.hw.gpu_count}"
        self.logln(info)
