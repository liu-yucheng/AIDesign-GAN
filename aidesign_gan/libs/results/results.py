"""Results."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import os

from aidesign_gan.libs import contexts
from aidesign_gan.libs import utils

_Context = contexts.Context
_flushlogs = utils.flushlogs
_logln = utils.logln
_logstr = utils.logstr
_makedirs = os.makedirs


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
        self.context = None
        """Context.

        Used to replace the context parameters of the context-involved results functions.
        """

    def ensure_folders(self):
        """Ensures the result folders."""
        _makedirs(self.path, exist_ok=True)
        self.logln(f"Ensured folder: {self.path}")

    def bind_context(self, context):
        """Binds a context to self.

        Args:
            context: the context to bind
        """
        self.context = context

    def check_context(self):
        """Check if self.context is not None.

        Raises:
            ValueError: if self.context is None
        """
        if self.context is None:
            raise ValueError("self.context cannot be None")

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

    def log_configs(self, coords_config, modelers_config):
        """Logs the coords and modelers config info.

        Args:
            coords_config: the coords config
            modelers_config: the modelers config
        """
        self.logln(f"Coords config: {coords_config.location}")
        self.logln(f"Modelers config: {modelers_config.location}")

    def log_rand(self):
        """Logs the random info."""
        self.check_context()
        c: _Context = self.context

        self.logln(f"Random seed ({c.rand.mode}): {c.rand.seed}")

    def log_hw(self):
        """Logs the torch hardware info."""
        self.check_context()
        c: _Context = self.context

        self.logln(f"Torch device: {c.hw.device}; GPU count: {c.hw.gpu_count}")
