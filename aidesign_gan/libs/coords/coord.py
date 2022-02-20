"""Coordinator."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng


class Coord:
    """Coordinator base class."""

    def __init__(self, model_path, logs):
        """Inits self with the given args.

        Args:
            model_path: the model path
            logs: the log file objects
        """
        self.model_path = model_path
        """Model path."""
        self.logs = logs
        """Log file objects"""

        self.coords_config = None
        """Coordinators configuration."""
        self.modelers_config = None
        """Modelers configuration."""
        self.results = None
        """Results."""
        self.context = None
        """Context."""
        self.results_ready = False
        """Whether self.results is ready."""
        self.context_ready = False
        """Whether self.context is ready."""

    def setup_results(self):
        """Supposed to set up self.result.

        Does nothing.
        """
        pass

    def setup_context(self):
        """Supposed to set up self.context.

        Does nothing."""
        pass
