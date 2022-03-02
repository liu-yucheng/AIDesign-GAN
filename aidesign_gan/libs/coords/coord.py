"""Coordinator."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng


class Coord:
    """Coordinator base class."""

    def __init__(self, model_path, logs, debug_level=0):
        """Inits self with the given args.

        Args:
            model_path: the model path
            logs: the log file objects
            debug_level: an optional debug level
        """
        model_path = str(model_path)
        logs = list(logs)
        debug_level = int(debug_level)

        self._model_path = model_path
        """Model path."""
        self._logs = logs
        """Log file objects"""
        self._debug_level = debug_level
        """Debug level."""
        self._cconfig = None
        """Coords config."""
        self._cconfig_loc = None
        """Coords config location."""
        self._mconfig = None
        """Modelers config."""
        self._mconfig_loc = None
        """Modelers config location."""
        self._results = None
        """Results."""
        self._context = None
        """Context."""
        self._results_ready = False
        """Whether self.results is ready."""
        self._context_ready = False
        """Whether self.context is ready."""
        self._prepared = False
        """Whether self is prepared for the upcoming start method call."""

    def _prep_results(self):
        """Supposed to prepare self.result.

        This method is abstract and does nothing.
        """
        pass

    def _prep_context(self):
        """Supposed to prepare self.context.

        This method is abstract and does nothing.
        """
        pass

    def prep(self):
        """Supposed to prepare everything that the start method needs.

        This method is abstract and does nothing.
        """
        pass

    def start(self):
        """Supposed to start the coordinator's action.

        This method is abstract and does nothing.
        """
        pass
