"""Algorithm."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng


class Algo:
    """Algorithm super class."""

    def __init__(self):
        """Inits self."""
        self.context = None
        """Context."""
        self.results = None
        """Results."""

    def bind_context_and_results(self, context, results):
        """Binds the context and results.

        Args:
            context: the context to bind
            results: the results to bind
        """
        self.context = context
        self.results = results

    def check_context_and_results(self):
        """Checks if the context and results are binded.

        Raises:
            ValueError: if the context or results are None
        """
        if self.context is None:
            raise ValueError("self.context cannot be None")
        if self.results is None:
            raise ValueError("self.results cannot be None")

    def start(self):
        """Starts the algorithm."""
        pass
