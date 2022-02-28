"""Algorithm."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import typing

from aidesign_gan.libs import contexts
from aidesign_gan.libs import results

_Context = contexts.Context
_Results = results.Results
_Union = typing.Union


class Algo:
    """Algorithm base class."""

    def __init__(self):
        """Inits self."""
        self.context: _Union[_Context, None] = None
        """Context.

        Used to replace the context parameters of the context-involved algo functions.
        """
        self.results: _Union[_Results, None] = None
        """Results.

        Used to replace the result parameters of the results-involved algo functions.
        """

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

    def find_results(self, results_arg):
        """Finds the results to use.

        Ensures that there is at least 1 results object to use.
        NOTE: Results usage priorities: results_arg > self.results

        Args:
            results_arg: the results argument

        Returns:
            results: the results to use

        Raises:
            ValueError: if both results_arg and self.results are None
        """
        results_arg: _Union[_Results, None] = results_arg

        if results_arg is None and self.results is None:
            err_info = str(
                f"At least 1 of the following items must be non-None:\n"
                f"  results_arg: {results_arg}\n"
                f"  self.results: {self.results}"
            )

            raise ValueError(err_info)
        # end if

        if results_arg is not None:
            results = results_arg
        elif self.results is not None:
            results = self.results
        # end if

        return results

    def start(self, context=None, results=None):
        """Supposed to start the algorithm.

        This method is abstract and effectively does nothing.

        Args:
            context: optional context
            results: optional results
        """
        _ = context
        _ = results
