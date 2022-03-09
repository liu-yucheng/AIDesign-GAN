"""Training coordinator."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng
from os import path as ospath

from aidesign_gan.libs import algos
from aidesign_gan.libs import configs
from aidesign_gan.libs import contexts
from aidesign_gan.libs import defaults
from aidesign_gan.libs import results
from aidesign_gan.libs.coords import coord

_AltSGDAlgo = algos.AltSGDAlgo
_Coord = coord.Coord
_CoordsConfig = configs.CoordsConfig
_FairPredAltSGDAlgo = algos.FairPredAltSGDAlgo
_join = ospath.join
_ModelersConfig = configs.ModelersConfig
_PredAltSGDAlgo = algos.PredAltSGDAlgo
_TrainContext = contexts.TrainContext
_TrainResults = results.TrainResults


class TrainCoord(_Coord):
    """Training coordinator."""

    def __init__(self, dataset_path, model_path, logs, debug_level=0):
        """Inits self with the given args.

        Args:
            dataset_path: the dataset path
            model_path: the model path
            logs: the log file objects
            debug_level: an optional debug level
        """
        super().__init__(model_path, logs, debug_level)

        self._dataset_path = dataset_path
        """Data path."""
        self._algo = None
        """Algorithm."""
        self._algo_ready = False
        """Whether self.algo is ready."""

    def _prep_results(self):
        """Prepares self.result."""
        super()._prep_results()

        path = _join(self._model_path, "Training-Results")
        self._results = _TrainResults(path, self._logs, self._debug_level)
        self._results.ensure_folders()
        self._results_ready = True
        self._results.logln("Coordinator prepared results")

    def _prep_context(self):
        """Sets up self.context."""
        super()._prep_context()

        if not self._results_ready:
            self._prep_results()

        self._cconfig = _CoordsConfig.load_from_path(self._model_path)
        self._cconfig_loc = _join(self._model_path, defaults.coords_config_name)
        self._cconfig = _CoordsConfig.verify(self._cconfig)

        self._mconfig = _ModelersConfig.load_from_path(self._model_path)
        self._mconfig_loc = _join(self._model_path, defaults.modelers_config_name)
        self._mconfig = _ModelersConfig.verify(self._mconfig)

        self._results.log_config_locs(self._cconfig_loc, self._mconfig_loc)

        self._context = _TrainContext()
        self._context.dataset_path = self._dataset_path
        self._context.model_path = self._model_path
        self._context.cconfig = self._cconfig
        self._context.mconfig = self._mconfig

        self._results.context = self._context

        self._context.setup_rand()
        self._results.log_rand()

        self._context.setup_hw()
        self._results.log_hw()

        self._context.setup_data()
        self._results.log_data()

        self._context.setup_mods()
        self._results.log_mods()

        self._context.setup_mode()
        self._results.log_mode()

        self._context.setup_labels()
        self._results.log_labels()

        self._context.setup_loops()
        self._context.setup_stats()
        self._context.setup_noises()

        self._context_ready = True
        self._results.logln("Coordinator prepared context")

    def _prep_algo(self):
        """Prepares self.algo.

        Raises:
            ValueError: if the algo's name is unknown
        """
        if not self._results_ready:
            self._prep_results()

        if not self._context_ready:
            self._prep_context()

        algo_name = self._cconfig["training"]["algorithm"]
        if algo_name == "alt_sgd_algo":
            self._algo = _AltSGDAlgo()
        elif algo_name == "pred_alt_sgd_algo":
            self._algo = _PredAltSGDAlgo()
        elif algo_name == "fair_pred_alt_sgd_algo":
            self._algo = _FairPredAltSGDAlgo()
        else:
            raise ValueError(f"Unknown algo: {algo_name}")
        # end if

        self._results.log_algo(algo_name)
        self._algo.context = self._context
        self._algo.results = self._results
        self._algo_ready = True
        self._results.logln("Coordinator prepared algorithm")

    def prep(self):
        """Prepares everything that the start method needs."""
        if not self._results_ready:
            self._prep_results()

        if not self._context_ready:
            self._prep_context()

        if not self._algo_ready:
            self._prep_algo()

        self._prepared = True
        self._results.logln("Coordinator completed preparation")
        self._results.flushlogs()

    def start(self):
        """Starts the training process."""
        if not self._prepared:
            self.prep()

        r = self._results

        info = str(
            "Started training\n"
            "-"
        )

        r.logln(info)
        self._algo.start()

        info = str(
            "-\n"
            "Completed training"
        )

        r.logln(info)
        r.flushlogs()
