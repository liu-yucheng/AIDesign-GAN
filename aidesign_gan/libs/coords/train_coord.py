"""Training coordinator."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng
from os import path as ospath

from aidesign_gan.libs import algos
from aidesign_gan.libs import configs
from aidesign_gan.libs import contexts
from aidesign_gan.libs import results
from aidesign_gan.libs import utils
from aidesign_gan.libs.coords import coord

_AltSGDAlgo = algos.AltSGDAlgo
_Coord = coord.Coord
_CoordsConfig = configs.CoordsConfig
_FairPredAltSGDAlgo = algos.FairPredAltSGDAlgo
_join = ospath.join
_ModelersConfig = configs.ModelersConfig
_PredAltSGDAlgo = algos.PredAltSGDAlgo
_TrainContext = contexts.TrainContext
_TrainResults = results.TrainingResults


class TrainCoord(_Coord):
    """Training coordinator."""

    def __init__(self, data_path, model_path, logs):
        """Inits self with the given args.

        Args:
            data_path: the data path
            model_path: the model path
            logs: the log file objects
        """
        super().__init__(model_path, logs)
        self.data_path = data_path
        """Data path."""
        self.algo = None
        """Algorithm."""
        self.algo_ready = False
        """Whether self.algo is ready."""

    def setup_results(self):
        """Sets up self.result."""
        path = _join(self.model_path, "Training-Results")
        self.results = _TrainResults(path, self.logs)
        self.results.ensure_folders()
        self.results_ready = True
        self.results.logln("Completed results setup")

    def setup_context(self):
        """Sets up self.context."""
        if not self.results_ready:
            self.setup_results()

        self.coords_config = _CoordsConfig(self.model_path)
        self.modelers_config = _ModelersConfig(self.model_path)

        self.coords_config.load()
        self.modelers_config.load()

        self.results.log_configs(self.coords_config, self.modelers_config)

        self.context = _TrainContext()
        self.results.bind_context(self.context)

        training_key = "training"
        config = self.coords_config[training_key]
        self.context.setup_rand(config)
        self.results.log_rand()

        config = self.coords_config[training_key]
        self.context.setup_hw(config)
        self.results.log_hw()

        config = self.coords_config[training_key]
        self.context.setup_data(self.data_path, config)
        self.results.log_data()

        config = self.modelers_config
        self.context.setup_mods(config)
        self.results.log_mods()

        config = self.coords_config[training_key]
        self.context.setup_mode(config)
        self.results.log_mode()

        labels_key = "labels"
        if labels_key in self.coords_config[training_key]:
            config = self.coords_config[training_key][labels_key]
            self.context.setup_labels(config=config)
        else:  # elif "labels" not in self.coords_config[training_key]:
            self.context.setup_labels()
        self.results.log_labels()

        config = self.coords_config[training_key]
        self.context.setup_loops(config)

        self.context.setup_stats()

        self.context.setup_noises()

        self.context_ready = True
        self.results.logln("Completed context setup")

    def setup_algo(self):
        """Sets up self.algo.

        Raises:
            ValueError: if the algo's name is unknown
        """
        if not self.results_ready:
            self.setup_results()

        if not self.context_ready:
            self.setup_context()

        algo_name = self.coords_config["training"]["algorithm"]
        if algo_name == "alt_sgd_algo":
            self.algo = _AltSGDAlgo()
        elif algo_name == "pred_alt_sgd_algo":
            self.algo = _PredAltSGDAlgo()
        elif algo_name == "fair_pred_alt_sgd_algo":
            self.algo = _FairPredAltSGDAlgo()
        else:
            raise ValueError(f"Unknown algo: {algo_name}")
        # end if

        self.results.log_algo(algo_name)
        self.algo.bind_context_and_results(self.context, self.results)
        self.algo_ready = True

        self.results.logln("Completed algo setup")

    def start_training(self):
        """Starts the training."""
        if not self.results_ready:
            self.setup_results()

        if not self.context_ready:
            self.setup_context()

        if not self.algo_ready:
            self.setup_algo()

        r = self.results
        r.logln("Started training")
        r.logln("-")
        self.algo.start()
        r.logln("Completed training")
