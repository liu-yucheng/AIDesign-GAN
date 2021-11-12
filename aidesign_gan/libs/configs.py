"""Module of the config (configuration) classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import pkg_resources

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils


class Config:
    """Super class of the config classes."""

    def __init__(self):
        """Inits self."""
        self.location = None
        """Config file location."""
        self.items = {}
        """Config items."""

    def __getitem__(self, key):
        """Finds the item corresponding to the given key.

        The function makes self[key] a shorthand of self.items[key].

        Args:
            key: key

        Returns:
            self.item[key]: the corresponding item
        """
        return self.items[key]

    def load(self):
        """Loads the config items from a JSON file.

        If the file does not exist, the function saves the current config.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        try:
            utils.load_json(self.location, self.items)
        except FileNotFoundError:
            utils.save_json(self.items, self.location)

    def save(self):
        """Saves the config to a JSON file.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        utils.save_json(self.items, self.location)


class CoordsConfig(Config):
    """Config of the training/generation coordinators."""

    def __init__(self, model_path):
        """Inits self with the given args.

        Args:
            model_path: the model path

        Raises:
            ValueError: if argument model_path is None
        """
        super().__init__()
        if model_path is None:
            raise ValueError("Argument model_path cannot be None")
        self.model_path = model_path
        self.location = utils.find_in_path(defaults.coords_config_name, model_path)
        self.items = {
            "training": {
                "mode": "resume",
                "algorithm": "pred_alt_sgd_algo",
                "manual_seed": None,
                "gpu_count": 1,
                "iteration_count": 1,
                "epochs_per_iteration": 1,
                "max_rollbacks": 1,
                "max_early_stops": 1,
                "datasets": {
                    "loader_worker_count": 0,
                    "percents_to_use": 1,
                    "images_per_batch": 32,
                    "image_resolution": 64,
                    "image_channel_count": 3,
                    "training_set_weight": 9,
                    "validation_set_weight": 1
                },
                "labels": {
                    "real": 1,
                    "fake": 0
                }
            },
            "generation": {
                "manual_seed": None,
                "gpu_count": 1,
                "image_count": 256,
                "images_per_batch": 32,
                "grid_mode": {
                    "enabled": True,
                    "images_per_grid": 64,
                    "padding": 2
                }
            }
        }


class ModelersConfig(Config):
    """Config of the discriminator/generator modelers."""

    def __init__(self, model_path):
        """Inits self with the given args.

        Args:
            model_path: the model path

        Raises:
            ValueError: if argument model_path is None
        """
        super().__init__()
        if model_path is None:
            raise ValueError("Argument model_path cannot be None")
        self.model_path = model_path
        self.location = utils.find_in_path(defaults.modelers_config_name, model_path)
        self.items = {
            "discriminator": {
                "image_resolution": 64,
                "image_channel_count": 3,
                "feature_map_size": 64,
                "struct_name": defaults.discriminator_struct_name,
                "state_name": defaults.discriminator_state_name,
                "optim_name": defaults.discriminator_optim_name,
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                },
                "params_init": {
                    "conv": {
                        "weight_mean": 0,
                        "weight_std": 0.02
                    },
                    "batch_norm": {
                        "weight_mean": 1,
                        "weight_std": 0.02,
                        "bias_mean": 0,
                        "bias_std": 0.0002
                    }
                }
            },
            "generator": {
                "noise_resolution": 2,
                "noise_channel_count": 32,
                "image_resolution": 64,
                "image_channel_count": 3,
                "feature_map_size": 64,
                "struct_name": defaults.generator_struct_name,
                "state_name": defaults.generator_state_name,
                "optim_name": defaults.generator_optim_name,
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                },
                "params_init": {
                    "conv": {
                        "weight_mean": 0,
                        "weight_std": 0.02
                    },
                    "batch_norm": {
                        "weight_mean": 1,
                        "weight_std": 0.02,
                        "bias_mean": 0,
                        "bias_std": 0.0002
                    }
                }
            }
        }


class FormatConfig(Config):
    """Config format."""

    def __init__(self, model_path):
        """Inits self with the given args.

        Args:
            model_path: model path

        Raises:
            ValueError: if argument model_path is None

        """
        super().__init__()

        if model_path is None:
            raise ValueError("Argument model_path cannot be None")

        self.location = utils.find_in_path(defaults.format_config, model_path)

        # Init version
        version = "<unknown version>"
        packages = pkg_resources.require("aidesign-gan")
        if len(packages) > 0:
            version = packages[0].version

        self.items = {
            "aidesign_gan_version": version,
            "aidesign_gan_repo_tag": "v" + version,
            "coords_config_version": version,
            "discriminator_struct_version": version,
            "format_config_version": version,
            "generator_struct_version": version,
            "modelers_config_version": version
        }
