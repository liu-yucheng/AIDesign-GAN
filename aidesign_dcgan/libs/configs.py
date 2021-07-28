"""Module of the config (configuration) classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

from aidesign_dcgan.libs import defaults
from aidesign_dcgan.libs import utils


class Config:
    """Super class of the config classes.

    Attributes:
        location: the config file location
        items: the config items, the settings
    """

    def __init__(self):
        """Inits self."""
        self.location = None
        self.items = {}

    def __getitem__(self, sub):
        """Finds the item corresponding to the given subscript.

        The function makes config[sub] a shorthand of config.items[sub].

        Args:
            sub: the subscript of the item

        Returns:
            the corresponding item
        """
        return self.items[sub]

    def load(self):
        """Loads the config items from a JSON file.

        If the file does not exist, the function saves the current config at the location.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        try:
            utils.load_json(self.location, self.items)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the config to a JSON file.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        utils.save_json(self.items, self.location)


class TrainConfig(Config):
    """Config of the train.py executable."""

    def __init__(self):
        super().__init__()
        self.location = utils.find_in_path(defaults.train_config_name, defaults.exes_path)
        self.items = {
            "data_path": defaults.data_path,
            "model_path": defaults.model_path
        }


class GenerateConfig(Config):
    """Config of the generate.py executable."""

    def __init__(self):
        super().__init__()
        self.location = utils.find_in_path(defaults.generate_config_name, defaults.exes_path)
        self.items = {
            "model_path": defaults.model_path
        }


class CoordsConfig(Config):
    """Config of the training/generation coordinators."""

    def __init__(self, model_path=None):
        """Inits self with the given args.

        Args:
            model_path: the model path
        """
        super().__init__()
        if model_path is None:
            model_path = defaults.model_path
        self.location = utils.find_in_path(defaults.coords_config_name, model_path)
        self.items = {
            "training": {
                "mode": "new",
                "algorithm": "batch_level_algo",
                "manual_seed": 0,
                "gpu_count": 1,
                "iteration_count": 2,
                "epochs_per_iteration": 2,
                "max_rollbacks": 1,
                "max_early_stops": 1,
                "data_sets": {
                    "loader_worker_count": 0,
                    "percents_to_use": 1,
                    "images_per_batch": 16,
                    "image_resolution": 64,
                    "image_channel_count": 3,
                    "training_set_weight": 8,
                    "validation_set_weight": 2
                }
            },
            "generation": {
                "manual_seed": None,
                "gpu_count": 1,
                "image_count": 64,
                "grid_mode": {
                    "enabled": True,
                    "images_per_grid": 64,
                    "padding": 2
                }
            }
        }


class ModelersConfig(Config):
    """Config of the discriminator/generator modelers."""

    def __init__(self, model_path=None):
        """Inits self with the given args.

        Args:
            model_path: the model path
        """
        super().__init__()
        if model_path is None:
            model_path = defaults.model_path
        self.location = utils.find_in_path(defaults.modelers_config_name, model_path)
        self.items = {
            "discriminator": {
                "feature_map_size": 64,
                "image_channel_count": 3,
                "struct_location": utils.find_in_path(defaults.discriminator_struct_name, model_path),
                "state_location": utils.find_in_path(defaults.discriminator_state_name, model_path),
                "optim_location": utils.find_in_path(defaults.discriminator_optim_name, model_path),
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                }
            },
            "generator": {
                "feature_map_size": 64,
                "image_channel_count": 3,
                "input_size": 100,
                "struct_location": utils.find_in_path(defaults.generator_struct_name, model_path),
                "state_location": utils.find_in_path(defaults.generator_state_name, model_path),
                "optim_location": utils.find_in_path(defaults.generator_optim_name, model_path),
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                }
            }
        }
