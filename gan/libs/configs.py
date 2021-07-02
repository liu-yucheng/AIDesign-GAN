"""Module of the config (configuration) classes."""

from gan.libs import defaults
from gan.libs import utils


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
        self.location = defaults.train_config_location
        self.items = {
            "data_path": defaults.data_path,
            "model_path": defaults.model_path
        }


class GenerateConfig(Config):
    """Config of the generate.py executable."""

    def __init__(self):
        super().__init__()
        self.location = defaults.generate_config_location
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
                "manual_seed": 0,
                "iteration_count": 5,
                "epochs_per_iteration": 2,
                "gpu_count": 1,
                "data_sets": {
                    "loader_worker_count": 0,
                    "percentage_to_use": 100,
                    "images_per_batch": 16,
                    "image_resolution": 64,
                    "training_set_weight": 8,
                    "validation_set_weight": 2
                }
            },
            "generation": {
                "image_count": 64,
                "manual_seed": None,
                "gpu_count": 1,
                "grid_mode": {
                    "enabled": True,
                    "padding": 2,
                    "images_per_grid": 64
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
                "image_channel_count": 3,
                "feature_map_size": 64,
                "struct_location": utils.find_in_path(defaults.discriminator_struct_name, model_path),
                "state_location": utils.find_in_path(defaults.discriminator_state_name, model_path),
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                }
            },
            "generator": {
                "input_size": 100,
                "feature_map_size": 64,
                "image_channel_count": 3,
                "struct_location": utils.find_in_path(defaults.generator_struct_name, model_path),
                "state_location": utils.find_in_path(defaults.generator_state_name, model_path),
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                }
            }
        }
