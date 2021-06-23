"""Module of the config (configuration) classes."""

import json
import pathlib

import gan_libs.defaults as defaults


class _Helpers:
    """Helpers for classes in the module."""

    @classmethod
    def load_json(cls, from_file, to_dict):
        """Loads the data from a json file to a dict.

        Only loads the contents with keys that can be found in the key set of
        the the dictionary.

        Args:
            from_file:   json file location
            to_dict:     dict object
        """
        file = open(from_file, "r")
        contents = json.load(file)
        for key in to_dict.keys():
            to_dict[key] = contents[key]
        file.close()

    @classmethod
    def save_json(cls, from_dict, to_file):
        """Saves the data from a dict to a json file.

        Args:
            from_dict:   dict object
            to_file:     json file location
        """
        file = open(to_file, "w+")
        json.dump(from_dict, file, indent=4)
        file.close()


class _Config:
    """Super class of the configs."""

    @classmethod
    def find_in_path(cls, name, path):
        """Finds the location of the config with a given name in a given path.

        Args:
            fname:  the given config file name
            path:   the given path

        Returns: the location of the config file
        """
        loc = str(pathlib.Path(path + "/" + name).resolve())
        return loc

    def __init__(self):
        """Initializes an object with the defaults."""
        self.location = None
        self.items = {}

    def __getitem__(self, item):
        """Finds the item corresponding to the given subscript.

        Makes config[item] a shorthand of config.items[item].
        """
        return self.items[item]

    def load(self):
        """Loads the config items from a JSON file.

        Saves the config items at the location if the file does not exist.
        """
        if self.location is None:
            raise ValueError("config.location cannot be None")

        try:
            _Helpers.load_json(self.location, self.items)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the config to a JSON file."""
        if self.location is None:
            raise ValueError("config.location cannot be None")

        _Helpers.save_json(self.items, self.location)


class TrainConfig(_Config):
    """Config of the train.py executable."""

    def __init__(self):
        super().__init__()

        self.location = defaults.train_config_location

        self.items = {
            "data_path": defaults.data_path,
            "model_path": defaults.model_path
        }


class GenerateConfig(_Config):
    """Config of the generate.py executable."""

    def __init__(self):
        super().__init__()

        self.location = defaults.generate_config_location

        self.items = {
            "model_path": defaults.model_path
        }


class CoordsConfig(_Config):
    """Config of the training/generation coordinators."""

    @classmethod
    def find_in_path(cls, path):
        return super().find_in_path("coords_config.json", path)

    def __init__(self):
        super().__init__()

        self.location = defaults.coords_config_location

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


class ModelersConfig(_Config):
    """Config of the discriminator/generator modelers."""

    @classmethod
    def find_in_path(cls, path):
        return super().find_in_path("modelers_config.json", path)

    def __init__(self):
        super().__init__()

        self.location = defaults.modelers_config_location

        self.items = {
            "discriminator": {
                "image_channel_count": 3,
                "feature_map_size": 64,
                "struct_location": defaults.discriminator_struct_location,
                "state_location": defaults.discriminator_state_location,
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
                "struct_location": defaults.generator_struct_location,
                "state_location": defaults.generator_state_location,
                "adam_optimizer": {
                    "learning_rate": 0.0002,
                    "beta1": 0.5,
                    "beta2": 0.999
                }
            }
        }
