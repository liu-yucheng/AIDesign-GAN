"""A module that includes the config (configuration) classes.

The config classes are used to set up the executables and training/generating
models.
"""

import json
import pathlib


class _Helpers:
    """Helpers for classes in the configs module."""

    this_path = str(pathlib.Path(__file__).parent.resolve())
    default_data_path = str(
        pathlib.Path(
            this_path + "/../../../AIDesign_Data/Default-Data"
        ).resolve()
    )
    default_model_path = str(
        pathlib.Path(
            this_path + "/../../../AIDesign_Models/Default-Model"
        ).resolve()
    )
    default_generator_structure_location = str(
        pathlib.Path(
            default_model_path + "/generator_structure.py"
        ).resolve()
    )
    default_generator_structure = r"""
# Convolutional Neural Network with Transposed Layers

# import torch.nn as nn

# z_size = generator_input_size
# gfm_size = generator_feature_map_size
# ic_count = image_channel_count
nn.Sequential(
    # (Layer) 0. in_channels=z_size, out_channels=8 * gfm_size, kernel_size=4
    nn.ConvTranspose2d(
        z_size, 8 * gfm_size, 4, stride=1, padding=0, bias=False
    ),
    nn.BatchNorm2d(8 * gfm_size),
    nn.ReLU(True),
    # 1. input state count: (8 * gfm_size) * (4 ** 2)
    nn.ConvTranspose2d(
        8 * gfm_size, 4 * gfm_size, 4, stride=2, padding=1, bias=False
    ),
    nn.BatchNorm2d(4 * gfm_size),
    nn.ReLU(True),
    # 2. input state count: (4 * gfm_size) * (8 ** 2)
    nn.ConvTranspose2d(
        4 * gfm_size, 2 * gfm_size, 4, stride=2, padding=1, bias=False
    ),
    nn.BatchNorm2d(2 * gfm_size),
    nn.ReLU(True),
    # 3. input state count: (2 * gfm_size) * (16 ** 2)
    nn.ConvTranspose2d(
        2 * gfm_size, gfm_size, 4, stride=2, padding=1, bias=False
    ),
    nn.BatchNorm2d(gfm_size),
    nn.ReLU(True),
    # 4. input state count: gfm_size * (32 ** 2)
    #    output state count: ic_count * (64 ** 2)
    nn.ConvTranspose2d(gfm_size, ic_count, 4, stride=2, padding=1, bias=False),
    nn.Tanh()
)
"""

    default_discriminator_structure_location = str(
        pathlib.Path(
            default_model_path + "/discriminator_structure.py"
        ).resolve()
    )
    # fmt: off
    default_discriminator_structure = r"""
# Convolutional Neural Network

# import torch.nn as nn

# dfm_size = discriminator_feature_map_size
# ic_count = image_channel_count
nn.Sequential(
    # (Layer) 0. in_channels=ic_count, out_channels=dfm_size, kernel_size=4
    #            input state count: ic_count * (64 ** 2)
    nn.Conv2d(ic_count, dfm_size, 4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    # 1. input state count: dfm_size * (32 ** 2)
    nn.Conv2d(dfm_size, 2 * dfm_size, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * dfm_size),
    nn.LeakyReLU(0.2, inplace=True),
    # 2. input state count: (2 * dfm_size) * (16 ** 2)
    nn.Conv2d(2 * dfm_size, 4 * dfm_size, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * dfm_size),
    nn.LeakyReLU(0.2, inplace=True),
    # 3. input state count: (4 * dfm_size) * (8 ** 2)
    nn.Conv2d(4 * dfm_size, 8 * dfm_size, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(8 * dfm_size),
    nn.LeakyReLU(0.2, inplace=True),
    # 4. input state count: (8 * dfm_size) * (4 ** 2)
    #    output state count: 1
    nn.Conv2d(8 * dfm_size, 1, 4, stride=1, padding=0, bias=False),
    nn.Sigmoid()
)
"""

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

    @classmethod
    def load_text_file(cls, from_file):
        """Loads the data from a file.

        Args: from_file: text file location

        Returns: the file contents
        """
        file = open(from_file, "r")
        contents = file.read()
        file.close()
        return contents

    @classmethod
    def save_text_file(cls, from_str, to_file):
        """Saves the data from a string to a file.

        Args:
            from_str:   string to save
            to_file:    text file location
        """
        file = open(to_file, "w+")
        file.write(from_str)
        file.close()


class TrainConfig:
    """Config of the gan/train.py executable."""

    default_location = str(
        pathlib.Path(
            _Helpers.this_path + "/../gan_exes/train_config.json"
        ).resolve()
    )

    def __init__(self):
        """Initializes a TrainConfig with the defaults."""
        self.location = TrainConfig.default_location

        self.items = {
            "data_path": _Helpers.default_data_path,
            "model_path": _Helpers.default_model_path
        }

    def load(self):
        """Loads the config from a JSON file.

        If the file does not exist, save the current config at the location.
        """
        try:
            _Helpers.load_json(self.location, self.items)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the config to a JSON file."""
        _Helpers.save_json(self.items, self.location)


class GenerateConfig:
    """Config of the gan/generate.py executable."""

    default_location = str(
        pathlib.Path(
            _Helpers.this_path + "/../gan_exes/generate_config.json"
        ).resolve()
    )

    def __init__(self):
        """Initializes a GenerateConfig with the defaults"""
        self.location = GenerateConfig.default_location

        self.items = {
            "image_count": 64,
            "manual_seed": None,
            "model_path": _Helpers.default_model_path
        }

    def load(self):
        """Loads the config from a JSON file.

        If the file does not exist, save the current config at the location.
        """
        try:
            _Helpers.load_json(self.location, self.items)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the config to a JSON file."""
        _Helpers.save_json(self.items, self.location)


class ModelConfig:
    """Config of a GAN model."""

    default_location = str(
        pathlib.Path(
            _Helpers.default_model_path + "/model_config.json"
        ).resolve()
    )

    def __init__(self):
        """Initializes a ModelConfig with the defaults"""
        self.location = ModelConfig.default_location

        self.items = {
            "training": {
                "mode": "new",
                "manual_seed": 0,
                "epoch_count": 10,
                "gpu_count": 1,
            },
            "training_set": {
                "loader_worker_count": 0,
                "images_per_batch": 16,
                "batch_count": 565,
                "image_resolution": 64,
                "image_channel_count": 3,
            },
            "model": {
                "generator_input_size": 100,
                "generator_feature_map_size": 64,
                "discriminator_feature_map_size": 64,
            },
            "adam_optimizer": {
                "learning_rate": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999
            },

            "generator_structure_location":
            _Helpers.default_generator_structure_location,

            "discriminator_structure_location":
            _Helpers.default_discriminator_structure_location
        }

    def load(self):
        """Loads the config from a JSON file.

        If the file does not exist, save the current config at the location.
        """
        try:
            _Helpers.load_json(self.location, self.items)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the config to a JSON file."""
        _Helpers.save_json(self.items, self.location)


class GeneratorStructure:
    """Structure of a generator."""

    default_location = _Helpers.default_generator_structure_location

    def __init__(self):
        """Initializes a GeneratorStructure with the defaults."""
        self.location = GeneratorStructure.default_location
        self.definition = _Helpers.default_generator_structure

    def load(self):
        """Loads the structure from a text file.

        If the file does not exist, save the current config at the location.
        """
        try:
            self.definition = _Helpers.load_text_file(self.location)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the structure to a text file."""
        _Helpers.save_text_file(self.definition, self.location)


class DiscriminatorStructure:
    """Structure of a discriminator."""

    default_location = _Helpers.default_discriminator_structure_location

    def __init__(self):
        """Initializes a DiscriminatorStructure with the defaults."""
        self.location = DiscriminatorStructure.default_location
        self.definition = _Helpers.default_discriminator_structure

    def load(self):
        """Loads the structure from a text file.

        If the file does not exist, save the current config at the location.
        """
        try:
            self.definition = _Helpers.load_text_file(self.location)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the structure to a text file."""
        _Helpers.save_text_file(self.definition, self.location)
