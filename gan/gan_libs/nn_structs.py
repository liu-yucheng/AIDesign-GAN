"""Module of the nn struct (neural network structure) classes."""

import gan_libs.defaults as defaults


class _Helpers:
    """Helpers for classes in the module."""

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


class _NNStruct:
    """Super class of the nn structs."""

    def __init__(self):
        """Initializes a struct."""
        self.location = None
        self.definition = ""

    def load(self):
        """Loads the struct definition from the struct file.

        Saves the current struct definition if the file does not exist.
        """
        if self.location is None:
            raise ValueError("struct.location cannot be None")

        try:
            self.definition = _Helpers.load_text_file(self.location)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the struct definition to a text file."""
        if self.location is None:
            raise ValueError("struct.location cannot be None")

        _Helpers.save_text_file(self.definition, self.location)


class DStruct(_NNStruct):
    """Discriminator structure."""

    def __init__(self):
        """Initializes a discriminator structure with the defaults."""
        super().__init__()

        self.location = defaults.discriminator_struct_location

        # fmt: off
        self.definition = r"""
# Convolutional Neural Network

# import torch.nn as nn

ic_count = self.config["image_channel_count"]
dfm_size = self.config["feature_map_size"]

self.model = nn.Sequential(
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
        # fmt: on


class GStruct(_NNStruct):
    """Generator structure."""

    def __init__(self):
        """Initializes a generator structure with the defaults."""
        super().__init__()

        self.location = defaults.generator_struct_location

        # fmt: off
        self.definition = r"""
# Convolutional Neural Network with Transposed Layers

# import torch.nn as nn

z_size = self.config["input_size"]
gfm_size = self.config["feature_map_size"]
ic_count = self.config["image_channel_count"]

self.model = nn.Sequential(
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
        # fmt: on
