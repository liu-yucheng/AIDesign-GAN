"""Module of the NN struct (neural network structure) classes."""

from gan_libs import defaults
from gan_libs import utils


class NNStruct:
    """Super class of the NN struct classes.

    Attributes:
        location: the structure file location
        definition: the structure definition
    """

    def __init__(self):
        """Inits self."""
        self.location = None
        self.definition = ""

    def load(self):
        """Loads the struct definition.

        If the file does not exist, the function saves the current stuct at the
        location.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        try:
            self.definition = utils.load_text_file(self.location)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the struct definition.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("struct.location cannot be None")

        utils.save_text_file(self.definition, self.location)


class DStruct(NNStruct):
    """Discriminator structure."""

    def __init__(self, model_path=None):
        """Inits self with the given args.

        Args:
            model_path: the model path
        """
        super().__init__()
        if model_path is None:
            model_path = defaults.model_path
        self.location = utils.\
            find_in_path(defaults.discriminator_struct_name, model_path)
        # fmt: off
        self.definition = r"""
# Convolutional Neural Network

from torch import nn

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


class GStruct(NNStruct):
    """Generator structure."""

    def __init__(self, model_path=None):
        """Inits self with the given args.

        Args:
            model_path: the model path
        """
        super().__init__()
        if model_path is None:
            model_path = defaults.model_path
        self.location = utils.\
            find_in_path(defaults.generator_struct_name, model_path)
        # fmt: off
        self.definition = r"""
# Convolutional Neural Network with Transposed Layers

from torch import nn

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
