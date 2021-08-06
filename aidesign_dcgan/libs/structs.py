"""Module of the NN struct (neural network structure) classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

from aidesign_dcgan.libs import defaults
from aidesign_dcgan.libs import utils


class Struct:
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

        If the file does not exist, the function saves the current stuct at the location.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        try:
            self.definition = utils.load_text(self.location)
        except FileNotFoundError:
            self.save()

    def save(self):
        """Saves the struct definition.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("struct.location cannot be None")

        utils.save_text(self.definition, self.location)


class DStruct(Struct):
    """Discriminator structure."""

    def __init__(self, model_path=None):
        """Inits self with the given args.

        Args:
            model_path: the model path
        """
        super().__init__()
        if model_path is None:
            model_path = defaults.model_path
        self.location = utils.find_in_path(defaults.discriminator_struct_name, model_path)
        # fmt: off
        self.definition = r"""# D (Discriminator)
# CNN (Convolutional Neural Network)

from torch import nn

self = self
ic = self.config["image_channel_count"]
fm = self.config["feature_map_size"]

self.model = nn.Sequential(
    # Layer group 1. input group
    #   input: x (the input image)
    #   input volume: 64*64, ic (width*length, height)
    #   params: in_channels=ic, out_channels=fm, kernel_size=4, ...
    nn.Conv2d(ic, fm, 4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    # 2. input volume: 32*32, fm
    nn.Conv2d(fm, 2 * fm, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * fm),
    nn.LeakyReLU(0.2, inplace=True),
    # 3. input volume: 16*16, 2*fm
    nn.Conv2d(2 * fm, 4 * fm, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * fm),
    nn.LeakyReLU(0.2, inplace=True),
    # 4. input volume: 8*8, 4*fm
    nn.Conv2d(4 * fm, 8 * fm, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(8 * fm),
    nn.LeakyReLU(0.2, inplace=True),
    # 5. output group
    #   input volume: 4*4, 8*fm
    #   output: D(x) (the predicted label value)
    nn.Conv2d(8 * fm, 1, 4, stride=1, padding=0, bias=False),
    nn.Sigmoid()
)
"""
        # fmt: on


class GStruct(Struct):
    """Generator structure."""

    def __init__(self, model_path=None):
        """Inits self with the given args.

        Args:
            model_path: the model path
        """
        super().__init__()
        if model_path is None:
            model_path = defaults.model_path
        self.location = utils.find_in_path(defaults.generator_struct_name, model_path)
        # fmt: off
        self.definition = r"""# G (Generator)
# CNN (Convolutional Neural Network)
# Deconvolution

from torch import nn

self = self
z = self.config["input_size"]
ic = self.config["image_channel_count"]
fm = self.config["feature_map_size"]

self.model = nn.Sequential(
    # (Layer group) 1. input group
    #   input: z (the input noise vector)
    #   input volume: 1*1, z (width*length, height)
    #   params: in_channels=z, out_channels=8*fm, kernel_size=4, ...
    nn.ConvTranspose2d(z, 8 * fm, 4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(8 * fm),
    nn.ReLU(True),
    # 2. input volume: 4*4, 8*fm
    nn.ConvTranspose2d(8 * fm, 4 * fm, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * fm),
    nn.ReLU(True),
    # 3. input volume: 8*8, 4*fm
    nn.ConvTranspose2d(4 * fm, 2 * fm, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * fm),
    nn.ReLU(True),
    # 4. input volume: 16*16, 2*fm
    nn.ConvTranspose2d(2 * fm, fm, 4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    # 5. output group
    #   input volume: 32*32, fm
    #   output: G(z) (the output image)
    nn.ConvTranspose2d(fm, ic, 4, stride=2, padding=1, bias=False),
    nn.Tanh()
)
"""
        # fmt: on
