"""Module of the NN struct (neural network structure) classes.

==== References ====
Odena, et al., 2016. Deconvolution and Checkerboard Artifacts. https://distill.pub/2016/deconv-checkerboard/
PyTorch DCGAN tutorial. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Radford, et al., 2016. Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks.
    https://arxiv.org/pdf/1511.06434.pdf
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils


class Struct:
    """Super class of the NN struct classes."""

    def __init__(self):
        """Inits self."""
        self.location = None
        """Structure file location."""
        self.definition = ""
        """Structure definition."""

    def load(self):
        """Loads the struct definition.

        If the file does not exist, the function saves the current stuct.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        try:
            self.definition = utils.load_text(self.location)
        except FileNotFoundError:
            utils.save_text(self.definition, self.location)

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

    def __init__(self, model_path):
        """Inits self with the given args.

        Args:
            model_path: the model path

        Raises:
            ValueError: if model_path is None
        """
        super().__init__()
        if model_path is None:
            raise ValueError("Argument model_path cannot be None")
        self.location = utils.find_in_path(defaults.discriminator_struct_name, model_path)
        # fmt: off
        self.definition = r"""# D (Discriminator)
# CNN (Convolutional Neural Network)
# Resize convolution

from torch import nn

self = self
ir = self.config["image_resolution"]
ic = self.config["image_channel_count"]
fm = self.config["feature_map_size"]

# NOTE:
# nn.Conv2d positional params: in_channels, out_channels, kernel_size, stride, padding
# nn.Upsample positional params: size
# nn.LeakyReLU positional params: negative_slope, inplace
# nn.BatchNorm2d positional params: num_features

_Conv2d = nn.Conv2d
_Upsample = nn.Upsample
_LeakyReLU = nn.LeakyReLU
_BatchNorm2d = nn.BatchNorm2d
_Sigmoid = nn.Sigmoid

self.model = nn.Sequential(
    # Layer group 1. input group
    _Conv2d(ic, fm, 5, 1, 2, bias=False),
    _Upsample(int(ir // 2), mode="bicubic", align_corners=False),
    _LeakyReLU(0.2, True),
    # 2.
    _Conv2d(fm, int(3 * fm), 3, 1, 1, bias=False),
    _Upsample(int(ir // 4), mode="bilinear", align_corners=False),
    _BatchNorm2d(int(3 * fm)),
    _LeakyReLU(0.2, True),
    # 3.
    _Conv2d(int(3 * fm), int(5 * fm), 3, 1, 1, bias=False),
    _Upsample(int(ir // 8), mode="bilinear", align_corners=False),
    _BatchNorm2d(int(5 * fm)),
    _LeakyReLU(0.2, True),
    # 4.
    _Conv2d(int(5 * fm), int(7 * fm), 3, 1, 1, bias=False),
    _Upsample(4, mode="bilinear", align_corners=False),
    _BatchNorm2d(int(7 * fm)),
    _LeakyReLU(0.2, True),
    # 5. output group
    _Conv2d(int(7 * fm), 1, 3, 1, 1, bias=False),
    _Upsample(1, mode="bicubic", align_corners=False),
    _Sigmoid()
)
"""
        # fmt: on


class GStruct(Struct):
    """Generator structure."""

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
        self.location = utils.find_in_path(defaults.generator_struct_name, model_path)
        # fmt: off
        self.definition = r"""# G (Generator)
# CNN (Convolutional Neural Network)
# Resize transposed convolution

from torch import nn

self = self
zr = self.config["noise_resolution"]
zc = self.config["noise_channel_count"]
ir = self.config["image_resolution"]
ic = self.config["image_channel_count"]
fm = self.config["feature_map_size"]

# NOTE:
# nn.ConvTranspose2d positional params: in_channels, out_channels, kernel_size, stride, padding
# nn.Upsample positional params: size
# nn.ReLU positional params: inplace
# nn.BatchNorm2d positional params: num_features

_ConvTranspose2d = nn.ConvTranspose2d
_Upsample = nn.Upsample
_ReLU = nn.ReLU
_BatchNorm2d = nn.BatchNorm2d
_Tanh = nn.Tanh

self.model = nn.Sequential(
    # Layer group 1. input group
    _Upsample(4, mode="bicubic", align_corners=False),
    _ConvTranspose2d(zc, int(7 * fm), 3, 1, 1, bias=False),
    _BatchNorm2d(int(7 * fm)),
    _ReLU(True),
    # 2.
    _Upsample(int(ir // 8), mode="bilinear", align_corners=False),
    _ConvTranspose2d(int(7 * fm), int(5 * fm), 3, 1, 1, bias=False),
    _BatchNorm2d(int(5 * fm)),
    _ReLU(True),
    # 3.
    _Upsample(int(ir // 4), mode="bilinear", align_corners=False),
    _ConvTranspose2d(int(5 * fm), int(3 * fm), 3, 1, 1, bias=False),
    _BatchNorm2d(int(3 * fm)),
    _ReLU(True),
    # 4.
    _Upsample(int(ir // 2), mode="bilinear", align_corners=False),
    _ConvTranspose2d(int(3 * fm), fm, 3, 1, 1, bias=False),
    _BatchNorm2d(fm),
    _ReLU(True),
    # 5. output group
    _Upsample(ir, mode="bicubic", align_corners=False),
    _ConvTranspose2d(fm, ic, 5, 1, 2, bias=False),
    _Tanh()
)
"""
        # fmt: on
