"""Module of the NN struct (neural network structure) classes.

==== References ====
Odena, et al., 2016. Deconvolution and Checkerboard Artifacts. https://distill.pub/2016/deconv-checkerboard/
PyTorch DCGAN tutorial. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
Radford, et al., 2016. Unsupervised Representation Learning With Deep Convolutional Genetrative Adversarial Networks.
    https://arxiv.org/pdf/1511.06434.pdf
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils


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

self.model = nn.Sequential(
    # Layer group 1. input group
    nn.Conv2d(ic, fm, 3, stride=1, padding=1, bias=False),
    nn.Upsample(size=ir // 2, mode="bilinear", align_corners=False),
    nn.LeakyReLU(0.2, inplace=True),
    # 2.
    nn.Conv2d(fm, 2 * fm, 3, stride=1, padding=1, bias=False),
    nn.Upsample(size=ir // 4, mode="bilinear", align_corners=False),
    nn.BatchNorm2d(2 * fm),
    nn.LeakyReLU(0.2, inplace=True),
    # 3.
    nn.Conv2d(2 * fm, 4 * fm, 3, stride=1, padding=1, bias=False),
    nn.Upsample(size=ir // 8, mode="bilinear", align_corners=False),
    nn.BatchNorm2d(4 * fm),
    nn.LeakyReLU(0.2, inplace=True),
    # 4.
    nn.Conv2d(4 * fm, 8 * fm, 3, stride=1, padding=1, bias=False),
    nn.Upsample(size=ir // 16, mode="bilinear", align_corners=False),
    nn.BatchNorm2d(8 * fm),
    nn.LeakyReLU(0.2, inplace=True),
    # 5. output group
    nn.Conv2d(8 * fm, 1, 3, stride=1, padding=1, bias=False),
    nn.Upsample(size=1, mode="bilinear", align_corners=False),
    nn.Sigmoid()
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
# Resize convolution

from torch import nn

self = self
z = self.config["input_size"]
ir = self.config["image_resolution"]
ic = self.config["image_channel_count"]
fm = self.config["feature_map_size"]

self.model = nn.Sequential(
    # Layer group 1. input group
    nn.Upsample(size=ir // 16, mode="bilinear", align_corners=False),
    nn.Conv2d(z, 8 * fm, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(8 * fm),
    nn.ReLU(True),
    # 2.
    nn.Upsample(size=ir // 8, mode="bilinear", align_corners=False),
    nn.Conv2d(8 * fm, 4 * fm, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(4 * fm),
    nn.ReLU(True),
    # 3.
    nn.Upsample(size=ir // 4, mode="bilinear", align_corners=False),
    nn.Conv2d(4 * fm, 2 * fm, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(2 * fm),
    nn.ReLU(True),
    # 4.
    nn.Upsample(size=ir // 2, mode="bilinear", align_corners=False),
    nn.Conv2d(2 * fm, fm, 3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(fm),
    nn.ReLU(True),
    # 5. output group
    nn.Upsample(size=ir, mode="bilinear", align_corners=False),
    nn.Conv2d(fm, ic, 3, stride=1, padding=1, bias=False),
    nn.Tanh()
)
"""
        # fmt: on
