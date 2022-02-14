"""Context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import numpy
import random
import torch

from aidesign_gan.libs import utils

_AttrDict = utils.AttrDict
_float32 = torch.float32
_FloatTensor = torch.FloatTensor
_nprandseed = numpy.random.seed
_randint = random.randint
_randseed = random.seed
_torch_device = torch.device
_torch_seed = torch.manual_seed
_torch_cuda_is_available = torch.cuda.is_available
_torch_cuda_seed_all = torch.cuda.manual_seed_all
_torch_set_default_dtype = torch.set_default_dtype
_torch_set_default_tensor_type = torch.set_default_tensor_type


class Context:
    """Context base class."""

    class Rand(_AttrDict):
        """Random info."""

        mode = None
        """Random mode."""
        seed = None
        """Random seed."""

    class Hw(_AttrDict):
        """Hardware info."""

        device = None
        """Device to use."""
        gpu_count = None
        """Number of GPUs to use."""

    def __init__(self):
        """Inits self."""

        self.rand = Context.Rand()
        """Random info attr dict."""
        self.hw = Context.Hw()
        """Hardware info attr dict."""

        # Explicitly set torch default dtype to float32 and default tensor type to FloatTensor
        _torch_set_default_dtype(_float32)
        _torch_set_default_tensor_type(_FloatTensor)

    def setup_rand(self, config):
        """Sets the random seeds with the given args.

        Set up seeds for numpy, random and torch. Set up self.rand and its attributes.

        Args:
            config: the training / generation coords config subset
        """
        mode = "manual"
        seed = config["manual_seed"]

        if hasattr(seed, "__int__"):
            seed = int(seed)
            seed = seed % (2 ** 32 - 1)
        else:  # elif not hasattr(seed, "__int__")
            mode = "auto"
            _randseed(None)
            seed = _randint(0, 2 ** 32 - 1)

        _nprandseed(seed)
        _randseed(seed)
        _torch_seed(seed)
        _torch_cuda_seed_all(seed)

        self.rand.mode = mode
        self.rand.seed = seed

    def setup_hw(self, config):
        """Sets up the torch hardware with the given args.

        Set up self.hw and its attributes.

        Args:
            config: the training / generation coords config subset
        """
        gpu_count = config["gpu_count"]

        device_name = "cpu"
        if _torch_cuda_is_available() and gpu_count > 0:
            device_name = "cuda:0"
        device = _torch_device(device_name)

        self.hw.device = device
        self.hw.gpu_count = gpu_count
