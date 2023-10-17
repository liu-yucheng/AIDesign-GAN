"""Context."""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import numpy
import random
import torch
import typing

from aidesign_gan.libs import configs
from aidesign_gan.libs import utils

_CoordsConfig = configs.CoordsConfig
_DotDict = utils.DotDict
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
_torch_set_default_device = torch.set_default_device
_torch_set_default_tensor_type = torch.set_default_tensor_type
_Union = typing.Union


class Context:
    """Context base class."""

    class Rand(_DotDict):
        """Random info."""

        mode = None
        """Random mode."""
        seed = None
        """Random seed."""

    class Hw(_DotDict):
        """Hardware info."""

        device = None
        """Device to use."""
        gpu_count = None
        """Number of GPUs to use."""

    def __init__(self):
        """Inits self."""

        self.model_path: _Union[str, None] = None
        """Model path.

        Used to replace the optional model_path arguments in the setup methods.
        """
        self.cconfig: _Union[dict, None] = None
        """Coords config.

        Used to replace the optional cconfig arguments in the setup methods.
        """
        self.mconfig: _Union[dict, None] = None
        """Modelers config.

        Used to replace the optional mconfig arguments in the setup methods.
        """

        self.rand = type(self).Rand()
        """Random."""
        self.hw = type(self).Hw()
        """Hardware."""

        # Explicitly set torch default dtype to float32.
        _torch_set_default_dtype(_float32)

        # Explicitly set torch default tensor type to FloatTensor.
        # UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1.
        # _torch_set_default_tensor_type(_FloatTensor)

    def find_model_path(self, model_path_arg):
        """Finds the model path to use.

        Ensures that there is at least 1 model path to use.
        NOTE: Path usage priorities: model_path_arg > self.model_path

        Args:
            model_path_arg: the model path argument

        Returns:
            model_path: the model path to use

        Raises:
            ValueError: if both model_path_arg and self.model_path are None
        """
        model_path_arg: _Union[str, None] = model_path_arg

        if model_path_arg is None and self.model_path is None:
            err_info = str(
                f"At least 1 of the following items must be non-None:\n"
                f"  model_path_arg: {model_path_arg}\n"
                f"  self.model_path: {self.model_path}"
            )

            raise ValueError(err_info)
        # end if

        if model_path_arg is not None:
            model_path = model_path_arg
        elif self.model_path is not None:
            model_path = self.model_path
        # end if

        return model_path

    def find_cconfig(self, cconfig_arg):
        """Finds the coords config to use.

        Ensures that there is at least 1 coords config to use.
        NOTE: Config usage priorities: cconfig_arg > self.cconfig

        Args:
            cconfig_arg: the coords config argument

        Returns:
            cconfig: the coords config to use

        Raises:
            ValueError: if both cconfig_arg and self.cconfig are None
        """
        cconfig_arg: _Union[dict, None] = cconfig_arg

        if cconfig_arg is None and self.cconfig is None:
            err_info = str(
                f"At least 1 of the following items must be non-None:\n"
                f"  cconfig_arg: {cconfig_arg}\n"
                f"  self.cconfig: {self.cconfig}"
            )

            raise ValueError(err_info)
        # end if

        if cconfig_arg is not None:
            cconfig = cconfig_arg
        elif self.cconfig is not None:
            cconfig = self.cconfig
        # end if

        return cconfig

    def find_mconfig(self, mconfig_arg):
        """Finds the modelers config to use.

        Ensures that there is at least 1 modelers config to use.
        NOTE: Config usage priorities: mconfig_arg > self.mconfig

        Args:
            mconfig_arg: the modelers config argument

        Returns:
            mconfig: the modelers config to use

        Raises:
            ValueError: if both mconfig_arg and self.mconfig are None
        """
        mconfig_arg: _Union[dict, None] = mconfig_arg

        if mconfig_arg is None and self.mconfig is None:
            err_info = str(
                f"At least 1 of the following items must be non-None:\n"
                f"  mconfig_arg: {mconfig_arg}\n"
                f"  self.mconfig: {self.mconfig}"
            )

            raise ValueError(err_info)
        # end if

        if mconfig_arg is not None:
            mconfig = mconfig_arg
        elif self.mconfig is not None:
            mconfig = self.mconfig
        # end if

        return mconfig

    def _setup_rand(self, cconfig_key, model_path=None, cconfig=None, mconfig=None):
        """Sets the random seeds with the given args.

        Set up seeds for numpy, random and torch. Set up self.rand and its attributes.

        Args:
            cconfig_key: coords config subdict key [training | generation | exportation]
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        cconfig_key = str(cconfig_key)
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        use_dc_config = cconfig_key not in cconfig and \
            cconfig_key != "training" and \
            cconfig_key != "generation"

        if use_dc_config:
            dc_config = _CoordsConfig.load_default()
            config = dc_config[cconfig_key]
        else:
            config = cconfig[cconfig_key]
        # end if

        manual_seed = config["manual_seed"]
        seed_max = 2 ** 32 - 1

        if hasattr(manual_seed, "__int__"):
            mode = "manual"
            seed = int(manual_seed)
            seed = seed % seed_max
        else:  # elif not hasattr(seed, "__int__"):
            mode = "auto"
            _randseed(None)
            seed = _randint(0, seed_max)
        # end if

        _nprandseed(seed)
        _randseed(seed)
        _torch_seed(seed)
        _torch_cuda_seed_all(seed)

        self.rand.mode = mode
        self.rand.seed = seed

    def _setup_hw(self, cconfig_key, model_path=None, cconfig=None, mconfig=None):
        """Sets up the torch hardware with the given args.

        Set up self.hw and its attributes.

        Args:
            cconfig_key: coords config subdict key [training | generation | exportation]
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        cconfig_key = str(cconfig_key)
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        use_dc_config = cconfig_key not in cconfig and \
            cconfig_key != "training" and \
            cconfig_key != "generation"

        if use_dc_config:
            dc_config = _CoordsConfig.load_default()
            config = dc_config[cconfig_key]
        else:
            config = cconfig[cconfig_key]
        # end if

        gpu_count = config["gpu_count"]

        if _torch_cuda_is_available() and gpu_count > 0:
            device_name = "cuda:0"
        else:
            device_name = "cpu"
        # end if

        device = _torch_device(device_name)

        # Keep the default device "cpu."
        # Explicitly set torch default device to the detected device.
        # _torch_set_default_device(device)

        self.hw.device = device
        self.hw.gpu_count = gpu_count
