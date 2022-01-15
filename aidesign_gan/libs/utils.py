"""Module of the util (utility) classes and functions."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from torch import nn
from torch import optim

import asyncio
import json
import os
import pathlib
import random
import shutil
import sys
import threading
import torch

from aidesign_gan.libs import optims

# ==== Classes ====


class AttrDict:
    """Attribute dictionary.

    The AIDesign-GAN ported version of LYC-PyUtils DotDict. A dict whose items can be accessed using the attrdict.key
    syntax, while compatible with Python standard library dict.
    """

    # Trailing underscores added to the public class method names to avoid attribute naming confusions.

    @classmethod
    def fromdict____(cls, dic):
        """Builds and gives an AttrDict from a Python library dict.

        Args:
            dic: the dictionary

        Returns:
            result: the resulting AttrDict
        """
        # print(f"fromdict____ dic: {dic}")  # Debug
        result = AttrDict()
        for key in dic:
            result.setattr____(key, dic[key])
        return result

    @classmethod
    def isprotected____(cls, key):
        """Finds if a key is protected.

        Args:
            key: the key

        Returns:
            result: the result
        """
        # print(f"isprotected____ key: {key}")  # Debug
        key = str(key)
        result = key[:1] == "_"  # See whether the key is private
        result = result or len(key) >= 4 and key[:2] == "__" and key[-2:] == "__"  # See whether the key is magic
        result = result or key[-4:] == "____"  # See whether the key is AttrDict reserved
        return result

    # Magic functions

    def __init__(self, *args, **kwargs):
        """Inits self with the given args and kwargs.

        Args:
            *args: the variable arguments
            **kwargs: the keyword arguments
        """
        super().__init__()
        selftype = type(self)

        # Set the inherited keys from the custom class level
        if selftype is not AttrDict and issubclass(selftype, AttrDict):
            classdict = type(self).__dict__
            for key in classdict:
                key = str(key)
                # print(f"__init__ classdict {key}: {classdict[key]}")  # Debug
                if not type(self).isprotected____(key):
                    value = classdict[key]
                    self.setattr____(key, value)

        # Set keys with the key names from the variable arguments
        for arg in args:
            arg = str(arg)
            # print(f"__init__ *args arg {arg}")  # Debug
            if not selftype.isprotected____(key):
                self.setattr____(arg, None)

        # Set keys with the key names and values from the keyword arguments
        for kw in kwargs:
            kw = str(kw)
            # print(f"__init__ **kwargs kw {kw}: {kwargs[kw]}")  # Debug
            if not selftype.isprotected____(key):
                self.setattr____(kw, kwargs[kw])

    def __getattr__(self, name):
        return self.getattr____(name)

    def __setattr__(self, name, value):
        return self.setattr____(name, value)

    def __str__(self):
        return self.str____()

    def __len__(self):
        return self.__dict__.__len__()

    def __iter__(self):
        return self.__dict__.__iter__()

    def __getstate__(self):
        return self.todict____()

    def __setstate__(self):
        return type(self).fromdict____(self.__dict__)

    # Trailing underscores added to the public function names to avoid attribute naming confusions.

    def getattr____(self, name):
        """Gets an attribute of self.

        Args:
            name: the name of the attribute

        Returns:
            value: the value of the attribute; or, a new AttrDict object, if the attribute does not exist

        Raises:
            AttributeError: if self does not have the attribute
        """
        if name not in self.__dict__:
            raise AttributeError(f"self does not have the attribute: {name}")

        value = self.__dict__[name]
        return value

    def setattr____(self, name, value):
        """Sets an attribute of self.

        All python library dict values are converted to AttrDict values recursively.

        Args:
            name: the name of the attribute
            value: the value of the attribute

        Returns:
            value: the value of the attribute
        """
        if isinstance(value, dict):
            value = AttrDict.fromdict____(value)
        if not type(self).isprotected____(name):
            self.__dict__[name] = value
        value = self.__dict__[name]
        return value

    def getclassattr____(self, name):
        """Gets the value of the class attribute with a name.

        This will also set self.name to type(self).__dict__[name].

        Args:
            name: the name

        Returns:
            value: the value

        Raises:
            AttributeError: if type(self) does not have the attribute
        """
        classdict = type(self).__dict__

        if name not in classdict:
            raise AttributeError(f"type(self) does not have the attribute: {name}")

        value = classdict[name]
        self.setattr____(name, value)
        return value

    def setclassattr____(self, name, value):
        """Sets the class attribute with a name to a value.

        This will first set self.name to value and then set type(self).__dict__[name] to value.

        Args:
            name: the name
            value: the value

        Returns:
            value: the value
        """
        selftype = type(self)
        if isinstance(value, AttrDict):
            value = value.todict____()
        if not selftype.isprotected____(name):
            self.setattr____(name, value)
            setattr(selftype, name, value)
        value = selftype.__dict__[name]
        return value

    def str____(self):
        """Finds and gives a string representation of self.

        Returns:
            result: the resulting string representation
        """
        result = ".{}"
        if len(self.__dict__) <= 0:
            return result

        result = ".{"
        for key in self.__dict__:
            result += f"{key.__str__()}: {self.__dict__[key].__str__()}, "

        result = result[:-2]  # Remove the trailing comma and space
        result += "}"

        return result

    def todict____(self):
        """Finds and gives a Python library dict version of self.

        All AttrDict values are converted to Python library dict values recursively.

        Returns:
            result: the result dict
        """
        result = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                value = value.todict____()
            result[key] = value
        return result


class TimedInput:
    """Timed input class.

    Python-native and platform-independent timed input prompt.
    """

    def __init__(self):
        """Inits self with the given args."""
        self._input_str = None
        self._subproc_code = r"""input_str = input()
print(input_str)
"""
        self._subproc = None

    async def _run_subproc(self):
        self._subproc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", self._subproc_code, stdin=sys.stdin, stdout=asyncio.subprocess.PIPE
        )
        data = await self._subproc.stdout.readline()
        self._input_str = data.decode("utf-8", "replace").rstrip()
        await self._subproc.wait()

    def _take(self):
        self._subproc = None
        asyncio.run(self._run_subproc())

    def take(self, timeout=5.0):
        """Takes and return a input string from the user with a given timeout.

        Args:
            timeout: the timeout period length in seconds

        Returns:
            self._input_str: the taken input string, or None if there is a timeout
        """
        timeout = float(timeout)
        self._input_str = None
        thread = threading.Thread(target=self._take)
        thread.start()
        thread.join(timeout)
        if self._input_str is None and self._subproc is not None:
            self._subproc.terminate()
        return self._input_str


# ==== Functions ====


def load_json(from_file, to_dict):
    """Loads the data from a JSON file to a dict.

    The function only loads the contents with keys in the key set of the given dict.

    Args:
        from_file: the JSON file location
        to_dict: the dict object
    """
    file = open(from_file, "r")
    contents = json.load(file)
    for key in to_dict:
        to_dict[key] = contents[key]
    file.close()


def save_json(from_dict, to_file):
    """Saves the data from a dict to a JSON file.

    Args:
        from_dict: the dict object
        to_file: the JSON file location
    """
    file = open(to_file, "w+")
    json.dump(from_dict, file, indent=4)
    file.close()


def find_in_path(name, path):
    """Finds the location of a file given its name and path.

    The path needs to be an existing path. But, the file needs not to be an existing file.

    Args:
        name: the given config file name
        path: the given path

    Returns:
        location: the location of the config file
    """
    path = str(pathlib.Path(path).resolve())
    location: str = str(pathlib.Path(path + "/" + name).absolute())
    return location


def load_text(from_file):
    """Loads the contents from a text file.

    Args:
        from_file: the text file location

    Returns:
        contents: the file contents
    """
    file = open(from_file, "r")
    contents = file.read()
    file.close()
    return contents


def save_text(from_str, to_file):
    """Saves a string to a text file.

    Args:
        from_str: the string to save
        to_file: the text file location
    """
    file = open(to_file, "w+")
    file.write(from_str)
    file.close()


def load_model(location, model):
    """Loads the states from a location into a model.

    Args:
        location: the model file location
        model: the model, a pytorch nn module
    """
    model.load_state_dict(torch.load(location))


def save_model(model, location):
    """Saves the states of a model to a location.

    Args:
        model: the model, a pytorch nn module
        location: the location to save the model
    """
    file = open(location, "w+")
    torch.save(model.state_dict(), location)
    file.close()


def load_optim(location, optim):
    """Loads the states from a location into an optimizer.

    Args:
        location: the optimizer file location
        optim: the optimizer, a pytorch optimizer
    """
    optim.load_state_dict(torch.load(location))


def save_optim(optim, location):
    """Saves the states of an optimizer to a location.

    Args:
        optim: the optimizer, a pytorch optimizer
        location: the location to save the optimizer
    """
    file = open(location, "w+")
    torch.save(optim.state_dict(), location)
    file.close()


def parallelize_model(model, device, gpu_count):
    """Finds the parallelized model with the given args.

    If the GPU count is 0 or 1, or the GPUs do not support CUDA, the function returns the original model.

    Args:
        model: the model, a pytorch nn module
        device: the device to use
        gpu_count: the number of GPUs to use

    Returns:
        model: the parallelized/original model
    """
    if device.type == "cuda" and gpu_count > 1:
        model = nn.DataParallel(model, list(range(gpu_count)))
    return model


def prep_batch_and_labels(batch, label, device):
    """Prepares batch and labels with the given device.

    Args:
        batch: the batch to prepare
        label: the target label
        device: the device to use

    Returns:
        batch: the prepared batch
        labels: the prepared labels
    """
    batch = batch.to(device)
    labels = torch.full((batch.size(0),), label, dtype=torch.float, device=device)
    return batch, labels


def bound_num(num, bound1, bound2):
    """Bounds a number with the given args.

    Args:
        num: the number
        bound1: the 1st bound
        bound2: the 2nd bound

    Returns:
        result: the number, if the number is bounded by the 2 bounds; or, the upper bound, if the number is greater
            than the bounds; or, the lower bound, if the number is less than the bounds
    """
    result = num
    lower = bound1
    upper = bound2
    if upper < lower:
        lower, upper = upper, lower
    if result < lower:
        result = lower
    if result > upper:
        result = upper
    return result


def concat_paths(path1, path2):
    """Concatenates 2 paths together.

    Args:
        path1: the 1st path
        path2: the 2nd path

    Returns:
        path: the concatenated path
    """
    path = str(pathlib.Path(path1 + "/" + path2).absolute())
    return path


def init_folder(path, clean=False):
    """Initializes a folder given a path.

    Args:
        path: the path to the folder
        clean: whether to clean up the folder
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    pathlib.Path(path).mkdir(exist_ok=True)


def logstr(logs, string=""):
    """Logs a string on the log file objects.

    Args:
        logs: the log file objects
        string: the string to log
    """
    for log in logs:
        log.write(string)


def logln(logs, line=""):
    """Logs a line on the log file objects.

    Args:
        logs: the log file objects
        line: the line to log
    """
    for log in logs:
        log.write(line + "\n")


def setup_adam(model, config):
    """Sets up an Adam optimizer with the given args.

    Args:
        model: the model, a pytorch nn module
        config: the adam optimizer config dict

    Returns:
        adam: the Adam optimizer
    """
    adam = optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"])
    )
    return adam


def setup_pred_adam(model, config):
    """Sets up a predictive Adam optimizer with the given args.

    Args:
        model: the model, a pytorch nn module
        config: the adam optimizer config dict

    Returns:
        pred_adam: the predictive Adam optimizer
    """
    pred_factor_key = "pred_factor"
    if pred_factor_key in config:
        pred_adam = optims.PredAdam(
            model.parameters(),
            lr=config["learning_rate"], betas=(config["beta1"], config["beta2"]),
            pred_factor=config["pred_factor"]
        )
    else:  # elif pred_factor_key not in config:
        pred_adam = optims.PredAdam(
            model.parameters(),
            lr=config["learning_rate"], betas=(config["beta1"], config["beta2"])
        )
    return pred_adam


def find_model_sizes(model):
    """Finds the total and trainable sizes of a model.

    Args:
        model: the model

    Returns:
        size: the total size
        training_size: the trainable size
    """
    training = model.training
    size = 0
    training_size = 0
    for param in model.parameters():
        param_size = param.numel()
        size += param_size
        if training and param.requires_grad:
            training_size += param_size
    return size, training_size


def half_float_nan_to_num(tensor):
    """Applies a half-precison float overflow-underflow-preventing nan_to_num operation to a tensor.

    Does nothing if the given tensor is not an instance of torch.Tensor.

    Args:
        tensor: the tensor
    """
    if isinstance(tensor, torch.Tensor):
        torch.nan_to_num_(tensor, nan=0.0, posinf=65504.0, neginf=-65504.0)


def nan_to_num_model(model):
    """Applies nan_to_num to a model.

    Args:
        model: the model, a pytorch nn module
    """
    class_name = model.__class__.__name__
    if class_name.find("Conv") != -1:
        half_float_nan_to_num(model.weight.data)
        half_float_nan_to_num(model.weight.grad.data)
    elif class_name.find("BatchNorm") != -1:
        half_float_nan_to_num(model.weight.data)
        half_float_nan_to_num(model.weight.grad.data)
        half_float_nan_to_num(model.bias.data)
        half_float_nan_to_num(model.bias.grad.data)


def nan_to_num_optim(optim):
    """Applies nan_to_num to an optimizer.

    Args:
        optim: the optimizer, a PyTorch optim
    """
    for group in optim.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.data
            half_float_nan_to_num(grad)
            state = optim.state[param]
            for key in state:
                half_float_nan_to_num(state[key])


def find_params_init_func(config=None):
    """Finds the parameters initialization function.

    Args:
        config: parameters initialization config

    Returns:
        result_func: resulting function
    """
    c_wmean = float(0)
    c_wstd = 0.02

    bn_wmean = float(1)
    bn_wstd = 0.02
    bn_bmean = float(0)
    bn_bstd = 0.0002

    if config is not None:
        c_wmean = float(config["conv"]["weight_mean"])
        c_wstd = float(config["conv"]["weight_std"])

        bn_wmean = float(config["batch_norm"]["weight_mean"])
        bn_wstd = float(config["batch_norm"]["weight_std"])
        bn_bmean = float(config["batch_norm"]["bias_mean"])
        bn_bstd = float(config["batch_norm"]["bias_std"])
    # end if

    def result_func(model):
        """Initializes model parameters.

        Args:
            model: the model
        """
        class_name = str(model.__class__.__name__)
        if class_name.find("Conv") != -1:
            nn.init.normal_(model.weight.data, c_wmean, c_wstd)
        elif class_name.find("BatchNorm") != -1:
            nn.init.normal_(model.weight.data, bn_wmean, bn_wstd)
            nn.init.normal_(model.bias.data, bn_bmean, bn_bstd)

    return result_func


def rand_bool():
    """Produce a random boolean value.

    Returns:
        result: the random boolean
    """
    result = bool(random.randint(0, 1))
    return result
