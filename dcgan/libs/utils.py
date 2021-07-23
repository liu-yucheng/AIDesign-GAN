"""Module of the util (utility) classes and functions."""

from torch import nn
from torch import optim
import json
import os
import pathlib
import shutil
import torch


class AttrDict:
    """Attribute dictionary.

    A dictionary whose items are accessed as attributes.
    """

    def __getattr__(self, name):
        return self.get_attr(name)

    def __setattr__(self, name, value):
        return self.set_attr(name, value)

    def __repr__(self):
        """Finds the string representation of self.

        Returns:
            result: the string representation of self.__dict__
        """
        result = self.__dict__
        if len(self.__dict__) == 0:
            result = "{ Empty AttrDict }"
        return result

    def get_attr(self, name):
        """Gets an attribute of self.

        Args:
            name: the name of the attribute

        Returns:
            attr: the value of the attribute; or, a new AttrDict object, if the attribute does not exist
        """
        attr = None
        if name not in self.__dict__:
            self.__dict__[name] = AttrDict()
        attr = self.__dict__[name]
        return attr

    def set_attr(self, name, value):
        """Sets an attribute of self.

        Args:
            name: the name of the attribute
            value: the value of the attribute

        Returns:
            value: the value of the attribute
        """
        self.__dict__[name] = value
        return value


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
    location = str(pathlib.Path(path + "/" + name).absolute())
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


def save_model(model, location):
    """Saves the states of a model to a location.

    Args:
        model: the model, a pytorch nn module
        location: the location to save the model
    """
    file = open(location, "w+")
    torch.save(model.state_dict(), location)
    file.close()


def load_model(location, model):
    """Loads the states from a location into a model.

    Args:
        location: the model file location
        model: the model, a pytorch nn module
    """
    model.load_state_dict(torch.load(location))
    model.eval()


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


def init_model_weights(model):
    """Inits the weights inside a model.

    Args:
        model: the model, a pytorch nn module
    """
    class_name = model.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def setup_adam(model, config, rollbacks=0):
    """Sets up an Adam optimizer with the given args.

    Args:
        model: the model, a pytorch nn module
        config: the adam optimizer config dict
        rollbacks: the number of times the model rollbacks

    Returns:
        adam: the Adam optimizer
    """
    adam = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"] / (2 ** rollbacks),
        betas=(config["beta1"], config["beta2"])
    )
    return adam


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
