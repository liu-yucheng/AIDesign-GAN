"""Module of the results classes and functions."""

import matplotlib.pyplot as pyplot
import numpy
import pathlib
import torch
import torchvision.utils as vision_utils


class _Helpers:
    """Helpers for classes in the module."""

    @classmethod
    def find_loc(cls, path, fname):
        """Finds the file location given the path and the file name."""
        location = str(pathlib.Path(path + "/" + fname).absolute())
        return location


def save_training_images(batch, path, device):
    """Saves the first batch of training images.

    Args:
        batch:  the first batch of the training images
        path:   the result path
        device: the device to use
    """
    pyplot.figure(figsize=(8, 8))
    pyplot.axis("off")
    pyplot.title("Training Images")
    grid = vision_utils.\
        make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu()
    pyplot.imshow(numpy.transpose(grid, (1, 2, 0)))
    loc = _Helpers.find_loc(path, "Training-Images.jpg")
    pyplot.savefig(loc)
    pyplot.close()
    print("Saved training images")


def save_validation_images(batch, path, device):
    """Saves the first batch of validation images.

    Args:
        batch:  the first batch of the validation images
        path:   the result path
        device: the device to use
    """
    pyplot.figure(figsize=(8, 8))
    pyplot.axis("off")
    pyplot.title("Validation Images")
    grid = vision_utils.\
        make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu()
    pyplot.imshow(numpy.transpose(grid, (1, 2, 0)))
    loc = _Helpers.find_loc(path, "Validation-Images.jpg")
    pyplot.savefig(loc)
    pyplot.close()
    print("Saved validation images")


def save_generated_images(g_model, noise, iter, epoch, path):
    pyplot.figure(figsize=(8, 8))
    pyplot.axis("off")
    pyplot.title("Generated Images: Iteration {} Epoch {}".format(iter + 1, epoch + 1))
    with torch.no_grad():
        images = g_model(noise).detach().cpu()
    images = vision_utils.make_grid(
        images, padding=2, normalize=True
    )
    pyplot.imshow(numpy.transpose(images, (1, 2, 0)), animated=True)
    plot_location = str(
        pathlib.Path(path + "/Iteration-{}_Epoch-{}.jpg".format(iter + 1, epoch + 1)).absolute()
    )
    pyplot.savefig(plot_location)
    pyplot.close()
    print("Saved generated images")
