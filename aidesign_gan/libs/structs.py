"""Structures.

Based on the GAN structures in [6] and [7].

NOTE: The [*] reference list is in AIDesign-GAN's main README.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils

_join = ospath.join
_load_text = utils.load_text
_save_text = utils.save_text


class Struct:
    """Struct base class."""

    default_loc = None
    """Default location."""
    default_name = None
    """Default name."""

    @classmethod
    def load(cls, from_file):
        """Loads the struct from a file to a str.

        Args:
            from_file: a file location

        Returns:
            to_str: the loaded str
        """
        from_file = str(from_file)
        to_str = _load_text(from_file)
        return to_str

    @classmethod
    def load_default(cls):
        """Loads the default struct.

        Returns:
            result: the result
        """
        result = cls.load(cls.default_loc)
        return result

    @classmethod
    def load_from_path(cls, path):
        """Loads the struct named cls.default_name from a path.

        Args:
            path: a path

        Returns:
            result: the result
        """
        path = str(path)
        loc = _join(path, cls.default_name)
        result = cls.load(loc)
        return result

    @classmethod
    def save(cls, from_str, to_file):
        """Saves a struct from a str to a file.

        Args:
            from_dict: a str to save
            to_file: a file location
        """
        from_str = str(from_str)
        to_file = str(to_file)
        _save_text(from_str, to_file)

    @classmethod
    def save_to_path(cls, from_str, path):
        """Saves the struct from a str to a file named cls.default_name in a path.

        Args:
            from_str: a str to save
            path: a path
        """
        from_str = str(from_str)
        path = str(path)
        loc = _join(path, cls.default_name)
        cls.save(from_str, loc)


class DiscStruct(Struct):
    """Discriminator structure."""

    default_loc = _join(defaults.default_gan_model_path, defaults.disc_struct_name)
    """Default location."""
    default_name = defaults.disc_struct_name
    """Default name."""


class GenStruct(Struct):
    """Generator structure."""

    default_loc = _join(defaults.default_gan_model_path, defaults.gen_struct_name)
    """Default location."""
    default_name = defaults.gen_struct_name
    """Default name."""
