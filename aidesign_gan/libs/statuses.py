"""Statuses."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils

_join = ospath.join
_load_json = utils.load_json
_save_json = utils.save_json


class Status:
    """Status base class."""

    default_loc = None
    """Default location."""
    default_name = None
    """Default name."""

    @classmethod
    def load(cls, from_file):
        """Loads the status from a file to a dict.

        Args:
            from_file: a file location

        Returns:
            to_dict: the loaded dict
        """
        from_file = str(from_file)
        to_dict = _load_json(from_file)
        to_dict = dict(to_dict)
        return to_dict

    @classmethod
    def load_default(cls):
        """Loads the default status.

        Returns:
            result: the result
        """
        result = cls.load(cls.default_loc)
        return result

    @classmethod
    def load_from_path(cls, path):
        """Loads the status named cls.default_name from a path.

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
    def save(cls, from_dict, to_file):
        """Saves a status from a dict to a file.

        Args:
            from_dict: a status dict to save
            to_file: a file location
        """
        from_dict = dict(from_dict)
        to_file = str(to_file)
        _save_json(from_dict, to_file)

    @classmethod
    def save_to_path(cls, from_dict, path):
        """Saves the status from a dict to a file named cls.default_name in a path.

        Args:
            from_dict: a dict to save
            path: a path
        """
        from_dict: dict = from_dict
        path = str(path)
        loc = _join(path, cls.default_name)
        cls.save(from_dict, loc)

    @classmethod
    def verify(cls, from_dict):
        """Verifies a status from a dict.

        Args:
            from_dict: a status dictionary to verify

        Returns:
            result: the verified dict"""
        from_dict: dict = from_dict
        result = from_dict
        return result

    @classmethod
    def _verify_str_nonable(cls, from_dict, key):
        val = from_dict[key]

        if val is not None:
            val = str(val)

        from_dict[key] = val


class GANTrainStatus(Status):
    """Status of the "gan train" command."""

    default_loc = _join(defaults.default_app_data_path, defaults.gan_train_status_name)
    """Default location."""
    default_name = defaults.gan_train_status_name
    """Default name."""

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        cls._verify_str_nonable(from_dict, "dataset_path")
        cls._verify_str_nonable(from_dict, "model_path")

        result = from_dict
        return result


class GANGenerateStatus(Status):
    """Status of the "gan generate" command."""

    default_loc = _join(defaults.default_app_data_path, defaults.gan_generate_status_name)
    """Default location."""
    default_name = defaults.gan_generate_status_name
    """Default name."""

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        cls._verify_str_nonable(from_dict, "model_path")

        result = from_dict
        return result


class GANExportStatus(Status):
    """Status of the "gan export <path-to-export>" command."""

    default_loc = _join(defaults.default_app_data_path, defaults.gan_export_status_name)
    """Default location."""
    default_name = defaults.gan_export_status_name
    """Default name."""

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        cls._verify_str_nonable(from_dict, "model_path")

        result = from_dict
        return result
