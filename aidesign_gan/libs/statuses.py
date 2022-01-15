"""Module of the app part statuses."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import pkg_resources

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils


class Status:
    """Super class of the status classes.

    Users can access all entries in the items attribute of this class with subscripts of this class.
    """

    def __init__(self):
        """Inits self."""
        self.location: str = None
        """Status file location."""
        self.items: dict = {}
        """Status items."""

    def __getitem__(self, key):
        """Finds the item corresponding to the given key.

        This function makes self[key] a shorthand of self.items[key].

        Args:
            key: the key

        Returns:
            self.items[key]: the corresponding item
        """
        return self.items[key]

    def __setitem__(self, key, value):
        """Sets the item corresponding to the key to a specified value.

        This function makes self[key] = value a shorthand of self.items[key] = value.

        Args:
            key: the key
        """
        self.items[key] = value

    def load(self):
        """Loads the status items from a JSON file.

        If the file does not exist, the function saves the current status.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        try:
            utils.load_json(self.location, self.items)
        except FileNotFoundError:
            utils.save_json(self.items, self.location)

    def save(self):
        """Saves the config to a JSON file.

        Raises:
            ValueError: if self.location is None
        """
        if self.location is None:
            raise ValueError("self.location cannot be None")
        utils.save_json(self.items, self.location)


class GANTrainStatus(Status):
    """Status of the "gan train" command."""

    def __init__(self):
        super().__init__()
        self.location = utils.find_in_path(defaults.gan_train_status_name, defaults.app_data_path)
        self.items = {
            "dataset_path": None,
            "model_path": None
        }


class GANGenerateStatus(Status):
    """Status of the "gan generate" command."""

    def __init__(self):
        super().__init__()
        self.location = utils.find_in_path(defaults.gan_generate_status_name, defaults.app_data_path)
        self.items = {
            "model_path": None
        }
