"""Module of the app part statuses."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng


from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils


class Status:
    """Super class of the status classes.

    Users can access all entries in the items attribute of this class with subscripts of this class.

    Attributes:
        location: the status file location
        items: the status items
    """

    def __init__(self):
        """Inits self."""
        self.location: str = None
        self.items: dict = {}

    def __getitem__(self, key: object) -> object:
        """Finds the item corresponding to the given key.

        This function makes status[key] a shorthand of status.items[key].

        Args:
            key: the key

        Returns:
            the item corresponding to the key
        """
        return self.items[key]

    def __setitem__(self, key: object, value: object) -> None:
        """Sets the item corresponding to the key to a specified value.
        
        This function makes status[key] = value a shorthand of status.items[key] = value.

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
    """Status of the following app parts: the "gan train" command."""

    def __init__(self):
        super().__init__()
        self.location = utils.find_in_path(defaults.gan_train_status_name, defaults.app_data_path)
        self.items = {
            "dataset_path": None,
            "model_path": None
        }


class GANGenerateStatus(Status):
    """Status of the following app parts: the "gan generate" command."""

    def __init__(self):
        super().__init__()
        self.location = utils.find_in_path(defaults.gan_generate_status_name, defaults.app_data_path)
        self.items = {
            "model_path": None
        }