"""Module of the context classes."""


class Context:
    """Super class of the context classes.

    Attributes:
        config: the config of the context, used for setup
    """

    def __init__(self, config):
        """Inits self with the given args.

        Args:
            config: the config used for setup

        Raises:
            ValueError: if the config is None
        """
        if config is None:
            raise ValueError("config cannot be None")
        self.config = config


class TrainingContext(Context):
    """Context used for training."""
