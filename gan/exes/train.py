"""Executable for model training."""

import sys

from gan.libs import configs
from gan.libs import coords


def main():
    """Starts the executable."""
    config = configs.TrainConfig()
    config.load()
    print(f"Training config: {config.location}")
    data_path = config["data_path"]
    model_path = config["model_path"]
    log = sys.stdout
    coord = coords.TrainingCoord(data_path, model_path, log)
    coord.setup_results()
    coord.setup_context()
    coord.start_training()


if __name__ == "__main__":
    main()
