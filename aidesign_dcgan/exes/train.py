"""Executable for model training."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import datetime
import sys

from aidesign_dcgan.libs import configs
from aidesign_dcgan.libs import coords


def main():
    """Starts the executable."""
    start_time = datetime.datetime.now()
    config = configs.TrainConfig()
    config.load()
    data_path = config["data_path"]
    model_path = config["model_path"]
    log = sys.stdout
    print(f"Training executable config: {config.location}")
    coord = coords.TrainingCoord(data_path, model_path, log)
    coord.setup_results()
    coord.setup_context()
    coord.start_training()
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    log.write(f"Command execution time: {execution_time} (days, hours:minutes:seconds)\n")


if __name__ == "__main__":
    main()
