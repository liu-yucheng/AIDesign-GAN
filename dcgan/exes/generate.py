"""Executable for image generation with the trained models."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import datetime
import sys

from dcgan.libs import configs
from dcgan.libs import coords


def main():
    """Starts the executable."""
    start_time = datetime.datetime.now()
    config = configs.GenerateConfig()
    config.load()
    model_path = config["model_path"]
    log = sys.stdout
    log.write(f"Generation executable config: {config.location}\n")
    coord = coords.GenerationCoord(model_path, log)
    coord.setup_results()
    coord.setup_context()
    coord.start_generation()
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    log.write(f"Command execution time: {execution_time} (days, hours:minutes:seconds)\n")


if __name__ == "__main__":
    main()
