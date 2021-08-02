"""Executable for image generation with the trained models."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import datetime
import sys

from aidesign_dcgan.libs import configs
from aidesign_dcgan.libs import coords
from aidesign_dcgan.libs import utils


def main():
    """Starts the executable."""
    start_time = datetime.datetime.now()
    config = configs.GenerateConfig()
    config.load()
    model_path = config["model_path"]
    log_file_location = utils.find_in_path("log.txt", model_path)
    log_file = open(log_file_location, "a+")
    logs = [sys.stdout, log_file]
    utils.logln(logs)
    utils.logln(logs, f"Executable: {__file__}")
    utils.logln(logs, f"Generation executable config: {config.location}")
    coord = coords.GenerationCoord(model_path, logs)
    coord.setup_results()
    coord.setup_context()
    coord.start_generation()
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    utils.logln(logs, f"Command execution time: {execution_time} (days, hours:minutes:seconds)")
    utils.logln(logs)
    log_file.close()


if __name__ == "__main__":
    main()
