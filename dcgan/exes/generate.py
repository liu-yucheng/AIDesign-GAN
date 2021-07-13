"""Executable for image generation with the trained models."""

import sys

from dcgan.libs import configs
from dcgan.libs import coords


def main():
    """Starts the executable."""
    config = configs.GenerateConfig()
    config.load()
    print(f"Generation executable config: {config.location}")
    model_path = config["model_path"]
    log = sys.stdout
    coord = coords.GenerationCoord(model_path, log)
    coord.setup_results()
    coord.setup_context()
    coord.start_generation()


if __name__ == "__main__":
    main()
