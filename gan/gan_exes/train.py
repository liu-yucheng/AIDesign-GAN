"""Executable for model training."""

from gan_libs import configs
from gan_libs import coords
import sys


def main():
    config = configs.TrainConfig()
    config.load()
    print(f"Training config: {config.location}")
    d_path = config["data_path"]
    m_path = config["model_path"]
    log = sys.stdout
    coord = coords.TCoord(d_path, m_path, log)
    coord.setup_results()
    coord.setup_context()
    coord.start_training()


if __name__ == "__main__":
    main()
