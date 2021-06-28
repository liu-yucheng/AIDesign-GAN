"""Executable for model training."""

import gan_libs.configs as configs
import gan_libs.coords as coords


def main():
    config = configs.TrainConfig()
    config.load()
    print(f"Training config: {config.location}")
    d_path = config["data_path"]
    m_path = config["model_path"]
    coord = coords.TCoord(d_path, m_path)
    coord.setup_results()
    coord.setup_context()


if __name__ == "__main__":
    main()
