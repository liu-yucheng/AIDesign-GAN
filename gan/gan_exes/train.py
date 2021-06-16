"""Executable for model training."""

import gan_libs.configs as configs
import gan_libs.models as models


def main():
    train_config = configs.TrainConfig()
    train_config.load()
    print("Loaded train_config from {}".format(train_config.location))
    print()

    training_model = models.TrainingModel(
        train_config["data_path"], train_config["model_path"]
    )
    training_model.setup_context()


if __name__ == "__main__":
    main()
