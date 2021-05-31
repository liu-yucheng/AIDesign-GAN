"""Executable for model training."""

from gan_exes._train_helpers import Funcs


def main():
    Funcs.load_configs()
    Funcs.set_random_seeds()
    Funcs.init_data_loader()
    Funcs.detect_device()
    Funcs.init_result_folder()

    Funcs.plot_training_images()

    Funcs.setup_training()
    Funcs.start_training()

    Funcs.plot_losses()
    Funcs.plot_generated_images()
    Funcs.plot_real_and_fake()


if __name__ == "__main__":
    main()
