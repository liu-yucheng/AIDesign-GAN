"""Executable for image generating with existing models."""

from gan_exes._generate_helpers import Funcs


def main():
    Funcs.load_configs()
    Funcs.set_random_seeds()
    Funcs.detect_device()
    Funcs.init_results_folder()
    Funcs.load_model()
    Funcs.generate_images()
    Funcs.save_images()


if __name__ == "__main__":
    main()
