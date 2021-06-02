"""Helpers for the main function in the generate module."""

# import system and third party modules
import pathlib
import random
import torch
import os
import shutil
import torch.nn as nn
import torchvision.utils as vision_utils

# import custom modules
import gan_libs.configs as configs
import gan_libs.neural_networks as neural_networks


class Vars:
    """Variables."""

    generate_config = None
    model_config = None

    gpu_count = None
    device = None

    results_path = None

    # g: Generator
    g = None
    # g_file_location: Generator file location
    g_file_location = None

    generated_images = None


class Funcs:
    """Functions."""

    @classmethod
    def load_configs(cls):
        """Loads the config files needed for image generation"""
        print("____ load_configs ____")

        # Load generate config
        Vars.generate_config = configs.GenerateConfig()
        print("generate_config.json location: {}".format(
            Vars.generate_config.location
        ))
        Vars.generate_config.load()
        print("Loaded generate_config.json")

        # Load model config
        Vars.model_config = configs.ModelConfig()
        Vars.model_config.location = str(
            pathlib.Path(
                Vars.generate_config.items["model_path"] + "/model_config.json"
            ).resolve()
        )
        print("model_config.json location: {}".format(
            Vars.model_config.location
        ))
        Vars.model_config.load()
        print("Loaded model_config.json")

        print()

    @classmethod
    def set_random_seeds(cls):
        """Sets the seeds of python and torch random number generator.

        Uses the seed specified in generate_config.json, * NOT * the seed in
        model_config.json. The latter seed is only used for training purposes.
        """
        print("____ set_random_seeds ____")

        manual_seed = Vars.generate_config.items["manual_seed"]
        if manual_seed is not None:
            print("Random seed (manual): {}".format(manual_seed))
            random.seed(manual_seed)
            torch.manual_seed(manual_seed)
        else:
            random.seed(None)
            seed = random.randint(
                -0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff
            )
            print("Random seed (auto): {}".format(seed))
            random.seed(seed)
            torch.manual_seed(seed)
        print("Completed setting random seeds")

        print()

    @classmethod
    def detect_device(cls):
        """Detects the image generation device."""
        print("____ detect_device ____")

        Vars.gpu_count = Vars.model_config.items["training"]["gpu_count"]
        print("GPU count: {}".format(Vars.gpu_count))

        if torch.cuda.is_available() and Vars.gpu_count > 0:
            device_name = "cuda:0"
        else:
            device_name = "cpu"
        Vars.device = torch.device(device_name)
        print("Image generation device: {}".format(device_name))

        print()

    @classmethod
    def init_results_folder(cls):
        """Initializes the generation results folder."""
        print("____ init_results_folder ____")

        Vars.results_path = str(
            pathlib.Path(
                Vars.generate_config.items["model_path"] +
                "/Generation-Results"
            ).absolute()
        )

        if os.path.exists(Vars.results_path):
            shutil.rmtree(Vars.results_path)

        pathlib.Path(Vars.results_path).mkdir(exist_ok=True)
        Vars.results_path = str(pathlib.Path(Vars.results_path).resolve())
        print("Results folder: {}".format(Vars.results_path))

        print()

    @classmethod
    def load_model(cls):
        """Loads the model used for image generation."""
        print("____ load_model ____")

        Vars.g = neural_networks.Generator(Vars.model_config.location).to(
            Vars.device
        )

        if Vars.device.type == "cuda" and Vars.gpu_count > 1:
            Vars.g = nn.DataParallel(Vars.g, list(range(Vars.gpu_count)))

        print("Completed generator setup")
        print("==== Generator Structure ====")
        print(Vars.g)
        print()

        Vars.g_file_location = str(
            pathlib.Path(
                Vars.generate_config.items["model_path"] + "/generator.pt"
            ).resolve()
        )

        Vars.g.load_state_dict(torch.load(Vars.g_file_location))
        Vars.g.eval()
        print("Loaded generator state dict")

        print()

    @classmethod
    def generate_images(cls):
        """Generates images with the model."""
        print("____ generate_images ____")

        # z_size: Generator input size
        z_size = Vars.model_config.items["model"]["generator_input_size"]
        image_count = Vars.generate_config.items["image_count"]
        noise = torch.randn(image_count, z_size, 1, 1, device=Vars.device)
        with torch.no_grad():
            Vars.generated_images = Vars.g(noise).detach().cpu()
        print("Generated {} images".format(image_count))

        print()

    @classmethod
    def save_images(cls):
        """Saves the generated images."""
        print("____ save_images ____")

        for index, image in enumerate(Vars.generated_images):
            image_location = str(
                pathlib.Path(
                    Vars.results_path + "/image-{}.jpg".format(index + 1)
                ).absolute()
            )
            vision_utils.save_image(image, image_location, "JPEG")
        print("Saved the generated images")

        print()
