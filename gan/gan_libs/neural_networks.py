"""A module that includes the neural network classes."""

import torch.nn as nn
import gan_libs.configs as configs


class Utils:
    """Utilities for neural network setups."""

    @classmethod
    def init_weights(cls, module):
        """Initializes the weights of the nodes in the neural network.

        Params: module: the neural network module
        """
        class_name = module.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    """Generator convolutional neural network."""

    def __init__(self, model_config_location=None):
        super(Generator, self).__init__()

        # Load model config
        self.model_config = configs.ModelConfig()
        if model_config_location is not None:
            self.model_config.location = model_config_location
            self.model_config.load()

        # Load structure definition
        structure_location = self.model_config.items[
            "generator_structure_location"
        ]
        self.structure = configs.GeneratorStructure()
        if structure_location is not None:
            self.structure.location = structure_location
            self.structure.load()

        # Load structure
        z_size = self.model_config.items["model"][
            "generator_input_size"
        ]
        gfm_size = self.model_config.items["model"][
            "generator_feature_map_size"
        ]
        ic_count = self.model_config.items["training_set"][
            "image_channel_count"
        ]
        self.main = eval(self.structure.definition)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """Discriminator convolutional neural network."""

    def __init__(self, model_config_location=None):
        super(Discriminator, self).__init__()

        # Load model config
        self.model_config = configs.ModelConfig()
        if model_config_location is not None:
            self.model_config.location = model_config_location
            self.model_config.load()

        # Load structure definition
        structure_location = self.model_config.items[
            "discriminator_structure_location"
        ]
        self.structure = configs.DiscriminatorStructure()
        if structure_location is not None:
            self.structure.location = structure_location
            self.structure.load()

        # Load structure
        dfm_size = self.model_config.items["model"][
            "discriminator_feature_map_size"
        ]
        ic_count = self.model_config.items["training_set"][
            "image_channel_count"
        ]
        self.main = eval(self.structure.definition)

    def forward(self, input):
        return self.main(input)
