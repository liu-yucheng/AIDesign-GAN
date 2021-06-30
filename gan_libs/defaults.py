"""Module of the default variables.

Attributes:
    train_config_location: the train config location
    generate_config_location: the generate config location
    data_path: the data path
    model_path: the model path
    coords_config_name: the coords config file name
    modelers_config_name: the modelers config file name
    discriminator_struct_name: the discriminator struct file name
    discriminator_state_name: the discriminator state file name
    generator_struct_name: the generator struct file name
    generator_state_name: the generator state file name
"""

import pathlib

_this_path = str(pathlib.Path(__file__).parent.resolve())
train_config_location = str(
    pathlib.Path(
        _this_path + "/../gan_exes/train_config.json"
    ).resolve()
)
generate_config_location = str(
    pathlib.Path(
        _this_path + "/../gan_exes/generate_config.json"
    ).resolve()
)
data_path = str(
    pathlib.Path(
        _this_path + "/../../../AIDesign_Data/Default-Data"
    ).resolve()
)
model_path = str(
    pathlib.Path(
        _this_path + "/../../../AIDesign_Models/Default-Model"
    ).resolve()
)
coords_config_name = "coords_config.json"
modelers_config_name = "modelers_config.json"
discriminator_struct_name = "discriminator_struct.py"
discriminator_state_name = "discriminator.pt"
generator_struct_name = "generator_struct.py"
generator_state_name = "generator.pt"
