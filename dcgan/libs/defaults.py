"""Module of the default variables.

Attributes:
    exes_path: the executables path
    data_path: the data path
    model_path: the model path
    train_config_name: the train config file name
    generate_config_name: the generate config file name
    coords_config_name: the coords config file name
    modelers_config_name: the modelers config file name
    discriminator_struct_name: the discriminator struct file name
    discriminator_state_name: the discriminator state file name
    generator_struct_name: the generator struct file name
    generator_state_name: the generator state file name
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import pathlib

_curr_path = str(pathlib.Path(__file__).parent.resolve())
_dcgan_path = str(pathlib.Path(_curr_path + "/..").resolve())
_aidesign_path = str(pathlib.Path(_curr_path + "/../../..").resolve())

exes_path = str(pathlib.Path(_dcgan_path + "/exes").absolute())
data_path = str(pathlib.Path(_aidesign_path + "/AIDesign_Data/Default-Data").absolute())
model_path = str(pathlib.Path(_aidesign_path + "/AIDesign_Models/Default-Model").absolute())

train_config_name = "train_config.json"
generate_config_name = "generate_config.json"
coords_config_name = "coords_config.json"
modelers_config_name = "modelers_config.json"
discriminator_struct_name = "discriminator_struct.py"
discriminator_state_name = "discriminator.pt"
generator_struct_name = "generator_struct.py"
generator_state_name = "generator.pt"
