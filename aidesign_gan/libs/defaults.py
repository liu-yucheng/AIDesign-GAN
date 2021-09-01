"""Module of the default variables.

Attributes:
    app_data_path: the app data folder's full path
    train_config_name: the train config file name
    generate_config_name: the generate config file name
    coords_config_name: the coords config file name
    modelers_config_name: the modelers config file name
    discriminator_struct_name: the discriminator struct file name
    discriminator_state_name: the discriminator state file name
    discriminator_optim_name: the discriminator optimizer file name
    generator_struct_name: the generator struct file name
    generator_state_name: the generator state file name
    generator_optim_name: the generator optimizer file name
"""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

import pathlib

Path = pathlib.Path

_aidesign_gan_libs_path = str(Path(__file__).parent)
_aidesign_gan_repo_path = str(Path(_aidesign_gan_libs_path).parent.parent)

app_data_path: str = str(Path(_aidesign_gan_repo_path + "/.aidesign_gan_app_data").absolute())

gan_train_status_name: str = "gan_train_status.json"
gan_generate_status_name: str = "gan_generate_status.json"

coords_config_name: str = "coords_config.json"
modelers_config_name: str = "modelers_config.json"
discriminator_struct_name: str = "discriminator_struct.py"
discriminator_state_name: str = "discriminator_state.pt"
discriminator_optim_name: str = "discriminator_optim.pt"
generator_struct_name: str = "generator_struct.py"
generator_state_name: str = "generator_state.pt"
generator_optim_name: str = "generator_optim.pt"
