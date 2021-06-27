"""Module of the default variables."""

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
