"""Module of the defaults."""

import pathlib

_this_path = str(pathlib.Path(__file__).parent.resolve())

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

coords_config_location = str(
    pathlib.Path(model_path + "/coords_config.json").absolute()
)

modelers_config_location = str(
    pathlib.Path(model_path + "/modelers_config.json").absolute()
)

discriminator_struct_location = str(
    pathlib.Path(model_path + "/discriminator_struct.py").absolute()
)

generator_struct_location = str(
    pathlib.Path(model_path + "/generator_struct.py").absolute()
)

discriminator_state_location = str(
    pathlib.Path(model_path + "/discriminator.pt").absolute()
)

generator_state_location = str(
    pathlib.Path(model_path + "/generator.pt").absolute()
)
