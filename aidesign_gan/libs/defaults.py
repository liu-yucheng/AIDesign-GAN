"""Default values.

Not supposed to be changed.
Change only if you know what you are doing.
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import pathlib
from os import path as ospath

_join = ospath.join
_Path = pathlib.Path

_libs_path = str(_Path(__file__).parent)
_repo_path = str(_Path(_libs_path).parent.parent)

app_data_path = _join(_repo_path, ".aidesign_gan_app_data")
"""App data path."""
test_data_path = _join(_repo_path, ".aidesign_gan_test_data")
"""Test data path."""

default_configs_path = _join(_repo_path, "aidesign_gan_default_configs")
"""Default configs path."""
default_app_data_path = _join(default_configs_path, "app_data")
"""Default app data path."""
default_gan_export_path = _join(default_configs_path, "gan_export")
"""Default GAN export path."""
default_gan_model_path = _join(default_configs_path, "gan_model")
"""Default GAN model path."""
default_test_data_path = _join(default_configs_path, "test_data")
"""Default test data path."""

gan_train_status_name = "gan_train_status.json"
""""gan train" satatus name."""
gan_generate_status_name = "gan_generate_status.json"
""""gan generate" status name."""
gan_export_status_name = "gan_export_status.json"
""""gan export <path-to-export>" status name."""

format_config_name = "format_config.json"
"""Format config name."""
coords_config_name = "coords_config.json"
"""Coords config name."""
modelers_config_name = "modelers_config.json"
"""Modelers config name."""
disc_struct_name = "discriminator_struct.py"
"""Discriminator struct name."""
gen_struct_name = "generator_struct.py"
"""Generator struct name."""

model_saves_name = "model_saves"
"""Model saves name."""
disc_state_name = "discriminator_state.pt"
"""Discriminator state name."""
disc_optim_name = "discriminator_optim.pt"
"""Discriminator optimizer name."""
gen_state_name = "generator_state.pt"
"""Generator state name."""
gen_optim_name = "generator_optim.pt"
"""Generator optimizer name."""

disc_config_name = "discriminator_config.json"
"""Discriminator config name."""
gen_config_name = "generator_config.json"
"""Generator config name."""
gen_preview_name = "generator_preview.jpg"
"""Generator preview name."""

disc_state_onnx_name = "discriminator_state.onnx"
"""Discriminator state ONNX name."""
disc_state_script_name = "discriminator_state_script.pt"
"""Discriminator TorchScript name."""
gen_state_onnx_name = "generator_state.onnx"
"""Generator state ONNX name."""
gen_state_script_name = "generator_state_script.pt"
"""Generator TorchScript name."""
