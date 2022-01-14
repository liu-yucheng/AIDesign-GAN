"""Module of the default variables."""

# Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.
# GNU GPL Version 3 copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by: liu-yucheng
# Last updated by: liu-yucheng

import pathlib

_Path = pathlib.Path

_aidesign_gan_libs_path = str(_Path(__file__).parent)
_aidesign_gan_repo_path = str(_Path(_aidesign_gan_libs_path).parent.parent)

app_data_path: str = str(_Path(_aidesign_gan_repo_path + "/.aidesign_gan_app_data").absolute())
"""App data folder's full path."""

gan_train_status_name: str = "gan_train_status.json"
"""\"gan train\" atatus file name."""
gan_generate_status_name: str = "gan_generate_status.json"
"""\"gan generate\" config file name."""

format_config: str = "format_config.json"
"""Format config file name."""
coords_config_name: str = "coords_config.json"
"""Coords config file name."""
modelers_config_name: str = "modelers_config.json"
"""Modelers config file name."""
discriminator_struct_name: str = "discriminator_struct.py"
"""Discriminator struct file name."""
generator_struct_name: str = "generator_struct.py"
"""Generator struct file name."""


discriminator_state_name: str = "discriminator_state.pt"
"""Discriminator state file name."""
discriminator_optim_name: str = "discriminator_optim.pt"
"""Discriminator optimizer file name."""
generator_state_name: str = "generator_state.pt"
"""Generator state file name."""
generator_optim_name: str = "generator_optim.pt"
"""Generator optimizer file name."""
