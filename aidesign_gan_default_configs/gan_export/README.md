<!---
Copyright 2022 Yucheng Liu. GNU GPL3 license.
GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
First added by username: liu-yucheng
Last updated by username: liu-yucheng
--->

# AIDesign-GAN Export Folder

A folder that holds the subfolders and files that an AIDesign-GAN export needs.

# Documentation Files

Texts and images.

## `README.md`

This file itself.

## `generator_preview.jpg`

Generator preview.

A preview of the images the generator can generate.

# Configuration Files

Texts.

## `discriminator_config.json`

Discriminator configuration.

Automatically configured. Not supposed to be edited manually.

Configuration items. Type `dict[str, typing.Union[dict, list, str, bool, int, float, None]]`.

Configuration item descriptions are listed below.

- `image_resolution`. Input image resolution in pixels. Type `int`. Range [1, ).
- `image_channel_count`. Input image channel count. Type `int`. Range [1, ).
- `feature_map_size` Layer 0 (first layer) output feature map count. Type `int`. Range [1, ).
- `struct_name`. Structure name. Type `str`.
- `state_name`. Model state name. Type `str`.

## `discriminator_struct.py`

Discriminator structure.

Automatically configured. Not supposed to be edited manually.

Python code fragment. Uses information from the loaded modeler configuration at `self.config` to setup the targeted model structure at `self.model`.

`self.config`. Loaded modeler configuration. Type `dict`.

`self.model`. Targeted model structure to setup. type `torch.nn.Module`.

## `format_config.json`

Format configuration.

Automatically configured. Not supposed to be edited manually. Serves as a model folder format reference.

Configuration items. Type `dict[str, typing.Union[dict, list, str, bool, int, float, None]]`.

Configuration item descriptions are listed below.

- `aidesign_gan_version`. Type `str`.
- `aisesign_gan_repo_tag`. Type `str`.

## `generator_config.json`

Generator configuration.

Automatically configured. Not supposed to be edited manually.

Configuration items. Type `dict[str, typing.Union[dict, list, str, bool, int, float, None]]`.

Configuration item descriptions are listed below.

- `noise_resolution`. Input noise resolution in pixels. Type `int`. Range [1, ).
- `noise_channel_count`. Input noise channel count. Type `int`. Range [1, ).
- `image_resolution`. Output image resolution in pixels. Type `int`. Range [1, ).
- `image_channel_count`. Output image channel count. Type `int`. Range [1, ).
- `feature_map_size`. Layer -1 (last layer) input feature map count. Type `int`. Range [1, ).
- `struct_name`. Structure name. Type `str`.
- `state_name`. Model state name. Type `str`.
- `preview_name`. Model preview name. Type `str`.

## `generator_struct.py`

Generator structure.

Automatically configured. Not supposed to be edited manually.

Python code fragment. Uses information from the loaded modeler configuration at `self.config` to setup the targeted model structure at `self.model`.

`self.config`. Loaded modeler configuration. Type `dict`.

`self.model`. Targeted model structure to setup. type `torch.nn.Module`.

# State Storage Files

Binaries.

## `discriminator_state.pt`

Discriminator state.

## `generator_state.pt`

Generator state.
