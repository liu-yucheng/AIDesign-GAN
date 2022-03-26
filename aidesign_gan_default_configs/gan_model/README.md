<!---
Copyright 2022 Yucheng Liu. GNU GPL3 license.
GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
First added by username: liu-yucheng
Last updated by username: liu-yucheng
--->

# AIDesign-GAN Model Folder

A folder that holds the subfolders and files that an AIDesign-GAN model needs.

# Documentation Files

Texts.

## `README.md`

This file itself.

# Configuration Files

Texts.

## `coords_config.json`

Coordinators configuration.

Configuration item descriptions are listed below.

- `training`. Training coordinator configuration. Type `dict`.
  - `mode`. Training mode. Type `str`. Values `"new", "resume"`.
  - `algorithm`. Training algorithm. Type `str`. Values `"alt_sgd_algo", "pred_alt_sgd_algo", "fair_pred_alt_sgd_algo"`.
  - `manual_seed`. Manual random seed. Type `typing.Union[None, int]`.
  - `gpu_count`. Type `int`. Range [0, ).
  - `iteration_count`. Iteration count. Each iteration contains multiple epochs. Type `int`. Range [0, ).
  - `epochs_per_iteration`. Epochs per iteration. Each epoch contains one complete pass of the training and validation datasets to use. Type `int`. Range [0, ).
  - `max_rollbacks`. Maximum rollbacks. Type `int`. Range [0, ).
  - `max_early_stops`. Maximum early stops. Type `int`. Range [0, ).
  - `datasets`. Datasets configuration. Type `dict`.
    - `loader_worker_count`. Type `int`. Range [0, ).
    - `percents_to_use`. Percentage of dataset to use for training. Type `float`. Range [0, 100].
    - `images_per_batch`. Type `int`. Range [1, ).
    - `image_resolution`. The resolution in pixels to resize the images used for training. Type `int`. Range [1, ).
    - `image_channel_count`. Type `int`. Range [1, ).
    - `training_set_weight`. Type `float`. Range [0, ].
    - `validation_set_weight`. Type `float`. Range [0, ].
  - `labels`. Labels configuration. Type `dict`.
    - `real`. The number used to label each real image from the real world. Type `float`. Range [0, 1].
    - `fake`. The number used to label each fake image from the generator. Type `float`. Range [0, 1].
  - `noise_models`. Model noising configuration. Type `dict`.
    - `before_each_iter`. Whether to inject noises into the model parameters before each iteration. Type `bool`.
    - `before_each_epoch`. Whether to inject noises into the model parameters before each epoch. Type `bool`.
    - `save_noised_images`. Whether to save the generated images after model noising. Type `bool`.
  - `epoch_collapses`. Epoch training collapse detection configurations. Type `dict`.
    - `max_loss`: Maximum loss allowed for a training batch to be marked as not collapsed. If the loss exceeds `max_loss`, the training batch will be marked as collapsed. Type `float`. Range [0, 100].
    - `percents_of_batches`. Maximum percentage of collapsed training batches allowed to mark an epoch as not collapsed. Type `float`. Range [0, 100].
- `generation`. Generation coordinator configuration. Type `dict`.
  - `manual_seed`. Manual random seed. Type `typing.Union[None, int]`.
  - `gpu_count`. Type `int`. Range [0, ).
  - `image_count`. Type `int`. Range [0, ).
  - `images_per_batch`. Type `int`. Range [1, ).
  - `grid_mode`. Grid generation mode configuration. Type `dict`.
    - `enabled`. Whether to enable the grid mode. Type `bool`.
    - `images_per_grid`. Type `int`. Range [1, ).
    - `padding`. Grid padding. Type `int`. Range [0, ).
- `exportation`. Exportation coordinator configuration. Type `dict`.
  - `manual_seed`. Manual random seed. Type `typing.Union[None, int]`.
  - `gpu_count`. Type `int`. Range [0, ).
  - `images_per_batch`. Type `int`. Range [1, ).
  - `preview_grids`. Preview grids configuration. Type `dict`.
    - `images_per_grid`. Type `int`. Range [1, ).
    - `padding`. Grid padding. Type `int`. Range [0, ).

## `discriminator_struct.py`

Discriminator structure.

## `format_config.json`

Format configuration.

## `generator_struct.py `

Generator structure.

## `modelers_config.json`

Modelers configuration.

# Result Subfolders And Files

Folders, texts, and images.

## `Generation-Results`

**Note:** Not present and ready until an AIDesign-GAN generation session completes.

Generation results.

## `Training-Results`

**Note:** Not present and ready until an AIDesign-GAN training session completes.

Training results.

## `log.txt`

**Note:** Not present and ready until an AIDesign-GAN training, generation, or exportation session completes.

Log.

# State Storage Files

Binaries.

## `discriminator_optim.pt`

**Note:** Not present and ready until an AIDesign-GAN training session completes.

Discriminator optimizer.

## `discriminator_state.pt`

**Note:** Not present and ready until an AIDesign-GAN training session completes.

Discriminator state.

## `generator_optim.pt`

**Note:** Not present and ready until an AIDesign-GAN training session completes.

Generator optimizer.

## `generator_state.pt`

**Note:** Not present and ready until an AIDesign-GAN training session completes.

Generator state.
