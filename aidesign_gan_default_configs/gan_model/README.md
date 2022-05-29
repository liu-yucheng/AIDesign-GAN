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

Configuration items. Type `dict[str, typing.Union[dict, list, str, bool, int, float, None]]`.

Configuration item descriptions are listed below.

- `training`. Training coordinator configuration. Type `dict`.
  - `mode`. Training mode. Type `str`. Values `"new", "resume"`.
  - `algorithm`. Training algorithm. Type `str`. Values `"alt_sgd_algo", "pred_alt_sgd_algo", "fair_pred_alt_sgd_algo"`.
  - `manual_seed`. Manual random seed. Type `typing.Union[None, int]`.
  - `gpu_count`. Type `int`. Range [0, ).
  - `iter_count`. Iteration count. Each iteration contains multiple epochs. Type `int`. Range [0, ). Compatibility alias `iteration_count`. Precedence `iter_count` > `iteration_count`.
  - `epochs_per_iter`. Epochs per iteration. Each epoch contains one complete pass of the training and validation datasets to use. Type `int`. Range [0, ). Compatibility alias `epochs_per_iteration`. Precedence `epochs_per_iter` > `epochs_per_iteration`.
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
  - `retrirals`. Epoch training exception retrials configurations. Type `dict`.
    - `max_count`. Maximum retrial count. Type `int`. Range [0, ].
    - `delay_seconds`. Delay in seconds, before starting a retrial. Type `float`. Range [0, ].
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

Python code fragment. Uses information from the loaded modeler configuration at `self.config` to setup the targeted model structure at `self.model`.

`self.config`. Loaded modeler configuration. Type `dict`.

`self.model`. Targeted model structure to setup. type `torch.nn.Module`.

**Warning:** Code in this file can possibly make `gan train` execute malicious code from this file itself and somewhere else. Please be careful.

## `format_config.json`

Format configuration.

Automatically configured. Not supposed to be edited manually. Serves as a model folder format reference.

Configuration items. Type `dict[str, typing.Union[dict, list, str, bool, int, float, None]]`.

Configuration item descriptions are listed below.

- `aidesign_gan_version`. Type `str`.
- `aidesign_gan_repo_tag`. Type `str`.

## `generator_struct.py `

Generator structure.

Python code fragment. Uses information from the loaded modeler configuration at `self.config` to setup the targeted model structure at `self.model`.

`self.config`. Loaded modeler configuration. Type `dict`.

`self.model`. Targeted model structure to setup. type `torch.nn.Module`.

**Warning:** Code in this file can possibly make `gan train`, `gan generate`, and `gan export ...` execute malicious code from this file itself and somewhere else. Please be careful.

## `modelers_config.json`

Modelers configuration.

Configuration items. Type `dict[str, typing.Union[dict, list, str, bool, int, float, None]]`.

Configuration item descriptions are listed below.

- `discriminator`. Discriminator modeler only configurations. Type `dict`.
  - `image_resolution`. Input image resolution in pixels. Type `int`. Range [1, ).
  - `image_channel_count`. Input image channel count. Type `int`. Range [1, ).
  - `feature_map_count`. Layer 0 (first layer) output feature map count. Type `int`. Range [1, ). Compatibility alias `feature_map_size`. Precedence `feature_map_count` > `feature_map_size`.
- `generator`. Generator modeler only configurations. Type `dict`.
  - `noise_resolution`. Input noise resolution in pixels. Type `int`. Range [1, ).
  - `noise_channel_count`. Input noise channel count. Type `int`. Range [1, ).
  - `image_resolution`. Output image resolution in pixels. Type `int`. Range [1, ).
  - `image_channel_count`. Output image channel count. Type `int`. Range [1, ).
  - `feature_map_count`. Layer -1 (last layer) input feature map count. Type `int`. Range [1, ). Compatibility alias `feature_map_size`. Precedence `feature_map_count` > `feature_map_size`.
- `discriminator` or `generator`. Discriminator and generator modelers common configurations. Type `dict`.
  - `struct_name`. Structure name. Type `str`.
  - `state_name`. Model state name. Type `str`.
  - `optim_name`. Optimizer state name. Type `str`.
  - `adam_optimizer`. Adam optimizer configuration. Type `dict`.
    - `learning_rate`. Type `float`. Range [0, ).
    - `beta1`. Momentum beta-1. Type `float`. Range [0, 1].
    - `beta2`. Momentum beta-2. Type `float`. Range [0, 1].
    - `pred_factor`. Prediction factor. Type `float`.
  - `params_init`. Parameters initialization configuration. Type `dict`.
    - `conv`. Convolution layers configuration. Type `dict`.
      - `weight_mean`. Type `float`.
      - `weight_std`. Weight standard deviation. Type `float`. Range [0, ).
    - `batch_norm`. Batch normalization layers configuration. Type `dict`.
      - `weight_mean`. Type `float`.
      - `weight_std`. Weight standard deviation. Type `float`. Range [0, ).
      - `bias_mean`. Type `float`.
      - `bias_std`. Bias standard deviation. Type `float`. Range [0, ).
  - `params_noising`. Parameters noising configuration. Type `dict`.
    - `conv`. Convolution layers configuration. Type `dict`.
      - `delta_weight_mean`. Weight incrementation mean. Type `float`.
      - `delta_weight_std`. Weight incrementation standard deviation. Type `float`. Range [0, ).
    - `batch_norm`. Batch normalization layers configuration. Type `dict`.
      - `delta_weight_mean`. Weight incrementation mean. Type `float`.
      - `delta_weight_std`. Weight incrementation standard deviation. Type `float`. Range [0, ).
      - `delta_bias_mean`. Bias incrementation mean. Type `float`.
      - `delta_bias_std`. Bias incrementation standard deviation. Type `float`. Range [0, ).
  - `fairness`. Fair loss factor configuration. Type `dict`.
    - `dx_factor`. D(X) factor. Type `float`.
    - `dgz_factor`. D(G(Z)) factor. Type `float`.
    - `cluster_dx_factor`. Cluster D(X) factor. Type `float`.
    - `cluster_dgz_factor`. Cluster D(G(Z)) factor. Type `float`.
    - `cluster_dx_overact_slope`. Cluster D(X) overact slope. Type `float`.
    - `cluster_dgz_overact_slope`. Cluster D(G(Z)) overact slope. Type `float`.

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
