<!---
Copyright 2022 Yucheng Liu. GNU GPL3 license.
GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
First added by username: liu-yucheng
Last updated by username: liu-yucheng
--->

# AIDesign-GAN

AIDesign GAN modeling application.

# Installation (Using `pip`)

1. Go to the root directory of this repository.
2. If you are using GUI, open a command line window in the directory.
3. Run the `pip install -r ./requirement.txt` command. This will install the dependencies.
4. See below. Choose your installation type and follow the instructions.

## Editable Installation

1. Run the `pip install -e ./` command. This will install the application under the editable mode.
2. If you change the source code, you do not need to reinstall the package to reflect the changes.

## ~~Deployment Installation~~ (Temporarily Unavailable)

1. ~~Run the `pip install ./` command. This will install the application.~~
2. ~~If you need to update the app or change the code, you will need to reinstall the package.~~

# Usage (From Command Line Shell)

`gan`: The main command, which provides you the access to all the subcommands of the app.

`gan help`: The help subcommand, which tells you the details about how to use the app.

# `gan help` Help Page

```powershell
> gan help
Usage: gan <command> ...
==== Commands ====
help:
    When:   You need help info. For example, now.
    How-to: gan help
create:
    When:   You create a new model with the defaults.
    How-to: gan create <path-to-model>
status:
    When:   You check the status of the train and generate commands.
    How-to: gan status
model:
    When:   You select the model for the next training or generation session.
    How-to: gan model <path-to-model>
dataset:
    When:   You select the dataset for the next training session.
    How-to: gan dataset <path-to-dataset>
train:
    When:   You start a training session.
    How-to: gan train
    Notes:
        You will be prompted with the command status. You need to confirm to continue. Depending on your training
        configs, the training session might take minutes, hours, or several days.
generate:
    When:   You start a generation session.
    How-to: gan generate
    Notes:
        You will be prompted with the command status. You need to confirm to continue. Depending on your generation
        configs, the generation session might take seconds or minutes.
export:
    When:   You want to export a selected model to a path for the use in other software products.
    How-to: gan export <path-to-export>
    Notes:  You can use the model export in any software product, including the proprietary ones.
reset:
    When:   You want to reset the app data, which includes the command statuses.
    How-to: gan reset
    Notes:  You will lose the current command statuses after the reset.
welcome:
    When:   You want to display the welcome message.
    How-to: gan welcome
```

# Dependencies

See `<this-repo>/requirements.txt`.

# Testing

You can test this application by running `python <this-repo>/test_all.py`.

# Python Code Style

Follows [PEP8](https://peps.python.org/pep-0008/) with the exceptions shown in the following VSCode `settings.json` code fragment.

```json
{
  ...,
  "python.formatting.provider": "autopep8",
  "python.formatting.autopep8Args": [
    "--max-line-length=119"
  ],
  "python.linting.enabled": true,
  "python.linting.pycodestyleEnabled": true,
  "python.linting.pycodestyleArgs": [
    "--max-line-length=119"
  ],
  ...
}
```

# Other `README` Files

Other `README.*` files in this repository are listed below.

- [GAN model `README`](aidesign_gan_default_configs/gan_model/README.md)
- [GAN export `README`](aidesign_gan_default_configs/gan_export/README.md)

# References

**Note:** References are in the IEEE style. The IEEE style guide is available at https://ieeeauthorcenter.ieee.org/wp-content/uploads/IEEE-Reference-Guide.pdf.

**Note:** Reference details are in the docstrings of the source code files.

**[1]** I. J. Goodfellow et al. (Jun. 2014). Generative Adversarial Networks. arXiv [Online]. Available: https://arxiv.org/abs/1406.2661

**[2]** A. Yadav, S. Shah, Z. Xu, D. Jacobs, and T. Goldstein. (Feb. 2018). Stabilizing Adversarial Nets With Prediction Methods. arXiv [Online]. Available: https://arxiv.org/abs/1705.07364

**[3]** M. Arjovsky, S. Chintala, and L. Bottou. (Dec. 2017). Wasserstein GAN. arXiv [Online]. Available: https://arxiv.org/abs/1701.07875

**[4]** "PyTorch DCGAN tutorial." PyTorch official website. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html (accessed Apr. 28, 2022).

**[5]** "PyTorch Adam optimizer source code." PyTorch official website. https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam (accessed Apr. 28, 2022)

**[6]** A. Odena, V. Dumoulin, and C. Olah. (Oct. 2016). Deconvolution and Checkerboard Artifacts. Distill [Online]. Available: https://distill.pub/2016/deconv-checkerboard/

**[7]** A. Radford, L. Metz, and S. Chintala. (Jan. 2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. arXiv [Online]. Available: https://arxiv.org/abs/1511.06434

# Miscellaneous
## Developer's Notes :memo: And Warnings :warning:
### Notes :memo:

This application is distributed under the **GNU GPL3 license**.

A subsequent work of this application is a work that satisfies **any one** of the following:

- Is a variant of any form of this application.
- Contains a part, some parts, or all parts of this application.
- Integrates a part, some parts, or all parts of this application.

All subsequent works of this application **must also be distributed under the GNU GPL3 license, and must also open their source codes to the public**.

An output of this application is a file that satisfies **all** of the following:

- Is directly produced by running one or more commands provided by this application.
- Is directly produced by conducting one or more operations on the GUI of this application.

The outputs of this application do not have to be distributed under the GNU GPL3 license.

The non-subsequent works that uses the outputs of this application do not have to be distributed under the GNU GPL3 license.

### Warnings :warning:

Making a **closed-source** subsequent work (as defined above) of this application, and distribute it to the public is **unlawful**, no matter if such work makes a profit.

Doing the above may result in severe civil and criminal penalties.

I reserve the rights, funds, time, and efforts to prosecute those who violate the license of this application to the maximum extent under applicable laws.

## Versions
### Versioning

```text
The versioning of this application is based on Semantic Versioning.
You can see the complete Semantic Versioning specification from
  https://semver.org/.
Basically, the version name of this application is in the form of:
  x.y.z
  Where x, y, and z are integers that are greater than or equal to 0.
  Where x, y, and z are separated by dots.
  x stands for the major version and indicates non-compatible major changes to
    the application.
  y stands for the minor version and indicates forward compatible minor
    changes to the application.
  z stands for the patch version and indicates bug fixes and patches to the
    application.
```

### Version Tags

```text
The version tags of this repository has the form of a letter "v" followed by a
  semantic version.
Given a semantic version:
  $x.$y.$z
  Where $x, $y, and $z are the semantic major, minor, and patch versions.
The corresponding version tag would be:
  v$x.$y.$z
The version tags are on the main branch.
```

## Copyright
### Short Version

```text
Copyright (C) 2022 Yucheng Liu. GNU GPL3 license (GNU General Public License
  Version 3).
You should have and keep a copy of the above license. If not, please get it
  from https://www.gnu.org/licenses/gpl-3.0.txt.
```

### Long Version

```text
AIDesign-GAN, AIDesign GAN modeling application.
Copyright (C) 2022 Yucheng Liu. GNU GPL3 license (GNU General Public License
  Version 3).

This program is free software: you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free
  Software Foundation, either version 3 of the License, or (at your option)
  any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
  more details.

You should have received a copy of the GNU General Public License along with
  this program. If not, see:
  1. The LICENSE file in this repository.
  2. https://www.gnu.org/licenses/#GPL.
  3. https://www.gnu.org/licenses/gpl-3.0.txt.
```
