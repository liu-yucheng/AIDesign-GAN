<!---
Copyright 2022 Yucheng Liu. GNU GPL3 lincense.
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

## Deployment Installation

1. Run the `pip install ./` command. This will install the application.
2. If you need to update the app or change the code, you will need to reinstall the package.

# Usage (From Command Line Shell)

`gan`: The main command, which provides you the access to all the subcommands of the app.

`gan help`: The help subcommand, which tells you the details about how to use the app.

# References

**Note:** The referenced works are listed below in alphabetical order. You can find the reference details in the docstring of the modules that are relevant to the referenced works.

Arjovsky, et al., 2017. *Wasserstein Generative Adversarial Networks.* https://arxiv.org/abs/1701.07875

Odena, et al., 2016. *Deconvolution and Checkerboard Artifacts.* https://distill.pub/2016/deconv-checkerboard/

*PyTorch Adam optimizer source code.* https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam

*PyTorch DCGAN tutorial.* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

Radford, et al., 2016. *Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks.* https://arxiv.org/pdf/1511.06434.pdf

Yadav, et al., 2018. *Stabilizing Adversarial Nets With Prediction Methods.* https://openreview.net/pdf?id=Skj8Kag0Z

# Miscellaneous
## Versioning

```text
The versioning of this app is based on Semantic Versioning.
You can see the complete Semantic Versioning specification from
  https://semver.org/.
Basically, the version name of this app is in the form of:
  x.y.z
  Where x, y, and z are integers that are greater than or equal to 0.
  Where x, y, and z are separated by dots.
  x stands for the major version and indicates non-compatible major changes to
    the app.
  y stands for the minor version and indicates forward compatible minor
    changes to the app.
  z stands for the patch version and indicates bug fixes and patches to the
    app.
```

## Version Tags

```text
The version tags of this repo has the form of a letter "v" followed by a
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
AIDesign-GAN, a GAN modeling application.
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
  2. https://www.gnu.org/licenses/.
  3. https://www.gnu.org/licenses/gpl-3.0.txt.
```
