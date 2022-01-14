<!---
Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.
GNU GPL Version 3 copy: https://www.gnu.org/licenses/gpl-3.0.txt
First added by: liu-yucheng
Last updated by: liu-yucheng
--->

# AIDesign-GAN

AIDesign GAN modeling application.

# Installing The App Using `pip`

1. Go to the root directory of this repository.
2. If you are using GUI, open a command line window in the directory.
3. Run the `pip install -r ./requirement.txt` command. This will install the dependencies.
4. See below. Choose your installation type and follow the instructions.

## For Development / Testing / Experimentation

1. Run the `pip install -e ./` command. This will install the application under the editable mode.
2. If you change the source code, you do not need to reinstall the package to reflect the changes.

## For Deployment / Use In Production

1. Run the `pip install ./` command. This will install the application.
2. If you need to update the app or change the code, you will need to reinstall the package.

# Using The App On The Command Line

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

# Copyright Notice

```plaintext
AIDesign-GAN, a GAN modeling application.
Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see:
    1. The LICENSE file in this repository.
    2. https://www.gnu.org/licenses/.
    3. https://www.gnu.org/licenses/gpl-3.0.txt.
```
