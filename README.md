[//]: # "Initially added by: liu-yucheng"
[//]: # "Last updated by: liu-yucheng"

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

