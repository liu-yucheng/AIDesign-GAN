# AIDesign_DCGAN

AI Design DCGAN (Deep Convolutional Generative Adversarial Network) application.

## Installation (with `pip`)

1. Go to the root directory of this repository.
2. If you are using GUI, open a command line window in the directory.
3. Run `pip install -r ./requirement.txt`. This will install the dependencies.
4. See below. Choose your installation type and follow the instructions.

### Development Installation

1. Run `pip install -e ./`. This will install the DCGAN application while making it editable.
2. If you change the source code, you do not need to reinstall the package to reflect the changes.

### Deployment Installation

1. Run `pip install ./`. This will install the DCGAN application.
2. If you need to update the app or change the code, you will need to reinstall the package.

## Usage

### Commands

#### `dcgan-train`

`dcgan-train` trains a model with the given data. You can specify the model and data paths in **`dcgan/exes/train_config.json`**.

#### `dcgan-generate`

`dcgan-generate` generates a set of images with a trained model. You can specify the model path in **`dcgan/exes/generate_config.json`**.


## References

Radford, Alec, Metz, Luke, and Chintala, Soumith. Unsupervised representation learning with deep convolutional generative adversarial networks. CoRR, abs/1511.06434, 2015. URL <http://arxiv.org/abs/1511.06434>.

