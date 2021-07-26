[//]: # "Initially added by: liu-yucheng"
[//]: # "Last updated by: liu-yucheng"

# AIDesign_DCGAN

AI Design DCGAN (Deep Convolutional Generative Adversarial Network) application.

# Installation (with `pip`)

1. Go to the root directory of this repository.
2. If you are using GUI, open a command line window in the directory.
3. Run `pip install -r ./requirement.txt`. This will install the dependencies.
4. See below. Choose your installation type and follow the instructions.

## Development Installation

1. Run `pip install -e ./`. This will install the DCGAN application while making it editable.
2. If you change the source code, you do not need to reinstall the package to reflect the changes.

## Deployment Installation

1. Run `pip install ./`. This will install the DCGAN application.
2. If you need to update the app or change the code, you will need to reinstall the package.

# Usage

## Commands

### `dcgan-train`

`dcgan-train` trains a model with the given data. You can specify the model and data paths in **`dcgan/exes/train_config.json`**.

### `dcgan-generate`

`dcgan-generate` generates a set of images with a trained model. You can specify the model path in **`dcgan/exes/generate_config.json`**.

# Versioning

The versioning of this app is based on the guidelines in Semantic Versioning (<https://semver.org/>). Please see the link for more details.

## Versioning Basics

Each version name has the form `v{x}.{y}.{z}`, where `x, y, z` are integers and `x, y, z >= 0`. Note that `x, y, z` can be integers that are `>= 10`, which means that `v12.34.56` or `v123.456.789` are two possible future version names. Also, **note that `v0.10.0` and `v1.0.0` are not the same version**.

The first released version is `v0.1.0`.

**In short**, if the latest version is `v{x}.{y}.{z}`, the versioning of the updates is the following. For the precise definitions, please see below.

1. **API** updates: `v0.{y + n}.{x + n} if x == 0` (where `n` is an integer and `n >= 1`) or `v{x + 1}.0.0 if x >= 1`.
2. **Function** updates: `v{x}.{y + 1}.0`.
3. **Bug fix** updates: `v{x}.{y}.{z + 1}`.

**In precise definitions**, the versioning of the updates is the following.

1. The **APIs** of this app are the commands and their configs (typically located in `dcgan/exes`). If the latest released version is `v0.{y}.{z}`, the API can change in any future versions before `v1.0.0`. If the latest released version is `v{x}.{y}.{z}` (where `x >= 1`) and the next version contains **any non-bug-fixing changes to the APIs**, the next version will be **`v{x + 1}.0.0`**.
2. The **functions** of this app are the libraries (typically located in `dcgan/libs`). If the latest released version is `v{x}.{y}.{z}` and the next version contains **any non-bug-fixing changes to the functions** and no changes described in (1), the next version will be **`v{x}.{y + 1}.0`**.
3. The **bugs** in this app have two definitions. First, the actual program behaviors that are different from the described or designed program behaviors. Second, the actual program behaviors that lead to errors and crashes when the app is used following the usage instructions. The bugs need to be fixed in the future versions. If the latest released version is `v{x}.{y}.{z}` and the next version **fixes any bugs** and contains no changes described in (1) and (2), the next version will be **`v{x}.{y}.{z + 1}`**.

# References

Radford, Alec, Metz, Luke, and Chintala, Soumith. Unsupervised representation learning with deep convolutional generative adversarial networks. CoRR, abs/1511.06434, 2015. URL <http://arxiv.org/abs/1511.06434>.

