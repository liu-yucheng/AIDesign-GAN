"""Training context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import numpy
import random
import torch
import typing

from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from aidesign_gan.libs import modelers
from aidesign_gan.libs import utils
from aidesign_gan.libs.contexts import context

_BCELoss = nn.BCELoss
_BICUBIC = transforms.InterpolationMode.BICUBIC
_CenterCrop = transforms.CenterCrop
_Compose = transforms.Compose
_Context = context.Context
_DataLoader = data.DataLoader
_DiscModeler = modelers.DiscModeler
_DotDict = utils.DotDict
_GenModeler = modelers.GenModeler
_ImageFolder = datasets.ImageFolder
_Normalize = transforms.Normalize
_nparray = numpy.array
_shuffle = random.shuffle
_Subset = data.Subset
_Resize = transforms.Resize
_Tensor = torch.Tensor
_ToTensor = transforms.ToTensor
_Union = typing.Union


class TrainContext(_Context):
    """Training context."""

    class Data(_DotDict):
        """Data."""

        class TrainValid(_DotDict):
            """Training validation subset."""

            loader = None
            """Subset data loader."""
            size = None
            """Total image count."""
            batch_count = None
            """Batch count."""

        size = None
        """Dataset's image count."""
        size_to_use = None
        """Image count to use."""
        batch_size = None
        """Image count for each mini-batch."""
        train = TrainValid()
        """Dataset training subset info."""
        valid = TrainValid()
        """Dataset validation subset info."""

    class Mods(_DotDict):
        """Modelers."""

        d: _Union[None, _DiscModeler] = None
        """Discriminator modeler instance."""
        g: _Union[None, _GenModeler] = None
        """Generator modeler instance."""

    class Labels(_DotDict):
        """Target labels."""

        real = None
        """Real label."""
        fake = None
        """Fake label."""

    class Loops(_DotDict):
        """Loop controls."""

        class IterationEpochBatch(_DotDict):
            """Iteration epoch batch."""

            count = None
            """Count."""
            index = None
            """Current index."""

        class RollbackEarlystop(_DotDict):
            """Rollback earlystop."""

            max = None
            """Maximum rollback / earlystop count."""
            d = None
            """Discriminator rollback / earlystop count."""
            g = None
            """Generator rollback / earlystop count."""

        class NoiseModels(_DotDict):
            """Model noising control."""

            before_iter = None
            """Whether to noise the model before each iteration."""
            before_epoch = None
            """Whether to noise the model before each epoch."""
            save_noised = None
            """Whether to save the noised images."""

        iteration = IterationEpochBatch()
        """Iteration control."""
        epoch = IterationEpochBatch()
        """Epoch control."""
        train = IterationEpochBatch()
        """Training batch control."""
        valid = IterationEpochBatch()
        """Validation batch control."""
        rb = RollbackEarlystop()
        """Rollback control."""
        es = RollbackEarlystop()
        """Earlystop control."""
        noise_models = NoiseModels()
        """Model noising control."""

    class Latest(_DotDict):
        """Latest batch result."""

        dx = None
        """Average D(X) while training D."""
        ldr = None
        """L(D, X), the loss of D on real."""
        dgz = None
        """Average D(G(Z)) while training D."""
        ldf = None
        """L(D, G(Z)), the loss of D on fake."""
        ldcr = None
        """L(D, Cluster, X), D cluster loss on real."""
        ldcf = None
        """L(D, Cluster, G(Z)), D cluster loss on fake."""
        ld = None
        """L(D), the loss of D."""

        dx2 = None
        """Average D(X) while training G."""
        lgr = None
        """L(G, X), the loss of G on real."""
        dgz2 = None
        """Average D(G(Z)) when training G."""
        lgf = None
        """L(G, G(Z)), the loss of G on fake."""
        lgcr = None
        """L(G, Cluster, X), G cluster loss on real."""
        lgcf = None
        """L(G, Cluster, G(Z)), G cluster loss on fake."""
        lg = None
        """L(G), the loss of G."""

    class Losses(_DotDict):
        """Epoch losses."""

        class Subset(_DotDict):
            """Data subset losses."""

            d = None
            """Discriminator epoch losses."""
            g = None
            """Generator epoch losses."""

        train = Subset()
        """Training losses."""
        valid = Subset()
        """Validation losses."""

    class Bests(_DotDict):
        """Best losses."""

        d = None
        """Discriminator best loss."""
        g = None
        """Generator best loss."""

    class Rbs(_DotDict):
        """Rollback epochs."""

        d = None
        """Discriminator rollback epoch number list."""
        g = None
        """Generator rollback epoch number list."""

    class Noises(_DotDict):
        """Fixed noises."""

        valid = None
        """Validation noise batch."""
        ref_batch = None
        """A reference batch."""

    class Collapses(_DotDict):
        """Training collapses."""

        epochs = None
        """Collapses epoch number list."""
        batch_count = None
        """Collapses batch count in the current epoch."""
        max_loss = None
        """The maximum loss allowed for the batch to pass the collapse detection."""
        batch_prop = None
        """The minimum proportion of nominal batches allowed for the epoch to pass the collapse detection."""
        max_batch_count = None
        """The maximum collapsed batch count allowed for the epoch to pass the collapse detection.

        This is found by calculating max_batch_count = int(self.collapses.factor * self.data.train.batch_count).
        """

    def __init__(self):
        """Inits self."""
        super().__init__()

        self.dataset_path: _Union[str, None] = None
        """Dataset path.

        Used to replace the optional dataset_path argument in the setup methods.
        """

        self.data = type(self).Data()
        """Data."""
        self.mods = type(self).Mods()
        """Modelers."""
        self.mode = None
        """Training mode."""
        self.labels = type(self).Labels()
        """Target labels."""
        self.loops = type(self).Loops()
        """Loop controls."""
        self.latest = type(self).Latest()
        """Latest batch."""
        self.losses = type(self).Losses()
        """Epoch losses."""
        self.bests = type(self).Bests()
        """Best losses."""
        self.rbs = type(self).Rbs()
        """Rollback epochs."""
        self.noises = type(self).Noises()
        """Reference noises."""
        self.collapses = type(self).Collapses()
        """Training collapses."""

    def find_dataset_path(self, dataset_path_arg):
        """Finds the dataset path to use.

        Ensures that there is at least 1 dataset path to use.
        NOTE: Path usage priorities: dataset_path_arg > self.dataset_path

        Args:
            dataset_path_arg: the dataset path argument

        Returns:
            dataset_path: the dataset path to use

        Raises:
            ValueError: if both dataset_path_arg and self.dataset_path are None
        """
        dataset_path_arg: _Union[str, None] = dataset_path_arg

        if dataset_path_arg is None and self.dataset_path is None:
            err_info = str(
                f"At least 1 of the following items must be non-None:\n"
                f"  dataset_path_arg: {dataset_path_arg}\n"
                f"  self.dataset_path: {self.dataset_path}"
            )

            raise ValueError(err_info)
        # end if

        if dataset_path_arg is not None:
            dataset_path = dataset_path_arg
        elif self.dataset_path is not None:
            dataset_path = self.dataset_path
        # end if

        return dataset_path

    def setup_rand(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets the random seeds with the given args.

        Set up seeds for numpy, random and torch. Set up self.rand and its attributes.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        _ = dataset_path
        model_path = self.find_model_path(model_path)
        cconfig = self.find_cconfig(cconfig)
        mconfig = self.find_mconfig(mconfig)

        self._setup_rand("training", model_path, cconfig, mconfig)

    def setup_hw(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up the torch hardware with the given args.

        Set up self.hw and its attributes.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        _ = dataset_path
        model_path = self.find_model_path(model_path)
        cconfig = self.find_cconfig(cconfig)
        mconfig = self.find_mconfig(mconfig)

        self._setup_hw("training", model_path, cconfig, mconfig)

    def setup_data(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up self.data and its attributes with the given args.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config

        Raises:
            ValueError: if there is no enough images to use
        """
        dataset_path = self.find_dataset_path(dataset_path)
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        path = dataset_path
        config = cconfig["training"]["datasets"]

        image_resolution = config["image_resolution"]
        channel_count = config["image_channel_count"]
        train_weight = config["training_set_weight"]
        valid_weight = config["validation_set_weight"]
        percents_to_use = config["percents_to_use"]
        worker_count = config["loader_worker_count"]
        batch_size = config["images_per_batch"]

        means = [0.5 for _ in range(channel_count)]
        sdevs = [0.667 for _ in range(channel_count)]

        dataset = _ImageFolder(
            root=path,
            transform=_Compose([
                _Resize(image_resolution, interpolation=_BICUBIC),
                _CenterCrop(image_resolution),
                _ToTensor(),
                _Normalize(means, sdevs)
            ])
        )

        subset_ratios = _nparray([train_weight, valid_weight])
        subset_ratios_sum = subset_ratios.sum()

        if subset_ratios_sum == 0:
            subset_ratios_sum = 1

        subset_ratios = subset_ratios / subset_ratios_sum
        prop_to_use = percents_to_use / 100

        size = len(dataset)
        indices = list(range(size))
        _shuffle(indices)

        size_to_use = int(prop_to_use * size)

        if size_to_use < 2:
            size_to_use = 2

        if size_to_use > size:
            info = str(
                f"No enough images to use\n"
                f"  size: {size}\n"
                f"  size_to_use: {size_to_use}"
            )

            raise ValueError(info)
        # end if

        indices_to_use = indices[0: size_to_use]
        # print(f"- indices_to_use\n{indices_to_use}\n-")  # Debug

        train_end = int(subset_ratios[0] * size_to_use)

        if train_end <= 0:
            train_end = 1

        if train_end >= size_to_use:
            train_end = size_to_use - 1

        train_start = 0
        train_indices = indices_to_use[train_start: train_end]
        train_set = _Subset(dataset, train_indices)

        valid_start, valid_end = train_end, size_to_use
        valid_indices = indices_to_use[valid_start: valid_end]
        valid_set = _Subset(dataset, valid_indices)

        train_loader = _DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
        train_size = len(train_set)
        train_batch_count = len(train_loader)

        valid_loader = _DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
        valid_size = len(valid_set)
        valid_batch_count = len(valid_loader)

        self.data.size = size
        self.data.size_to_use = size_to_use
        self.data.batch_size = batch_size

        self.data.train.loader = train_loader
        self.data.train.size = train_size
        self.data.train.batch_count = train_batch_count

        self.data.valid.loader = valid_loader
        self.data.valid.size = valid_size
        self.data.valid.batch_count = valid_batch_count

    def setup_mods(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up self.mods and its attributes with the given args.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config

        Raises:
            ValueError: if self.hw.device is None
        """
        if self.hw.device is None:
            raise ValueError("self.hw.device cannot be None")

        _ = dataset_path
        model_path = self.find_model_path(model_path)
        _ = cconfig
        mconfig = self.find_mconfig(mconfig)

        disc_config = mconfig["discriminator"]
        gen_config = mconfig["generator"]
        loss_func = _BCELoss()

        disc = _DiscModeler(model_path, disc_config, self.hw.device, self.hw.gpu_count, loss_func)
        gen = _GenModeler(model_path, gen_config, self.hw.device, self.hw.gpu_count, loss_func)

        self.mods.d = disc
        self.mods.g = gen

    def setup_mode(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up self.mode with the given args.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config

        Raises:
            ValueError: If self.mods.d is None, or;
                If the training mode is unknown (other than "new" and "resume").
        """
        if self.mods.d is None:
            raise ValueError("self.mods.d cannot be None")

        _ = dataset_path
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        mode = cconfig["training"]["mode"]

        if mode == "new":
            self.mods.d.save()
            self.mods.g.save()
        elif mode == "resume":
            self.mods.d.load()
            self.mods.g.load()
        else:
            raise ValueError(f"Unknown training mode {mode}")
        # end if

        self.mode = mode

    def setup_labels(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up the labels.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config
        """
        _ = dataset_path
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        train = cconfig["training"]

        if "labels" in train:
            config = train["labels"]
        else:
            config = None
        # end if

        if config is None:
            real = float(1)
            fake = float(0)
        else:  # elif config is not None:
            real = config["real"]
            fake = config["fake"]
        # end if

        self.labels.real = real
        self.labels.fake = fake

    def setup_loops(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up the loop control variables.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config

        Raises:
            ValueError: if self.data.size is None
        """
        if self.data.size is None:
            raise ValueError("self.data.size cannot be None")

        _ = dataset_path
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        config = cconfig["training"]

        iteration_count = config["iteration_count"]
        epoch_count = config["epochs_per_iteration"]
        max_rbs = config["max_rollbacks"]
        max_ess = config["max_early_stops"]

        noise_models_key = "noise_models"

        if noise_models_key in config:
            noise_iter = config[noise_models_key]["before_each_iter"]
            noise_epoch = config[noise_models_key]["before_each_epoch"]
            save_noised = config[noise_models_key]["save_noised_images"]
        else:
            noise_iter = False
            noise_epoch = False
            save_noised = False
        # end if

        # Avoid double noising
        if noise_iter and noise_epoch:
            noise_iter = False

        self.loops.iteration.count = iteration_count
        self.loops.iteration.index = 0
        self.loops.epoch.count = epoch_count
        self.loops.epoch.index = 0
        self.loops.train.count = self.data.train.batch_count
        self.loops.train.index = 0
        self.loops.valid.count = self.data.valid.batch_count
        self.loops.valid.index = 0
        self.loops.rb.max = max_rbs
        self.loops.rb.d = 0
        self.loops.rb.g = 0
        self.loops.es.max = max_ess
        self.loops.es.d = 0
        self.loops.es.g = 0
        self.loops.noise_models.before_iter = noise_iter
        self.loops.noise_models.before_epoch = noise_epoch
        self.loops.noise_models.save_noised = save_noised

    def setup_stats(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up the statistics.

        The statistics include self.latest, self.losses, self.bests, self.rbs, self.collapses, and their attributes.

        Args:
            dataset_path: an optional dataset path
            model_path: an optional model path
            cconfig: an optional coords config
            mconfig: an optional modelers config

        Raises:
            ValueError: if self.data.size is None
        """
        if self.data.size is None:
            raise ValueError("self.data.size cannot be None")

        _ = dataset_path
        _ = model_path
        cconfig = self.find_cconfig(cconfig)
        _ = mconfig

        self.latest.dx = None
        self.latest.ldr = None
        self.latest.dgz = None
        self.latest.ldf = None
        self.latest.ldcr = None
        self.latest.ldcf = None
        self.latest.ld = None

        self.latest.dx2 = None
        self.latest.lgr = None
        self.latest.dgz2 = None
        self.latest.lgf = None
        self.latest.lgcr = None
        self.latest.lgcf = None
        self.latest.lg = None

        self.losses.train.d = []
        self.losses.train.g = []
        self.losses.valid.d = []
        self.losses.valid.g = []

        self.bests.d = None
        self.bests.g = None

        self.rbs.d = []
        self.rbs.g = []

        collapses_key = "epoch_collapses"

        if collapses_key in cconfig:
            config = cconfig[collapses_key]
            max_loss = float(config["max_loss"])
            percents_of_batches = float(config["percents_of_batches"])
        else:
            max_loss = float(99)
            percents_of_batches = float(67)
        # end if

        self.collapses.epochs = []
        self.collapses.batch_count = 0
        self.collapses.max_loss = max_loss
        self.collapses.batch_prop = float(percents_of_batches / 100)
        self.collapses.max_batch_count = int(self.collapses.batch_prop * self.data.train.batch_count)

    def setup_noises(self, dataset_path=None, model_path=None, cconfig=None, mconfig=None):
        """Sets up self.noises.

        Raises:
            ValueError: if self.data.size or self.mods.d is None
        """
        if self.data.size is None:
            raise ValueError("self.data.size cannot be None")

        if self.mods.d is None:
            raise ValueError("self.mods.d cannot be None")

        _ = dataset_path
        _ = model_path
        _ = cconfig
        _ = mconfig

        valid = []

        for real_batch in self.data.valid.loader:
            real_data, real_indices = real_batch
            real_data: _Tensor
            _ = real_indices
            # print(f"data.size(): {data.size()}")  # Debug
            # print(f"real_indices: {real_indices}")  # Debug
            batch_image_count = real_data.size()[0]
            noise_batch = self.mods.g.gen_noises(batch_image_count)
            valid.append(noise_batch)
        # end for

        ref_batch = self.mods.g.gen_noises(self.data.batch_size)

        self.noises.valid = valid
        self.noises.ref_batch = ref_batch
