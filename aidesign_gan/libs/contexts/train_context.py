"""Training context."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import numpy
import random
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
_clamp = utils.clamp_float
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
_ToTensor = transforms.ToTensor
_Union = typing.Union


class TrainContext(_Context):
    """Training context."""

    class Data(_DotDict):
        """Data info."""

        class TrainValid(_DotDict):
            """Training validation subset info."""

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
        """Modelers info."""

        d: _Union[None, _DiscModeler] = None
        """Discriminator modeler instance."""
        g: _Union[None, _GenModeler] = None
        """Generator modeler instance."""

    class Labels(_DotDict):
        """Target labels info."""

        real = None
        """Real label."""
        fake = None
        """Fake label."""

    class Loops(_DotDict):
        """Loop controls info."""

        class IterationEpochBatch(_DotDict):
            """Iteration epoch batch info."""

            count = None
            """Count."""
            index = None
            """Current index."""

        class RollbackEarlystop(_DotDict):
            """Rollback earlystop info."""

            max = None
            """Maximum rollback / earlystop count."""
            d = None
            """Discriminator rollback / earlystop count."""
            g = None
            """Generator rollback / earlystop count."""

        iteration = IterationEpochBatch()
        """Iteration control info."""
        epoch = IterationEpochBatch()
        """Epoch control info."""
        train = IterationEpochBatch()
        """Training batch control info."""
        valid = IterationEpochBatch()
        """Validation batch control info."""
        rb = RollbackEarlystop()
        """Rollback control info."""
        es = RollbackEarlystop()
        """Earlystop control info."""

    class Latest(_DotDict):
        """Latest batch result info."""

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
        """Average D(X) while training G"""
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
        """Epoch losses info."""

        class Subset(_DotDict):
            """Data subset losses info."""

            d = None
            """Discriminator epoch losses."""
            g = None
            """Generator epoch losses."""

        train = Subset()
        """Training losses info."""
        valid = Subset()
        """Validation losses info."""

    class Bests(_DotDict):
        """Best losses info."""

        d = None
        """Discriminator best loss."""
        g = None
        """Generator best loss."""

    class Rbs(_DotDict):
        """Rollback epochs info."""

        d = None
        """Discriminator rollback epoch number list."""
        g = None
        """Generator rollback epoch number list."""

    class Noises(_DotDict):
        """Fixed noises info."""

        valid = None
        """Validation noise batch."""
        ref_batch = None
        """A reference batch."""

    class Collapses(_DotDict):
        """Training collapses info."""

        epochs = None
        """Collapses epoch number list."""
        batch_count = None
        """Collapses batch count in the current epoch."""
        max_loss = None
        """The maximum loss allowed for the batch to pass the collapse detection."""
        factor = None
        """The collapse batch factor."""
        max_batch_count = None
        """The maximum collapsed batch count allowed for the epoch to pass the collapse detection.

        This is found by calculating max_batch_count = int(self.collapses.factor * self.data.train.batch_count).
        """

    def __init__(self):
        """Inits self."""

        super().__init__()

        self.data = TrainContext.Data()
        """Data info attr dict."""
        self.mods = TrainContext.Mods()
        """Modelers info attr dict."""
        self.mode = None
        """Training mode name."""
        self.labels = TrainContext.Labels()
        """Target labels info attr dict."""
        self.loops = TrainContext.Loops()
        """Loop controls info attr dict."""
        self.latest = TrainContext.Latest()
        """Latest batch result info attr dict."""
        self.losses = TrainContext.Losses()
        """Epoch losses info attr dict."""
        self.bests = TrainContext.Bests()
        """Best losses info attr dict."""
        self.rbs = TrainContext.Rbs()
        """Rollback epochs info attr dict."""
        self.noises = TrainContext.Noises()
        """Fixed noises info attr dict."""
        self.collapses = TrainContext.Collapses()
        """Training collapses info attr dict."""

    def setup_data(self, path, config):
        """Sets up self.data and its attributes with the given args.

        Args:
            path: the data path
            config: the training coords config subset
        """
        config = config["datasets"]

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

        size = len(dataset)
        subset_ratio = _nparray([train_weight, valid_weight])
        subset_ratio = subset_ratio / subset_ratio.sum()
        prop_to_use = percents_to_use / 100
        prop_to_use = _clamp(prop_to_use, 0, 1)
        size_to_use = int(prop_to_use * size)
        train_start, train_end = 0, int(subset_ratio[0] * size_to_use)
        valid_start, valid_end = train_end, size_to_use
        indices = list(range(size_to_use))
        _shuffle(indices)
        train_indices = indices[train_start: train_end]
        valid_indices = indices[valid_start: valid_end]
        train_set = _Subset(dataset, train_indices)
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

    def setup_mods(self, model_path, config):
        """Sets up self.mods and its attributes with the given args.

        Args:
            config: the modelers config

        Raises:
            ValueError: if self.hw.device is None
        """
        if self.hw.device is None:
            raise ValueError("self.hw.device cannot be None")

        model_path = str(model_path)
        d_config = config["discriminator"]
        g_config = config["generator"]
        loss_func = _BCELoss()

        d = _DiscModeler(model_path, d_config, self.hw.device, self.hw.gpu_count, loss_func)
        g = _GenModeler(model_path, g_config, self.hw.device, self.hw.gpu_count, loss_func)

        self.mods.d = d
        self.mods.g = g

    def setup_mode(self, config):
        """Sets up self.mode with the given args.

        Args:
            config: the training coords config subset

        Raises:
            ValueError: if self.mods.d is None; or, if the training mode is unknown (other than "new" and "resume")
        """
        if self.mods.d is None:
            raise ValueError("self.mods.d cannot be None")

        mode = config["mode"]
        if mode == "new":
            self.mods.d.save()
            self.mods.g.save()
        elif mode == "resume":
            self.mods.d.load()
            self.mods.g.load()
        else:
            raise ValueError(f"Unknown training mode {mode}")

        self.mode = mode

    def setup_labels(self, config=None):
        """Sets up the labels.

        Args:
            config: the coords training labels config dict
        """
        if config is None:
            self.labels.real = float(1)
            self.labels.fake = float(0)
        else:  # elif config is not None:
            real = _clamp(config["real"], 0, 1)
            fake = _clamp(config["fake"], 0, 1)

            self.labels.real = real
            self.labels.fake = fake
        # end if

    def setup_loops(self, config):
        """Sets up the loop control variables.

        Args:
            config: the training coords config subset

        Raises:
            ValueError: if self.data.size is None
        """
        if self.data.size is None:
            raise ValueError("self.data.size cannot be None")

        iteration_count = config["iteration_count"]
        epoch_count = config["epochs_per_iteration"]
        max_rb = config["max_rollbacks"]
        max_es = config["max_early_stops"]

        self.loops.iteration.count = iteration_count
        self.loops.iteration.index = 0
        self.loops.epoch.count = epoch_count
        self.loops.epoch.index = 0
        self.loops.train.count = self.data.train.batch_count
        self.loops.train.index = 0
        self.loops.valid.count = self.data.valid.batch_count
        self.loops.valid.index = 0
        self.loops.rb.max = max_rb
        self.loops.rb.d = 0
        self.loops.rb.g = 0
        self.loops.es.max = max_es
        self.loops.es.d = 0
        self.loops.es.g = 0

    def setup_stats(self):
        """Sets up the statistics.

        The statistics include self.latest, self.losses, self.bests, self.rbs, self.collapses, and their attributes.

        Raises:
            ValueError: if self.data.size is None
        """
        if self.data.size is None:
            raise ValueError("self.data.size cannot be None")

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

        self.collapses.epochs = []
        self.collapses.batch_count = 0
        self.collapses.max_loss = float(99)
        self.collapses.factor = 0.67
        self.collapses.max_batch_count = int(self.collapses.factor * self.data.train.batch_count)

    def setup_noises(self):
        """Sets up self.noises.

        Raises:
            ValueError: if self.data.size or self.mods.d is None
        """
        if self.data.size is None:
            raise ValueError("self.data.size cannot be None")

        if self.mods.d is None:
            raise ValueError("self.mods.d cannot be None")

        valid = []
        for real_batch in self.data.valid.loader:
            real_batch = real_batch[0]
            batch_size = real_batch.size()[0]
            # print(f"[Debug] Batch size: {batch_size}")

            noise_batch = self.mods.g.generate_noises(batch_size)
            valid.append(noise_batch)

        ref_batch = self.mods.g.generate_noises(self.data.batch_size)

        self.noises.valid = valid
        self.noises.ref_batch = ref_batch
