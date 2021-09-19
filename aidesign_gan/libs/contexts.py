"""Module of the context classes."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

import numpy
import random
import torch

from aidesign_gan.libs import modelers
from aidesign_gan.libs import utils

AttrDict = utils.AttrDict


class Context:
    """Super class of the context classes."""
    class Rand(AttrDict):
        """Random info."""
        mode = None
        """Random mode."""
        seed = None
        """Random seed."""

    class Hw(AttrDict):
        """Hardware info."""
        device = None
        """Device to use."""
        gpu_count = None
        """Number of GPUs to use."""

    def __init__(self):
        """Inits self."""
        self.rand = Context.Rand()
        """Random info attr dict."""
        self.hw = Context.Hw()
        """Hardware info attr dict."""

    def setup_rand(self, config):
        """Sets the random seeds with the given args.

        Set up seeds for numpy, random and torch. Set up self.rand and its attributes.

        Args:
            config: the training / generation coords config subset
        """
        mode = "manual"
        seed = config["manual_seed"]

        if hasattr(seed, "__int__"):
            seed = int(seed)
            seed = seed % (2 ** 32 - 1)
        else:  # elif not hasattr(seed, "__int__")
            mode = "auto"
            random.seed(None)
            seed = random.randint(0, 2 ** 32 - 1)

        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.rand.mode = mode
        self.rand.seed = seed

    def setup_hw(self, config):
        """Sets up the torch hardware with the given args.

        Set up self.hw and its attributes.

        Args:
            config: the training / generation coords config subset
        """
        gpu_count = config["gpu_count"]

        device_name = "cpu"
        if torch.cuda.is_available() and gpu_count > 0:
            device_name = "cuda:0"
        device = torch.device(device_name)

        self.hw.device = device
        self.hw.gpu_count = gpu_count


class TrainingContext(Context):
    """Training context."""

    class Data(AttrDict):
        """Data info."""

        class TrainValid(AttrDict):
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

    class Mods(AttrDict):
        """Modelers info."""

        d = None
        """Discriminator modeler instance."""
        g = None
        """Generator modeler instance."""

    class Labels(AttrDict):
        """Target labels info."""

        real = None
        """Real label."""
        fake = None
        """Fake label."""

    class Loops(AttrDict):
        """Loop controls info."""

        class IterationEpochBatch(AttrDict):
            """Iteration epoch batch info."""

            count = None
            """Count."""
            index = None
            """Current index."""

        class RollbackEarlystop(AttrDict):
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

    class Latest(AttrDict):
        """Latest batch result info."""

        dx = None
        """Average D(X)."""
        dgz = None
        """Average D(G(Z)) when training D."""
        dgz2 = None
        """Average D(G(Z)) when training G."""
        ld = None
        """L(D), the loss of D, with range [0, 200]."""
        lg = None
        """L(G), the loss of G, with range [0, 100]."""

    class Losses(AttrDict):
        """Epoch losses info."""

        class Subset(AttrDict):
            """Data subset losses info."""

            d = None
            """Discriminator epoch losses."""
            g = None
            """Generator epoch losses."""

        train = Subset()
        """Training losses info."""
        valid = Subset()
        """Validation losses info."""

    class Bests(AttrDict):
        """Best losses info."""

        d = None
        """Discriminator best loss."""
        g = None
        """Generator best loss."""

    class Rbs(AttrDict):
        """Rollback epochs info."""

        d = None
        """Discriminator rollback epoch number list."""
        g = None
        """Generator rollback epoch number list."""

    class Noises(AttrDict):
        """Fixed noises info."""
        valid = None
        """Validation noise batch."""
        batch_of_64 = None
        """A batch of 64 fixed noises."""

    class Collapses(AttrDict):
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
        self.data = TrainingContext.Data()
        """Data info attr dict."""
        self.mods = TrainingContext.Mods()
        """Modelers info attr dict."""
        self.mode = None
        """Training mode name."""
        self.labels = TrainingContext.Labels()
        """Target labels info attr dict."""
        self.loops = TrainingContext.Loops()
        """Loop controls info attr dict."""
        self.latest = TrainingContext.Latest()
        """Latest batch result info attr dict."""
        self.losses = TrainingContext.Losses()
        """Epoch losses info attr dict."""
        self.bests = TrainingContext.Bests()
        """Best losses info attr dict."""
        self.rbs = TrainingContext.Rbs()
        """Rollback epochs info attr dict."""
        self.noises = TrainingContext.Noises()
        """Fixed noises info attr dict."""
        self.collapses = TrainingContext.Collapses()
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

        norm_list = [0.5 for _ in range(channel_count)]
        dataset = datasets.ImageFolder(
            root=path,
            transform=transforms.Compose([
                transforms.Resize(image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize(norm_list, norm_list)
            ])
        )

        size = len(dataset)
        subset_ratio = numpy.array([train_weight, valid_weight])
        subset_ratio = subset_ratio / subset_ratio.sum()
        prop_to_use = percents_to_use / 100
        prop_to_use = utils.bound_num(prop_to_use, 0, 1)
        size_to_use = int(prop_to_use * size)
        train_start, train_end = 0, int(subset_ratio[0] * size_to_use)
        valid_start, valid_end = train_end, size_to_use
        indices = list(range(size_to_use))
        random.shuffle(indices)
        train_indices = indices[train_start: train_end]
        valid_indices = indices[valid_start: valid_end]
        train_set = data.Subset(dataset, train_indices)
        valid_set = data.Subset(dataset, valid_indices)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
        train_size = len(train_set)
        train_batch_count = len(train_loader)
        valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
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

    def setup_mods(self, config):
        """Sets up self.mods and its attributes with the given args.

        Args:
            config: the modelers config

        Raises:
            ValueError: if self.hw.device is None
        """
        if self.hw.device is None:
            raise ValueError("self.hw.device cannot be None")

        d_config = config["discriminator"]
        g_config = config["generator"]
        loss_func = nn.BCELoss()

        d = modelers.DModeler(config.model_path, d_config, self.hw.device, self.hw.gpu_count, loss_func)
        g = modelers.GModeler(config.model_path, g_config, self.hw.device, self.hw.gpu_count, loss_func)

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

    def setup_labels(self):
        """Sets up the labels."""
        self.labels.real = float(1)
        self.labels.fake = float(0)

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
        self.latest.dgz = None
        self.latest.dgz2 = None
        self.latest.ld = None
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
        for _ in range(self.data.valid.batch_count):
            noise_batch = self.mods.g.generate_noises(self.data.batch_size)
            valid.append(noise_batch)

        batch_of_64 = self.mods.g.generate_noises(64)

        self.noises.valid = valid
        self.noises.batch_of_64 = batch_of_64


class GenerationContext(Context):
    """Generation context."""

    class Images(utils.AttrDict):
        """Images info."""

        count = None
        """Image count."""
        per_batch = None
        """Image count of each batch."""
        to_save = None
        """Images to save."""

    class Grids(utils.AttrDict):
        """Grid mode info."""

        enabled = None
        """Whether grid mode is enabled."""
        size_each = None
        """Size of each grid."""
        padding = None
        """Grid padding width."""

    class BatchProg(utils.AttrDict):
        """Generation batch progress info."""

        count = None
        """Count."""
        index = None
        """Current index."""

    def __init__(self):
        """Inits self."""
        super().__init__()
        self.g = None
        """Generator modeler instance."""
        self.images = GenerationContext.Images()
        """Images info attr dict."""
        self.grids = GenerationContext.Grids()
        """Grid mode info attr dict."""
        self.noise_batches = None
        """Noise batch list."""
        self.batch_prog = GenerationContext.BatchProg()
        """Generation batch progress info."""

    def setup_all(self, coords_config, modelers_config):
        """Sets up the entire context.

        Sets up self.g, self.images, self.grids, and self.noises.

        Args:
            coords_config: the coords config
            modelers_config: the modelers config

        Raises:
            ValueError: if self.hw.device is None
        """
        if self.hw.device is None:
            raise ValueError("self.hw.device cannot be None")

        # Setup self.g
        model_path = modelers_config.model_path
        config = modelers_config["generator"]
        loss_func = nn.BCELoss()
        self.g = modelers.GModeler(model_path, config, self.hw.device, self.hw.gpu_count, loss_func, train=False)
        self.g.load()

        # Setup self.images
        config = coords_config["generation"]
        self.images.count = config["image_count"]
        self.images.per_batch = config["images_per_batch"]
        self.images.to_save = []

        # Setup self.grids
        config = config["grid_mode"]
        self.grids.enabled = config["enabled"]
        self.grids.size_each = config["images_per_grid"]
        self.grids.padding = config["padding"]

        # Setup self.noise_batches
        noises_left = self.images.count
        noise_batches = []
        while noises_left > 0:
            noise_count = min(noises_left, self.images.per_batch)
            noise_batch = self.g.generate_noises(noise_count)
            noise_batches.append(noise_batch)
            noises_left -= noise_count
        self.noise_batches = noise_batches

        # Setup self.batch_prog
        self.batch_prog.count = len(self.noise_batches)
        self.batch_prog.index = 0
