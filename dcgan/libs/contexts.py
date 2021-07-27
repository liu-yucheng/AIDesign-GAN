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

from dcgan.libs import modelers
from dcgan.libs import utils


class Context:
    """Super class of the context classes.

    Attributes:
        rand: the random number generators attr dict  \n
            `rand.mode`: the mode of the random seed \n
            `rand.seed`: the value of the random seed \n
        hw: the hardware attr dict \n
            `hw.device`: the torch device to use \n
            `hw.gpu_count`: the number of GPUs to use \n
    """

    def __init__(self):
        """Inits self."""
        self.hw = None
        self.rand = None

    def setup_rand(self, config):
        """Sets the random seeds with the given args.

        Set up seeds for numpy, random and torch. Set up self.rand and its attributes.

        Args:
            config: the training/generation coords config subset
        """
        mode = "manual"
        seed = config["manual_seed"]
        if seed is None:
            mode = "auto"
            random.seed(None)
            seed = random.randint(0, 2 ** 32 - 1)
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.rand = utils.AttrDict()
        self.rand.mode = mode
        self.rand.seed = seed

    def setup_hw(self, config):
        """Sets up the torch hardware with the given args.

        Set up self.hw and its attributes.

        Args:
            config: the training/generation coords config subset
        """
        gpu_count = config["gpu_count"]
        device_name = "cpu"
        if torch.cuda.is_available() and gpu_count > 0:
            device_name = "cuda:0"
        device = torch.device(device_name)
        self.hw = utils.AttrDict()
        self.hw.device = device
        self.hw.gpu_count = gpu_count


class TrainingContext(Context):
    """Training context.

    Attributes:
        data: the data attr dict \n
            `data.size`: the dataset's total size \n
            `data.size_to_use`: the size of the data portion to use \n
            `data.batch_size`: the size of each batch \n
            `data.train.loader`: the training data loader \n
            `data.train.size`: the training set's size \n
            `data.train.batch_count`: the number of training batches \n
            `data.valid.loader`: the validation data loader \n
            `data.valid.size`: the validation set's size \n
            `data.valid.batch_count`: the number of validation batches \n
        mods: the modelers attr dict \n
            `mods.d`: the discriminator modeler \n
            `mods.g`: the generator modeler \n
        mode: the training mode string \n
        labels: the labels attr dict \n
            `labels.real`: the real label \n
            `labels.fake`: the fake label \n
        loops: the loop control variables attr dict \n
            `loops.iter_count`: the total number of iterations \n
            `loops.iter`: the current iteration number \n
            `loops.epoch_count`: the total number of epochs \n
            `loops.epoch`: the current epoch number \n
            `loops.train_index`: the current training batch index \n
            `loops.valid_index`: the current validation batch index \n
            `loops.rb.max`: the maximum rollback count \n
            `loops.rb.d`: the discriminator rollback count \n
            `loops.rb.g`: the generator rollback count \n
            `loops.es.max`: the maximum early stop count \n
            `loops.es.d`: the discriminator early stop count \n
            `loops.es.g`: the generator early stop count \n
        latest: the latest output attr dict \n
            `latest.dx`: the latest average D(X) \n
            `latest.dgz`: the latest average D(G(Z)) \n
            `latest.dgz2`: another latest average D(G(Z)) \n
            `latest.ld`: the latest discriminator loss, L(D) \n
            `latest.lg`: the latest generator loss, L(G) \n
        losses: the losses attr dict \n
            `losses.train.d`: discriminator's training losses \n
            `losses.train.g`: generator's training losses \n
            `losses.valid.d`: discriminator's validation losses \n
            `losses.valid.g`: generator's validation losses \n
        bests: the best losses attr dict \n
            `bests.d`: the best discriminator loss overall \n
            `bests.g`: the best generator loss overall \n
        rbs: the rollbacks attr dict \n
            `rbs.d`: the discriminator rollback list \n
            `rbs.g`: the generator rollback list \n
        collapses: the training collapses attr dict \n
            `collapses.epochs`: the collapse epoch list \n
            `collapses.batch_count`: the collapses batch count in an epoch \n
            `collapses.max_loss`: the collapses maximum loss for each training batch (in a training batch, if any of
                the losses is >= collapses.max_loss, the training batch has collapsed) \n
            `collapses.factor`: the collapse factor (in an epoch, if `collapses.batch_count` is >=
                `int(collapses.factor * data.train.batch_count)`, the epoch has collapsed) \n
            `collapses.max_batch_count`: `int(collapses.factor * data.train.batch_count)` \n
        noises: the fixed generator inputs attr dict \n
            `noises.valid_set`: the validation batches of generator inputs \n
            `noises.batch_64`: a batch of 64 generator inputs \n
    """

    def __init__(self):
        """Inits self."""
        super().__init__()
        self.data = None
        self.mods = None
        self.mode = None
        self.labels = None
        self.loops = None
        self.latest = None
        self.losses = None
        self.bests = None
        self.rbs = None
        self.noises = None
        self.collapses = None

    def setup_data(self, path, config):
        """Sets up self.data and its attributes with the given args.

        Args:
            path: the data path
            config: the training coords config subset
        """
        config = config["data_sets"]
        image_resolution = config["image_resolution"]
        channel_count = config["image_channel_count"]
        train_weight = config["training_set_weight"]
        valid_weight = config["validation_set_weight"]
        percents_to_use = config["percents_to_use"]
        worker_count = config["loader_worker_count"]
        batch_size = config["images_per_batch"]

        norm_list = [0.5 for _ in range(channel_count)]
        data_set = datasets.ImageFolder(
            root=path,
            transform=transforms.Compose([
                transforms.Resize(image_resolution),
                transforms.CenterCrop(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize(norm_list, norm_list)
            ])
        )

        size = len(data_set)
        subset_ratio = numpy.array([train_weight, valid_weight])
        subset_ratio = subset_ratio / subset_ratio.sum()
        prop_to_use = percents_to_use / 100
        prop_to_use = utils.bound_num(prop_to_use, 0, 1)
        size_to_use = int(prop_to_use * size)
        train_start, train_end = 0, int(subset_ratio[0] * size_to_use)
        valid_start, valid_end = train_end, size_to_use
        train_indices = list(range(train_start, train_end))
        valid_indices = list(range(valid_start, valid_end))
        train_set = data.Subset(data_set, train_indices)
        valid_set = data.Subset(data_set, valid_indices)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
        train_size = len(train_set)
        train_batch_count = len(train_loader)
        valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=worker_count)
        valid_size = len(valid_set)
        valid_batch_count = len(valid_loader)

        self.data = utils.AttrDict()
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
            ValueError: if self.hw is None
        """
        if self.hw is None:
            raise ValueError("self.hw cannot be None")
        d_config = config["discriminator"]
        g_config = config["generator"]
        loss_func = nn.BCELoss()
        d = modelers.DModeler(d_config, self.hw.device, self.hw.gpu_count, loss_func)
        g = modelers.GModeler(g_config, self.hw.device, self.hw.gpu_count, loss_func)
        self.mods = utils.AttrDict()
        self.mods.d = d
        self.mods.g = g

    def setup_mode(self, config):
        """Sets up self.mode with the given args.

        Args:
            config: the training coords config subset

        Raises:
            ValueError: if self.mods is None; or, if the training mode is unknown (other than "new" and "resume")
        """
        if self.mods is None:
            raise ValueError("self.mods cannot be None")
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
        self.labels = utils.AttrDict()
        self.labels.real = 1.0
        self.labels.fake = 0.0

    def setup_loops(self, config):
        """Sets up the loop control variables.

        Args:
            config: the training coords config subset
        """
        iter_count = config["iteration_count"]
        epoch_count = config["epochs_per_iteration"]
        max_rb = config["max_rollbacks"]
        max_es = config["max_early_stops"]
        self.loops = utils.AttrDict()
        self.loops.iter_count = iter_count
        self.loops.iter = 0
        self.loops.epoch_count = epoch_count
        self.loops.epoch = 0
        self.loops.train_index = 0
        self.loops.valid_index = 0
        self.loops.rb.max = max_rb
        self.loops.rb.d = 0
        self.loops.rb.g = 0
        self.loops.es.max = max_es
        self.loops.es.d = 0
        self.loops.es.g = 0

    def setup_stats(self):
        """Sets up the statistics.

        The statistics consists of self.latest, self.losses, self.bests, self.rbs, self.collapses, and their
        attributes.

        Raises:
            ValueError: if self.data is None
        """
        if self.data is None:
            raise ValueError("self.data cannot be None")
        self.latest = utils.AttrDict()
        self.latest.dx = None
        self.latest.dgz = None
        self.latest.dgz2 = None
        self.latest.ld = None
        self.latest.lg = None
        self.losses = utils.AttrDict()
        self.losses.train.d = []
        self.losses.train.g = []
        self.losses.valid.d = []
        self.losses.valid.g = []
        self.bests = utils.AttrDict()
        self.bests.d = None
        self.bests.g = None
        self.rbs = utils.AttrDict()
        self.rbs.d = []
        self.rbs.g = []
        self.collapses = utils.AttrDict()
        self.collapses.epochs = []
        self.collapses.batch_count = 0
        self.collapses.max_loss = 100
        self.collapses.factor = 0.67
        self.collapses.max_batch_count = int(self.collapses.factor * self.data.train.batch_count)

    def setup_noises(self):
        """Sets up the noises.

        Raises:
            ValueError: if self.data of self.mods is None
        """
        if self.data is None:
            raise ValueError("self.data cannot be None")
        if self.mods is None:
            raise ValueError("self.mods cannot be None")
        valid_set = []
        for _ in range(self.data.valid.batch_count):
            noise_batch = self.mods.g.generate_noises(self.data.batch_size)
            valid_set.append(noise_batch)
        batch_64 = self.mods.g.generate_noises(64)
        self.noises = utils.AttrDict()
        self.noises.valid_set = valid_set
        self.noises.batch_64 = batch_64


class GenerationContext(Context):
    """Generation context.

    Attributes:
        g: the generator modeler
        images: the images attr dict \n
            `images.count`: the image count
            `images.list`: the images to save
        grids: the grids attr dict \n
            `grids.enabled`: whether the grid mode is enabled \n
            `grids.each_size`: the size of each grid \n
            `grids.padding`: the padding of the grids \n
        noises: the generator inputs
    """

    def __init__(self):
        """Inits self."""
        super().__init__()
        self.g = None
        self.images = None
        self.grids = None
        self.noises = None

    def setup_all(self, coords_config, modelers_config):
        """Sets up the entire context.

        Sets up self.g, self.images, self.grids, and self.noises.

        Args:
            coords_config: the coords config
            modelers_config: the modelers config

        Raises:
            ValueError: if self.hw is None
        """
        if self.hw is None:
            raise ValueError("self.hw cannot be None")
        config = modelers_config["generator"]
        loss_func = nn.BCELoss()
        self.g = modelers.GModeler(config, self.hw.device, self.hw.gpu_count, loss_func, training=False)
        self.g.load()
        config = coords_config["generation"]
        self.images = utils.AttrDict()
        self.images.count = config["image_count"]
        self.images.list = None
        config = config["grid_mode"]
        self.grids = utils.AttrDict()
        self.grids.enabled = config["enabled"]
        self.grids.each_size = config["images_per_grid"]
        self.grids.padding = config["padding"]
        self.noises = self.g.generate_noises(self.images.count)
