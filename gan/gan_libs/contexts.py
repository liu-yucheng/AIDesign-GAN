"""Module of the context classes."""

from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import numpy
import random
import sys
import torch

from gan_libs import modelers
from gan_libs import utils


class Context:
    """Super class of the context classes.

    Attributes:
        hw: the hardware attr dict
        hw.dev: the torch device to be used
        hw.gpu_cnt: the number of GPUs to be used
        rand: the random number generators attr dict
        rand.mode: the mode of the random seed
        rand.seed: the value of the random seed
    """

    def __init__(self):
        """Inits self."""
        self.hw = None
        self.rand = None

    def set_rand_seeds(self, config):
        """Sets the random seeds with the given args.

        Sets seeds for numpy, random and torch.

        Args:
            config: the training/generation coords config subset
        """
        mode = "manual"
        val = config["manual_seed"]
        if val is None:
            mode = "auto"
            random.seed(None)
            val = random.randint(0, 2 ** 32 - 1)
        numpy.random.seed(val)
        random.seed(val)
        torch.manual_seed(val)
        self.rand = utils.AttrDict()
        self.rand.mode = mode
        self.rand.seed = val

    def setup_device(self, config):
        """Sets up the torch device with the given args.

        Sets self.env.dev and self.env.gpu_count.

        Args:
            config: the training/generation coords config subset
        """
        gpu_cnt = config["gpu_count"]
        dev_name = "cpu"
        if torch.cuda.is_available() and gpu_cnt > 0:
            dev_name = "cuda:0"
        dev = torch.device(dev_name)
        self.hw = utils.AttrDict()
        self.hw.dev = dev
        self.hw.gpu_cnt = gpu_cnt


class TContext(Context):
    """Training context.

    Attributes:
        data: the data attr dict
        data.tdl: the training dataloader
        data.vdl: the validation dataloader
        data.total_size: the dataset's total size
        data.size_to_use: the size of the portion to be used
        data.t_size: the training set's total size
        data.v_size: the validation set's total size
        data.batch_size: the size of each batch
        data.t_batch_cnt: the number of training batches
        data.v_batch_cnt: the number of validation batches
        mods: the modelers attr dict
        mods.d: the discriminator modeler
        mods.d_str: the discriminator model structure string
        mods.g: the generator modeler
        mods.g_str: the generator model structure string
        mods.loss_fn: the loss function
        mode: the training mode string
        labels: the labels attr dict
        labels.r: the real label
        labels.f: the fake label
        loops: the loop control variables attr dict
        loops.iter_cnt: the total number of iterations
        loops.epoch_cnt: the total number of epochs
        loops.iter: the current iteration number
        loops.epoch: the current epoch number
        loops.t_idx: the current training batch index
        loops.v_idx: the current validation batch index
        outs: the latest output attr dict
        outs.dx: the latest average D(X)
        outs.dgz: the latest average D(G(Z))
        losses: the losses attr dict
        losses.td: discriminator's training losses
        losses.tg: generator's training losses
        losses.vd: discriminator's validation losses
        losses.vg: generator's validation losses
        bests: the best losses attr dict
        bests.d: the best discriminator loss overall
        bests.g: the best generator loss overall
        bests.iter_d: the best discriminator loss for the current iter
        bests.iter_g: the best generator loss for the current iter
    """

    def __init__(self):
        """Inits self."""
        super().__init__()
        self.data = None
        self.mods = None
        self.mode = None
        self.labels = None
        self.loops = None
        self.outs = None
        self.losses = None
        self.bests = None

    def setup_data_loaders(self, path, config):
        """Sets up the data loaders with the given args.

        Args:
            path: the data path
            config: the training coords config subset
        """
        config = config["data_sets"]
        img_res = config["image_resolution"]
        t_weight = config["training_set_weight"]
        v_weight = config["validation_set_weight"]
        percents_to_use = config["percentage_to_use"]
        worker_count = config["loader_worker_count"]
        batch_size = config["images_per_batch"]

        data_set = datasets.ImageFolder(
            root=path, transform=transforms.Compose([
                transforms.Resize(img_res),
                transforms.CenterCrop(img_res),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
        total_size = len(data_set)
        subset_ratio = numpy.array([t_weight, v_weight])
        subset_ratio = subset_ratio / subset_ratio.sum()
        prop_to_use = percents_to_use / 100
        prop_to_use = utils.bound_num(prop_to_use, 0, 1)
        size_to_use = int(prop_to_use * total_size)
        t_start, t_end = 0, int(subset_ratio[0] * size_to_use)
        v_start, v_end = t_end, size_to_use
        t_indices = list(range(t_start, t_end))
        v_indices = list(range(v_start, v_end))
        t_set = data.Subset(data_set, t_indices)
        v_set = data.Subset(data_set, v_indices)
        t_size = len(t_set)
        v_size = len(v_set)
        t_loader = data.DataLoader(
            t_set, batch_size=batch_size, shuffle=True,
            num_workers=worker_count)
        v_loader = data.DataLoader(
            v_set, batch_size=batch_size, shuffle=True,
            num_workers=worker_count)
        t_batch_cnt = len(t_loader)
        v_batch_cnt = len(v_loader)

        self.data = utils.AttrDict()
        self.data.tdl = t_loader
        self.data.vdl = v_loader
        self.data.total_size = total_size
        self.data.size_to_use = size_to_use
        self.data.t_size = t_size
        self.data.v_size = v_size
        self.data.batch_size = batch_size
        self.data.t_batch_cnt = t_batch_cnt
        self.data.v_batch_cnt = v_batch_cnt

    def setup_modelers(self, config):
        """Sets up the modelers with the given args.

        Args:
            config: the modelers config

        Raises:
            ValueError: if self.hw is None
        """
        if self.hw is None:
            raise ValueError("self.hw cannot be None")
        d_config = config["discriminator"]
        g_config = config["generator"]
        loss_fn = nn.BCELoss()
        d = modelers.\
            DModeler(d_config, self.hw.dev, self.hw.gpu_cnt, loss_fn)
        g = modelers.\
            GModeler(g_config, self.hw.dev, self.hw.gpu_cnt, loss_fn)
        self.mods = utils.AttrDict()
        self.mods.d = d
        self.mods.d_str = str(d.model)
        self.mods.g = g
        self.mods.g_str = str(g.model)
        self.mods.loss_fn = loss_fn

    def setup_mode(self, config):
        """Sets up the training mode with the given args.

        Args:
            config: the training coords config subset

        Raises:
            ValueError: if self.mods is None;
                or, if the training mode is unknown (other than "new" and
                "resume")
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
        self.labels.r = 1.0
        self.labels.f = 0.0

    def setup_loops(self, config):
        """Sets up the loop control variables.

        Args:
            config: the training coords config subset
        """
        iter_cnt = config["iteration_count"]
        epoch_cnt = config["epochs_per_iteration"]
        self.loops = utils.AttrDict()
        self.loops.iter_cnt = iter_cnt
        self.loops.epoch_cnt = epoch_cnt
        self.loops.iter = 0
        self.loops.epoch = 0
        self.loops.t_idx = 0
        self.loops.v_idx = 0

    def setup_stats(self):
        """Sets up the statistics.

        The statistics include self.outs, self.losses, and self.bests.
        """
        self.outs = utils.AttrDict()
        self.outs.dx = None
        self.outs.dgz = None
        self.losses = utils.AttrDict()
        self.losses.td = []
        self.losses.tg = []
        self.losses.vd = []
        self.losses.vg = []
        self.bests = utils.AttrDict()
        self.bests.d = sys.maxsize
        self.bests.g = sys.maxsize
        self.bests.iter_d = sys.maxsize
        self.bests.iter_g = sys.maxsize
