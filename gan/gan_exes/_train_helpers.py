"""Helpers for the main function in the train module."""

# import system and third party modules
import pathlib
import random
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as pyplot
import numpy
import torchvision.utils as vision_utils
import torch.nn as nn
import torch.optim as optim

# import custom modules
import gan_libs.configs as configs
import gan_libs.neural_networks as neural_networks


class Vars:
    """Variables."""
    train_config = None
    model_config = None

    data_loader = None

    gpu_count = None
    device = None

    result_path = None

    # g_file_name: generator file name
    g_file_name = None
    # d_file_name: discriminator file name
    d_file_name = None
    training_mode = None
    # g: generator
    g = None
    # d: discriminator
    d = None
    fake_label = None
    real_label = None
    criterion = None
    # g_optim: generator optimizer
    g_optim = None
    # d_optim: discriminator optimizer
    d_optim = None
    # z_size: generator input size
    z_size = None
    fixed_noise = None

    generated_images = None
    # g_losses: generator losses
    g_losses = None
    # d_losses: discriminator losses
    d_losses = None


class Funcs:
    """Functions."""
    @classmethod
    def load_configs(cls):
        """Loads the config files needed for the training process."""
        print("____ load_configs ____")

        # Load train config
        Vars.train_config = configs.TrainConfig()
        print("train_config.json location: {}".format(
            Vars.train_config.location
        ))
        Vars.train_config.load()
        print("Loaded train_config.json")

        # Load model config
        Vars.model_config = configs.ModelConfig()
        Vars.model_config.location = str(
            pathlib.Path(
                Vars.train_config.items["model_path"] + "/model_config.json"
            ).resolve()
        )
        print("model_config.json location: {}".format(
            Vars.model_config.location
        ))
        Vars.model_config.load()
        print("Loaded model_config.json")

        print()

    @classmethod
    def set_random_seeds(cls):
        """Sets the seeds of python and torch random number generator."""
        print("____ set_random_seeds ____")

        manual_seed = Vars.model_config.items["training"]["manual_seed"]
        print("Random seed: {}".format(manual_seed))

        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        print("Completed setting random seeds")

        print()

    @classmethod
    def init_data_loader(cls):
        """Initializes the training dataset loader."""
        print("____ init_data_loader ____")

        data_path = Vars.train_config.items["data_path"]
        image_resolution = Vars.model_config.items["training_set"][
            "image_resolution"
        ]
        data_set = datasets.ImageFolder(
            root=data_path,
            transform=transforms.Compose([
                transforms.Resize(image_resolution),
                transforms.CenterCrop(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )

        images_per_batch = Vars.model_config.items["training_set"][
            "images_per_batch"
        ]
        loader_worker_count = Vars.model_config.items["training_set"][
            "loader_worker_count"
        ]
        Vars.data_loader = data.DataLoader(
            data_set,
            batch_size=images_per_batch,
            shuffle=True,
            num_workers=loader_worker_count
        )
        print("Initialized training set loader")

        print()

    @classmethod
    def detect_device(cls):
        """Detects the training device."""
        print("____ detect_device ____")

        Vars.gpu_count = Vars.model_config.items["training"]["gpu_count"]
        print("GPU Count: {}".format(Vars.gpu_count))

        if torch.cuda.is_available() and Vars.gpu_count > 0:
            device_name = "cuda:0"
        else:
            device_name = "cpu"
        Vars.device = torch.device(device_name)
        print("Training device: {}".format(device_name))

        print()

    @classmethod
    def init_result_folder(cls):
        """Initializes the training result folder."""
        print("____ init_result_folder ____")

        Vars.result_path = str(
            pathlib.Path(
                Vars.train_config.items["model_path"] + "/Training-Results"
            ).absolute()
        )
        pathlib.Path(Vars.result_path).mkdir(exist_ok=True)
        Vars.result_path = str(pathlib.Path(Vars.result_path).resolve())
        print("Result folder: {}".format(Vars.result_path))

        print()

    @classmethod
    def plot_training_images(cls):
        """Plot the first batch of training images."""
        print("____ plot_training_images ____")

        first_batch = next(iter(Vars.data_loader))
        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        pyplot.title("Training Images")

        pyplot.imshow(numpy.transpose(
            vision_utils.make_grid(
                first_batch[0].to(Vars.device)[:64],
                padding=2,
                normalize=True
            ).cpu(),
            (1, 2, 0)
        ))

        plot_location = str(
            pathlib.Path(
                Vars.result_path + "/training_images.jpg"
            ).absolute()
        )
        pyplot.savefig(plot_location)
        print("Plotted training images")
        print("Plot location: {}".format(plot_location))

        print()

    @classmethod
    def _save_model(cls, model, file_name):
        location = str(
            pathlib.Path(
                Vars.train_config.items["model_path"] + "/" + file_name
            ).absolute()
        )
        file = open(location, "w+")
        torch.save(model.state_dict(), location)
        file.close()

    @classmethod
    def _load_model(cls, file_name, model):
        location = str(
            pathlib.Path(
                Vars.train_config.items["model_path"] + "/" + file_name
            ).absolute()
        )
        model.load_state_dict(torch.load(location))
        model.eval()

    @classmethod
    def _setup_neural_networks(cls):
        # Setup generator and discriminator file names
        Vars.g_file_name = "generator.pt"
        Vars.d_file_name = "discriminator.pt"

        device = Vars.device
        gpu_count = Vars.gpu_count
        # Create the generator and discriminator
        # g: generator
        g = neural_networks.Generator(Vars.model_config.location).to(device)
        if device.type == "cuda" and gpu_count > 1:
            g = nn.DataParallel(g, list(range(gpu_count)))
        g.apply(neural_networks.Utils.init_weights)
        # d: discriminator
        d = neural_networks.Discriminator(
            Vars.model_config.location
        ).to(device)
        if device.type == "cuda" and gpu_count > 1:
            d = nn.DataParallel(d, list(range(gpu_count)))
        d.apply(neural_networks.Utils.init_weights)

        # Save or load the models based on the training mode
        Vars.training_mode = Vars.model_config.items["training"]["mode"]
        print("Training mode: {}".format(Vars.training_mode))
        if Vars.training_mode == "new":
            Funcs._save_model(g, Vars.g_file_name)
            Funcs._save_model(d, Vars.d_file_name)
        elif Vars.training_mode == "resume":
            Funcs._load_model(Vars.g_file_name, g)
            Funcs._load_model(Vars.d_file_name, d)
        else:
            raise ValueError(
                "Unknown training mode: {}".format(Vars.training_mode)
            )

        # Complete generator and discriminator setups
        Vars.g = g
        print("Completed generator setup")
        print("==== Generator Structure ====")
        print(Vars.g)
        print()
        Vars.d = d
        print("Completed discriminator setup")
        print("==== Discriminator Structure ====")
        print(Vars.d)
        print()

    @classmethod
    def setup_training(cls):
        """Sets up the training environment."""
        print("____ setup_training ____")

        # Setup the neural networks
        Funcs._setup_neural_networks()

        # Setup the labels
        Vars.fake_label = 0.0
        Vars.real_label = 1.0
        print("Completed label setup")

        # Setup the BCELoss function
        Vars.criterion = nn.BCELoss()
        print("Completed criterion setup")

        # Setup Adam optimizers for the generator and discriminator
        # adam_lr: adam optimizer learning rate
        adam_lr = Vars.model_config.items["adam_optimizer"]["learning_rate"]
        adam_beta1 = Vars.model_config.items["adam_optimizer"]["beta1"]
        adam_beta2 = Vars.model_config.items["adam_optimizer"]["beta2"]
        Vars.g_optim = optim.Adam(
            Vars.g.parameters(),
            lr=adam_lr,
            betas=(adam_beta1, adam_beta2)
        )
        Vars.d_optim = optim.Adam(
            Vars.d.parameters(),
            lr=adam_lr,
            betas=(adam_beta1, adam_beta2)
        )
        print("Completed optimizer setup")

        # Setup generator input
        Vars.z_size = Vars.model_config.items["model"]["generator_input_size"]
        Vars.fixed_noise = torch.randn(
            64, Vars.z_size, 1, 1, device=Vars.device
        )
        print("Completed generator input setup")

        print("^^^^ setup_training ^^^^")
        print()

    @classmethod
    def _train_neural_networks(cls, real_batch):
        """Returns: (d_loss, g_loss, d_x, d_g_z1, d_g_z2)"""
        d = Vars.d
        device = Vars.device
        real_label = Vars.real_label
        criterion = Vars.criterion
        fake_label = Vars.fake_label
        g = Vars.g

        # ==== Update the discriminator ====
        # Maximize log(D(x)) + log( 1 - D(G(z)) )
        # Train the discriminator with all-real batch
        d.zero_grad()
        # Format the real batch
        real_batch = real_batch[0].to(device)
        batch_size = real_batch.size(0)
        label = torch.full(
            (batch_size,), real_label, dtype=torch.float, device=device
        )
        # Forward pass the real batch through the discriminator
        output = d(real_batch).view(-1)
        # Find discriminator loss on the real batch
        # d_loss_on_real: discriminator loss on the real batch
        d_loss_on_real = criterion(output, label)
        # Find the gradients in backward pass
        d_loss_on_real.backward()
        # d_x: D(x)
        d_x = output.mean().item()

        # Train discriminator with all-fake batch
        # Generate generator input noise (latent vectors)
        noise = torch.randn(batch_size, Vars.z_size, 1, 1, device=device)
        # Generate the fake batch with the generator
        fake_batch = g(noise)
        label.fill_(fake_label)
        # Forward pass the fake batch through the discriminator
        output = d(fake_batch.detach()).view(-1)
        # Find the discriminator loss on the fake batch
        # d_loss_on_fake: discriminator loss on the fake batch
        d_loss_on_fake = criterion(output, label)
        # Find the gradients for the fake batch,
        # summing up with the gradients on the real batch
        d_loss_on_fake.backward()
        # d_g_z1: D(G(z1))
        d_g_z1 = output.mean().item()

        # d_loss: discriminator loss
        d_loss = d_loss_on_real + d_loss_on_fake
        # Optimize the discriminator
        Vars.d_optim.step()

        # ==== Update the generator ====
        # Maximize log( D(G(z)) )
        g.zero_grad()
        # Use real label when finding the generator loss
        label.fill_(real_label)
        # Forward pass the fake batch through the discriminator,
        # since we already updated the discriminator
        output = d(fake_batch).view(-1)

        # g_loss: generator loss
        g_loss = criterion(output, label)
        # Find the gradients in backward pass
        g_loss.backward()
        # d_g_z2: D(G(z2))
        d_g_z2 = output.mean().item()
        # Optimize the generator
        Vars.g_optim.step()

        return d_loss, g_loss, d_x, d_g_z1, d_g_z2

    @classmethod
    def _print_training_stats(
        cls, index, data_loader, d_loss, g_loss, d_x, d_g_z1, d_g_z2
    ):
        """Params:
            index, data_loader,
            d_loss: discriminator loss
            g_loss: generator loss
            d_x:    D(x)
            d_g_z1: D(G(z1))
            d_g_z2: D(G(z2))
        """
        stats = "[batch {} of {}] ".format(index + 1, len(data_loader))
        stats += "d_loss: {:.4f} g_loss: {:.4f} ".format(
            d_loss.item(), g_loss.item()
        )
        stats += "D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}".format(
            d_x, d_g_z1, d_g_z2
        )
        print(stats)

    @classmethod
    def _record_generated_images(cls, g):
        """Params: g: generator"""
        with torch.no_grad():
            fake_images = g(Vars.fixed_noise).detach().cpu()
        Vars.generated_images.append(vision_utils.make_grid(
            fake_images, padding=2, normalize=True
        ))

    @classmethod
    def start_training(cls):
        """Starts the training"""
        print("____ start_training ____")

        epoch_count = Vars.model_config.items["training"]["epoch_count"]
        # g_file_name: generator file name
        g_file_name = Vars.g_file_name
        # g: generator
        g = Vars.g
        # d_file_name: discriminator file name
        d_file_name = Vars.d_file_name
        # d: discriminator
        d = Vars.d
        data_loader = Vars.data_loader
        batch_count = Vars.model_config.items["training_set"]["batch_count"]

        Vars.generated_images = []
        Vars.g_losses = []
        Vars.d_losses = []

        # curr_iter: current iteration
        curr_iter = 0

        for epoch in range(epoch_count):
            print("==== Epoch {} of {} ====".format(epoch + 1, epoch_count))
            Funcs._load_model(g_file_name, g)
            Funcs._load_model(d_file_name, d)
            for index, real_batch in enumerate(data_loader):
                # Break out of the loop when reaching the batch count limit
                if index >= batch_count:
                    break
                # Train the neural networks
                d_loss, g_loss, d_x, d_g_z1, d_g_z2 = \
                    Funcs._train_neural_networks(real_batch)
                # Print the training stats
                if index == 0 or (index + 1) % 50 == 0:
                    Funcs._print_training_stats(
                        index, data_loader, d_loss, g_loss, d_x, d_g_z1, d_g_z2
                    )
                # Record the generator and discriminator losses
                Vars.g_losses.append(g_loss.item())
                Vars.d_losses.append(d_loss.item())
                # Record the generated images
                if curr_iter == 0 or (curr_iter + 1) % 500 == 0 or (
                    epoch == epoch_count - 1 and index == len(data_loader) - 1
                ):
                    Funcs._record_generated_images(g)
                curr_iter += 1
            Funcs._save_model(g, g_file_name)
            Funcs._save_model(d, d_file_name)

        print("^^^^ start_training ^^^^")
        print()

    @classmethod
    def plot_losses(cls):
        print("____ plot_losses ____")

        pyplot.figure(figsize=(10, 5))
        pyplot.title("Generator and Discriminator Training Losses")
        pyplot.plot(Vars.g_losses, label="Generator")
        pyplot.plot(Vars.d_losses, label="Discriminator")
        pyplot.xlabel("Iteration No.")
        pyplot.ylabel("Loss")
        pyplot.legend()
        plot_location = str(
            pathlib.Path(Vars.result_path + "/losses-plot.jpg").absolute()
        )
        pyplot.savefig(plot_location)
        print("Loss plot location: {}".format(plot_location))

        print()

    @classmethod
    def plot_generated_images(cls):
        print("____ plot_generated_images ____")

        pyplot.figure(figsize=(8, 8))
        pyplot.axis("off")
        generated_images_path = str(
            pathlib.Path(Vars.result_path + "/Generated-Images").absolute()
        )
        pathlib.Path(generated_images_path).mkdir(exist_ok=True)
        for index, image in enumerate(Vars.generated_images):
            pyplot.imshow(numpy.transpose(image, (1, 2, 0)), animated=True)
            plot_location = str(
                pathlib.Path(
                    generated_images_path + "/generated-image_" +
                    str(index + 1) + ".jpg"
                ).absolute()
            )
            pyplot.savefig(plot_location)
        print("Generated images path: {}".format(generated_images_path))

        print()

    @classmethod
    def plot_real_and_fake(cls):
        print("____ plot_real_and_fake ____")

        first_batch = next(iter(Vars.data_loader))

        # Plot the real (training) images
        pyplot.figure(figsize=(15, 15))
        pyplot.subplot(1, 2, 1)
        pyplot.axis("off")
        pyplot.title("Real (Training) Images")
        pyplot.imshow(numpy.transpose(
            vision_utils.make_grid(
                first_batch[0].to(Vars.device)[:64],
                padding=5,
                normalize=True
            ).cpu(),
            (1, 2, 0)
        ))

        # Plot the fake (generated) images from the last epoch
        pyplot.subplot(1, 2, 2)
        pyplot.axis("off")
        pyplot.title("Fake (Generated) Images")
        pyplot.imshow(numpy.transpose(Vars.generated_images[-1], (1, 2, 0)))

        plot_location = str(
            pathlib.Path(Vars.result_path + "/real-and-fake.jpg").absolute()
        )
        pyplot.savefig(plot_location)
        print("Real and fake plot location: {}".format(plot_location))

        print()
