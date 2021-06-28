"""Module of the coordinator classes."""

import sys

from gan_libs import configs
from gan_libs import contexts
from gan_libs import results
from gan_libs import utils


class TCoord:
    """Training coordinator.

    Attributes:
        d_path: the data path
        m_path: the model path
        c_config: the coords config
        m_config: the modelers config
        rst: the result
        ctx: the context
        rst_ready: whether the results are ready
        ctx_ready: whether the context is ready
    """

    def __init__(self, d_path, m_path):
        """Inits self with the given args.

        Args:
            d_path: the data path
            m_path: the model path
        """
        self.d_path = d_path
        self.m_path = m_path
        self.c_config = None
        self.m_config = None
        self.rst = None
        self.ctx = None

    def setup_results(self):
        """Sets up the results, self.rst."""
        path = utils.concat_paths(self.m_path, "Training-Results")
        log = sys.stdout
        self.rst = results.TResults(path, log)
        self.rst.init_folders()
        self.rst_ready = True
        self.rst.logln("Completed results setup")

    def setup_context(self):
        """Sets up the context, self.ctx."""
        if not self.rst_ready:
            self.setup_results()
        self.c_config = configs.CoordsConfig(self.m_path)
        self.c_config.load()
        self.m_config = configs.ModelersConfig(self.m_path)
        self.m_config.load()
        self.rst.log_configs(self.c_config, self.m_config)
        self.ctx = contexts.TContext()
        config = self.c_config["training"]
        self.ctx.set_rand_seeds(config)
        self.rst.log_rand_seeds(self.ctx)
        self.ctx.setup_device(config)
        self.rst.log_device(self.ctx)
        self.ctx.setup_data_loaders(self.d_path, config)
        self.rst.log_data_loaders(self.ctx)
        config = self.m_config
        self.ctx.setup_modelers(config)
        self.rst.log_modelers(self.ctx)
        config = self.c_config["training"]
        self.ctx.setup_mode(config)
        self.rst.log_mode(self.ctx)
        self.ctx.setup_labels()
        self.ctx.setup_loops(config)
        self.ctx.setup_stats()
        self.rst.logln("Completed context setup")

    def train_d(self):
        """Trains the discriminator with the training set."""
        print("== {}.{}. Train D ==".format(self.iter + 1, self.epoch + 1))
        self.t_index = 0
        for real_batch in self.t_loader:
            real_batch = real_batch[0]
            real_output, real_loss = self.d.train(real_batch, self.real)
            fake_batch = self.g.test(self.batch_size)
            fake_output, fake_loss = self.d.train(fake_batch, self.fake)
            loss = real_loss + fake_loss
            if self.t_index == 0 or (self.t_index + 1) % 50 == 0 or self.t_index == self.t_batch_count - 1:
                print(
                    "Batch {}/{}: Average D(X): {:.4f} Averge D(G(Z)): {:.4f} Loss(D): {:.4f}".format(
                        self.t_index + 1, self.t_batch_count,
                        real_output.mean().item(), fake_output.mean().item(),
                        loss
                    )
                )
            self.t_index += 1

    def train_g(self):
        """Trains the generator with the training set."""
        print("== {}.{}. Train G ==".format(self.iter + 1, self.epoch + 1))
        self.t_index = 0
        while self.t_index < self.t_batch_count:
            output, loss = self.g.train(
                self.d.model, 2 * self.batch_size, self.real)
            if self.t_index == 0 or (self.t_index + 1) % 50 == 0 or self.t_index == self.t_batch_count - 1:
                print(
                    "Batch {}/{}: Averge D(G(Z)): {:.4f} Loss(G): {:.4f}".format(
                        self.t_index + 1, self.t_batch_count, output.mean().item(), loss
                    )
                )
            self.t_index += 1

    def train_d_and_g(self):
        print("== {}.{}. Train D and G ==".format(
            self.iter + 1, self.epoch + 1))
        self.t_index = 0
        for real_batch in self.t_loader:
            real_batch = real_batch[0]
            real_output, real_loss = self.d.train(real_batch, self.real)
            fake_batch = self.g.test(self.batch_size)
            fake_output, fake_loss = self.d.train(fake_batch, self.fake)
            d_loss = real_loss + fake_loss

            g_output, g_loss = self.g.train(
                self.d.model, 2 * self.batch_size, self.real)

            if self.t_index == 0 or (self.t_index + 1) % 50 == 0 or self.t_index == self.t_batch_count - 1:
                print(
                    "Batch {}/{}:\n"
                    "\tTrain D: Average D(X): {:.4f} Averge D(G(Z)): {:.4f} Loss(D): {:.4f}\n"
                    "\tTrain G: Average D(G(Z)): {:.4f} Loss(G): {:.4f}".format(
                        self.t_index + 1, self.t_batch_count,
                        real_output.mean().item(), fake_output.mean().item(), d_loss,
                        g_output.mean().item(), g_loss
                    )
                )
            self.t_index += 1

    def validate_d(self):
        """Validate the discriminator with the validation set."""
        print("== {}.{}. Validate D ==".format(self.iter + 1, self.epoch + 1))
        self.v_index = 0
        losses = numpy.array([])
        for real_batch in self.v_loader:
            real_batch = real_batch[0]
            real_output, real_loss = self.d.validate(real_batch, self.real)
            fake_batch = self.g.test(self.batch_size)
            fake_output, fake_loss = self.d.validate(fake_batch, self.fake)
            loss = real_loss + fake_loss
            losses = numpy.append(losses, [loss.cpu()])
            if self.v_index == 0 or (self.v_index + 1) % 50 == 0 or self.v_index == self.v_batch_count - 1:
                print(
                    "Batch {}/{}: Average D(X): {:.4f} Averge D(G(Z)): {:.4f} Loss(D): {:.4f}".format(
                        self.v_index + 1, self.v_batch_count,
                        real_output.mean().item(), fake_output.mean().item(),
                        loss
                    )
                )
            self.v_index += 1
        average_loss = losses.mean()
        if average_loss < self.d_iter_best_loss:
            self.d_iter_best_loss = average_loss

    def validate_g(self):
        """Validate the generator with the validation set."""
        print("== {}.{}. Validate G ==".format(self.iter + 1, self.epoch + 1))
        self.v_index = 0
        losses = numpy.array([])
        while self.v_index < self.v_batch_count:
            output, loss = self.g.validate(
                self.d.model, 2 * self.batch_size, self.real)
            losses = numpy.append(losses, [loss.cpu()])
            if self.v_index == 0 or (self.v_index + 1) % 50 == 1 or self.v_index == self.v_batch_count - 1:
                print(
                    "Batch {}/{}: Averge D(G(Z)): {:.4f} Loss(G): {:.4f}".format(
                        self.v_index + 1, self.v_batch_count, output.mean().item(), loss
                    )
                )
            self.v_index += 1
        average_loss = losses.mean()
        if average_loss < self.g_iter_best_loss:
            self.g_iter_best_loss = average_loss

    def run_epoch(self):
        """Runs an epoch."""
        print("==== {}. Epoch {}/{} ====".format(
            self.iter + 1, self.epoch + 1, self.epoch_count
        ))

        self.dt_losses[-1].append([])
        self.dv_losses[-1].append([])
        self.gt_losses[-1].append([])
        self.gv_losses[-1].append([])

        # self.train_d()
        # self.train_g()
        self.train_d_and_g()

        results.save_generated_images(self.g.model, self.g.gen_noises(
            64), self.iter, self.epoch, self.generated_images_path)

        self.validate_d()
        self.validate_g()

        print()

    def run_iter(self):
        """Runs an iteration."""
        print("==== ==== Iter {}/{} ==== ====".format(self.iter + 1, self.iter_count))

        self.dt_losses.append([])
        self.dv_losses.append([])
        self.gt_losses.append([])
        self.gv_losses.append([])
        self.d_iter_best_loss = sys.maxsize
        self.g_iter_best_loss = sys.maxsize

        self.epoch = 0
        while self.epoch < self.epoch_count:
            self.run_epoch()
            self.epoch += 1

        if self.d_iter_best_loss < self.d_best_loss:
            self.d.save()
            self.d_best_loss = self.d_iter_best_loss
            print("Saved D")
        else:
            self.d.rollback()
            print("Rollbacked (count: {}) D".format(self.d.rollbacks))

        if self.g_iter_best_loss < self.g_best_loss:
            self.g.save()
            self.g_best_loss = self.g_iter_best_loss
            print("Saved G")
        else:
            self.g.rollback()
            print("Rollbacked (count: {}) G".format(self.g.rollbacks))

        print()

    def start_training(self):
        """Starts the training."""
        print("____ Training ____")

        if not self.completed_context_setup:
            self.setup_context()

        results.save_training_images(
            next(iter(self.t_loader)), self.results_path, self.device
        )
        results.save_validation_images(
            next(iter(self.v_loader)), self.results_path, self.device
        )

        self.iter = 0
        while self.iter < self.iter_count:
            self.run_iter()
            self.iter += 1

        print("^^^^ Training ^^^^")
        print()
