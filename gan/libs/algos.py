"""Module of the training algos (algorithms)."""

import numpy


class Algo:
    """Super class of the algo classes.

    Attributes:
        context: the binded training context
        results: the binded training results
    """

    def __init__(self):
        """Inits self."""
        self.context = None
        self.results = None

    def bind_context_and_results(self, context, results):
        """Binds the context and results.

        Args:
            context: the context to bind
            results: the results to bind
        """
        self.context = context
        self.results = results

    def check_context_and_results(self):
        """Checks if the context and results are binded.

        Raises:
            ValueError: if the context or results are None
        """
        if self.context is None:
            raise ValueError("self.context cannot be None")
        if self.results is None:
            raise ValueError("self.results cannot be None")

    def start_training(self):
        """Starts the training algorithm.

        Raises:
            NotImplementedError: always
        """
        raise NotImplementedError("start_training not implemented")


class IterLevelAlgo(Algo):
    """Algorithm that trains D and G together at iter level."""

    def __init__(self):
        """Inits self."""
        super().__init__()

    def train_d(self):
        """Trains D with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training D")
        lds_on_real = []
        c.loops.train_index = 0
        for real_batch in c.data.train.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.train(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            lds_on_real.append(ld)
            r.log_batch("d", "tr")
            c.loops.train_index += 1
        lds_on_fake = []
        c.loops.train_index = 0
        while c.loops.train_index < c.data.train.batch_count:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.train(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            lds_on_fake.append(ld)
            r.log_batch("d", "tf")
            c.loops.train_index += 1
        lds = []
        for index in range(c.data.train.batch_count):
            lds.append(lds_on_real[index] + lds_on_fake[index])
        epoch_ld = numpy.array(lds).mean()
        c.losses.train.d.append(epoch_ld)
        r.log_epoch_loss("td")

    def valid_d(self):
        """Validates D with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating D")
        lds_on_real = []
        c.loops.valid_index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            lds_on_real.append(ld)
            r.log_batch("d", "vr")
            c.loops.valid_index += 1
        lds_on_fake = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            lds_on_fake.append(ld)
            r.log_batch("d", "vf")
            c.loops.valid_index += 1
        lds = []
        for index in range(c.data.valid.batch_count):
            lds.append(lds_on_real[index] + lds_on_fake[index])
        epoch_ld = numpy.array(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")

    def save_best_d(self):
        """Saves the D model that performs the best."""
        r = self.results
        c = self.context
        r.log_best_losses("d")
        curr_ld = c.losses.valid.d[-1]
        if c.bests.d is None or curr_ld <= c.bests.d:
            c.bests.d = curr_ld
            c.mods.d.save()
            r.log_model_action("save", "d")
        elif c.loops.es.d < c.loops.es.max:
            c.loops.es.d += 1
            r.log_model_action("es", "d")
        elif c.loops.rb.d < c.loops.rb.max:
            c.loops.rb.d += 1
            c.rbs.d.append((c.loops.iter, c.loops.epoch))
            c.loops.es.d = 0
            c.mods.d.rollback(c.loops.rb.d)
            r.log_model_action("rb", "d")
        else:
            c.mods.d.load()
            r.log_model_action("load", "d")

    def train_g(self):
        """Trains G with the training set."""
        r = self.results
        c = self.context
        r.logln("Started training G")
        lgs = []
        c.loops.train_index = 0
        while c.loops.train_index < c.data.train.batch_count:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            dgz, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz, c.latest.lg = dgz, lg
            lgs.append(lg)
            r.log_batch("g", "t")
            c.loops.train_index += 1
        epoch_lg = numpy.array(lgs).mean()
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("tg")

    def valid_g(self):
        """Validates G with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating G")
        lgs = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            dgz, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz, c.latest.lg = dgz, lg
            lgs.append(lg)
            r.log_batch("g", "v")
            c.loops.valid_index += 1
        epoch_lg = numpy.array(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")

    def save_best_g(self):
        """Saves the G model that performs the best."""
        r = self.results
        c = self.context
        r.log_best_losses("g")
        curr_lg = c.losses.valid.g[-1]
        if c.bests.g is None or curr_lg <= c.bests.g:
            c.bests.g = curr_lg
            c.mods.g.save()
            r.log_model_action("save", "g")
        elif c.loops.es.g < c.loops.es.max:
            c.loops.es.g += 1
            r.log_model_action("es", "g")
        elif c.loops.rb.g < c.loops.rb.max:
            c.loops.rb.g += 1
            c.rbs.g.append((c.loops.iter, c.loops.epoch))
            c.loops.es.g = 0
            c.mods.g.rollback(c.loops.rb.g)
            r.log_model_action("rb", "g")
        else:
            c.mods.g.load()
            r.log_model_action("load", "g")

    def run_d_iter(self):
        """Runs a iter of training, validating, and saving D."""
        r = self.results
        c = self.context
        c.loops.epoch = 0
        while c.loops.epoch < c.loops.epoch_count:
            r.log_epoch("Started", "d")
            self.train_d()
            self.valid_d()
            self.save_best_d()
            r.save_d_losses()
            r.logln()
            c.loops.epoch += 1

    def run_g_iter(self):
        """Runs a iter of training, validating, and saving G."""
        r = self.results
        c = self.context
        c.loops.epoch = 0
        while c.loops.epoch < c.loops.epoch_count:
            r.log_epoch("Started", "g")
            self.train_g()
            self.valid_g()
            self.save_best_g()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln()
            c.loops.epoch += 1

    def start_training(self):
        """Starts the training algorithm."""
        self.check_context_and_results()
        r = self.results
        c = self.context
        r.logln("Started training")
        r.save_training_images()
        r.save_validation_images()
        r.logln()
        c.loops.iter = 0
        while c.loops.iter < c.loops.iter_count:
            r.log_iter("Started")
            self.run_d_iter()
            self.run_g_iter()
            r.logln()
            c.loops.iter += 1
        r.logln("Completed training")
