"""Module of the training algos (algorithms)."""

# Initially added by: liu-yucheng
# Last updated by: liu-yucheng

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
        lds = []
        c.loops.train_index = 0
        for real_batch in c.data.train.loader:
            noises = c.mods.g.generate_noises(c.data.batch_size)
            real_batch = real_batch[0]
            fake_batch = c.mods.g.test(noises)
            dx, ldr = c.mods.d.train(real_batch, c.labels.real)
            dgz, ldf = c.mods.d.train(fake_batch, c.labels.fake)
            ld = ldr + ldf
            c.latest.dx, c.latest.dgz, c.latest.ld = dx, dgz, ld
            lds.append(ld)
            r.log_batch("d", "t")
            c.loops.train_index += 1
        epoch_ld = numpy.array(lds).mean()
        c.losses.train.d.append(epoch_ld)
        r.log_epoch_loss("td")

    def valid_d(self):
        """Validates D with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating D")
        ldrs = []
        c.loops.valid_index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            ldrs.append(ld)
            r.log_batch("d", "vr")
            c.loops.valid_index += 1
        ldfs = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            ldfs.append(ld)
            r.log_batch("d", "vf")
            c.loops.valid_index += 1
        lds = []
        for index in range(c.data.valid.batch_count):
            lds.append(ldrs[index] + ldfs[index])
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
        """
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
        """

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
            dgz, lg = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
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
        """
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
        """

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
            r.logln("-")
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
            r.logln("-")
            c.loops.epoch += 1

    def start_training(self):
        """Starts the training algorithm."""
        self.check_context_and_results()
        r = self.results
        c = self.context
        r.logln("Started iter level algorithm")
        r.logln("-")
        r.save_training_images()
        r.save_validation_images()
        c.loops.iter = 0
        while c.loops.iter < c.loops.iter_count:
            r.log_iter("Started")
            self.run_d_iter()
            self.run_g_iter()
            r.logln("-")
            c.loops.iter += 1


class BatchLevelAlgo(Algo):
    """Algorithm that trains D and G together at batch level."""

    def __init__(self):
        """Inits self."""
        super().__init__()

    def train_both(self):
        """Trains both D and G together."""
        r = self.results
        c = self.context
        r.logln("Started training both D and G")
        lds = []
        lgs = []
        c.collapses.batch_count = 0
        c.loops.train_index = 0
        for real_batch in c.data.train.loader:
            # Train D on a real batch
            real_batch = real_batch[0]
            dx, ldr = c.mods.d.train(real_batch, c.labels.real)
            noises = c.mods.g.generate_noises(c.data.batch_size)
            # Train D on a fake batch
            fake_batch = c.mods.g.test(noises)
            dgz, ldf = c.mods.d.train(fake_batch, c.labels.fake)
            # Train G on a batch generated by itself
            noises = c.mods.g.generate_noises(c.data.batch_size)
            dgz2, lg = c.mods.g.train(c.mods.d.model, noises, c.labels.real)
            # Detect training collapse
            collapsed = bool(ldr >= c.collapses.max_loss)
            collapsed = collapsed or bool(ldf >= c.collapses.max_loss)
            collapsed = collapsed or bool(lg >= c.collapses.max_loss)
            if collapsed:
                c.collapses.batch_count += 1
            # Update the statistics
            ld = ldr + ldf
            c.latest.dx, c.latest.dgz, c.latest.ld = dx, dgz, ld
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lds.append(ld)
            lgs.append(lg)
            r.log_batch_2("t")
            c.loops.train_index += 1
        epoch_collapsed = c.collapses.batch_count >= c.collapses.max_batch_count
        if epoch_collapsed:
            c.collapses.epochs.append((c.loops.iter, c.loops.epoch))
            r.logln("Epoch training collapsed")
        epoch_ld = numpy.array(lds).mean()
        epoch_lg = numpy.array(lgs).mean()
        c.losses.train.d.append(epoch_ld)
        c.losses.train.g.append(epoch_lg)
        r.log_epoch_loss("td")
        r.log_epoch_loss("tg")

    def valid_d(self):
        """Validates D with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating D")
        ldrs = []
        c.loops.valid_index = 0
        for real_batch in c.data.valid.loader:
            real_batch = real_batch[0]
            dx, ld = c.mods.d.valid(real_batch, c.labels.real)
            c.latest.dx, c.latest.ld = dx, ld
            ldrs.append(ld)
            r.log_batch_2("vdr")
            c.loops.valid_index += 1
        ldfs = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            fake_batch = c.mods.g.test(noises)
            dgz, ld = c.mods.d.valid(fake_batch, c.labels.fake)
            c.latest.dgz, c.latest.ld = dgz, ld
            ldfs.append(ld)
            r.log_batch_2("vdf")
            c.loops.valid_index += 1
        lds = []
        for index in range(c.data.valid.batch_count):
            lds.append(ldrs[index] + ldfs[index])
        epoch_ld = numpy.array(lds).mean()
        c.losses.valid.d.append(epoch_ld)
        r.log_epoch_loss("vd")

    def valid_g(self):
        """Validates G with the validation set."""
        r = self.results
        c = self.context
        r.logln("Started validating G")
        lgs = []
        c.loops.valid_index = 0
        for noises in c.noises.valid_set:
            dgz2, lg = c.mods.g.valid(c.mods.d.model, noises, c.labels.real)
            c.latest.dgz2, c.latest.lg = dgz2, lg
            lgs.append(lg)
            r.log_batch_2("vg")
            c.loops.valid_index += 1
        epoch_lg = numpy.array(lgs).mean()
        c.losses.valid.g.append(epoch_lg)
        r.log_epoch_loss("vg")

    def save_best_d(self):
        """Saves the best D."""
        r = self.results
        c = self.context
        r.log_best_losses("d")
        curr_ld = c.losses.valid.d[-1]
        if c.bests.d is None or curr_ld <= c.bests.d:
            c.bests.d = curr_ld
        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iter, c.loops.epoch) == c.collapses.epochs[-1]
        if epoch_collapsed:
            c.mods.d.load()
            r.log_model_action("load", "d")
        else:
            c.mods.d.save()
            r.log_model_action("save", "d")

    def save_best_g(self):
        """Saves the best G."""
        r = self.results
        c = self.context
        r.log_best_losses("g")
        curr_lg = c.losses.valid.g[-1]
        if c.bests.g is None or curr_lg <= c.bests.g:
            c.bests.g = curr_lg
        epoch_collapsed = False
        if len(c.collapses.epochs) > 0:
            epoch_collapsed = (c.loops.iter, c.loops.epoch) == c.collapses.epochs[-1]
        if epoch_collapsed:
            c.mods.g.load()
            r.log_model_action("load", "g")
        else:
            c.mods.g.save()
            r.log_model_action("save", "g")

    def run_iter(self):
        """Runs a iter of multiple epochs."""
        r = self.results
        c = self.context
        c.loops.epoch = 0
        while c.loops.epoch < c.loops.epoch_count:
            r.log_epoch("Started", "")
            self.train_both()
            self.valid_d()
            self.save_best_d()
            self.valid_g()
            self.save_best_g()
            r.save_d_losses()
            r.save_g_losses()
            r.save_generated_images()
            r.save_tvg()
            r.logln("-")
            c.loops.epoch += 1

    def start_training(self):
        """Starts the training algorithm."""
        self.check_context_and_results()
        r = self.results
        c = self.context
        r.logln("Started batch level algorithm")
        r.logln("-")
        r.save_training_images()
        r.save_validation_images()
        r.save_images_before_training()
        c.loops.iter = 0
        while c.loops.iter < c.loops.iter_count:
            r.log_iter("Started")
            self.run_iter()
            r.logln("-")
            c.loops.iter += 1
