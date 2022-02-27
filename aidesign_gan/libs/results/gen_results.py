"""Generation results."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import datetime
from os import path as ospath
from torchvision import utils as vutils

from aidesign_gan.libs import contexts
from aidesign_gan.libs.results import results

_GenContext = contexts.GenContext
_join = ospath.join
_now = datetime.datetime.now
_save_image = vutils.save_image
_Results = results.Results


class GenResults(_Results):
    """Generation results."""

    def __init__(self, path, logs):
        """Inits self with the given args.

        Args:
            path: the root path of the results
            logs: the log file objects
        """
        super().__init__(path, logs)

    def log_g(self, context=None):
        """Logs the G modelers info.

        Args:
            context: optional context
        """
        c: _GenContext = self.find_context(context)

        self.logln(f"G's size: {c.g.size}")
        self.logln(f"G's training size: {c.g.training_size}")
        self.logln(f"==== G's struct ====")
        self.logln(str(c.g.model))

    def log_batch(self, context=None):
        """Logs the batch info.

        Args:
            context: optional context
        """
        c: _GenContext = self.find_context(context)

        needs_log = c.batch_prog.index == 0
        needs_log = needs_log or (c.batch_prog.index + 1) % 15 == 0
        needs_log = needs_log or c.batch_prog.index == c.batch_prog.count - 1
        if not needs_log:
            return

        self.logln(f"Generated image batch {c.batch_prog.index + 1} / {c.batch_prog.count}")

    def save_generated_images(self, context=None):
        """Saves the generated images.

        Args:
            context: optional context
        """
        c: _GenContext = self.find_context(context)

        for index, image in enumerate(c.images.to_save):
            now = _now()
            timestamp = f"Time-{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-"\
                f"{now.microsecond:06}"

            name = ""
            if c.grids.enabled:
                name += "Grid-"
            else:
                name += "Image-"
            name += f"{index + 1}-{timestamp}.jpg"

            location = _join(self.path, name)
            _save_image(image, location, "JPEG")

        count = len(c.images.to_save)
        if c.grids.enabled:
            self.logln(f"Generated {count} grids, each has {c.grids.size_each} images")
        else:
            self.logln(f"Generated {count} images")
