"""Configurations."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

import pkg_resources
from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import utils

_clamp_float = utils.clamp_float
_join = ospath.join
_load_json = utils.load_json
_require = pkg_resources.require
_save_json = utils.save_json


class Config:
    """Config base class."""

    default_loc = None
    """Default location."""
    default_name = None
    """Default name."""

    @classmethod
    def load(cls, from_file):
        """Loads the config from a file to a dict.

        Args:
            from_file: a file location

        Returns:
            to_dict: the loaded dict
        """
        from_file = str(from_file)
        to_dict = _load_json(from_file)
        to_dict = dict(to_dict)
        return to_dict

    @classmethod
    def load_default(cls):
        """Loads the default config.

        Returns:
            result: the result
        """
        result = cls.load(cls.default_loc)
        return result

    @classmethod
    def load_from_path(cls, path):
        """Loads the config named cls.default_name from a path.

        Args:
            path: a path

        Returns:
            result: the result
        """
        path = str(path)
        loc = _join(path, cls.default_name)
        result = cls.load(loc)
        return result

    @classmethod
    def save(cls, from_dict, to_file):
        """Saves a config from a dict to a file.

        Args:
            from_dict: a status dict to save
            to_file: a file location
        """
        from_dict = dict(from_dict)
        to_file = str(to_file)
        _save_json(from_dict, to_file)

    @classmethod
    def save_to_path(cls, from_dict, path):
        """Saves the config from a dict to a file named cls.default_name in a path.

        Args:
            from_dict: a dict to save
            path: a path
        """
        from_dict = dict(from_dict)
        loc = _join(path, cls.default_name)
        cls.save(from_dict, loc)

    @classmethod
    def verify(cls, from_dict):
        """Verifies a config from a dict.

        Args:
            from_dict: a status dictionary to verify

        Returns:
            result: the verified dict"""
        from_dict = dict(from_dict)
        result = from_dict
        return result

    @classmethod
    def _verify_str(cls, from_dict, key):
        val = from_dict[key]
        val = str(val)
        from_dict[key] = val

    @classmethod
    def _verify_int_ge_0(cls, from_dict, key):
        val = from_dict[key]
        val = int(val)

        if val < 0:
            val *= -1

        from_dict[key] = val

    @classmethod
    def _verify_int_ge_1(cls, from_dict, key):
        val = from_dict[key]
        val = int(val)

        if val < 0:
            val *= -1

        if val < 1:
            val = 1

        from_dict[key] = val

    @classmethod
    def _verify_float(cls, from_dict, key):
        val = from_dict[key]
        val = float(val)
        from_dict[key] = val

    @classmethod
    def _verify_float_ge_0(cls, from_dict, key):
        val = from_dict[key]
        val = float(val)

        if val < 0:
            val *= -1

        from_dict[key] = val

    @classmethod
    def _verify_float_clamp(cls, from_dict, key, bound1, bound2):
        val = from_dict[key]
        val = float(val)
        val = _clamp_float(val, bound1, bound2)
        from_dict[key] = val


class CoordsConfig(Config):
    """Coordinators config."""

    default_loc = _join(defaults.default_gan_model_path, defaults.coords_config_name)
    """Default location."""
    default_name = defaults.coords_config_name
    """Default name."""

    @classmethod
    def _verify_int_nonable(cls, from_dict, key):
        val = from_dict[key]

        if val is not None:
            val = int(val)

        from_dict[key] = val

    @classmethod
    def _verify_bool(cls, from_dict, key):
        val = from_dict[key]
        val = bool(val)
        from_dict[key] = val

    @classmethod
    def verify(cls, from_dict):
        result = super().verify(from_dict)

        train = result["training"]
        cls._verify_str(train, "mode")
        cls._verify_str(train, "algorithm")
        cls._verify_int_nonable(train, "manual_seed")
        cls._verify_int_ge_0(train, "gpu_count")
        cls._verify_int_ge_0(train, "iteration_count")
        cls._verify_int_ge_0(train, "epochs_per_iteration")
        cls._verify_int_ge_0(train, "max_rollbacks")
        cls._verify_int_ge_0(train, "max_early_stops")

        datasets = train["datasets"]
        cls._verify_int_ge_0(datasets, "loader_worker_count")
        cls._verify_float_clamp(datasets, "percents_to_use", 0, 100)
        cls._verify_int_ge_1(datasets, "images_per_batch")
        cls._verify_int_ge_1(datasets, "image_resolution")
        cls._verify_int_ge_1(datasets, "image_channel_count")
        cls._verify_float_ge_0(datasets, "training_set_weight")
        cls._verify_float_ge_0(datasets, "validation_set_weight")

        labels = train["labels"]
        cls._verify_float_clamp(labels, "real", 0, 1)
        cls._verify_float_clamp(labels, "fake", 0, 1)

        if "noise_models" in train:
            noise = train["noise_models"]
            cls._verify_bool(noise, "before_each_iter")
            cls._verify_bool(noise, "before_each_epoch")
            cls._verify_bool(noise, "save_noised_images")

        if "epoch_collapses" in train:
            collapses = train["epoch_collapses"]
            cls._verify_float_clamp(collapses, "max_loss", 0, 100)
            cls._verify_float_clamp(collapses, "percents_of_batches", 0, 100)

        gen = result["generation"]
        cls._verify_int_nonable(gen, "manual_seed")
        cls._verify_int_ge_0(gen, "gpu_count")
        cls._verify_int_ge_0(gen, "image_count")
        cls._verify_int_ge_1(gen, "images_per_batch")

        grid = gen["grid_mode"]
        cls._verify_bool(grid, "enabled")
        cls._verify_int_ge_1(grid, "images_per_grid")
        cls._verify_int_ge_0(grid, "padding")

        return result


class ModelersConfig(Config):
    """Modelers config."""

    default_loc = _join(defaults.default_gan_model_path, defaults.modelers_config_name)
    """Default location."""
    default_name = defaults.modelers_config_name
    """Default name."""

    @classmethod
    def _verify_modeler(cls, from_dict):
        cls._verify_int_ge_1(from_dict, "image_resolution")
        cls._verify_int_ge_1(from_dict, "image_channel_count")
        cls._verify_int_ge_1(from_dict, "feature_map_size")
        cls._verify_str(from_dict, "struct_name")
        cls._verify_str(from_dict, "state_name")
        cls._verify_str(from_dict, "optim_name")

        optim = from_dict["adam_optimizer"]
        cls._verify_float_ge_0(optim, "learning_rate")
        cls._verify_float_clamp(optim, "beta1", 0, 1)
        cls._verify_float_clamp(optim, "beta2", 0, 1)

        if "pred_factor" in optim:
            cls._verify_float(optim, "pred_factor")

        if "params_init" in from_dict:
            params = from_dict["params_init"]

            conv = params["conv"]
            cls._verify_float(conv, "weight_mean")
            cls._verify_float_ge_0(conv, "weight_std")

            bn = params["batch_norm"]
            cls._verify_float(bn, "weight_mean")
            cls._verify_float_ge_0(bn, "weight_std")
            cls._verify_float(bn, "bias_mean")
            cls._verify_float_ge_0(bn, "bias_std")

        if "params_noising" in from_dict:
            noise = from_dict["params_noising"]

            conv = noise["conv"]
            cls._verify_float(conv, "delta_weight_mean")
            cls._verify_float_ge_0(conv, "delta_weight_std")

            bn = noise["batch_norm"]
            cls._verify_float(bn, "delta_weight_mean")
            cls._verify_float_ge_0(bn, "delta_weight_std")
            cls._verify_float(bn, "delta_bias_mean")
            cls._verify_float_ge_0(bn, "delta_bias_std")

        if "fairness" in from_dict:
            fair = from_dict["fairness"]
            cls._verify_float(fair, "dx_factor")
            cls._verify_float(fair, "dgz_factor")
            cls._verify_float(fair, "cluster_dx_factor")
            cls._verify_float(fair, "cluster_dgz_factor")

    @classmethod
    def verify(cls, from_dict):
        result = super().verify(from_dict)

        disc = result["discriminator"]
        cls._verify_modeler(disc)

        gen = result["generator"]
        cls._verify_modeler(gen)

        return result


class FormatConfig(Config):
    """Format config."""

    default_loc = _join(defaults.default_gan_model_path, defaults.format_config_name)
    """Default location."""
    default_name = defaults.format_config_name
    """Default name."""

    @classmethod
    def _find_version(cls):
        version = "<unknown version>"

        try:
            packages = _require("aidesign-gan")

            if len(packages) > 0:
                version = packages[0].version
        except Exception as _:
            pass
        # end try

        return version

    @classmethod
    def load_default(cls):
        result = super().load_default()

        version = cls._find_version()
        result["aidesign_gan_version"] = version
        result["aidesign_gan_repo_tag"] = "v" + version

        return result

    @classmethod
    def verify(cls, from_dict):
        result = super().verify(from_dict)

        cls._verify_str(result, "aidesign_gan_version")
        cls._verify_str(result, "aidesign_gan_repo_tag")

        return result
