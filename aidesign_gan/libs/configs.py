"""Configurations."""

# Copyright 2022-2023 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from os import path as ospath

from aidesign_gan.libs import defaults
from aidesign_gan.libs import pack_info
from aidesign_gan.libs import utils

_clamp_float = utils.clamp_float
_join = ospath.join
_load_json = utils.load_json
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
        from_dict: dict = from_dict
        path = str(path)
        loc = _join(path, cls.default_name)
        cls.save(from_dict, loc)

    @classmethod
    def verify(cls, from_dict):
        """Verifies a config from a dict.

        Args:
            from_dict: a status dictionary to verify

        Returns:
            result: the verified dict"""
        from_dict: dict = from_dict
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
        from_dict = super().verify(from_dict)

        train = from_dict["training"]
        cls._verify_str(train, "mode")
        cls._verify_str(train, "algorithm")
        cls._verify_int_nonable(train, "manual_seed")
        cls._verify_int_ge_0(train, "gpu_count")

        iter_count_key = "iter_count"
        iteration_count_key = "iteration_count"

        if iter_count_key in train:
            cls._verify_int_ge_0(train, iter_count_key)
            train[iteration_count_key] = train[iter_count_key]
        elif iteration_count_key in train:
            cls._verify_int_ge_0(train, iteration_count_key)
            train[iter_count_key] = train[iteration_count_key]
        else:
            raise KeyError(f"train has no key \"{iter_count_key}\" nor \"{iteration_count_key}\"")
        # end if

        epochs_per_iter_key = "epochs_per_iter"
        epochs_per_iteration_key = "epochs_per_iteration"

        if epochs_per_iter_key in train:
            cls._verify_int_ge_0(train, epochs_per_iter_key)
            train[epochs_per_iteration_key] = train[epochs_per_iter_key]
        elif epochs_per_iteration_key in train:
            cls._verify_int_ge_0(train, epochs_per_iteration_key)
            train[epochs_per_iter_key] = train[epochs_per_iteration_key]
        else:
            raise KeyError(f"train has no key \"{epochs_per_iter_key}\" nor \"{epochs_per_iteration_key}\"")
        # end if

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

        if "retrials" in train:
            retrials = train["retrials"]
            cls._verify_int_ge_0(retrials, "max_count")
            cls._verify_float_ge_0(retrials, "delay_seconds")

        gen = from_dict["generation"]
        cls._verify_int_nonable(gen, "manual_seed")
        cls._verify_int_ge_0(gen, "gpu_count")
        cls._verify_int_ge_0(gen, "image_count")
        cls._verify_int_ge_1(gen, "images_per_batch")

        grid = gen["grid_mode"]
        cls._verify_bool(grid, "enabled")
        cls._verify_int_ge_1(grid, "images_per_grid")
        cls._verify_int_ge_0(grid, "padding")

        if "exportation" in from_dict:
            export = from_dict["exportation"]
            cls._verify_int_nonable(export, "manual_seed")
            cls._verify_int_ge_0(export, "gpu_count")
            cls._verify_int_ge_1(export, "images_per_batch")

            preview = export["preview_grids"]
            cls._verify_int_ge_1(preview, "images_per_grid")
            cls._verify_int_ge_0(preview, "padding")
        # end if

        result = from_dict
        return result


class ModelersConfig(Config):
    """Modelers config."""

    default_loc = _join(defaults.default_gan_model_path, defaults.modelers_config_name)
    """Default location."""
    default_name = defaults.modelers_config_name
    """Default name."""

    @classmethod
    def _verify_modeler_commons(cls, from_dict):
        cls._verify_int_ge_1(from_dict, "image_resolution")
        cls._verify_int_ge_1(from_dict, "image_channel_count")

        fm_count_key = "feature_map_count"
        fm_size_key = "feature_map_size"

        if fm_count_key in from_dict:
            cls._verify_int_ge_1(from_dict, fm_count_key)
            from_dict[fm_size_key] = from_dict[fm_count_key]
        elif fm_size_key in from_dict:
            cls._verify_int_ge_1(from_dict, fm_size_key)
            from_dict[fm_count_key] = from_dict[fm_size_key]
        else:
            raise KeyError(f"from_dict has no key \"{fm_count_key}\" nor \"{fm_size_key}\"")
        # end if

        cls._verify_str(from_dict, "struct_name")
        cls._verify_str(from_dict, "state_name")
        cls._verify_str(from_dict, "optim_name")

        optim = from_dict["adam_optimizer"]
        cls._verify_float_ge_0(optim, "learning_rate")
        cls._verify_float_clamp(optim, "beta1", 0, 1)
        cls._verify_float_clamp(optim, "beta2", 0, 1)

        if "pred_factor" in optim:
            cls._verify_float(optim, "pred_factor")

        wm_key = "weight_mean"
        ws_key = "weight_std"
        bm_key = "bias_mean"
        bs_key = "bias_std"

        if "params_init" in from_dict:
            params = from_dict["params_init"]

            conv = params["conv"]
            cls._verify_float(conv, wm_key)
            cls._verify_float_ge_0(conv, ws_key)

            if bm_key in conv:
                cls._verify_float(conv, bm_key)

            if bs_key in conv:
                cls._verify_float_ge_0(conv, bs_key)

            bn = params["batch_norm"]
            cls._verify_float(bn, wm_key)
            cls._verify_float_ge_0(bn, ws_key)
            cls._verify_float(bn, bm_key)
            cls._verify_float_ge_0(bn, bs_key)

            if "others" in params:
                others = params["others"]
                cls._verify_float(others, wm_key)
                cls._verify_float_ge_0(others, ws_key)
                cls._verify_float(others, bm_key)
                cls._verify_float_ge_0(others, bs_key)
            # end if
        # end if

        dwm_key = "delta_weight_mean"
        dws_key = "delta_weight_std"
        dbm_key = "delta_bias_mean"
        dbs_key = "delta_bias_std"

        if "params_noising" in from_dict:
            noise = from_dict["params_noising"]

            conv = noise["conv"]
            cls._verify_float(conv, dwm_key)
            cls._verify_float_ge_0(conv, dws_key)

            if dbm_key in conv:
                cls._verify_float(conv, dbm_key)

            if dbs_key in conv:
                cls._verify_float_ge_0(conv, dbs_key)

            bn = noise["batch_norm"]
            cls._verify_float(bn, dwm_key)
            cls._verify_float_ge_0(bn, dws_key)
            cls._verify_float(bn, dbm_key)
            cls._verify_float_ge_0(bn, dbs_key)

            if "others" in noise:
                others = noise["others"]

                cls._verify_float(others, dwm_key)
                cls._verify_float_ge_0(others, dws_key)
                cls._verify_float(others, dbm_key)
                cls._verify_float_ge_0(others, dbs_key)
            # end if
        # end if

        if "fairness" in from_dict:
            fair = from_dict["fairness"]
            cls._verify_float(fair, "dx_factor")
            cls._verify_float(fair, "dgz_factor")
            cls._verify_float(fair, "cluster_dx_factor")
            cls._verify_float(fair, "cluster_dgz_factor")

            if "cluster_dx_overact_slope" in fair:
                cls._verify_float(fair, "cluster_dx_overact_slope")

            if "cluster_dgz_overact_slope" in fair:
                cls._verify_float(fair, "cluster_dgz_overact_slope")
            # end if
        # end if

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        disc = from_dict["discriminator"]
        label_res_key = "label_resolution"

        if label_res_key in disc:
            cls._verify_int_ge_1(disc, label_res_key)
        else:
            disc[label_res_key] = 1
        # end if

        label_ch_count_key = "label_channel_count"

        if label_ch_count_key in disc:
            cls._verify_int_ge_1(disc, label_ch_count_key)
        else:
            disc[label_ch_count_key] = 1
        # end if

        cls._verify_modeler_commons(disc)

        gen = from_dict["generator"]
        cls._verify_int_ge_1(gen, "noise_resolution")
        cls._verify_int_ge_1(gen, "noise_channel_count")
        cls._verify_modeler_commons(gen)

        result = from_dict
        return result


class DiscConfig(Config):
    """Discriminator config."""

    default_loc = _join(defaults.default_gan_export_path, defaults.disc_config_name)
    """Default location."""
    default_name = defaults.disc_config_name
    """Default name."""

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        cls._verify_int_ge_1(from_dict, "image_resolution")
        cls._verify_int_ge_1(from_dict, "image_channel_count")

        fm_count_key = "feature_map_count"
        fm_size_key = "feature_map_size"

        if fm_count_key in from_dict:
            cls._verify_int_ge_1(from_dict, fm_count_key)
            from_dict[fm_size_key] = from_dict[fm_count_key]
        elif fm_size_key in from_dict:
            cls._verify_int_ge_1(from_dict, fm_size_key)
            from_dict[fm_count_key] = from_dict[fm_size_key]
        else:
            raise KeyError(f"from_dict has no key \"{fm_count_key}\" nor \"{fm_size_key}\"")
        # end if

        if fm_size_key in from_dict:
            del from_dict[fm_size_key]

        cls._verify_str(from_dict, "struct_name")
        cls._verify_str(from_dict, "state_script_name")
        cls._verify_str(from_dict, "state_onnx_name")

        result = from_dict
        return result


class GenConfig(Config):
    """Generator config."""

    default_loc = _join(defaults.default_gan_export_path, defaults.gen_config_name)
    """Default location."""
    default_name = defaults.gen_config_name
    """Default name."""

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        cls._verify_int_ge_1(from_dict, "noise_resolution")
        cls._verify_int_ge_1(from_dict, "noise_channel_count")
        cls._verify_int_ge_1(from_dict, "image_resolution")
        cls._verify_int_ge_1(from_dict, "image_channel_count")

        fm_count_key = "feature_map_count"
        fm_size_key = "feature_map_size"

        if fm_count_key in from_dict:
            cls._verify_int_ge_1(from_dict, fm_count_key)
            from_dict[fm_size_key] = from_dict[fm_count_key]
        elif fm_size_key in from_dict:
            cls._verify_int_ge_1(from_dict, fm_size_key)
            from_dict[fm_count_key] = from_dict[fm_size_key]
        else:
            raise KeyError(f"from_dict has no key \"{fm_count_key}\" nor \"{fm_size_key}\"")
        # end if

        if fm_size_key in from_dict:
            del from_dict[fm_size_key]

        cls._verify_str(from_dict, "struct_name")
        cls._verify_str(from_dict, "state_script_name")
        cls._verify_str(from_dict, "state_onnx_name")
        cls._verify_str(from_dict, "preview_name")

        result = from_dict
        return result


class FormatConfig(Config):
    """Format config."""

    default_loc = _join(defaults.default_gan_model_path, defaults.format_config_name)
    """Default location."""
    default_name = defaults.format_config_name
    """Default name."""
    ver = pack_info.ver
    """Format version."""

    @classmethod
    def load_default(cls):
        result = super().load_default()

        version = cls.ver
        result["aidesign_gan_version"] = version
        result["aidesign_gan_repo_tag"] = "v" + version

        return result

    @classmethod
    def verify(cls, from_dict):
        from_dict = super().verify(from_dict)

        cls._verify_str(from_dict, "aidesign_gan_version")
        cls._verify_str(from_dict, "aidesign_gan_repo_tag")

        result = from_dict
        return result
