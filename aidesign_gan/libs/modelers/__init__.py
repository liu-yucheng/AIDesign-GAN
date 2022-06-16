"""Modelers."""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from aidesign_gan.libs.modelers import disc_modeler
from aidesign_gan.libs.modelers import gen_modeler
from aidesign_gan.libs.modelers import _helpers
from aidesign_gan.libs.modelers import modeler

# Shortcuts

DiscModeler = disc_modeler.DiscModeler
GenModeler = gen_modeler.GenModeler
Modeler = modeler.Modeler

load_state_dict = _helpers.load_state_dict
save_state_dict = _helpers.save_state_dict
load_torch_script = _helpers.load_torch_script
save_torch_script = _helpers.save_torch_script
save_onnx = _helpers.save_onnx
find_model_sizes = _helpers.find_model_sizes

# End of shortcuts
