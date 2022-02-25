"""Modelers.

==== References ====
Arjovsky, et al., 2017. Wasserstein Generative Adversarial Networks. https://arxiv.org/abs/1701.07875
"""

# Copyright 2022 Yucheng Liu. GNU GPL3 license.
# GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by username: liu-yucheng
# Last updated by username: liu-yucheng

from aidesign_gan.libs.modelers import disc_modeler
from aidesign_gan.libs.modelers import gen_modeler
from aidesign_gan.libs.modelers import helpers
from aidesign_gan.libs.modelers import modeler

# Shortcuts

DiscModeler = disc_modeler.DiscModeler
GenModeler = gen_modeler.GenModeler
Modeler = modeler.Modeler

load_model = helpers.load_model
save_model = helpers.save_model
load_optim = helpers.load_optim
save_optim = helpers.save_optim
find_model_sizes = helpers.find_model_sizes

# End of shortcuts
