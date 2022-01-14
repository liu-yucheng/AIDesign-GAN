"""Algorithms.

Training algorithms.
"""

# Copyright (C) 2022 Yucheng Liu. GNU GPL Version 3.
# GNU GPL Version 3 copy: https://www.gnu.org/licenses/gpl-3.0.txt
# First added by: liu-yucheng
# Last updated by: liu-yucheng

from aidesign_gan.libs.algos import algo
from aidesign_gan.libs.algos import alt_sgd_algo
from aidesign_gan.libs.algos import fair_pred_alt_sgd_algo
from aidesign_gan.libs.algos import pred_alt_sgd_algo

# Shortcuts

Algo = algo.Algo
AltSGDAlgo = alt_sgd_algo.AltSGDAlgo
FairPredAltSGDAlgo = fair_pred_alt_sgd_algo.FairPredAltSGDAlgo
PredAltSGDAlgo = pred_alt_sgd_algo.PredAltSGDAlgo

# End of shortcuts
