"""Stable pickle location for solver-demonstration samples (also under -m)."""
from dataclasses import dataclass

import numpy as np

from .features import InstanceFeatures


@dataclass
class InstanceSample:
    feats: InstanceFeatures
    pos: np.ndarray
    feas_mask: np.ndarray
    n_feasible_solutions: int
    pos_w: np.ndarray = None
    pos_mode: np.ndarray = None
