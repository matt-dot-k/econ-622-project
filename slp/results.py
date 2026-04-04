from dataclasses import dataclass
import numpy as np

@dataclass
class LPResults:
    beta: np.ndarray
    H:    int
    k:    int
