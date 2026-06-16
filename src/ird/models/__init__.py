"""Short-rate models.

Modules:
    vasicek.py     - analytical bond pricing, closed-form MLE calibration
    hull_white.py  - exact term-structure fit, Jamshidian swaption pricing
    calibration.py - least-squares fit of (a, sigma) to a swaption vol surface
"""

from __future__ import annotations

from ird.models.calibration import (
    HWCalibrationResult,
    calibrate_hull_white,
    model_atm_normal_vol,
    synth_vol_surface,
)
from ird.models.hull_white import HullWhite1F
from ird.models.vasicek import VasicekModel, calibrate_vasicek_mle, simulate_vasicek

__all__ = [
    "HWCalibrationResult",
    "HullWhite1F",
    "VasicekModel",
    "calibrate_hull_white",
    "calibrate_vasicek_mle",
    "model_atm_normal_vol",
    "simulate_vasicek",
    "synth_vol_surface",
]
