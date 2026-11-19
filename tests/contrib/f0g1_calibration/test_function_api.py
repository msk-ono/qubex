"""Behavior of the contributed f0g1 calibration workflow."""

import numpy as np
import pytest

from qubex.contrib import F0G1CalibrationOptions
from qubex.contrib.experiment.f0g1_calibration import select_f0g1_drive


def test_selects_frequency_and_amplitude_together():
    """The best ground population selects both measured drive coordinates."""
    assert select_f0g1_drive(
        [4.0, 4.1], [0.3, 0.6], np.array([[0.2, 0.9], [0.8, 0.4]])
    ) == (4.0, 0.6)


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_rejects_invalid_ramp(value):
    """Invalid ramp durations fail before hardware access."""
    with pytest.raises(ValueError, match="ramp_duration"):
        F0G1CalibrationOptions(ramp_duration=value)
