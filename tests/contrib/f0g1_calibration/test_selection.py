"""Selection contracts for reset calibration."""

import numpy as np

from qubex.contrib.experiment.f0g1_calibration import _select_duration


def test_shortest_duration_satisfies_every_state():
    """Select the shortest measured duration satisfying all preparation states."""
    populations = np.array([[0.02, 0.0, 0.0], [0.0, 0.005, 0.009]])
    assert _select_duration(populations, 0.01) == 1


def test_no_acceptable_duration():
    """Return no index when every candidate exceeds the target."""
    assert _select_duration(np.array([[0.0, 0.02, 0.0]]), 0.01) is None
