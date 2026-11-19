"""Test the single-pole coordinates used in the normalized reset diagram."""

import numpy as np
import pytest
from scipy.linalg import expm

from qubex.contrib.experiment._f0g1_physics import (
    RAD,
    fit_transfer,
    mode_parameters,
    normalized_reset_rate,
)


def test_rate_map_uses_population_decay_and_hamiltonian_omega():
    """The ideal exceptional point reaches one third and disconnected drives give zero."""
    x = (2 / 27) ** 0.5
    y = (1 / 108) ** 0.5
    rates = normalized_reset_rate(np.array([[0, x]]), np.array([[0], [y]]))
    np.testing.assert_allclose(rates, [[0, 0], [0, 1 / 3]], atol=1e-5)


def test_single_mode_fit_uses_the_hybrid_width():
    """A fit to three detunings recovers coupling to the selected photon pole in MHz."""
    ckp = [6288.2, 6253.9, 21.19, 39.11]
    poles, widths = mode_parameters(ckp)
    fsum, g, shift, gamma, j = 8141.0, 1.5, -3.0, 0.00007, 1
    t = np.r_[0, np.geomspace(16, 2500, 22)]
    fs = fsum - poles[j] + shift + np.array([-4.0, 0, 4.0])
    data = []
    for fd in fs:
        h = np.array(
            [
                [-0.5j * gamma, RAD * g],
                [RAD * g, RAD * (fd - fsum + poles[j] - shift - 0.5j * widths[j])],
            ]
        )
        data.append([0.12 + 0.68 * abs(expm(-1j * h * time)[0, 0]) ** 2 for time in t])
    fit = fit_transfer(fs, t, np.array(data), fsum, ckp, 0, gamma, mode=j)
    assert fit["valid"]
    assert fit["g"] == pytest.approx(g, rel=0.001)
    assert fit["shift"] == pytest.approx(shift, abs=0.001)


def test_coordinates_match_both_tones_and_keep_the_rabi_factor_of_two():
    """Only the selected waveform gets a star and interpolated rates keep their provenance."""
    from qubex.contrib.experiment._f0g1_physics import reset_map_points

    base = dict(j=1, A=0.6, fd=1840.0)
    fit = dict(**base, g=2.0, valid=True)
    direct = dict(**base, b=0.02, fe=3969.0, rate=8.0, valid=True, edge=False)
    selected = dict(**base, b=0.01, fe=3968.9, rate=4.0, ideal_worst=0.3)
    points = reset_map_points([fit], [direct], [20, 8], selected)
    assert [
        (p["g_over_kappa"], p["omega_over_kappa"], p["selected"], p["source"])
        for p in points
    ] == [(0.25, 0.5, False, "chevron"), (0.25, 0.25, True, "interpolated")]
    assert reset_map_points([fit], [direct], [20, 8], direct)[0]["selected"]
    unknown = dict(selected, j=2)
    points = reset_map_points([fit], [direct], [20, 8], unknown)
    assert len(points) == 1
    assert not points[0]["selected"]
    assert reset_map_points([dict(fit, valid=False)], [direct], [20, 8], selected) == []
