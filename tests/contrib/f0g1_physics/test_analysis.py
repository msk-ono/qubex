"""Physics and acquisition-design regressions for the adaptive notebook."""

import numpy as np
import pytest

from qubex.contrib.experiment._f0g1_physics import (
    chevron,
    fit_chevron,
    fit_transfer,
    mode_parameters,
    peaks,
    transfer,
)


@pytest.mark.parametrize(("center", "rate"), [(0.15, 6.45), (-29.2, 4.29)])
def test_sparse_chevron(center, rate):
    """A local sparse map should resolve both weakly and strongly shifted lines."""
    frequencies = center + np.arange(-8, 9, 2)
    times = np.arange(32, 641, 16)
    data = chevron(frequencies, times, center, rate, 0.6, 0.08, 16, 0.0005)
    fit = fit_chevron(frequencies, times, data)
    assert fit["valid"]
    assert fit["center"] == pytest.approx(center, abs=0.1)
    assert fit["rate"] == pytest.approx(rate, abs=0.1)


def test_edge_and_branches():
    """An edge signal should request extension while preserving two interior peaks."""
    x = np.arange(-40, 41, 2)
    y = sum(np.exp(-(((x - c) / 4) ** 2)) for c in [-42, -14, 22])
    found, edges = peaks(x, np.asarray(y), 0.1)
    np.testing.assert_array_equal(found, [-14, 22])
    assert edges == (True, False)


def test_shared_filter_coupling():
    """Detuned traces should recover the bare coupling to a hybridized resonator."""
    ckp = (6296.77, 6276.84, 20.17, 29.63)
    poles, widths = mode_parameters(ckp)
    np.testing.assert_allclose(widths.sum(), ckp[-1], atol=1e-8)
    f = 8000 - poles[1] + np.array([-5, 0, 5])
    t = np.r_[0, np.geomspace(32, 2000, 18)]
    p = transfer(f, t, 8000, ckp, 2.1, -8, 0.0001)
    result = fit_transfer(f, t, 0.1 + 0.64 * p, 8000, ckp, -8, 0.0001)
    assert result["valid"]
    assert result["g"] == pytest.approx(2.1, abs=0.05)


@pytest.mark.parametrize(("target", "carrier"), [("Q032", 1.903), ("Q108", 1.861)])
def test_two_tone_schedule(target, carrier):
    """Both ports' waveforms should encode the requested offsets on fixed carriers."""
    from qubex.contrib.experiment._f0g1_physics import build_schedule
    from qubex.pulse import FlatTop

    labels = (target, target + "-ef", target + "-f0g1", target + "-ef-reset")
    prep = FlatTop(duration=32, amplitude=0.1, tau=8, sampling_period=0.8)
    ps = build_schedule(
        "f",
        labels,
        (prep, prep),
        (0.8, 0.2),
        (carrier * 1000 - 8, 3986),
        (carrier * 1000, 3981),
        160,
        16,
        100,
        0.8,
    )
    assert ps.get_frequency(labels[2]) == pytest.approx(carrier, abs=1e-9)
    assert ps.get_frequency(labels[3]) == pytest.approx(3.981, abs=1e-9)
    for label, offset, amp in zip(labels[2:], [-8, 5], [0.8, 0.2], strict=True):
        waveform = ps.get_sampled_sequence(label)
        plateau = waveform[np.isclose(abs(waveform), amp, rtol=1e-5, atol=1e-8)]
        assert len(plateau) > 100
        step = np.angle(plateau[1:] * plateau[:-1].conj())
        np.testing.assert_allclose(step, -2 * np.pi * 0.8 * offset / 1000, atol=1e-6)
    assert ps.duration == pytest.approx(324.0, abs=1e-6)


def test_ideal_reset_without_ef():
    """An initial e excitation stays put when only the f0g1 tone is applied."""
    from qubex.contrib.experiment._f0g1_physics import reset_prediction

    t = np.array([0.0, 100.0, 1000.0])
    p, rate = reset_prediction((6300, 6270, 20, 30), 8000, 1700, 0, 2, 0, t)
    np.testing.assert_allclose(p[0], 1, atol=1e-10)
    assert rate == pytest.approx(0, abs=1e-12)
    assert p[1, -1] < p[1, 0]


def test_flat_chevron():
    """Constant assignments should never be accepted as a resolved oscillation."""
    f = np.arange(-10.0, 11, 2)
    t = np.arange(32.0, 640, 16)
    assert not fit_chevron(f, t, np.full((len(f), len(t)), 0.2))["valid"]
