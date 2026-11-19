"""Empirical calibration, spectroscopy and independent verification contracts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.contrib.experiment import _f0g1_search as search
from qubex.contrib.experiment._f0g1_search import _probe_line


def test_off_reference_spans_split_lobes_and_rejects_flat_signal():
    """A Rabi map resolves a central resonance even when its time-average has two maxima."""
    frequencies = 3.969 + np.arange(-6, 7) * 0.004
    times = np.array([32.0, 48.0, 73.6, 96.0, 122.0, 184.0, 260.0])
    r = np.hypot(frequencies[:, None] - 3.969, 0.0195)
    data = 0.08 + 0.68 * (0.0195 / r * np.sin(np.pi * r * (times - 16.0))) ** 2
    line = _probe_line(frequencies, times, data, 16.0, 192)
    assert line.frequency == pytest.approx(3.969, abs=0.001)
    assert line.rabi_frequency == pytest.approx(0.0195, abs=0.001)
    flat = _probe_line(frequencies, times, np.full_like(data, 0.1), 16.0, 192)
    assert flat.frequency is None


@pytest.mark.parametrize(
    ("verification_fails", "hidden_shift"),
    [(False, False), (True, False), (False, True)],
)
def test_remote_peak_stark_shift_and_independent_reset(
    monkeypatch, verification_fails, hidden_shift
):
    """The remote peak wins on all preparations and needs fresh finite-shot evidence."""
    opts = search.F0G1SearchOptions(
        frequency_offsets=tuple(np.arange(-0.08, 0.081, 0.01)),
        ef_frequency_offsets=tuple(np.arange(-0.064, 0.065, 0.008)),
        amplitudes=(0.8, 0.95, 1.0),
        ef_amplitudes=(0.005, 0.02, 0.05),
        durations=(500.0, 1000.0, 2000.0),
        target_excitation=0.08,
        n_shots=96,
        verification_shots=512,
    )
    exp = Mock(
        qubits={
            "Q": SimpleNamespace(control_frequency_ef=4.0, control_frequency_ge=4.2)
        },
        resonators={"Q": SimpleNamespace(frequency=6.4)},
        ctx=SimpleNamespace(measurement=SimpleNamespace(sampling_period=0.4)),
    )
    verified = []
    monkeypatch.setattr(search, "register_f0g1_targets", lambda *a, **kw: None)
    monkeypatch.setattr(search, "build_f0g1_reset", lambda *a, **kw: "pulse")

    async def measure(
        exp, target, parameters, options, states=("g", "e", "f"), *, verification=False
    ):
        n = options.verification_shots if verification else options.n_shots
        counts = np.zeros((len(parameters), len(states), 3), dtype=int)
        for i, point in enumerate(parameters):
            a, b = point.f0g1_amplitude, point.ef_amplitude
            f, e, t = point.f0g1_frequency, point.ef_frequency, point.duration
            assert a + b <= 1.0 + 1e-12
            near = np.exp(-(((f - 1.8) / 0.012) ** 2))
            remote = np.exp(-(((f - 1.74) / 0.01) ** 2))
            resonance = 4.0 - 0.048 * remote if a > 0 else 4.0
            for j, s in enumerate(states):
                if b == 0:
                    g = 0.98 if s == "g" else 0.05
                    if s == "f":
                        g += a * (0.7 * remote + 0.8 * near)
                    pf = 1 - g if s == "f" else 0.01
                elif s == "e" and t <= 400:
                    r = np.hypot(e - resonance, 0.014)
                    pf = 0.05 + 0.7 * (0.014 / r * np.sin(np.pi * r * (t - 16.0))) ** 2
                    if hidden_shift and a:
                        pf = 0.08
                    g = 0.03
                else:
                    operating_center = 4.032 if hidden_shift else resonance
                    optimal_b = 0.05 if hidden_shift else 0.02
                    strength = (
                        remote
                        * np.exp(-(((e - operating_center) / 0.012) ** 2))
                        * np.exp(-(((b - optimal_b) / 0.015) ** 2))
                    )
                    g = (
                        0.98
                        if s == "g"
                        else 0.97 - (0.94 * np.exp(-t * strength / 240))
                    )
                    if verification and verification_fails and s == "g":
                        g = 0.6
                    pf = (1 - g) / 2
                ng, nf = round(g * n), round(pf * n)
                counts[i, j] = [ng, n - ng - nf, nf]
        if verification:
            verified.extend(parameters)
        return search.F0G1PopulationScan(parameters, states, counts, ())

    monkeypatch.setattr(search, "_measure_populations", measure)
    result = asyncio.run(search.search_f0g1_reset(exp, "Q", options=opts))
    assert result.success is (not verification_fails)
    assert (result.pulse is None) is verification_fails
    assert result.parameters is not None
    assert result.parameters.f0g1_frequency == pytest.approx(1.74, abs=0.005)
    assert result.parameters.ef_frequency == pytest.approx(
        4.032 if hidden_shift else 3.952, abs=0.005
    )
    if hidden_shift:
        assert all(row.shift is None for row in result.ef_shifts)
    else:
        assert any(
            row.shift == pytest.approx(-0.048, abs=0.003) for row in result.ef_shifts
        )
    assert any(point.f0g1_amplitude == 0 for point in verified)
    assert len({point.duration for point in verified if point.ef_amplitude > 0}) == 3


def test_refinement_retains_bracketed_saturated_transfer():
    """A saturated top within a coarse bracket stays eligible while a rising edge does not."""
    from qubex.contrib.experiment._f0g1_search import _refined_peak_index

    assert _refined_peak_index(np.ones(7)) == 3
    assert _refined_peak_index(np.arange(7)) is None


def test_overlapping_trajectories_do_not_report_a_stark_shift():
    """A nearby doublet prevents assignment of a single measured Stark displacement."""
    from qubex.contrib.experiment._f0g1_search import F0G1TransferPeak, _connect_peaks

    old = [F0G1TransferPeak(0.5, 2.01, 0.7), F0G1TransferPeak(0.5, 2.018, 0.65)]
    current = [F0G1TransferPeak(0.7, 2.015, 0.8)]
    assert _connect_peaks(old, current, 0.012)[0].shift is None
    assert _connect_peaks(old[:1], current, 0.012)[0].shift == pytest.approx(0.005)


def test_budget_extends_featureless_spectrum_toward_a_remote_peak():
    """One extension can find a disconnected peak just beyond the starting interval."""
    from qubex.contrib.experiment._f0g1_search import _spectrum

    async def measure(frequencies):
        response = np.exp(-(((frequencies - 1.94) / 0.004) ** 2))
        return response[:, None]

    x, y = asyncio.run(
        _spectrum(
            2.0,
            tuple(np.linspace(-0.04, 0.04, 9)),
            1,
            0.1,
            measure,
            lambda data: data.mean(axis=1),
        )
    )
    assert x[np.argmax(y[:, 0])] == pytest.approx(1.94)
    assert len(x) == 17
