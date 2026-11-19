"""Contracts for identifying a CKP mode and measuring full-drive coupling."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.contrib.experiment import _f0g1_characterization as physics
from qubex.contrib.experiment._f0g1_physics import mode_parameters, transfer

CKP_MHZ = (6296.77, 6276.84, 20.17, 29.63)
TIMES = np.r_[0, np.geomspace(32, 3600, 20)]


def test_shifted_pole_and_effective_coupling():
    """A displaced line should recover its local pole and separately fitted effective coupling."""
    params = physics.F0G1CKP(*CKP_MHZ)
    poles, widths = mode_parameters(CKP_MHZ)
    frequencies = 8100 + 87 - poles[1] + np.array([-10.8, 0, 10.8])
    data = 0.12 + 0.68 * transfer(
        frequencies, np.maximum(0, TIMES - 16), 8100, CKP_MHZ, 0.74, 87, 0.00003
    )
    point = physics.analyze_f0g1_transfer(
        frequencies,
        TIMES,
        data,
        amplitude=1,
        transition_sum_mhz=8100,
        ckp=params,
        off_decay_per_ns=0.00003,
    )
    assert point.mode == 1
    assert point.bare_coupling_mhz == pytest.approx(0.74, rel=1e-3)
    assert point.g_over_kappa == pytest.approx(point.mode_coupling_mhz / widths[1])
    assert point.mode_coupling_mhz != pytest.approx(point.bare_coupling_mhz, rel=1e-2)
    assert point.heldout_rmse < 0.001
    assert len(point.windows) == 2


def test_no_contrast_keeps_the_ratio_unresolved():
    """Flat assigned populations should preserve measurements and withhold the ratio."""
    point = physics.analyze_f0g1_transfer(
        [1800, 1805, 1810],
        TIMES,
        np.full((3, len(TIMES)), 0.6),
        amplitude=0.45,
        transition_sum_mhz=8100,
        ckp=physics.F0G1CKP(*CKP_MHZ),
        off_decay_per_ns=0.0,
    )
    assert point.mode is None
    assert point.g_over_kappa is None
    assert point.status != "identified"
    assert point.assigned_f.shape == (3, len(TIMES))


@pytest.mark.parametrize("amplitudes", [(0.2, 1.01), (0.5, 0.5)])
def test_validate_amplitudes(amplitudes):
    """Invalid amplitude grids should fail before targets are registered."""
    with pytest.raises(ValueError, match="amplitude"):
        physics.F0G1PhysicsOptions(amplitudes=amplitudes)


def test_passive_filter_parameters():
    """Nonpositive filter loss should be rejected before fitting."""
    with pytest.raises(ValueError, match="positive"):
        physics.F0G1CKP(6300, 6200, 20, 0)


@pytest.mark.parametrize("has_line", [True, False])
def test_acquire_full_amplitude_even_when_unresolved(monkeypatch, has_line):
    """Full amplitude must be surveyed with ef off and report missing lines explicitly."""
    from qubex.contrib.experiment._f0g1_acquisition import F0G1PopulationScan

    exp = Mock(
        ctx=SimpleNamespace(measurement=SimpleNamespace(sampling_period=0.4)),
        qubits={
            "Q": SimpleNamespace(control_frequency_ge=4.2, control_frequency_ef=3.9)
        },
    )
    poles, _ = mode_parameters(CKP_MHZ)
    center = 8100 + 70 - poles[1]
    amplitudes = []
    monkeypatch.setattr(physics, "register_f0g1_targets", lambda *a, **kw: None)

    async def measure(exp, target, parameters, options, states):
        counts = np.zeros((len(parameters), len(states), 3), dtype=np.int64)
        for i, p in enumerate(parameters):
            amplitudes.append(p.f0g1_amplitude)
            assert p.ef_amplitude == 0
            for j, state in enumerate(states):
                f, t = p.f0g1_frequency * 1000, p.duration
                surv = np.exp(-0.00006 * t)
                if state == "f" and p.f0g1_amplitude and has_line:
                    if len({q.duration for q in parameters}) == 1:
                        surv *= 1 - 0.9 * np.exp(-(((f - center) / 5) ** 2))
                    else:
                        surv = transfer(
                            np.array([f]),
                            np.array([max(0, t - 16)]),
                            8100,
                            CKP_MHZ,
                            1.2,
                            70,
                            0.00006,
                        )[0, 0]
                pf = 0.05 + (0.78 * surv if state == "f" else 0)
                counts[i, j, 2] = round(pf * options.n_shots)
                counts[i, j, 0] = options.n_shots - counts[i, j, 2]
        return F0G1PopulationScan(parameters, states, counts, ())

    monkeypatch.setattr(physics, "_measure_populations", measure)
    result = asyncio.run(
        physics.characterize_f0g1(
            exp,
            "Q",
            ckp=physics.F0G1CKP(*CKP_MHZ),
            options=physics.F0G1PhysicsOptions(amplitudes=(1.0,)),
        )
    )
    assert 1.0 in amplitudes
    assert result.spectra[0].amplitude == 1.0
    assert result.off_decay_per_ns == pytest.approx(0.00006, rel=0.1)
    assert bool(result.at_full_amplitude) == has_line
    assert result.full_amplitude_status == (
        "identified" if has_line else "no confirmed peak"
    )
    if has_line:
        point = result.at_full_amplitude[0]
        assert point.g_over_kappa is not None
        marker = result.plot_reset_map().to_dict()["data"][1]
        assert marker["x"] == [point.g_over_kappa, point.g_over_kappa]
        series = point.plot().to_dict()["data"]
        assert series[0]["line"]["color"] == series[1]["marker"]["color"]


def test_single_pole_center_must_match_two_pole_feature(monkeypatch):
    """A single-pole fit to a distinct center should withhold the normalized coupling."""
    ck = physics.F0G1CKP(*CKP_MHZ)
    poles, widths = ck.modes
    center = 8100 - poles[0]
    frequencies = center + np.array([-4, 0, 4])
    data = np.full((3, len(TIMES)), 0.5)

    def fit(f, t, pf, fsum, ckp, shift, gamma, *, mode=None):
        prediction = data if mode is not None or abs(shift) < 1 else data + 0.1
        return dict(
            prediction=prediction,
            valid=True,
            g=1.0,
            g_se=0.05,
            shift=0.6 * widths[0] if mode is not None else shift,
            heldout_rmse=float(np.sqrt(np.mean((prediction - data) ** 2))),
        )

    monkeypatch.setattr(physics, "fit_transfer", fit)
    point = physics.analyze_f0g1_transfer(
        frequencies,
        TIMES,
        data,
        amplitude=1,
        transition_sum_mhz=8100.0,
        ckp=ck,
        off_decay_per_ns=0,
    )
    assert point.status == "single- and two-pole centers disagree"
    assert point.mode == 0
    assert point.single_detuning_mhz == pytest.approx(0.6 * widths[0])
    assert point.g_over_kappa is None


@pytest.mark.parametrize("width", [0.02, 6.0])
def test_survey_resolution_budget_includes_both_edge_extensions(width):
    """A finite initial budget should also bound a narrow-line survey extended on both edges."""
    batches = []

    async def measure(f, shots):
        batches.append(f)
        if len(batches) == 1:
            # Two unbracketed lines, both active at the initial limits.
            return 0.1 + 0.65 * (abs(f - f.mean()) > np.ptp(f) * 0.43)
        return np.full_like(f, 0.1)

    result = asyncio.run(
        physics.survey_transfer(
            measure,
            [1800.0, 1850.0],
            [width, 36.0],
            (),
            0.1,
            frequency_step_mhz=max(0.1, width / 2),
            max_frequency_points=15,
        )
    )
    assert len(batches[0]) == 15
    assert 15 < len(result["frequencies"]) <= 30


def test_unresolved_reset_map_has_no_x_marker():
    """An empty full-amplitude result should display only the theoretical rate surface."""
    result = physics.F0G1PhysicsResult("Q", physics.F0G1CKP(*CKP_MHZ), 0.0, (), (), ())
    assert result.full_amplitude_status == "not surveyed"
    fig = result.plot_reset_map()
    assert len(fig.to_dict()["data"]) == 1
    assert not fig.to_dict()["layout"].get("shapes")


def test_saturated_peak_is_confirmed_against_measured_wings():
    """A flat-topped transfer peak should survive confirmation using independently measured shoulders."""
    calls = []

    async def transfer_spectrum(f, shots):
        calls.append(f)
        return 0.05 + 0.9 * np.clip(2.5 - abs(f - 1840) / 8, 0, 1)

    found = asyncio.run(
        physics.survey_transfer(
            transfer_spectrum,
            [1840, 1880],
            [4, 25],
            (),
            0.05,
            frequency_step_mhz=2,
            max_frequency_points=129,
        )
    )
    assert [r["frequency"] for r in found["confirmed"]] == pytest.approx([1840], abs=2)
    assert len(calls) == 2
    assert max(calls[1]) - min(calls[1]) > 24
