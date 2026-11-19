"""Discoverable F0G1 workflows and measured-data figures."""

from dataclasses import replace

import numpy as np
import pytest

from qubex import contrib
from qubex.contrib import experiment
from qubex.contrib.experiment import f0g1


def test_both_workflows_are_discoverable():
    """The public contrib namespaces lead to the same calibration and physics entry points."""
    for namespace in (experiment, contrib):
        assert namespace.search_f0g1_reset is f0g1.search_f0g1_reset
        assert namespace.characterize_f0g1 is f0g1.characterize_f0g1


def test_independent_verification_plots_elapsed_time_and_state():
    """The verification figure uses physical time and distinct ON/OFF counts for each state."""
    base = f0g1.F0G1PulseParameters(100, 1.85, 0.8, 3.93, 0.05)
    points = (base, replace(base, duration=120))
    counts = np.array(
        [[[9, 1, 0], [8, 1, 1], [7, 1, 2]], [[10, 0, 0], [9, 0, 1], [8, 2, 0]]]
    )
    scan = f0g1.F0G1PopulationScan(points, ("g", "e", "f"), counts, ())
    result = f0g1.F0G1ResetSearchResult(
        False,
        "test",
        base,
        None,
        (),
        (),
        {
            "verification": scan,
            "verification_off": replace(scan, counts=counts[..., ::-1]),
        },
        f0g1.F0G1SearchOptions(),
    )
    traces = f0g1.plot_f0g1_verification(result).to_plotly_json()["data"]
    assert len(traces) == 6
    assert list(traces[0]["x"]) == [100, 120]
    assert list(traces[0]["y"]) == pytest.approx([0.1, 0])
    assert traces[0]["name"] != traces[3]["name"]


def test_ef_map_keeps_frequency_horizontal_and_uncertain_shift_absent():
    """An unresolved driven line keeps its measured heatmap and suppresses its shift."""
    frequencies = np.array([3.910, 3.915, 3.920])
    durations = np.array([32.0, 64.0])
    populations = np.array([[0.1, 0.2], [0.4, 0.7], [0.2, 0.1]])
    off = f0g1.F0G1ProbeLine(
        3.915, 0.010, 0.03, "resolved", frequencies, durations, populations
    )
    on = replace(off, frequency=None, status="unresolved")
    estimate = f0g1.F0G1StarkEstimate(0.9, 1.8, 0.02, off, on)
    fig = f0g1.plot_f0g1_ef_probe(estimate).to_plotly_json()
    maps = [r for r in fig["data"] if r["type"] == "heatmap"]
    assert len(maps) == 2
    np.testing.assert_allclose(maps[1]["x"], frequencies * 1000)
    np.testing.assert_array_equal(maps[1]["y"], durations)
    np.testing.assert_array_equal(maps[1]["z"], populations.T)
    result = f0g1.F0G1ResetSearchResult(
        False, "no reset", None, None, (), (estimate,), {}, f0g1.F0G1SearchOptions()
    )
    assert not len(
        f0g1.plot_f0g1_stark_shifts(result).to_plotly_json()["data"][-1]["x"]
    )
