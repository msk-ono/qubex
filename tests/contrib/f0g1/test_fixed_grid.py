"""Legacy explicit grids share the calibrated hardware acquisition contract."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from qubex.contrib.experiment import f0g1_calibration as grid


def test_two_tone_grid_respects_headroom(monkeypatch):
    """A full-amplitude transfer point remains single-tone in a joint drive scan."""
    exp = Mock(qubits={"Q": SimpleNamespace(control_frequency_ef=4.0)})
    opts = grid.F0G1CalibrationOptions(
        amplitudes=(0.98, 1.0), ef_amplitudes=(0.01, 0.04), ef_frequency_offsets=(0.0,)
    )
    points = tuple(
        grid.F0G1PulseParameters(1000, 1.8, a, 4, 0) for a in opts.amplitudes
    )
    scan = grid.F0G1PopulationScan(
        points, ("f",), np.array([[[8, 1, 1]], [[9, 0, 1]]]), ()
    )
    acquire = AsyncMock()
    monkeypatch.setattr(grid, "_measure_populations", acquire)
    asyncio.run(grid.scan_f0g1_ef_drive(exp, "Q", scan, options=opts))
    assert acquire.await_args is not None
    assert [(p.f0g1_amplitude, p.ef_amplitude) for p in acquire.await_args.args[2]] == [
        (0.98, 0.01)
    ]
