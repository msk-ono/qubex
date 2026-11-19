"""Offline reset workflow contracts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from qubex.backend.quel3 import Quel3BackendController
from qubex.contrib import F0G1CalibrationOptions, calibrate_f0g1_pulse
from qubex.pulse import Rect


@pytest.mark.parametrize("failure_stage", [None, "duration", "verification"])
def test_independent_verification_controls_success(failure_stage):
    """Only an independently verified candidate exposes a reset schedule."""
    exp = Mock()
    exp.backend_controller = Mock(spec=Quel3BackendController)
    exp.targets = {
        label: SimpleNamespace(frequency=freq)
        for label, freq in [
            ("Q", 5.0),
            ("Q-ef", 4.8),
            ("Q-f0g1", 2.8),
            ("Q-ef-reset", 4.8),
        ]
    }
    exp.ctx.measurement.sampling_period = 0.4
    exp.qubits = {
        "Q": SimpleNamespace(control_frequency_ge=5.0, control_frequency_ef=4.8)
    }
    exp.resonators = {"Q": SimpleNamespace(frequency=7.0)}
    classifier = Mock(n_states=3, label_map={0: 0, 1: 1, 2: 2})
    classifier.predict.side_effect = lambda iq: iq.real.astype(int)
    exp.classifiers = {"Q": classifier}
    exp.get_pi_pulse.return_value = Rect(
        duration=20.0, amplitude=0.1, sampling_period=0.4
    )
    schedules = []

    async def acquire(schedule, *, sweep_values, n_shots, **kwargs):
        schedules.extend(schedule(int(i)) for i in sweep_values)
        value = int(
            (failure_stage == "verification" and n_shots == 4)
            or (failure_stage == "duration" and len(sweep_values) == 2)
        )
        return SimpleNamespace(
            data={"Q": [np.full((len(sweep_values), n_shots), value, dtype=complex)]}
        )

    exp.run_sweep_measurement = AsyncMock(side_effect=acquire)
    options = F0G1CalibrationOptions(
        durations=(100.0, 200.0),
        frequency_offsets=(0.0,),
        amplitudes=(0.5,),
        ef_frequency_offsets=(0.0,),
        ef_amplitudes=(0.005,),
        n_shots=2,
        verification_shots=4,
    )
    result = asyncio.run(calibrate_f0g1_pulse(exp, "Q", options=options))
    assert result.success is (failure_stage is None)
    assert (result.pulse is None) is (failure_stage is not None)
    assert result.parameters is not None
    if failure_stage == "duration":
        assert "verification" not in result.scans
        assert result.parameters.duration == pytest.approx(options.search_duration)
    else:
        assert result.parameters.duration == pytest.approx(100.0)
        assert result.scans["verification"].counts.sum() == 12
    assert all(s.get_frequency("Q-ef") is None for s in schedules if "Q-ef" in s.labels)
