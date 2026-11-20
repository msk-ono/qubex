"""Hardware contracts shared by reset calibration and transfer spectroscopy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from qubex.backend.quel3 import Quel3BackendController
from qubex.contrib.experiment._f0g1_acquisition import (
    F0G1AcquisitionOptions,
    F0G1PulseParameters,
    _measure_populations,
    build_f0g1_reset,
)
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.measurement_schedule_builder import MeasurementScheduleBuilder
from qubex.pulse import Rect, Waveform


@pytest.fixture
def experiment(monkeypatch):
    """Return calibrated reset aliases and a three-state assignment model."""
    exp = Mock()
    exp.ctx.measurement.sampling_period = 0.4
    monkeypatch.setattr(
        Waveform, "SAMPLING_PERIOD", exp.ctx.measurement.sampling_period
    )
    exp.backend_controller = Mock(spec=Quel3BackendController)
    exp.targets = {
        label: SimpleNamespace(frequency=freq)
        for label, freq in (
            ("Q", 4.2),
            ("Q-ef", 4.0),
            ("Q-f0g1", 1.85),
            ("Q-ef-reset", 4.0),
        )
    }
    classifier = Mock(n_states=3, label_map={0: 0, 1: 1, 2: 2})
    classifier.predict.side_effect = lambda iq: iq.real.astype(int)
    exp.classifiers = {"Q": classifier}
    exp.get_pi_pulse.return_value = Rect(
        duration=20, amplitude=0.1, sampling_period=0.4
    )
    return exp


@pytest.mark.parametrize(("state", "prep_ns"), [("g", 0), ("e", 20), ("f", 40)])
@pytest.mark.parametrize("wait_ns", [100, 0])
def test_final_capture_follows_preparation_drive_and_wait(
    experiment, state, prep_ns, wait_ns
):
    """OFF references and driven points must reach final readout after the full sequence."""
    opts = F0G1AcquisitionOptions(wait_duration=wait_ns, n_shots=4)
    points = (
        F0G1PulseParameters(0, 1.85, 0, 4.0, 0),
        F0G1PulseParameters(91.6, 1.85, 0, 4.0, 0),
        F0G1PulseParameters(91.6, 1.85, 0.8, 4.0, 0.1),
    )
    builder = MeasurementScheduleBuilder(
        control_params=Mock(readout_amplitude={"RQ": 0.1}),
        pulse_factory=Mock(
            readout_pulse=Mock(return_value=Rect(duration=16, amplitude=0.1))
        ),
        targets={label: Mock(is_pump=False) for label in experiment.targets},
        mux_dict={},
        target_registry=Mock(resolve_read_label=Mock(return_value="RQ")),
        constraint_profile=MeasurementConstraintProfile.quel3(
            experiment.ctx.measurement.sampling_period
        ),
    )

    async def acquire(
        schedule, *, sweep_values, n_shots, frequencies, final_measurement, **_
    ):
        for i in sweep_values:
            measured = builder.build(
                schedule=schedule(i),
                frequencies=frequencies,
                final_measurement=final_measurement,
            )
            (capture,) = measured.capture_schedule.captures
            assert capture.channels == ["RQ"]
            assert capture.start_time == pytest.approx(
                prep_ns + points[i].duration + wait_ns
            )
        return SimpleNamespace(
            data={"Q": [np.zeros((len(sweep_values), n_shots), dtype=complex)]}
        )

    experiment.run_sweep_measurement = acquire
    asyncio.run(_measure_populations(experiment, "Q", points, opts, (state,)))


def test_off_control_has_same_elapsed_time_as_two_tone(experiment):
    """A zero-drive reference waits the full envelope and the post-drive delay."""
    opts = F0G1AcquisitionOptions(wait_duration=10)
    off = F0G1PulseParameters(100, 1.8, 0, 3.95, 0)
    on = F0G1PulseParameters(100, 1.8, 0.9, 3.95, 0.05)
    for p in (off, on):
        pulse = build_f0g1_reset(experiment, "Q", p, options=opts)
        assert pulse.duration == pytest.approx(110)
        assert not np.any(pulse.get_sampled_sequence("Q"))
    pulse = build_f0g1_reset(experiment, "Q", on, options=opts)
    for alias, carrier, amplitude in (("Q-f0g1", 1.85, 0.9), ("Q-ef-reset", 4, 0.05)):
        assert pulse.get_frequency(alias) == pytest.approx(carrier)
        samples = pulse.get_sampled_sequence(alias)
        plateau = samples[np.isclose(abs(samples), amplitude)]
        np.testing.assert_allclose(
            np.angle(plateau[1:] * plateau[:-1].conj()),
            2 * np.pi * 0.05 * 0.4,
            atol=1e-6,
        )


def test_shared_port_budget_and_digital_bandwidth(experiment):
    """The two tones share unit amplitude and offsets below digital Nyquist."""
    with pytest.raises(ValueError, match="amplitude"):
        F0G1PulseParameters(100, 1.8, 0.98, 3.95, 0.04)
    with pytest.raises(ValueError, match="Nyquist"):
        build_f0g1_reset(experiment, "Q", F0G1PulseParameters(100, 3.5, 0.8, 4, 0))


def test_chunked_sweep_uses_registered_profiles_and_retains_order(experiment):
    """Fixed carriers span hardware chunks while rows retain their preparation and frequency order."""
    opts = F0G1AcquisitionOptions(n_shots=6)
    points = tuple(
        F0G1PulseParameters(50, 1.72 + i * 0.005, 0.6, 4.0, 0.04) for i in range(53)
    )
    sizes = []

    async def acquire(schedule, *, sweep_values, n_shots, frequencies, **_):
        sizes.append(len(sweep_values))
        assert frequencies == {"Q-f0g1": 1.85, "Q-ef-reset": 4.0}
        assigned = []
        for i in sweep_values:
            ps = schedule(int(i))
            assert ps.get_frequency("Q-ef") is None
            assert ps.get_frequency("Q-f0g1") == frequencies["Q-f0g1"]
            samples = ps.get_sampled_sequence("Q-f0g1")
            plateau = samples[np.isclose(abs(samples), 0.6)]
            offset = -np.angle(plateau[1] * plateau[0].conj()) / (2 * np.pi * 0.4)
            idx = round((frequencies["Q-f0g1"] + offset - 1.72) / 0.005)
            assigned.append(np.full(n_shots, idx % 3, dtype=complex))
        return SimpleNamespace(data={"Q": [np.stack(assigned)]})

    experiment.run_sweep_measurement = acquire
    scan = asyncio.run(_measure_populations(experiment, "Q", points, opts, ("f", "e")))
    assert max(sizes) <= 48
    assert sum(sizes) == 106
    assert scan.parameters == points
    np.testing.assert_array_equal(
        scan.counts.argmax(axis=-1), np.tile(np.arange(53)[:, None] % 3, (1, 2))
    )
