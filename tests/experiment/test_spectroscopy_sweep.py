"""Spectroscopy sweep execution contracts."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.experiment.services.characterization_service import CharacterizationService


@pytest.fixture
def spectroscopy(monkeypatch):
    """Provide backend-independent sweep execution with deterministic IQ."""
    batches = []
    preparations = []

    @contextmanager
    def prepare_frequency_sweep(label, **kwargs):
        preparations.append((label, kwargs))
        yield kwargs.get("center_frequency") or 6.0

    def build_measurement_schedule(schedule, **kwargs):
        return SimpleNamespace(pulse_schedule=schedule, options=kwargs)

    async def run_sweep_measurement(schedule, *, sweep_values, **kwargs):
        points = [schedule(value) for value in sweep_values]
        batches.append((points, kwargs))
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    data={
                        "Q00": [
                            SimpleNamespace(
                                kerneled=np.exp(
                                    -2j
                                    * np.pi
                                    * point.options["frequencies"].get("R00", 6.0)
                                    + 0.3j * (len(batches) - 1)
                                )
                            )
                        ]
                    }
                )
                for point in points
            ]
        )

    box = SimpleNamespace(
        traits=SimpleNamespace(
            default_readout_frequency_range=(6.0, 6.04, 0.01),
            default_control_frequency_range=(5.0, 5.04, 0.01),
        )
    )
    ctx = SimpleNamespace(
        qubit_labels=["Q00"],
        resolve_qubit_label=lambda _: "Q00",
        resolve_read_label=lambda _: "R00",
        params=SimpleNamespace(
            control_amplitude={"Q00": 0.1}, readout_amplitude={"Q00": 0.2}
        ),
        targets={"R00": SimpleNamespace(frequency=6.0)},
        experiment_system=SimpleNamespace(
            get_readout_box_for_qubit=lambda _: box,
            get_control_box_for_qubit=lambda _: box,
            get_mux_by_qubit=lambda _: SimpleNamespace(label="MUX0"),
        ),
        system_manager=SimpleNamespace(prepare_frequency_sweep=prepare_frequency_sweep),
        reset_awg_and_capunits=lambda **_: None,
    )
    service = CharacterizationService(
        experiment_context=cast(Any, ctx),
        measurement_service=cast(
            Any,
            SimpleNamespace(
                build_measurement_schedule=build_measurement_schedule,
                run_sweep_measurement=run_sweep_measurement,
            ),
        ),
        calibration_service=cast(Any, None),
        pulse_service=cast(Any, None),
    )
    monkeypatch.setattr(
        "qubex.experiment.services.characterization_service.ExperimentUtil.split_frequency_range",
        lambda *, frequency_range, **_: [
            np.asarray(frequency_range)[:2],
            np.asarray(frequency_range)[2:],
        ],
    )
    return service, batches, preparations


def test_electrical_delay_uses_one_sweep_and_resolves_custom_labels(spectroscopy):
    """Electrical delay measures one ordered sweep using resolved schedule labels."""
    service, batches, preparations = spectroscopy
    delay = service.measure_electrical_delay(
        "custom-target",
        f_start=6.0,
        df=0.001,
        n_samples=4,
        plot=False,
        confirm=False,
    )
    assert delay == pytest.approx(1.0, abs=1e-10)
    assert len(batches) == len(preparations) == 1
    points, _ = batches[0]
    assert preparations[0][0] == "R00"
    assert all(p.options["readout_amplitudes"] == {"Q00": 1.0} for p in points)


def test_electrical_delay_cancellation_does_not_measure(spectroscopy, monkeypatch):
    """Declining frequency preparation returns without submitting measurements."""
    service, batches, _ = spectroscopy

    @contextmanager
    def cancelled(*args, **kwargs):
        yield None

    monkeypatch.setattr(
        service.ctx.system_manager, "prepare_frequency_sweep", cancelled
    )
    assert service.measure_electrical_delay("Q00", plot=False) is None
    assert batches == []
