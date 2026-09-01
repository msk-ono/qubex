"""Tests for backend-aware frequency-sweep planning."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.measurement import Measurement
from qubex.system import PortType


class _SettingsRecorder:
    """Record temporary QuEL-1 backend settings."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.active = False

    @contextmanager
    def modified_backend_settings(self, label: str, **settings: Any):
        """Record one temporary backend-settings context."""
        self.calls.append({"label": label, **settings})
        self.active = True
        try:
            yield
        finally:
            self.active = False


class _Quel3ConfigurationStub:
    """Provide deployed-range metadata for QuEL-3 planning."""

    def __init__(self, lower_hz: float, upper_hz: float) -> None:
        self.lower_hz = lower_hz
        self.upper_hz = upper_hz
        self.temporary_calls: list[dict[str, Any]] = []
        self.active = False

    def get_deployed_frequency_range(
        self,
        *,
        box_id: str,
        target_label: str,
    ) -> tuple[float, float]:
        """Return the configured frequency range."""
        _ = (box_id, target_label)
        return self.lower_hz, self.upper_hz

    @contextmanager
    def temporary_frequency_range(self, **kwargs: Any):
        """Record one temporary frequency-range context."""
        self.temporary_calls.append(kwargs)
        self.active = True
        try:
            yield
        finally:
            self.active = False


class _Quel3ExecutionStub:
    """Record resolver invalidation."""

    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate_instrument_resolver(self) -> None:
        """Record one resolver invalidation."""
        self.invalidations += 1


def _make_measurement(
    *,
    backend_controller: object,
    system_manager: object,
    target_frequency: float = 5.0,
    fine_frequency: float = 5.0,
) -> Measurement:
    """Build a Measurement with a minimal frequency-planning runtime."""
    measurement = Measurement(
        chip_id="TEST",
        qubits=["Q00"],
        load_configs=False,
        connect_devices=False,
    )
    port = SimpleNamespace(box_id="BOX0", type=PortType.CTRL)
    target = SimpleNamespace(
        label="Q00",
        frequency=target_frequency,
        fine_frequency=fine_frequency,
        sideband="U",
        channel=SimpleNamespace(port=port),
    )
    experiment_system = SimpleNamespace(
        get_target=lambda _label: target,
        get_box=lambda _box_id: SimpleNamespace(
            traits=SimpleNamespace(readout_cnco_center=1_500_000_000)
        ),
    )
    context = SimpleNamespace(
        backend_controller=backend_controller,
        experiment_system=experiment_system,
        system_manager=system_manager,
    )
    session_service = SimpleNamespace(backend_controller=backend_controller)
    measurement.__dict__["_context"] = context
    measurement.__dict__["_session_service"] = session_service
    measurement.execution_service.__dict__["_context"] = context
    measurement.execution_service.__dict__["_session_service"] = session_service
    return measurement


def test_quel1_plan_uses_current_configuration_when_all_frequencies_fit() -> None:
    """A QuEL-1 sweep already covered by the current NCO should not retune."""
    recorder = _SettingsRecorder()
    controller = object.__new__(Quel1BackendController)
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=recorder,
    )

    plan = measurement.plan_frequency_sweep(
        "Q00",
        frequencies=[4.8, 5.0, 5.2],
        max_segment_width=0.1,
    )

    assert plan.frequencies == (4.8, 5.0, 5.2)
    assert [segment.frequencies for segment in plan.segments] == [(4.8, 5.0, 5.2)]
    assert plan.bounds == (4.8, 5.2)
    assert plan.requires_reconfiguration is False
    with plan.activate(plan.segments[0]):
        assert recorder.active is False
    assert recorder.calls == []


def test_quel1_plan_minimizes_retuned_segments_with_width_hint() -> None:
    """A wide QuEL-1 sweep should use the longest contiguous hinted segments."""
    recorder = _SettingsRecorder()
    controller = object.__new__(Quel1BackendController)
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=recorder,
    )

    plan = measurement.plan_frequency_sweep(
        "Q00",
        frequencies=[4.0, 4.1, 4.2, 4.3],
        max_segment_width=0.2,
    )

    assert [segment.frequencies for segment in plan.segments] == [
        (4.0, 4.1, 4.2),
        (4.3,),
    ]
    assert plan.bounds == (4.0, 4.2, 4.3)
    assert plan.requires_reconfiguration is True
    for segment in plan.segments:
        with plan.activate(segment):
            assert recorder.active is True
        assert recorder.active is False
    assert [call["label"] for call in recorder.calls] == ["Q00", "Q00"]


def test_quel1_plan_preserves_descending_nonuniform_frequency_order() -> None:
    """A descending nonuniform QuEL-1 sweep should keep its original order."""
    recorder = _SettingsRecorder()
    controller = object.__new__(Quel1BackendController)
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=recorder,
    )

    plan = measurement.plan_frequency_sweep(
        "Q00",
        frequencies=[4.4, 4.2, 4.1, 3.9],
        max_segment_width=0.3,
    )

    assert plan.frequencies == (4.4, 4.2, 4.1, 3.9)
    assert [segment.frequencies for segment in plan.segments] == [
        (4.4, 4.2, 4.1),
        (3.9,),
    ]


def test_quel1_plan_uses_fine_frequency_as_generated_start() -> None:
    """A generated QuEL-1 sweep should start from the current fine frequency."""
    recorder = _SettingsRecorder()
    controller = object.__new__(Quel1BackendController)
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=recorder,
        target_frequency=6.0,
        fine_frequency=5.0,
    )

    plan = measurement.plan_frequency_sweep(
        "Q00",
        frequency_step=-0.1,
        frequency_count=3,
    )

    assert plan.frequencies == pytest.approx((5.0, 4.9, 4.8))
    assert plan.requires_reconfiguration is False


def test_quel3_plan_uses_target_frequency_as_generated_start() -> None:
    """A generated QuEL-3 sweep should start from the logical target frequency."""
    controller = object.__new__(Quel3BackendController)
    controller.__dict__["_configuration_manager"] = _Quel3ConfigurationStub(
        5.5e9,
        6.5e9,
    )
    controller.__dict__["_execution_manager"] = _Quel3ExecutionStub()
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=SimpleNamespace(),
        target_frequency=6.0,
        fine_frequency=1.0,
    )

    plan = measurement.plan_frequency_sweep(
        "Q00",
        frequency_step=0.1,
        frequency_count=3,
    )

    assert plan.frequencies == pytest.approx((6.0, 6.1, 6.2))
    assert [segment.frequencies for segment in plan.segments] == [
        pytest.approx((6.0, 6.1, 6.2))
    ]
    assert plan.requires_reconfiguration is False


def test_quel3_plan_temporarily_expands_out_of_range_sweep() -> None:
    """An out-of-range QuEL-3 sweep should expand and restore its deployment."""
    configuration = _Quel3ConfigurationStub(4.9e9, 5.1e9)
    execution = _Quel3ExecutionStub()
    controller = object.__new__(Quel3BackendController)
    controller.__dict__["_configuration_manager"] = configuration
    controller.__dict__["_execution_manager"] = execution
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=SimpleNamespace(),
    )

    plan = measurement.plan_frequency_sweep(
        "Q00",
        frequencies=[4.8, 5.2],
    )

    assert plan.requires_reconfiguration is True
    with plan.activate(plan.segments[0]):
        assert configuration.active is True
    assert configuration.active is False
    assert configuration.temporary_calls == [
        {
            "box_id": "BOX0",
            "target_label": "Q00",
            "frequency_range_min_hz": 4.8e9,
            "frequency_range_max_hz": 5.2e9,
        }
    ]
    assert execution.invalidations == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "frequencies"),
        ({"frequencies": []}, "non-empty"),
        (
            {"frequencies": [5.0], "frequency_step": 0.1},
            "cannot be combined",
        ),
        (
            {"frequency_step": 0.1, "frequency_count": 0},
            "positive",
        ),
    ],
)
def test_frequency_sweep_plan_rejects_invalid_frequency_specs(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    """Invalid explicit or generated sweep specifications should be rejected."""
    recorder = _SettingsRecorder()
    controller = object.__new__(Quel1BackendController)
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=recorder,
    )

    with pytest.raises(ValueError, match=message):
        measurement.plan_frequency_sweep("Q00", **kwargs)


def test_frequency_sweep_plan_rejects_foreign_segment() -> None:
    """A plan should reject activation of a segment owned by another plan."""
    recorder = _SettingsRecorder()
    controller = object.__new__(Quel1BackendController)
    measurement = _make_measurement(
        backend_controller=controller,
        system_manager=recorder,
    )
    first = measurement.plan_frequency_sweep("Q00", frequencies=[5.0])
    second = measurement.plan_frequency_sweep("Q00", frequencies=[5.1])

    with (
        pytest.raises(ValueError, match="does not belong"),
        first.activate(second.segments[0]),
    ):
        pass
