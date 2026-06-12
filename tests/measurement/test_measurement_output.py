"""Tests for canonical measurement output label resolution."""

from __future__ import annotations

from types import SimpleNamespace

from qubex.measurement.measurement_output import resolve_measurement_output_label


def test_resolve_measurement_output_label_prefers_registry_output_label() -> None:
    """Given registry output label, resolver should use it as canonical target."""

    class _TargetRegistry:
        @staticmethod
        def measurement_output_label(target_label: str) -> str:
            return "Q17" if target_label == "raw-readout-target" else target_label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda _target_label: "wrong-target",
    )

    assert (
        resolve_measurement_output_label(
            experiment_system=experiment_system,
            target_label="raw-readout-target",
        )
        == "Q17"
    )


def test_resolve_measurement_output_label_uses_experiment_resolver_fallback() -> None:
    """Given no registry output label, resolver should use experiment resolver."""
    experiment_system = SimpleNamespace(
        resolve_qubit_label=lambda target_label: (
            "Q00" if target_label == "custom-readout" else target_label
        ),
    )

    assert (
        resolve_measurement_output_label(
            experiment_system=experiment_system,
            target_label="custom-readout",
        )
        == "Q00"
    )


def test_resolve_measurement_output_label_falls_back_when_registry_keeps_label() -> (
    None
):
    """Given unresolved registry output, resolver should still allow legacy mapping."""

    class _TargetRegistry:
        @staticmethod
        def measurement_output_label(target_label: str) -> str:
            return target_label

    experiment_system = SimpleNamespace(
        target_registry=_TargetRegistry(),
        resolve_qubit_label=lambda target_label: (
            "Q00" if target_label == "RQ00" else target_label
        ),
    )

    assert (
        resolve_measurement_output_label(
            experiment_system=experiment_system,
            target_label="RQ00",
        )
        == "Q00"
    )


def test_resolve_measurement_output_label_uses_legacy_readout_fallback() -> None:
    """Given legacy readout label, resolver should strip the leading R."""
    assert (
        resolve_measurement_output_label(
            experiment_system=None,
            target_label="RQ00",
        )
        == "Q00"
    )
