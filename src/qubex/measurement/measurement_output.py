"""Utilities for canonical measurement output labels."""

from __future__ import annotations

from typing import Any


def _call_string_resolver(resolver: object, target_label: str) -> str | None:
    """Return string resolver output, or None when the resolver cannot map a label."""
    if not callable(resolver):
        return None
    try:
        return str(resolver(target_label))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def resolve_measurement_output_label(
    *,
    experiment_system: object | None,
    target_label: str,
) -> str:
    """Resolve the canonical `MeasurementResult.data` key for one capture target."""
    target_registry: Any | None = None
    if experiment_system is not None:
        target_registry = getattr(experiment_system, "target_registry", None)

    if target_registry is not None:
        resolved = _call_string_resolver(
            getattr(target_registry, "measurement_output_label", None),
            target_label,
        )
        if resolved is not None and resolved != target_label:
            return resolved

    if experiment_system is not None:
        resolved = _call_string_resolver(
            getattr(experiment_system, "resolve_qubit_label", None),
            target_label,
        )
        if resolved is not None:
            return resolved

    if target_label.startswith("R") and len(target_label) > 1:
        return target_label[1:]
    return target_label
