"""Execution-result payload for QuEL-1 backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quel1BackendExecutionResult:
    """
    Backend-level status, data, and config returned from qube-calib execution.

    Notes
    -----
    Each `data[target]` list preserves the target capture-window execution order.
    """

    status: dict
    data: dict
    config: dict
