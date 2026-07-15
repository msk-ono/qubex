"""QuEL-3 backend-specific hardware constants."""

from __future__ import annotations

from typing import Final

SAMPLING_PERIOD_NS: Final[float] = 0.4
READOUT_SAMPLING_PERIOD_NS: Final[float] = 0.8
CAPTURE_DECIMATION_FACTOR: Final[int] = 1
MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ: Final[float] = 0.9
MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ: Final[float] = 1.8
