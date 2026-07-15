"""Result models for QuEL-3 spectroscopy sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .payload import Quel3CaptureMode
from .result import Quel3BackendExecutionResult


@dataclass(frozen=True)
class Quel3ResonatorSpectroscopyResult:
    """
    Store one packed QuEL-3 resonator frequency scan result.

    Notes
    -----
    `iq` has shape `(frequency,)` for averaged modes and `(frequency, shot)`
    for `VALUES_PER_ITER`. `sampling_period_ns` is set only for
    `AVERAGED_WAVEFORM`.
    """

    frequency_range: npt.NDArray[np.float64]
    capture_mode: Quel3CaptureMode
    captures: tuple[npt.NDArray[np.complex128], ...]
    iq: npt.NDArray[np.complex128]
    sampling_period_ns: float | None
    backend_results: tuple[Quel3BackendExecutionResult, ...]


@dataclass(frozen=True)
class Quel3QubitSpectroscopyResult:
    """
    Store one packed QuEL-3 qubit frequency scan result.

    Notes
    -----
    `iq` has shape `(frequency,)` for averaged modes and `(frequency, shot)`
    for `VALUES_PER_ITER`. `sampling_period_ns` is set only for
    `AVERAGED_WAVEFORM`.
    """

    frequency_range: npt.NDArray[np.float64]
    readout_frequency: float
    capture_mode: Quel3CaptureMode
    captures: tuple[npt.NDArray[np.complex128], ...]
    iq: npt.NDArray[np.complex128]
    sampling_period_ns: float | None
    backend_results: tuple[Quel3BackendExecutionResult, ...]
