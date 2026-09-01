"""Backend-aware frequency-sweep planning."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from qubex.backend import BackendController
from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.system import ExperimentSystem, MixingUtil, PortType, SystemManager, Target
from qubex.system.quel1.quel1_system_constants import (
    AWG_MAX_HZ,
    CNCO_CENTER_CTRL_HZ,
)

_GHZ_TO_HZ = 1e9
_AWG_MAX_GHZ = AWG_MAX_HZ / _GHZ_TO_HZ
_FLOAT_TOLERANCE_GHZ = 1e-12
_ActivationFactory = Callable[[], AbstractContextManager[None]]


@dataclass(frozen=True)
class FrequencySweepSegment:
    """One contiguous frequency-sweep segment in execution order."""

    frequencies: tuple[float, ...]


class FrequencySweepPlan:
    """Hold backend-aware frequency segments and their activation contexts."""

    def __init__(
        self,
        *,
        target: str,
        segments: Sequence[FrequencySweepSegment],
        activations: Sequence[_ActivationFactory],
        reconfigurations: Sequence[bool],
    ) -> None:
        """Initialize one session-bound frequency-sweep plan."""
        if not segments or len(segments) != len(activations):
            raise ValueError("A frequency-sweep plan requires matching segments.")
        if len(segments) != len(reconfigurations):
            raise ValueError("Frequency-sweep reconfiguration metadata is invalid.")
        self._target = target
        self._segments = tuple(segments)
        self._activations = tuple(activations)
        self._reconfigurations = tuple(reconfigurations)
        self._active = False

    @property
    def target(self) -> str:
        """Return the planned target label."""
        return self._target

    @property
    def frequencies(self) -> tuple[float, ...]:
        """Return all planned frequencies in execution order in GHz."""
        return tuple(
            frequency for segment in self._segments for frequency in segment.frequencies
        )

    @property
    def segments(self) -> tuple[FrequencySweepSegment, ...]:
        """Return the backend-compatible frequency segments."""
        return self._segments

    @property
    def bounds(self) -> tuple[float, ...]:
        """Return the first frequency and each segment's final frequency in GHz."""
        return (
            self._segments[0].frequencies[0],
            *(segment.frequencies[-1] for segment in self._segments),
        )

    @property
    def requires_reconfiguration(self) -> bool:
        """Return whether any segment changes backend configuration."""
        return any(self._reconfigurations)

    @contextmanager
    def activate(self, segment: FrequencySweepSegment) -> Iterator[None]:
        """
        Activate the backend configuration for one planned segment.

        Parameters
        ----------
        segment : FrequencySweepSegment
            Segment returned by this plan.

        Yields
        ------
        None
            Context where the segment's backend configuration is active.

        Raises
        ------
        ValueError
            If the segment belongs to a different plan.
        RuntimeError
            If another segment from this plan is already active.
        """
        try:
            index = next(
                index
                for index, planned_segment in enumerate(self._segments)
                if planned_segment is segment
            )
        except StopIteration as exc:
            raise ValueError(
                "Frequency-sweep segment does not belong to this plan."
            ) from exc
        if self._active:
            raise RuntimeError("A frequency-sweep plan segment is already active.")

        self._active = True
        try:
            with self._activations[index]():
                yield
        finally:
            self._active = False


class FrequencySweepPlanner:
    """Build frequency-sweep plans for the active measurement backend."""

    def __init__(
        self,
        *,
        system_manager: SystemManager,
        experiment_system: ExperimentSystem,
        backend_controller: BackendController,
    ) -> None:
        """Initialize a planner from the active measurement runtime."""
        self._system_manager = system_manager
        self._experiment_system = experiment_system
        self._backend_controller = backend_controller

    def plan(
        self,
        target: str,
        *,
        frequencies: ArrayLike | None = None,
        start_frequency: float | None = None,
        frequency_step: float | None = None,
        frequency_count: int | None = None,
        max_segment_width: float | None = None,
    ) -> FrequencySweepPlan:
        """Build a backend-compatible frequency-sweep plan in GHz."""
        target_model = self._experiment_system.get_target(target)
        resolved_frequencies = self._resolve_frequencies(
            target=target_model,
            frequencies=frequencies,
            start_frequency=start_frequency,
            frequency_step=frequency_step,
            frequency_count=frequency_count,
        )
        width = self._normalize_max_segment_width(max_segment_width)

        if isinstance(self._backend_controller, Quel1BackendController):
            return self._plan_quel1(
                target=target_model,
                frequencies=resolved_frequencies,
                max_segment_width=width,
            )
        if isinstance(self._backend_controller, Quel3BackendController):
            return self._plan_quel3(
                target=target_model,
                frequencies=resolved_frequencies,
            )
        raise TypeError("Active backend does not support frequency-sweep planning.")

    def _resolve_frequencies(
        self,
        *,
        target: Target,
        frequencies: ArrayLike | None,
        start_frequency: float | None,
        frequency_step: float | None,
        frequency_count: int | None,
    ) -> tuple[float, ...]:
        """Normalize explicit or generated sweep frequencies."""
        generation_values = (start_frequency, frequency_step, frequency_count)
        if frequencies is not None:
            if any(value is not None for value in generation_values):
                raise ValueError(
                    "`frequencies` cannot be combined with generated sweep options."
                )
            values = np.asarray(frequencies, dtype=np.float64)
            if values.ndim != 1 or values.size == 0:
                raise ValueError("`frequencies` must be a non-empty 1D array.")
        else:
            if frequency_step is None or frequency_count is None:
                raise ValueError(
                    "Provide `frequencies` or both `frequency_step` and "
                    "`frequency_count`."
                )
            if frequency_count <= 0:
                raise ValueError("`frequency_count` must be positive.")
            if not math.isfinite(float(frequency_step)) or frequency_step == 0:
                raise ValueError("`frequency_step` must be finite and non-zero.")
            if start_frequency is None:
                start_frequency = self._default_start_frequency(target)
            if not math.isfinite(float(start_frequency)):
                raise ValueError("`start_frequency` must be finite.")
            values = float(start_frequency) + float(frequency_step) * np.arange(
                frequency_count,
                dtype=np.float64,
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Sweep frequencies must all be finite.")
        return tuple(float(value) for value in values)

    def _default_start_frequency(self, target: Target) -> float:
        """Resolve the backend-specific generated sweep start frequency in GHz."""
        if isinstance(self._backend_controller, Quel1BackendController):
            return float(target.fine_frequency)
        if isinstance(self._backend_controller, Quel3BackendController):
            return float(target.frequency)
        raise TypeError("Active backend does not support frequency-sweep planning.")

    @staticmethod
    def _normalize_max_segment_width(value: float | None) -> float | None:
        """Validate an optional maximum segment-width hint in GHz."""
        if value is None:
            return None
        width = float(value)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("`max_segment_width` must be finite and positive.")
        return width

    def _plan_quel1(
        self,
        *,
        target: Target,
        frequencies: tuple[float, ...],
        max_segment_width: float | None,
    ) -> FrequencySweepPlan:
        """Build a QuEL-1 plan that minimizes coarse-frequency retunes."""
        current_frequency = float(target.fine_frequency)
        if all(
            abs(frequency - current_frequency) <= _AWG_MAX_GHZ + _FLOAT_TOLERANCE_GHZ
            for frequency in frequencies
        ):
            return FrequencySweepPlan(
                target=target.label,
                segments=(FrequencySweepSegment(frequencies),),
                activations=(nullcontext,),
                reconfigurations=(False,),
            )

        width = min(
            2 * _AWG_MAX_GHZ,
            max_segment_width if max_segment_width is not None else math.inf,
        )
        frequency_segments = self._split_longest_contiguous(
            frequencies,
            max_width=width,
        )
        segments = tuple(FrequencySweepSegment(values) for values in frequency_segments)
        activations = tuple(
            self._quel1_activation(target=target, frequencies=values)
            for values in frequency_segments
        )
        return FrequencySweepPlan(
            target=target.label,
            segments=segments,
            activations=activations,
            reconfigurations=(True,) * len(segments),
        )

    @staticmethod
    def _split_longest_contiguous(
        frequencies: tuple[float, ...],
        *,
        max_width: float,
    ) -> tuple[tuple[float, ...], ...]:
        """Split values into the longest ordered slices within one width."""
        segments: list[tuple[float, ...]] = []
        start = 0
        while start < len(frequencies):
            stop = start + 1
            lower = upper = frequencies[start]
            while stop < len(frequencies):
                candidate = frequencies[stop]
                next_lower = min(lower, candidate)
                next_upper = max(upper, candidate)
                if next_upper - next_lower > max_width + _FLOAT_TOLERANCE_GHZ:
                    break
                lower, upper = next_lower, next_upper
                stop += 1
            segments.append(frequencies[start:stop])
            start = stop
        return tuple(segments)

    def _quel1_activation(
        self,
        *,
        target: Target,
        frequencies: tuple[float, ...],
    ) -> _ActivationFactory:
        """Build one QuEL-1 coarse-frequency activation context."""
        sideband = target.sideband
        if sideband not in ("U", "L"):
            raise ValueError(
                f"QuEL-1 frequency sweeps require a sideband for `{target.label}`."
            )
        port = target.channel.port
        if port.type == PortType.READ_OUT:
            cnco_center = self._experiment_system.get_box(
                port.box_id
            ).traits.readout_cnco_center
            if cnco_center is None:
                raise ValueError(
                    f"Readout CNCO center is unavailable for `{target.label}`."
                )
        else:
            cnco_center = CNCO_CENTER_CTRL_HZ
        center_hz = (min(frequencies) + max(frequencies)) / 2 * _GHZ_TO_HZ
        lo, cnco, _ = MixingUtil.calc_lo_cnco(
            center_hz,
            ssb=sideband,
            cnco_center=cnco_center,
        )

        def activate() -> AbstractContextManager[None]:
            return self._system_manager.modified_backend_settings(
                label=target.label,
                lo_freq=lo,
                cnco_freq=cnco,
                fnco_freq=0,
            )

        return activate

    def _plan_quel3(
        self,
        *,
        target: Target,
        frequencies: tuple[float, ...],
    ) -> FrequencySweepPlan:
        """Build one direct-frequency QuEL-3 segment."""
        controller = self._backend_controller
        if not isinstance(controller, Quel3BackendController):
            raise TypeError("QuEL-3 sweep planning requires a QuEL-3 controller.")
        port = target.channel.port
        lower_hz, upper_hz = (
            controller.configuration_manager.get_deployed_frequency_range(
                box_id=port.box_id,
                target_label=target.label,
            )
        )
        requested_lower_hz = min(frequencies) * _GHZ_TO_HZ
        requested_upper_hz = max(frequencies) * _GHZ_TO_HZ
        requires_reconfiguration = (
            requested_lower_hz < lower_hz or requested_upper_hz > upper_hz
        )

        def activate() -> AbstractContextManager[None]:
            if not requires_reconfiguration:
                return nullcontext()
            return controller.temporary_frequency_range(
                box_id=port.box_id,
                target_label=target.label,
                frequency_range_min_hz=requested_lower_hz,
                frequency_range_max_hz=requested_upper_hz,
            )

        return FrequencySweepPlan(
            target=target.label,
            segments=(FrequencySweepSegment(frequencies),),
            activations=(activate,),
            reconfigurations=(requires_reconfiguration,),
        )
