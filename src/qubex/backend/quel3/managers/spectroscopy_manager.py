"""Packed spectroscopy manager for the QuEL-3 backend."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from numbers import Integral
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3.models import (
    InstrumentDeployRequest,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3CaptureWindow,
    Quel3ExecutionPayload,
    Quel3FixedTimeline,
    Quel3QubitSpectroscopyResult,
    Quel3ResonatorSpectroscopyResult,
    Quel3Waveform,
    Quel3WaveformEvent,
    RoleName,
)
from qubex.backend.quel3.quel3_backend_constants import (
    MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ,
    MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ,
)
from qubex.core.async_bridge import DEFAULT_TIMEOUT_SECONDS, get_shared_async_bridge

from .configuration_manager import Quel3ConfigurationManager
from .execution_manager import Quel3ExecutionManager

T = TypeVar("T")

_SUPPORTED_CAPTURE_MODES = (
    Quel3CaptureMode.AVERAGED_WAVEFORM,
)


def _run_async(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T:
    """Run one awaitable factory from a synchronous spectroscopy API."""
    bridge = get_shared_async_bridge(key="quel3-spectroscopy")
    return bridge.run(factory, timeout=timeout)


class Quel3SpectroscopyManager:
    """Partition, exclusively deploy, and execute packed spectroscopy scans."""

    def __init__(
        self,
        *,
        execution_manager: Quel3ExecutionManager,
        configuration_manager: Quel3ConfigurationManager,
        control_sampling_period_ns: float,
        readout_sampling_period_ns: float,
    ) -> None:
        self._execution_manager = execution_manager
        self._configuration_manager = configuration_manager
        self._control_sampling_period_ns = float(control_sampling_period_ns)
        self._readout_sampling_period_ns = float(readout_sampling_period_ns)
        if self._control_sampling_period_ns <= 0.0:
            raise ValueError("control_sampling_period_ns must be positive.")
        if self._readout_sampling_period_ns <= 0.0:
            raise ValueError("readout_sampling_period_ns must be positive.")

    def scan_resonator_frequencies(
        self,
        *,
        target: str,
        frequency_range: Sequence[float] | npt.NDArray[np.float64],
        readout_amplitude: float,
        readout_duration: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int = 200,
        capture_mode: Quel3CaptureMode = Quel3CaptureMode.AVERAGED_WAVEFORM,
        parallel: bool = True,
    ) -> Quel3ResonatorSpectroscopyResult:
        """Run a packed resonator scan across instrument-sized spans."""
        return _run_async(
            lambda: self.scan_resonator_frequencies_async(
                target=target,
                frequency_range=frequency_range,
                readout_amplitude=readout_amplitude,
                readout_duration=readout_duration,
                capture_delay=capture_delay,
                capture_length=capture_length,
                point_interval=point_interval,
                n_shots=n_shots,
                shot_interval=shot_interval,
                max_points_per_batch=max_points_per_batch,
                capture_mode=capture_mode,
                parallel=parallel,
            )
        )

    async def scan_resonator_frequencies_async(
        self,
        *,
        target: str,
        frequency_range: Sequence[float] | npt.NDArray[np.float64],
        readout_amplitude: float,
        readout_duration: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int = 200,
        capture_mode: Quel3CaptureMode = Quel3CaptureMode.AVERAGED_WAVEFORM,
        parallel: bool = True,
    ) -> Quel3ResonatorSpectroscopyResult:
        """Run a packed resonator scan across instrument-sized spans."""
        frequencies = self._normalize_frequencies(frequency_range)
        capture_mode = self._validate_capture_mode(capture_mode)
        batch_size = self._validate_acquisition_settings(
            readout_amplitude=readout_amplitude,
            readout_duration=readout_duration,
            capture_delay=capture_delay,
            capture_length=capture_length,
            point_interval=point_interval,
            occupied_length=max(readout_duration, capture_delay + capture_length),
            n_shots=n_shots,
            shot_interval=shot_interval,
            max_points_per_batch=max_points_per_batch,
        )
        self._validate_duration_on_grid(
            duration_ns=readout_duration,
            sampling_period_ns=self._readout_sampling_period_ns,
            parameter_name="readout_duration",
        )
        frequency_sections = self._partition_frequency_spans(
            frequencies=frequencies,
            maximum_span_ghz=MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ,
        )
        capture_sections: list[tuple[npt.NDArray[np.complex128], ...]] = []
        iq_sections: list[npt.NDArray[np.complex128]] = []
        typed_results: list[Quel3BackendExecutionResult] = []
        sampling_period_ns: float | None = None
        packing_batch_size = (
            batch_size if capture_mode is Quel3CaptureMode.AVERAGED_WAVEFORM else 1
        )
        for section in frequency_sections:
            self._validate_chunk_detunings(
                frequencies=section,
                sampling_period_ns=self._readout_sampling_period_ns,
                max_points_per_batch=packing_batch_size,
            )
            section_request = self._build_deploy_request(
                target=target,
                frequencies=section,
                profile_width_ghz=MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ,
                role="TRANSCEIVER",
            )
            alias = await self._deploy_and_resolve_alias(
                request=section_request,
                parallel=parallel,
            )
            requests = self._build_resonator_requests(
                instrument_alias=alias,
                frequencies=section,
                readout_amplitude=float(readout_amplitude),
                readout_duration=float(readout_duration),
                capture_delay=float(capture_delay),
                capture_length=float(capture_length),
                point_interval=float(point_interval),
                n_shots=int(n_shots),
                shot_interval=float(shot_interval),
                max_points_per_batch=packing_batch_size,
                capture_mode=capture_mode,
            )
            backend_results = await self._execution_manager.execute_batch_async(
                requests=requests,
                parallel=parallel,
            )
            section_captures, section_iq, section_period, section_results = (
                self._collect_captures(
                    capture_alias=alias,
                    frequencies=section,
                    requests=requests,
                    backend_results=backend_results,
                    capture_mode=capture_mode,
                    n_shots=int(n_shots),
                    demodulate=True,
                )
            )
            sampling_period_ns = self._merge_sampling_period(
                current=sampling_period_ns,
                section=section_period,
            )
            capture_sections.append(section_captures)
            iq_sections.append(section_iq)
            typed_results.extend(section_results)
        return Quel3ResonatorSpectroscopyResult(
            frequency_range=frequencies.copy(),
            capture_mode=capture_mode,
            captures=tuple(
                capture
                for section_captures in capture_sections
                for capture in section_captures
            ),
            iq=np.concatenate(iq_sections),
            sampling_period_ns=sampling_period_ns,
            backend_results=tuple(typed_results),
        )

    def scan_qubit_frequencies(
        self,
        *,
        target: str,
        readout_target: str,
        frequency_range: Sequence[float] | npt.NDArray[np.float64],
        readout_frequency: float,
        control_amplitude: float,
        control_duration: float,
        readout_amplitude: float,
        readout_duration: float,
        control_to_readout_gap: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int = 200,
        capture_mode: Quel3CaptureMode = Quel3CaptureMode.AVERAGED_WAVEFORM,
        parallel: bool = True,
    ) -> Quel3QubitSpectroscopyResult:
        """Run a packed qubit scan across control-instrument-sized spans."""
        return _run_async(
            lambda: self.scan_qubit_frequencies_async(
                target=target,
                readout_target=readout_target,
                frequency_range=frequency_range,
                readout_frequency=readout_frequency,
                control_amplitude=control_amplitude,
                control_duration=control_duration,
                readout_amplitude=readout_amplitude,
                readout_duration=readout_duration,
                control_to_readout_gap=control_to_readout_gap,
                capture_delay=capture_delay,
                capture_length=capture_length,
                point_interval=point_interval,
                n_shots=n_shots,
                shot_interval=shot_interval,
                max_points_per_batch=max_points_per_batch,
                capture_mode=capture_mode,
                parallel=parallel,
            )
        )

    async def scan_qubit_frequencies_async(
        self,
        *,
        target: str,
        readout_target: str,
        frequency_range: Sequence[float] | npt.NDArray[np.float64],
        readout_frequency: float,
        control_amplitude: float,
        control_duration: float,
        readout_amplitude: float,
        readout_duration: float,
        control_to_readout_gap: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int = 200,
        capture_mode: Quel3CaptureMode = Quel3CaptureMode.AVERAGED_WAVEFORM,
        parallel: bool = True,
    ) -> Quel3QubitSpectroscopyResult:
        """Run a packed qubit scan across control-instrument-sized spans."""
        frequencies = self._normalize_frequencies(frequency_range)
        capture_mode = self._validate_capture_mode(capture_mode)
        if not np.isfinite(readout_frequency) or readout_frequency <= 0.0:
            raise ValueError("readout_frequency must be positive and finite.")
        self._validate_amplitude(
            amplitude=control_amplitude,
            parameter_name="control_amplitude",
        )
        if control_duration <= 0.0 or not np.isfinite(control_duration):
            raise ValueError("control_duration must be positive.")
        self._validate_duration_on_grid(
            duration_ns=control_duration,
            sampling_period_ns=self._control_sampling_period_ns,
            parameter_name="control_duration",
            grid_name="control sampling grid",
        )
        if control_to_readout_gap < 0.0 or not np.isfinite(control_to_readout_gap):
            raise ValueError("control_to_readout_gap must be non-negative.")
        readout_start_ns = control_duration + control_to_readout_gap
        occupied_length = max(
            control_duration,
            readout_start_ns + readout_duration,
            readout_start_ns + capture_delay + capture_length,
        )
        batch_size = self._validate_acquisition_settings(
            readout_amplitude=readout_amplitude,
            readout_duration=readout_duration,
            capture_delay=capture_delay,
            capture_length=capture_length,
            point_interval=point_interval,
            occupied_length=occupied_length,
            n_shots=n_shots,
            shot_interval=shot_interval,
            max_points_per_batch=max_points_per_batch,
        )
        self._validate_duration_on_grid(
            duration_ns=readout_duration,
            sampling_period_ns=self._readout_sampling_period_ns,
            parameter_name="readout_duration",
        )
        frequency_sections = self._partition_frequency_spans(
            frequencies=frequencies,
            maximum_span_ghz=MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ,
        )
        readout_request = self._build_deploy_request(
            target=readout_target,
            frequencies=np.asarray([readout_frequency], dtype=np.float64),
            profile_width_ghz=MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ,
            role="TRANSCEIVER",
        )
        capture_sections: list[tuple[npt.NDArray[np.complex128], ...]] = []
        iq_sections: list[npt.NDArray[np.complex128]] = []
        typed_results: list[Quel3BackendExecutionResult] = []
        sampling_period_ns: float | None = None
        for section in frequency_sections:
            self._validate_chunk_detunings(
                frequencies=section,
                sampling_period_ns=self._control_sampling_period_ns,
                max_points_per_batch=batch_size,
            )
            control_request = self._build_deploy_request(
                target=target,
                frequencies=section,
                profile_width_ghz=MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ,
                role="TRANSMITTER",
            )
            if control_request.port_id == readout_request.port_id:
                raise ValueError(
                    "Control and readout spectroscopy ports must be different."
                )
            control_alias, readout_alias = await self._deploy_and_resolve_aliases(
                requests=(control_request, readout_request),
                parallel=parallel,
            )
            requests = self._build_qubit_requests(
                control_alias=control_alias,
                readout_alias=readout_alias,
                frequencies=section,
                readout_frequency=float(readout_frequency),
                control_amplitude=float(control_amplitude),
                control_duration=float(control_duration),
                readout_amplitude=float(readout_amplitude),
                readout_duration=float(readout_duration),
                control_to_readout_gap=float(control_to_readout_gap),
                capture_delay=float(capture_delay),
                capture_length=float(capture_length),
                point_interval=float(point_interval),
                occupied_length=float(occupied_length),
                n_shots=int(n_shots),
                shot_interval=float(shot_interval),
                max_points_per_batch=batch_size,
                capture_mode=capture_mode,
            )
            backend_results = await self._execution_manager.execute_batch_async(
                requests=requests,
                parallel=parallel,
            )
            section_captures, section_iq, section_period, section_results = (
                self._collect_captures(
                    capture_alias=readout_alias,
                    frequencies=section,
                    requests=requests,
                    backend_results=backend_results,
                    capture_mode=capture_mode,
                    n_shots=int(n_shots),
                    demodulate=False,
                )
            )
            sampling_period_ns = self._merge_sampling_period(
                current=sampling_period_ns,
                section=section_period,
            )
            capture_sections.append(section_captures)
            iq_sections.append(section_iq)
            typed_results.extend(section_results)
        return Quel3QubitSpectroscopyResult(
            frequency_range=frequencies.copy(),
            readout_frequency=float(readout_frequency),
            capture_mode=capture_mode,
            captures=tuple(
                capture
                for section_captures in capture_sections
                for capture in section_captures
            ),
            iq=np.concatenate(iq_sections),
            sampling_period_ns=sampling_period_ns,
            backend_results=tuple(typed_results),
        )

    def _validate_acquisition_settings(
        self,
        *,
        readout_amplitude: float,
        readout_duration: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        occupied_length: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int,
    ) -> int:
        """Validate settings shared by both spectroscopy scans."""
        self._validate_amplitude(
            amplitude=readout_amplitude,
            parameter_name="readout_amplitude",
        )
        if readout_duration <= 0.0 or not np.isfinite(readout_duration):
            raise ValueError("readout_duration must be positive.")
        if capture_delay < 0.0 or not np.isfinite(capture_delay):
            raise ValueError("capture_delay must be non-negative.")
        if capture_length <= 0.0 or not np.isfinite(capture_length):
            raise ValueError("capture_length must be positive.")
        if point_interval < occupied_length or not np.isfinite(point_interval):
            raise ValueError(
                "point_interval must be at least the occupied length "
                f"({occupied_length:g} ns)."
            )
        if isinstance(n_shots, bool) or not isinstance(n_shots, Integral):
            raise TypeError("n_shots must be an integer.")
        if n_shots <= 0:
            raise ValueError("n_shots must be a positive integer.")
        if shot_interval < 0.0 or not np.isfinite(shot_interval):
            raise ValueError("shot_interval must be non-negative.")
        if isinstance(max_points_per_batch, bool) or not isinstance(
            max_points_per_batch, Integral
        ):
            raise TypeError("max_points_per_batch must be an integer.")
        if max_points_per_batch <= 0:
            raise ValueError("max_points_per_batch must be a positive integer.")
        return int(max_points_per_batch)

    @staticmethod
    def _partition_frequency_spans(
        *,
        frequencies: npt.NDArray[np.float64],
        maximum_span_ghz: float,
    ) -> tuple[npt.NDArray[np.float64], ...]:
        """Partition consecutive frequencies into instrument-sized spans."""
        sections: list[npt.NDArray[np.float64]] = []
        start = 0
        section_minimum = float(frequencies[0])
        section_maximum = section_minimum
        for index in range(1, frequencies.size):
            frequency = float(frequencies[index])
            candidate_minimum = min(section_minimum, frequency)
            candidate_maximum = max(section_maximum, frequency)
            candidate_span = candidate_maximum - candidate_minimum
            if candidate_span > maximum_span_ghz and not np.isclose(
                candidate_span,
                maximum_span_ghz,
                rtol=0.0,
                atol=1e-12,
            ):
                sections.append(frequencies[start:index])
                start = index
                section_minimum = frequency
                section_maximum = frequency
            else:
                section_minimum = candidate_minimum
                section_maximum = candidate_maximum
        sections.append(frequencies[start:])
        return tuple(sections)

    def _build_deploy_request(
        self,
        *,
        target: str,
        frequencies: npt.NDArray[np.float64],
        profile_width_ghz: float,
        role: RoleName,
    ) -> InstrumentDeployRequest:
        """Build a fixed-width deploy request from current target state."""
        box_id, target_label, instrument_info = (
            self._configuration_manager.resolve_target_instrument(target)
        )
        port_id = str(instrument_info.port_id).strip()
        alias = str(instrument_info.definition.alias).strip()
        if not port_id or not alias:
            raise ValueError(
                "Spectroscopy target must resolve to an instrument with a port and "
                "alias."
            )
        unit_label = port_id.split(":", maxsplit=1)[0]
        alias_prefix = f"{unit_label}:"
        if alias.startswith(alias_prefix):
            alias = alias.removeprefix(alias_prefix)
        center_hz = float(np.min(frequencies) + np.max(frequencies)) * 0.5e9
        half_width_hz = profile_width_ghz * 0.5e9
        minimum_hz = center_hz - half_width_hz
        maximum_hz = center_hz + half_width_hz
        if minimum_hz <= 0.0:
            raise ValueError(
                "Spectroscopy frequencies are too low for the required instrument "
                f"profile width of {profile_width_ghz:g} GHz."
            )
        return InstrumentDeployRequest(
            port_id=port_id,
            role=role,
            frequency_range_min_hz=minimum_hz,
            frequency_range_max_hz=maximum_hz,
            alias=alias,
            target_labels=(target_label,),
            box_id=box_id,
        )

    @staticmethod
    def _merge_sampling_period(
        *,
        current: float | None,
        section: float | None,
    ) -> float | None:
        """Merge one section sampling period into the scan-wide value."""
        if section is None:
            return current
        if current is not None and not np.isclose(current, section):
            raise ValueError(
                "Spectroscopy results across instrument spans must agree on "
                "sampling period."
            )
        return section

    async def _deploy_and_resolve_alias(
        self,
        *,
        request: InstrumentDeployRequest,
        parallel: bool,
    ) -> str:
        """Deploy one dedicated instrument and return its runtime alias."""
        return (
            await self._deploy_and_resolve_aliases(
                requests=(request,),
                parallel=parallel,
            )
        )[0]

    async def _deploy_and_resolve_aliases(
        self,
        *,
        requests: tuple[InstrumentDeployRequest, ...],
        parallel: bool,
    ) -> tuple[str, ...]:
        """Replace selected ports with dedicated instruments and resolve aliases."""
        await self._configuration_manager.deploy_instruments_async(
            requests=requests,
            parallel=parallel,
        )
        alias_map = self._configuration_manager.target_alias_map
        aliases = []
        for request in requests:
            target_label = request.target_labels[0]
            key = (request.box_id, target_label)
            runtime_alias = alias_map.get(key)
            if not isinstance(runtime_alias, str) or not runtime_alias.strip():
                raise ValueError(
                    "Dedicated spectroscopy deployment did not publish a runtime "
                    f"alias for target `{target_label}`."
                )
            aliases.append(runtime_alias)
        return tuple(aliases)

    @staticmethod
    def _normalize_frequencies(
        frequency_range: Sequence[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return a validated one-dimensional GHz frequency array."""
        frequencies = np.asarray(frequency_range, dtype=np.float64)
        if frequencies.ndim != 1:
            raise ValueError("frequency_range must be one-dimensional.")
        if frequencies.size == 0:
            raise ValueError("frequency_range must not be empty.")
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("frequency_range must contain only finite values.")
        if not np.all(frequencies > 0.0):
            raise ValueError("frequency_range values must be positive.")
        return frequencies

    @staticmethod
    def _validate_amplitude(*, amplitude: float, parameter_name: str) -> None:
        """Validate one normalized waveform amplitude."""
        if not np.isfinite(amplitude) or abs(amplitude) > 1.0:
            raise ValueError(f"{parameter_name} must be within [-1, 1].")

    @staticmethod
    def _validate_capture_mode(capture_mode: Quel3CaptureMode) -> Quel3CaptureMode:
        """Return a supported spectroscopy capture mode."""
        if not isinstance(capture_mode, Quel3CaptureMode):
            raise TypeError("capture_mode must be a `Quel3CaptureMode` value.")
        if capture_mode not in _SUPPORTED_CAPTURE_MODES:
            supported = ", ".join(mode.name for mode in _SUPPORTED_CAPTURE_MODES)
            raise ValueError(
                f"capture_mode must be one of the supported modes: {supported}."
            )
        return capture_mode

    @staticmethod
    def _validate_duration_on_grid(
        *,
        duration_ns: float,
        sampling_period_ns: float,
        parameter_name: str,
        grid_name: str = "sampling grid",
    ) -> None:
        """Validate one duration against a waveform sampling grid."""
        samples = duration_ns / sampling_period_ns
        if not np.isclose(samples, round(samples)):
            raise ValueError(f"{parameter_name} must be on the {grid_name}.")

    @staticmethod
    def _validate_chunk_detunings(
        *,
        frequencies: npt.NDArray[np.float64],
        sampling_period_ns: float,
        max_points_per_batch: int,
    ) -> None:
        """Reject packed waveform detunings at or beyond Nyquist."""
        nyquist_ghz = 0.5 / sampling_period_ns
        for start in range(0, frequencies.size, max_points_per_batch):
            chunk = frequencies[start : start + max_points_per_batch]
            detunings = chunk - np.mean(chunk)
            if np.any(np.abs(detunings) >= nyquist_ghz):
                raise ValueError(
                    "A packed spectroscopy detuning reaches the waveform Nyquist limit."
                )

    def _build_resonator_requests(
        self,
        *,
        instrument_alias: str,
        frequencies: npt.NDArray[np.float64],
        readout_amplitude: float,
        readout_duration: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int,
        capture_mode: Quel3CaptureMode,
    ) -> tuple[BackendExecutionRequest, ...]:
        """Build packed resonator execution requests."""
        requests = []
        for start in range(0, frequencies.size, max_points_per_batch):
            chunk = frequencies[start : start + max_points_per_batch]
            center_frequency_ghz = float(np.mean(chunk))
            waveform_library: dict[str, Quel3Waveform] = {}
            events = []
            captures = []
            for local_index, frequency_ghz in enumerate(chunk):
                global_index = start + local_index
                waveform_name = f"resonator_drive_{global_index}"
                waveform_library[waveform_name] = Quel3Waveform(
                    iq_array=self._rectangular_waveform(
                        duration_ns=readout_duration,
                        amplitude=readout_amplitude,
                        detuning_ghz=float(frequency_ghz) - center_frequency_ghz,
                        sampling_period_ns=self._readout_sampling_period_ns,
                    ),
                    sampling_period_ns=self._readout_sampling_period_ns,
                )
                point_start_ns = local_index * point_interval
                events.append(
                    Quel3WaveformEvent(
                        waveform_name=waveform_name,
                        start_offset_ns=point_start_ns,
                    )
                )
                captures.append(
                    Quel3CaptureWindow(
                        name=f"resonator_capture_{global_index}",
                        start_offset_ns=point_start_ns + capture_delay,
                        length_ns=capture_length,
                    )
                )
            occupied_length = max(readout_duration, capture_delay + capture_length)
            timeline_length_ns = (chunk.size - 1) * point_interval + occupied_length
            payload = Quel3ExecutionPayload(
                waveform_library=waveform_library,
                fixed_timelines={
                    instrument_alias: Quel3FixedTimeline(
                        events=tuple(events),
                        capture_windows=tuple(captures),
                        length_ns=timeline_length_ns,
                        frequency_hz=center_frequency_ghz * 1e9,
                    )
                },
                n_iterations=n_shots,
                shot_interval_ns=shot_interval,
                capture_mode=capture_mode,
            )
            requests.append(BackendExecutionRequest(payload=payload))
        return tuple(requests)

    def _build_qubit_requests(
        self,
        *,
        control_alias: str,
        readout_alias: str,
        frequencies: npt.NDArray[np.float64],
        readout_frequency: float,
        control_amplitude: float,
        control_duration: float,
        readout_amplitude: float,
        readout_duration: float,
        control_to_readout_gap: float,
        capture_delay: float,
        capture_length: float,
        point_interval: float,
        occupied_length: float,
        n_shots: int,
        shot_interval: float,
        max_points_per_batch: int,
        capture_mode: Quel3CaptureMode,
    ) -> tuple[BackendExecutionRequest, ...]:
        """Build packed qubit and readout execution requests."""
        requests = []
        for start in range(0, frequencies.size, max_points_per_batch):
            chunk = frequencies[start : start + max_points_per_batch]
            center_frequency_ghz = float(np.mean(chunk))
            batch_index = start // max_points_per_batch
            readout_waveform_name = f"qubit_readout_{batch_index}"
            waveform_library = {
                readout_waveform_name: Quel3Waveform(
                    iq_array=self._rectangular_waveform(
                        duration_ns=readout_duration,
                        amplitude=readout_amplitude,
                        detuning_ghz=0.0,
                        sampling_period_ns=self._readout_sampling_period_ns,
                    ),
                    sampling_period_ns=self._readout_sampling_period_ns,
                )
            }
            control_events = []
            readout_events = []
            captures = []
            for local_index, frequency_ghz in enumerate(chunk):
                global_index = start + local_index
                control_waveform_name = f"qubit_drive_{global_index}"
                waveform_library[control_waveform_name] = Quel3Waveform(
                    iq_array=self._rectangular_waveform(
                        duration_ns=control_duration,
                        amplitude=control_amplitude,
                        detuning_ghz=float(frequency_ghz) - center_frequency_ghz,
                        sampling_period_ns=self._control_sampling_period_ns,
                    ),
                    sampling_period_ns=self._control_sampling_period_ns,
                )
                point_start_ns = local_index * point_interval
                readout_start_ns = (
                    point_start_ns + control_duration + control_to_readout_gap
                )
                control_events.append(
                    Quel3WaveformEvent(
                        waveform_name=control_waveform_name,
                        start_offset_ns=point_start_ns,
                    )
                )
                readout_events.append(
                    Quel3WaveformEvent(
                        waveform_name=readout_waveform_name,
                        start_offset_ns=readout_start_ns,
                    )
                )
                captures.append(
                    Quel3CaptureWindow(
                        name=f"qubit_capture_{global_index}",
                        start_offset_ns=readout_start_ns + capture_delay,
                        length_ns=capture_length,
                    )
                )
            timeline_length_ns = (chunk.size - 1) * point_interval + occupied_length
            payload = Quel3ExecutionPayload(
                waveform_library=waveform_library,
                fixed_timelines={
                    control_alias: Quel3FixedTimeline(
                        events=tuple(control_events),
                        capture_windows=(),
                        length_ns=timeline_length_ns,
                        frequency_hz=center_frequency_ghz * 1e9,
                    ),
                    readout_alias: Quel3FixedTimeline(
                        events=tuple(readout_events),
                        capture_windows=tuple(captures),
                        length_ns=timeline_length_ns,
                        frequency_hz=readout_frequency * 1e9,
                    ),
                },
                n_iterations=n_shots,
                shot_interval_ns=shot_interval,
                capture_mode=capture_mode,
            )
            requests.append(BackendExecutionRequest(payload=payload))
        return tuple(requests)

    @staticmethod
    def _rectangular_waveform(
        *,
        duration_ns: float,
        amplitude: float,
        detuning_ghz: float,
        sampling_period_ns: float,
    ) -> npt.NDArray[np.complex128]:
        """Return a detuned rectangular waveform on one sample grid."""
        sample_count = round(duration_ns / sampling_period_ns)
        time_ns = np.arange(sample_count, dtype=np.float64) * sampling_period_ns
        return np.asarray(
            amplitude * np.exp(2j * np.pi * detuning_ghz * time_ns),
            dtype=np.complex128,
        )

    @staticmethod
    def _collect_captures(
        *,
        capture_alias: str,
        frequencies: npt.NDArray[np.float64],
        requests: tuple[BackendExecutionRequest, ...],
        backend_results: Sequence[Quel3BackendExecutionResult],
        capture_mode: Quel3CaptureMode,
        n_shots: int,
        demodulate: bool,
    ) -> tuple[
        tuple[npt.NDArray[np.complex128], ...],
        npt.NDArray[np.complex128],
        float | None,
        tuple[Quel3BackendExecutionResult, ...],
    ]:
        """Collect captures in input order and optionally demodulate them."""
        if len(backend_results) != len(requests):
            raise ValueError(
                "Spectroscopy backend result count does not match packed payloads."
            )
        captures: list[npt.NDArray[np.complex128]] = []
        sampling_period_ns: float | None = None
        frequency_index = 0
        typed_results = []
        for request, backend_result in zip(requests, backend_results, strict=True):
            if not isinstance(backend_result, Quel3BackendExecutionResult):
                raise TypeError(
                    "Spectroscopy execution must return `Quel3BackendExecutionResult`."
                )
            typed_results.append(backend_result)
            timeline = request.payload.fixed_timelines[capture_alias]
            captured_waveforms = backend_result.data.get(capture_alias)
            expected_count = len(timeline.capture_windows)
            if captured_waveforms is None or len(captured_waveforms) != expected_count:
                raise ValueError(
                    "Spectroscopy capture count does not match packed frequency points."
                )
            chunk_sampling_period: float | None = None
            if capture_mode is Quel3CaptureMode.AVERAGED_WAVEFORM:
                configured_period = backend_result.config.get("sampling_period_ns")
                if not isinstance(configured_period, (int, float)):
                    raise TypeError(
                        "Spectroscopy backend result sampling period must be numeric."
                    )
                chunk_sampling_period = float(configured_period)
                if sampling_period_ns is None:
                    sampling_period_ns = chunk_sampling_period
                elif not np.isclose(sampling_period_ns, chunk_sampling_period):
                    raise ValueError(
                        "Packed spectroscopy results must agree on sampling period."
                    )
            center_frequency_ghz = (
                float(timeline.frequency_hz) * 1e-9 if demodulate else 0.0
            )
            for captured in captured_waveforms:
                capture = np.asarray(captured, dtype=np.complex128)
                if capture.ndim != 1 or capture.size == 0:
                    raise ValueError(
                        "Spectroscopy captures must be non-empty and one-dimensional."
                    )
                if capture_mode is Quel3CaptureMode.AVERAGED_VALUE:
                    if capture.size != 1:
                        raise ValueError(
                            "AVERAGED_VALUE spectroscopy captures must contain one "
                            "value per frequency."
                        )
                elif capture_mode is Quel3CaptureMode.VALUES_PER_ITER:
                    if capture.size != n_shots:
                        raise ValueError(
                            "VALUES_PER_ITER spectroscopy captures must contain one "
                            "value per shot."
                        )
                elif demodulate:
                    if chunk_sampling_period is None:
                        raise ValueError(
                            "AVERAGED_WAVEFORM spectroscopy requires a sampling period."
                        )
                    detuning_ghz = (
                        float(frequencies[frequency_index]) - center_frequency_ghz
                    )
                    time_ns = (
                        np.arange(capture.size, dtype=np.float64)
                        * chunk_sampling_period
                    )
                    capture = np.asarray(
                        capture * np.exp(-2j * np.pi * detuning_ghz * time_ns),
                        dtype=np.complex128,
                    )
                captures.append(capture)
                frequency_index += 1
        if frequency_index != frequencies.size:
            raise ValueError(
                "Spectroscopy capture count does not match requested frequency points."
            )
        if capture_mode is Quel3CaptureMode.AVERAGED_WAVEFORM:
            iq = np.asarray(
                [np.mean(capture) for capture in captures],
                dtype=np.complex128,
            )
        elif capture_mode is Quel3CaptureMode.AVERAGED_VALUE:
            iq = np.asarray(
                [capture[0] for capture in captures],
                dtype=np.complex128,
            )
        else:
            iq = np.asarray(np.stack(captures, axis=0), dtype=np.complex128)
        return (
            tuple(captures),
            iq,
            sampling_period_ns,
            tuple(typed_results),
        )
