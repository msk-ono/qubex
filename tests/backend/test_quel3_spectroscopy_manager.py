"""Tests for grouped QuEL-3 spectroscopy instrument pools."""

from __future__ import annotations

import asyncio
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qubex.backend import BackendExecutionRequest
from qubex.backend.quel3 import (
    InstrumentDeployRequest,
    Quel3BackendExecutionResult,
    Quel3CaptureMode,
    Quel3QubitSpectroscopyResult,
    Quel3ResonatorSpectroscopyResult,
)
from qubex.backend.quel3.managers.spectroscopy_manager import (
    Quel3SpectroscopyManager,
)
from qubex.backend.quel3.quel3_backend_constants import (
    MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ,
    MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ,
)


class _RecordingExecutionManager:
    """Execution-manager stub that records both spectroscopy strategies."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._events = events

    async def execute_async(
        self,
        *,
        request: BackendExecutionRequest,
        parallel: bool,
    ) -> Quel3BackendExecutionResult:
        """Return constant data for every capture in one packed payload."""
        self.calls.append(
            {"method": "execute_async", "request": request, "parallel": parallel}
        )
        if self._events is not None:
            self._events.append("execute")
        return self._result_for(request)

    async def execute_batch_async(
        self,
        *,
        requests: tuple[BackendExecutionRequest, ...],
        parallel: bool,
    ) -> list[Quel3BackendExecutionResult]:
        """Return constant data for every capture in packed batch requests."""
        self.calls.append(
            {
                "method": "execute_batch_async",
                "requests": requests,
                "parallel": parallel,
            }
        )
        if self._events is not None:
            self._events.append("execute")
        return [self._result_for(request) for request in requests]

    @staticmethod
    def _result_for(request: BackendExecutionRequest) -> Quel3BackendExecutionResult:
        """Return one constant backend result for a request."""
        payload = request.payload
        if payload.capture_mode is Quel3CaptureMode.AVERAGED_WAVEFORM:
            capture = np.ones(4, dtype=np.complex128)
        elif payload.capture_mode is Quel3CaptureMode.AVERAGED_VALUE:
            capture = np.array([2.0 + 1.0j], dtype=np.complex128)
        elif payload.capture_mode is Quel3CaptureMode.VALUES_PER_ITER:
            capture = np.arange(
                1,
                payload.n_iterations + 1,
                dtype=np.complex128,
            )
        else:
            raise AssertionError(f"Unexpected capture mode: {payload.capture_mode}")
        data = {
            alias: [capture.copy() for _ in timeline.capture_windows]
            for alias, timeline in payload.fixed_timelines.items()
            if timeline.capture_windows
        }
        return Quel3BackendExecutionResult(
            status={},
            data=data,
            config={"sampling_period_ns": 0.8},
        )


class _RecordingConfigurationManager:
    """Configuration-manager stub that records replacement deployments."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.target_alias_map: dict[tuple[str, str], str] = {}
        self._events = events

    async def deploy_instruments_async(
        self,
        *,
        requests: tuple[InstrumentDeployRequest, ...],
        parallel: bool,
    ) -> dict[str, tuple[object, ...]]:
        """Record requests and publish their unit-qualified runtime aliases."""
        self.calls.append({"requests": requests, "parallel": parallel})
        if self._events is not None:
            self._events.append("deploy")
        deployed = {}
        for request in requests:
            unit_label = request.port_id.split(":", maxsplit=1)[0]
            runtime_alias = f"{unit_label}:{request.alias}"
            for target_label in request.target_labels:
                self.target_alias_map[(request.box_id, target_label)] = runtime_alias
            deployed[request.alias] = (
                _instrument_info(port_id=request.port_id, alias=request.alias),
            )
        return deployed

    def resolve_target_instrument(
        self,
        target: str,
    ) -> tuple[str, str, SimpleNamespace]:
        """Resolve one test target from the cached deployment state."""
        targets = {
            "Q00": (
                "BOX1",
                "Q00",
                _instrument_info(port_id="unit-a:tx_p04", alias="qubit"),
            ),
            "R00": (
                "BOX1",
                "R00",
                _instrument_info(port_id="unit-a:trx_p00p01", alias="readout"),
            ),
            "unit-a:qubit": (
                "BOX1",
                "Q00",
                _instrument_info(port_id="unit-a:tx_p04", alias="qubit"),
            ),
            "unit-a:readout": (
                "BOX1",
                "R00",
                _instrument_info(port_id="unit-a:trx_p00p01", alias="readout"),
            ),
        }
        return targets[target]


def _expected_readout_request(
    *, group_minimum: float, group_maximum: float, index: int
) -> InstrumentDeployRequest:
    """Return one resonator request covering a full frequency group."""
    return InstrumentDeployRequest(
        port_id="unit-a:trx_p00p01",
        role="TRANSCEIVER",
        frequency_range_min_hz=group_minimum * 1e9,
        frequency_range_max_hz=group_maximum * 1e9,
        alias="readout" if index == 0 else f"readout-spectroscopy-{index}",
        target_labels=("R00",),
        box_id="BOX1",
    )


def _instrument_info(*, port_id: str, alias: str) -> SimpleNamespace:
    """Return cached instrument info used to resolve a spectroscopy port."""
    return SimpleNamespace(
        port_id=port_id,
        definition=SimpleNamespace(alias=alias),
    )


def _manager(
    execution_manager: _RecordingExecutionManager,
    configuration_manager: _RecordingConfigurationManager | None = None,
) -> Quel3SpectroscopyManager:
    """Return a spectroscopy manager with QuEL-3 control/readout grids."""
    return Quel3SpectroscopyManager(
        execution_manager=execution_manager,  # type: ignore[arg-type]
        configuration_manager=(
            configuration_manager or _RecordingConfigurationManager()
        ),  # type: ignore[arg-type]
        control_sampling_period_ns=0.4,
        readout_sampling_period_ns=0.8,
    )


def test_resonator_averaged_waveform_uses_detuned_packed_instrument() -> None:
    """Averaged-waveform resonator scans should detune one packed instrument."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.array([6.0, 6.1])

    result = asyncio.run(
        manager.scan_resonator_frequencies_async(
            target="R00",
            frequency_range=frequencies,
            readout_amplitude=0.5,
            readout_duration=3.2,
            capture_delay=0.0,
            capture_length=3.2,
            point_interval=3.2,
            n_shots=1,
            shot_interval=0.0,
        )
    )

    assert len(configuration_manager.calls) == 1
    (deploy_request,) = configuration_manager.calls[0]["requests"]
    group_center = frequencies.mean()
    assert deploy_request.frequency_range_min_hz == pytest.approx(
        (group_center - MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ / 2) * 1e9
    )
    assert deploy_request.frequency_range_max_hz == pytest.approx(
        (group_center + MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ / 2) * 1e9
    )
    assert len(execution_manager.calls) == 1
    call = execution_manager.calls[0]
    assert call["method"] == "execute_batch_async"
    (request,) = call["requests"]
    timeline = request.payload.fixed_timelines["unit-a:readout"]
    assert len(timeline.events) == 2
    assert timeline.frequency_hz == pytest.approx(frequencies.mean() * 1e9)
    first_waveform = request.payload.waveform_library[
        timeline.events[0].waveform_name
    ].iq_array
    detuning = frequencies[0] - frequencies.mean()
    expected_transmit = 0.5 * np.exp(2j * np.pi * detuning * np.arange(4) * 0.8)
    np.testing.assert_allclose(first_waveform, expected_transmit)
    expected_capture = np.exp(-2j * np.pi * detuning * np.arange(4) * 0.8)
    np.testing.assert_allclose(result.captures[0], expected_capture)
    assert result.sampling_period_ns == pytest.approx(0.8)


def test_resonator_averaged_waveform_avoids_large_instrument_pool() -> None:
    """Averaged-waveform resonator batches should reuse one instrument."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(
            manager.scan_resonator_frequencies_async(
                target="R00",
                frequency_range=np.linspace(6.0, 6.2, 201),
                readout_amplitude=0.25,
                readout_duration=3.2,
                capture_delay=0.0,
                capture_length=3.2,
                point_interval=3.2,
                n_shots=1,
                shot_interval=0.0,
                max_points_per_batch=200,
            )
        )

    assert len(configuration_manager.calls) == 1
    assert len(configuration_manager.calls[0]["requests"]) == 1
    assert len(execution_manager.calls) == 1
    assert execution_manager.calls[0]["method"] == "execute_batch_async"
    assert len(execution_manager.calls[0]["requests"]) == 2


def test_resonator_scan_reuses_group_instruments_across_batches() -> None:
    """A resonator group should deploy one batch-sized reusable instrument pool."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.linspace(6.0, 6.2, 201)

    with pytest.warns(RuntimeWarning, match="200 spectroscopy instruments"):
        result = asyncio.run(
            manager.scan_resonator_frequencies_async(
                target="R00",
                frequency_range=frequencies,
                readout_amplitude=0.25,
                readout_duration=200.0,
                capture_delay=100.0,
                capture_length=80.0,
                point_interval=400.0,
                n_shots=32,
                shot_interval=10_000.0,
                max_points_per_batch=200,
                capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
                parallel=False,
            )
        )

    assert isinstance(result, Quel3ResonatorSpectroscopyResult)
    assert len(configuration_manager.calls) == 1
    deploy_requests = configuration_manager.calls[0]["requests"]
    assert len(deploy_requests) == 200
    assert deploy_requests[0] == _expected_readout_request(
        group_minimum=frequencies[0],
        group_maximum=frequencies[-1],
        index=0,
    )
    assert deploy_requests[1] == _expected_readout_request(
        group_minimum=frequencies[0],
        group_maximum=frequencies[-1],
        index=1,
    )
    assert all(call["parallel"] is False for call in configuration_manager.calls)
    assert len(execution_manager.calls) == 2
    assert all(call["parallel"] is False for call in execution_manager.calls)

    first_payload = execution_manager.calls[0]["request"].payload
    assert len(first_payload.fixed_timelines) == 200
    assert len(first_payload.waveform_library) == 1
    first_timeline = first_payload.fixed_timelines["unit-a:readout"]
    second_timeline = first_payload.fixed_timelines["unit-a:readout-spectroscopy-1"]
    assert len(first_timeline.events) == 1
    assert len(first_timeline.capture_windows) == 1
    assert first_timeline.frequency_hz == pytest.approx(frequencies[0] * 1e9)
    assert second_timeline.frequency_hz == pytest.approx(frequencies[1] * 1e9)
    assert first_payload.n_iterations == 32
    assert first_payload.shot_interval_ns == pytest.approx(10_000.0)
    assert second_timeline.events[0].start_offset_ns == pytest.approx(400.0)
    assert second_timeline.capture_windows[0].start_offset_ns == pytest.approx(500.0)

    second_payload = execution_manager.calls[1]["request"].payload
    assert tuple(second_payload.fixed_timelines) == ("unit-a:readout",)
    assert second_payload.fixed_timelines[
        "unit-a:readout"
    ].frequency_hz == pytest.approx(frequencies[-1] * 1e9)

    first_event = first_timeline.events[0]
    first_waveform = first_payload.waveform_library[first_event.waveform_name].iq_array
    np.testing.assert_allclose(first_waveform, np.full(250, 0.25 + 0.0j))

    np.testing.assert_allclose(result.frequency_range, frequencies)
    assert result.capture_mode is Quel3CaptureMode.AVERAGED_VALUE
    assert len(result.captures) == 201
    assert result.iq.shape == (201,)
    assert result.sampling_period_ns is None
    assert len(result.backend_results) == 2


def test_resonator_scan_uses_exact_carriers_without_demodulation() -> None:
    """A resonator scan should use exact carriers and leave captures unchanged."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.array([6.0, 6.1])

    result = asyncio.run(
        manager.scan_resonator_frequencies_async(
            target="R00",
            frequency_range=frequencies,
            readout_amplitude=0.5,
            readout_duration=3.2,
            capture_delay=0.0,
            capture_length=3.2,
            point_interval=3.2,
            n_shots=1,
            shot_interval=0.0,
            capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
        )
    )

    deploy_requests = configuration_manager.calls[0]["requests"]
    assert deploy_requests == (
        _expected_readout_request(group_minimum=6.0, group_maximum=6.1, index=0),
        _expected_readout_request(group_minimum=6.0, group_maximum=6.1, index=1),
    )

    payload = execution_manager.calls[0]["request"].payload
    first_timeline = payload.fixed_timelines["unit-a:readout"]
    second_timeline = payload.fixed_timelines["unit-a:readout-spectroscopy-1"]
    assert first_timeline.frequency_hz == pytest.approx(6.0e9)
    assert second_timeline.frequency_hz == pytest.approx(6.1e9)
    first_event = first_timeline.events[0]
    first_waveform = payload.waveform_library[first_event.waveform_name].iq_array
    np.testing.assert_allclose(
        first_waveform,
        np.full(4, 0.5 + 0.0j),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.captures[0],
        np.array([2.0 + 1.0j]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.iq[0] == pytest.approx(2.0 + 1.0j)


def test_resonator_scan_returns_hardware_averaged_values() -> None:
    """Averaged-value scans should share the instrument-per-frequency payload."""
    execution_manager = _RecordingExecutionManager()
    manager = _manager(execution_manager)
    frequencies = np.array([6.0, 6.1])

    result = asyncio.run(
        manager.scan_resonator_frequencies_async(
            target="R00",
            frequency_range=frequencies,
            readout_amplitude=0.25,
            readout_duration=3.2,
            capture_delay=0.0,
            capture_length=3.2,
            point_interval=3.2,
            n_shots=4,
            shot_interval=0.0,
            capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
        )
    )

    request = execution_manager.calls[0]["request"]
    assert request.payload.capture_mode is Quel3CaptureMode.AVERAGED_VALUE
    assert [
        timeline.frequency_hz for timeline in request.payload.fixed_timelines.values()
    ] == pytest.approx(frequencies * 1e9)
    assert all(len(capture) == 1 for capture in result.captures)
    np.testing.assert_allclose(result.iq, np.full(2, 2.0 + 1.0j))
    assert result.capture_mode is Quel3CaptureMode.AVERAGED_VALUE
    assert result.sampling_period_ns is None


def test_resonator_scan_returns_values_per_iteration_from_shared_payload() -> None:
    """Values-per-iteration scans should preserve shots in one shared payload."""
    execution_manager = _RecordingExecutionManager()
    manager = _manager(execution_manager)

    result = asyncio.run(
        manager.scan_resonator_frequencies_async(
            target="R00",
            frequency_range=[6.0, 6.1],
            readout_amplitude=0.25,
            readout_duration=3.2,
            capture_delay=0.0,
            capture_length=3.2,
            point_interval=3.2,
            n_shots=3,
            shot_interval=0.0,
            capture_mode=Quel3CaptureMode.VALUES_PER_ITER,
        )
    )

    request = execution_manager.calls[0]["request"]
    assert len(request.payload.fixed_timelines) == 2
    assert request.payload.capture_mode is Quel3CaptureMode.VALUES_PER_ITER
    np.testing.assert_allclose(result.captures[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.iq, np.array([[1.0, 2.0, 3.0]] * 2))
    assert result.sampling_period_ns is None


def test_resonator_scan_redeploys_each_frequency_group() -> None:
    """A resonator scan should redeploy only when the frequency span changes."""
    events: list[str] = []
    execution_manager = _RecordingExecutionManager(events)
    configuration_manager = _RecordingConfigurationManager(events)
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.array([5.0, 5.4, 5.8, 6.2])

    result = asyncio.run(
        manager.scan_resonator_frequencies_async(
            target="R00",
            frequency_range=frequencies,
            readout_amplitude=0.25,
            readout_duration=3.2,
            capture_delay=0.0,
            capture_length=3.2,
            point_interval=3.2,
            n_shots=1,
            shot_interval=0.0,
            max_points_per_batch=2,
            capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
            parallel=False,
        )
    )

    assert len(configuration_manager.calls) == 2
    assert [len(call["requests"]) for call in configuration_manager.calls] == [2, 1]
    for call, frequency_group in zip(
        configuration_manager.calls,
        (frequencies[:3], frequencies[3:]),
        strict=True,
    ):
        deploy_requests = call["requests"]
        assert [request.frequency_range_min_hz for request in deploy_requests] == (
            pytest.approx(np.full(len(deploy_requests), np.min(frequency_group) * 1e9))
        )
        assert [request.frequency_range_max_hz for request in deploy_requests] == (
            pytest.approx(np.full(len(deploy_requests), np.max(frequency_group) * 1e9))
        )
    assert events == ["deploy", "execute", "execute", "deploy", "execute"]
    assert len(execution_manager.calls) == 3
    np.testing.assert_allclose(result.frequency_range, frequencies)
    assert len(result.captures) == 4
    assert result.iq.shape == (4,)
    assert len(result.backend_results) == 3


def test_qubit_averaged_waveform_uses_detuned_packed_control() -> None:
    """Averaged-waveform qubit scans should detune one packed control instrument."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.array([4.0, 4.1])

    result = asyncio.run(
        manager.scan_qubit_frequencies_async(
            target="Q00",
            readout_target="R00",
            frequency_range=frequencies,
            readout_frequency=6.2,
            control_amplitude=0.2,
            control_duration=1.6,
            readout_amplitude=0.25,
            readout_duration=1.6,
            control_to_readout_gap=0.0,
            capture_delay=0.0,
            capture_length=1.6,
            point_interval=3.2,
            n_shots=1,
            shot_interval=0.0,
        )
    )

    assert len(configuration_manager.calls) == 1
    control_request, readout_request = configuration_manager.calls[0]["requests"]
    group_center = frequencies.mean()
    assert control_request.frequency_range_min_hz == pytest.approx(
        (group_center - MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ / 2) * 1e9
    )
    assert control_request.frequency_range_max_hz == pytest.approx(
        (group_center + MAX_QUBIT_SPECTROSCOPY_SPAN_GHZ / 2) * 1e9
    )
    assert readout_request.frequency_range_min_hz == pytest.approx(
        (6.2 - MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ / 2) * 1e9
    )
    assert readout_request.frequency_range_max_hz == pytest.approx(
        (6.2 + MAX_RESONATOR_SPECTROSCOPY_SPAN_GHZ / 2) * 1e9
    )
    call = execution_manager.calls[0]
    assert call["method"] == "execute_batch_async"
    (request,) = call["requests"]
    control_timeline = request.payload.fixed_timelines["unit-a:qubit"]
    assert len(control_timeline.events) == 2
    assert control_timeline.frequency_hz == pytest.approx(frequencies.mean() * 1e9)
    first_waveform = request.payload.waveform_library[
        control_timeline.events[0].waveform_name
    ].iq_array
    detuning = frequencies[0] - frequencies.mean()
    expected_control = 0.2 * np.exp(2j * np.pi * detuning * np.arange(4) * 0.4)
    np.testing.assert_allclose(first_waveform, expected_control)
    np.testing.assert_allclose(result.captures[0], np.ones(4))
    assert result.sampling_period_ns == pytest.approx(0.8)


def test_qubit_averaged_waveform_avoids_large_instrument_pool() -> None:
    """Averaged-waveform qubit batches should reuse one control instrument."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(
            manager.scan_qubit_frequencies_async(
                target="Q00",
                readout_target="R00",
                frequency_range=np.linspace(4.0, 4.2, 201),
                readout_frequency=6.2,
                control_amplitude=0.2,
                control_duration=1.6,
                readout_amplitude=0.25,
                readout_duration=1.6,
                control_to_readout_gap=0.0,
                capture_delay=0.0,
                capture_length=1.6,
                point_interval=3.2,
                n_shots=1,
                shot_interval=0.0,
                max_points_per_batch=200,
            )
        )

    assert len(configuration_manager.calls) == 1
    assert len(configuration_manager.calls[0]["requests"]) == 2
    assert len(execution_manager.calls) == 1
    assert execution_manager.calls[0]["method"] == "execute_batch_async"
    assert len(execution_manager.calls[0]["requests"]) == 2


def test_qubit_scan_reuses_group_instruments_across_batches() -> None:
    """A qubit group should deploy one batch-sized reusable control pool."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.linspace(4.0, 4.2, 201)

    with pytest.warns(RuntimeWarning, match="200 spectroscopy instruments"):
        result = asyncio.run(
            manager.scan_qubit_frequencies_async(
                target="Q00",
                readout_target="R00",
                frequency_range=frequencies,
                readout_frequency=6.2,
                control_amplitude=0.2,
                control_duration=4.0,
                readout_amplitude=0.25,
                readout_duration=3.2,
                control_to_readout_gap=2.0,
                capture_delay=1.0,
                capture_length=2.4,
                point_interval=10.0,
                n_shots=16,
                shot_interval=100.0,
                max_points_per_batch=200,
                capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
                parallel=False,
            )
        )

    assert isinstance(result, Quel3QubitSpectroscopyResult)
    assert len(configuration_manager.calls) == 1
    deploy_requests = configuration_manager.calls[0]["requests"]
    assert len(deploy_requests) == 201
    first_control_request = deploy_requests[0]
    second_control_request = deploy_requests[1]
    readout_request = deploy_requests[-1]
    assert first_control_request.alias == "qubit"
    assert first_control_request.target_labels == ("Q00",)
    assert first_control_request.frequency_range_min_hz == pytest.approx(4.0e9)
    assert first_control_request.frequency_range_max_hz == pytest.approx(4.2e9)
    assert second_control_request.alias == "qubit-spectroscopy-1"
    assert second_control_request.target_labels == ("Q00",)
    assert second_control_request.frequency_range_min_hz == pytest.approx(4.0e9)
    assert second_control_request.frequency_range_max_hz == pytest.approx(4.2e9)
    assert readout_request.alias == "readout"
    assert readout_request.frequency_range_min_hz == pytest.approx(6.2e9)
    assert readout_request.frequency_range_max_hz == pytest.approx(6.2e9)

    assert len(execution_manager.calls) == 2
    payload = execution_manager.calls[0]["request"].payload
    assert len(payload.fixed_timelines) == 201
    assert len(payload.waveform_library) == 2
    control_timeline = payload.fixed_timelines["unit-a:qubit"]
    second_control_timeline = payload.fixed_timelines["unit-a:qubit-spectroscopy-1"]
    readout_timeline = payload.fixed_timelines["unit-a:readout"]
    assert len(control_timeline.events) == 1
    assert control_timeline.capture_windows == ()
    assert control_timeline.frequency_hz == pytest.approx(frequencies[0] * 1e9)
    assert second_control_timeline.frequency_hz == pytest.approx(frequencies[1] * 1e9)
    assert len(readout_timeline.events) == 200
    assert len(readout_timeline.capture_windows) == 200
    assert readout_timeline.frequency_hz == pytest.approx(6.2e9)
    assert second_control_timeline.events[0].start_offset_ns == pytest.approx(10.0)
    assert readout_timeline.events[0].start_offset_ns == pytest.approx(6.0)
    assert readout_timeline.events[1].start_offset_ns == pytest.approx(16.0)
    assert readout_timeline.capture_windows[0].start_offset_ns == pytest.approx(7.0)
    assert payload.n_iterations == 16
    assert payload.shot_interval_ns == pytest.approx(100.0)

    second_payload = execution_manager.calls[1]["request"].payload
    assert tuple(second_payload.fixed_timelines) == (
        "unit-a:qubit",
        "unit-a:readout",
    )
    assert second_payload.fixed_timelines["unit-a:qubit"].frequency_hz == pytest.approx(
        frequencies[-1] * 1e9
    )

    first_control_event = control_timeline.events[0]
    first_control_waveform = payload.waveform_library[
        first_control_event.waveform_name
    ].iq_array
    np.testing.assert_allclose(
        first_control_waveform,
        np.full(10, 0.2 + 0.0j),
        rtol=1e-12,
        atol=1e-12,
    )

    readout_waveform_names = {event.waveform_name for event in readout_timeline.events}
    assert len(readout_waveform_names) == 1
    np.testing.assert_allclose(result.frequency_range, frequencies)
    np.testing.assert_allclose(result.captures[0], np.array([2.0 + 1.0j]))
    np.testing.assert_allclose(result.iq, np.full(201, 2.0 + 1.0j))
    assert result.readout_frequency == pytest.approx(6.2)
    assert result.sampling_period_ns is None
    assert len(result.backend_results) == 2


def test_qubit_scan_uses_each_frequency_as_its_control_carrier() -> None:
    """Each qubit frequency should be the carrier of its dedicated instrument."""
    execution_manager = _RecordingExecutionManager()
    manager = _manager(execution_manager)
    frequencies = np.array([4.0, 4.1, 4.4])

    asyncio.run(
        manager.scan_qubit_frequencies_async(
            target="Q00",
            readout_target="R00",
            frequency_range=frequencies,
            readout_frequency=6.2,
            control_amplitude=0.2,
            control_duration=1.6,
            readout_amplitude=0.25,
            readout_duration=1.6,
            control_to_readout_gap=0.0,
            capture_delay=0.0,
            capture_length=1.6,
            point_interval=3.2,
            n_shots=1,
            shot_interval=0.0,
            max_points_per_batch=2,
            capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
        )
    )

    first_payload = execution_manager.calls[0]["request"].payload
    second_payload = execution_manager.calls[1]["request"].payload
    first_control = first_payload.fixed_timelines["unit-a:qubit"]
    second_control = first_payload.fixed_timelines["unit-a:qubit-spectroscopy-1"]
    third_control = second_payload.fixed_timelines["unit-a:qubit"]
    assert first_control.frequency_hz == pytest.approx(4.0e9)
    assert second_control.frequency_hz == pytest.approx(4.1e9)
    assert third_control.frequency_hz == pytest.approx(4.4e9)


def test_qubit_scan_returns_values_per_iteration() -> None:
    """Values-per-iteration qubit scans should preserve every shot value."""
    execution_manager = _RecordingExecutionManager()
    manager = _manager(execution_manager)
    frequencies = np.array([4.0, 4.1])

    result = asyncio.run(
        manager.scan_qubit_frequencies_async(
            target="Q00",
            readout_target="R00",
            frequency_range=frequencies,
            readout_frequency=6.2,
            control_amplitude=0.2,
            control_duration=1.6,
            readout_amplitude=0.25,
            readout_duration=1.6,
            control_to_readout_gap=0.0,
            capture_delay=0.0,
            capture_length=1.6,
            point_interval=3.2,
            n_shots=3,
            shot_interval=0.0,
            capture_mode=Quel3CaptureMode.VALUES_PER_ITER,
        )
    )

    payload = execution_manager.calls[0]["request"].payload
    assert payload.capture_mode is Quel3CaptureMode.VALUES_PER_ITER
    assert len(result.captures) == 2
    np.testing.assert_allclose(result.captures[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(result.iq, np.array([[1.0, 2.0, 3.0]] * 2))
    assert result.capture_mode is Quel3CaptureMode.VALUES_PER_ITER
    assert result.sampling_period_ns is None


def test_qubit_scan_redeploys_each_control_frequency_group() -> None:
    """A qubit scan should redeploy only when the control span changes."""
    events: list[str] = []
    execution_manager = _RecordingExecutionManager(events)
    configuration_manager = _RecordingConfigurationManager(events)
    manager = _manager(execution_manager, configuration_manager)
    frequencies = np.array([3.0, 3.8, 4.6, 5.4])

    result = asyncio.run(
        manager.scan_qubit_frequencies_async(
            target="Q00",
            readout_target="R00",
            frequency_range=frequencies,
            readout_frequency=6.2,
            control_amplitude=0.2,
            control_duration=1.6,
            readout_amplitude=0.25,
            readout_duration=1.6,
            control_to_readout_gap=0.0,
            capture_delay=0.0,
            capture_length=1.6,
            point_interval=3.2,
            n_shots=1,
            shot_interval=0.0,
            max_points_per_batch=2,
            capture_mode=Quel3CaptureMode.AVERAGED_VALUE,
            parallel=False,
        )
    )

    assert len(configuration_manager.calls) == 2
    assert [len(call["requests"]) for call in configuration_manager.calls] == [3, 2]
    for call, frequency_group in zip(
        configuration_manager.calls,
        (frequencies[:3], frequencies[3:]),
        strict=True,
    ):
        *control_requests, readout_request = call["requests"]
        assert [request.frequency_range_min_hz for request in control_requests] == (
            pytest.approx(np.full(len(control_requests), np.min(frequency_group) * 1e9))
        )
        assert [request.frequency_range_max_hz for request in control_requests] == (
            pytest.approx(np.full(len(control_requests), np.max(frequency_group) * 1e9))
        )
        assert readout_request.frequency_range_min_hz == pytest.approx(6.2e9)
        assert readout_request.frequency_range_max_hz == pytest.approx(6.2e9)
    assert events == ["deploy", "execute", "execute", "deploy", "execute"]
    assert len(execution_manager.calls) == 3
    np.testing.assert_allclose(result.frequency_range, frequencies)
    assert len(result.captures) == 4
    assert result.iq.shape == (4,)
    assert len(result.backend_results) == 3


def test_instrument_pool_warning_threshold_is_strict() -> None:
    """A 20-instrument spectroscopy pool should not emit a warning."""
    execution_manager = _RecordingExecutionManager()
    configuration_manager = _RecordingConfigurationManager()
    manager = _manager(execution_manager, configuration_manager)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(
            manager.scan_resonator_frequencies_async(
                target="R00",
                frequency_range=np.linspace(6.0, 6.1, 20),
                readout_amplitude=0.25,
                readout_duration=3.2,
                capture_delay=0.0,
                capture_length=3.2,
                point_interval=3.2,
                n_shots=1,
                shot_interval=0.0,
                max_points_per_batch=20,
                capture_mode=Quel3CaptureMode.VALUES_PER_ITER,
            )
        )

    assert len(configuration_manager.calls[0]["requests"]) == 20


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frequency_range": []}, "frequency_range must not be empty"),
        ({"readout_amplitude": 1.1}, "readout_amplitude"),
        ({"readout_duration": 1.0}, "sampling grid"),
        ({"point_interval": 10.0}, "occupied length"),
        ({"n_shots": 0}, "n_shots"),
        ({"max_points_per_batch": 0}, "max_points_per_batch"),
        ({"capture_mode": Quel3CaptureMode.RAW_WAVEFORMS}, "capture_mode"),
        ({"capture_mode": Quel3CaptureMode.UNSPECIFIED}, "capture_mode"),
    ],
)
def test_resonator_scan_rejects_invalid_settings(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Invalid resonator settings should fail before backend execution."""
    execution_manager = _RecordingExecutionManager()
    manager = _manager(execution_manager)
    kwargs: dict[str, object] = {
        "target": "R00",
        "frequency_range": [6.0],
        "readout_amplitude": 0.25,
        "readout_duration": 16.0,
        "capture_delay": 8.0,
        "capture_length": 8.0,
        "point_interval": 16.0,
        "n_shots": 1,
        "shot_interval": 0.0,
        "max_points_per_batch": 200,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            manager.scan_resonator_frequencies_async(**kwargs)  # type: ignore[arg-type]
        )

    assert execution_manager.calls == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"readout_target": "Q00"},
            "ports must be different",
        ),
        ({"control_amplitude": -1.1}, "control_amplitude"),
        ({"control_duration": 1.0}, "control sampling grid"),
        ({"control_to_readout_gap": -1.0}, "control_to_readout_gap"),
        ({"point_interval": 9.0}, "occupied length"),
    ],
)
def test_qubit_scan_rejects_invalid_settings(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Invalid qubit settings should fail before backend execution."""
    execution_manager = _RecordingExecutionManager()
    manager = _manager(execution_manager)
    kwargs: dict[str, object] = {
        "target": "Q00",
        "readout_target": "R00",
        "frequency_range": [4.0],
        "readout_frequency": 6.2,
        "control_amplitude": 0.2,
        "control_duration": 4.0,
        "readout_amplitude": 0.25,
        "readout_duration": 3.2,
        "control_to_readout_gap": 2.0,
        "capture_delay": 1.0,
        "capture_length": 2.4,
        "point_interval": 10.0,
        "n_shots": 1,
        "shot_interval": 0.0,
        "max_points_per_batch": 200,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            manager.scan_qubit_frequencies_async(**kwargs)  # type: ignore[arg-type]
        )

    assert execution_manager.calls == []
