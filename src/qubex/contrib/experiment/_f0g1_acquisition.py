"""Pulse construction and single-shot acquisition shared by F0G1 experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from qubex.backend.quel3 import Quel3BackendController
from qubex.experiment import Experiment
from qubex.measurement import SweepMeasurementResult
from qubex.pulse import Blank, FlatTop, PulseSchedule
from qubex.system import TargetType

State = Literal["g", "e", "f"]


@dataclass(frozen=True)
class F0G1AcquisitionOptions:
    """
    Configure a two-tone measurement on a shared QuEL-3 control port.

    Times are in ns. The post-drive wait is included in returned pulses.
    Frequencies are in GHz. An explicit reset resonator frequency overrides
    the configured readout frequency used to seed the carrier search.
    """

    ramp_duration: float = 16.0
    shot_interval: float = 500_000.0
    n_shots: int = 1024
    verification_shots: int = 4096
    wait_duration: float = 0.0
    reset_resonator_frequency: float | None = None

    def __post_init__(self) -> None:
        """Validate settings shared by all acquisitions."""
        for name in ("ramp_duration", "shot_interval"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.wait_duration) or self.wait_duration < 0:
            raise ValueError("wait_duration must be finite and nonnegative")
        for value in (self.n_shots, self.verification_shots):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Shot counts must be positive integers")
        frequency = self.reset_resonator_frequency
        if frequency is not None and (not np.isfinite(frequency) or frequency <= 0):
            raise ValueError("Resonator frequency must be positive and finite")


@dataclass(frozen=True)
class F0G1PulseParameters:
    """
    Describe one pair of reset tones and their common duration.

    Parameters
    ----------
    duration : float
        Nonnegative tone duration in ns, including both ramps and excluding
        the post-drive wait. Zero creates a reference with neither tone.
    f0g1_frequency : float
        Positive f0g1 carrier frequency in GHz.
    f0g1_amplitude : float
        Normalized command amplitude in [0, 1]; zero disables this tone.
    ef_frequency : float
        Positive carrier frequency in GHz for the reset ef tone, independent
        of the frequency used for state-preparation pi pulses.
    ef_amplitude : float
        Normalized command amplitude in [0, 1]; zero disables this tone.

    Notes
    -----
    All values must be finite. A nonzero duration must accommodate the ramps
    when building a schedule. Acquired scan parameters record effective,
    sampling-grid-aligned durations rather than the requested off-grid values.
    """

    duration: float
    f0g1_frequency: float
    f0g1_amplitude: float
    ef_frequency: float
    ef_amplitude: float

    def __post_init__(self) -> None:
        """Validate physical coordinates, allowing zero-duration reference points."""
        if self.f0g1_amplitude + self.ef_amplitude > 1 + 1e-12:
            raise ValueError("Reset tones exceed the shared-port amplitude budget")
        if not np.isfinite(self.duration) or self.duration < 0:
            raise ValueError("duration must be finite and nonnegative")
        for frequency in (self.f0g1_frequency, self.ef_frequency):
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("Frequencies must be finite and positive")
        for amplitude in (self.f0g1_amplitude, self.ef_amplitude):
            if not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
                raise ValueError("Amplitudes must be finite and in [0, 1]")


@dataclass(frozen=True)
class F0G1PopulationScan:
    """
    Keep pulse coordinates, assignment counts, and raw measurements together.

    Attributes
    ----------
    parameters : tuple[F0G1PulseParameters, ...]
        Effective pulse coordinates in acquisition order.
    initial_states : tuple[State, ...]
        Preparation order: `("f",)` for f0g1-only, otherwise `("g", "e", "f")`.
    counts : NDArray[np.int64]
        Counts shaped `(point, preparation, assigned state)`. The final axis
        always means g/e/f. Fractions retain preparation and readout errors.
    measurements : tuple[SweepMeasurementResult, ...]
        Raw sweeps in preparation order, then in consecutive acquisition chunks.

    Notes
    -----
    Manually created results must have consistent axes and positive shot totals;
    this container does not validate them. Frozen fields do not freeze arrays.
    """

    parameters: tuple[F0G1PulseParameters, ...]
    initial_states: tuple[State, ...]
    counts: NDArray[np.int64]
    measurements: tuple[SweepMeasurementResult, ...]

    @property
    def populations(self) -> NDArray[np.float64]:
        """Return assignment fractions shaped `(point, initial state, g/e/f)`."""
        return self.counts / self.counts.sum(axis=-1, keepdims=True)

    @property
    def excitation(self) -> NDArray[np.float64]:
        """Return assigned `P(e) + P(f)` shaped `(point, initial state)`."""
        return self.populations[..., 1:].sum(axis=-1)

    def plot(self) -> go.Figure:
        """
        Plot assigned excitation against the acquisition-point index.

        Returns
        -------
        go.Figure
            One trace per preparation state. Values are fractions, not percent.
            Look up physical coordinates in `parameters` using the point index;
            multidimensional scans are displayed in flattened acquisition order.
            Call `.show()` explicitly to display the figure.
        """
        figure = go.Figure()
        for i, state in enumerate(self.initial_states):
            figure.add_scatter(y=self.excitation[:, i], name=f"Initial {state}")
        figure.update_layout(
            xaxis_title="Measured point", yaxis_title="Assigned excitation"
        )
        return figure


def _estimate_f0g1_frequency(
    exp: Experiment, target: str, options: F0G1AcquisitionOptions
) -> float:
    """Estimate the undressed f0g1 carrier from transition energy balance."""
    resonator = options.reset_resonator_frequency
    if resonator is None:
        resonator = exp.resonators[target].frequency
    q = exp.qubits[target]
    return q.control_frequency_ge + q.control_frequency_ef - resonator


def register_f0g1_targets(
    exp: Experiment, target: str, *, options: F0G1AcquisitionOptions | None = None
) -> None:
    """
    Register separate f0g1 and ef-reset aliases on the qubit control port.

    Parameters
    ----------
    exp : Experiment
        QuEL-3 experiment after `configure()`, with the qubit target registered.
    target : str
        Qubit label. Creates `<target>-f0g1` and `<target>-ef-reset` aliases.
    options : F0G1AcquisitionOptions | None
        Optional reset-resonator frequency used for the initial f0g1 carrier.
        `None` uses default options.

    Raises
    ------
    TypeError
        If the backend is not QuEL-3 or the generator port number is not an integer.
    ValueError
        If an existing alias has a different port, qubit, or target type.

    Notes
    -----
    This operation deploys instruments to hardware and modifies the session's
    registry; it does not persist configuration files. Compatible aliases are
    reused without changing their existing frequency. Both aliases are checked
    for conflicts before registration begins, but hardware failures are not
    transactional: registration of the first alias may already have succeeded.
    A separate ef-reset alias prevents ef sweeps from detuning preparation pulses.
    """
    if not isinstance(exp.backend_controller, Quel3BackendController):
        raise TypeError("F0G1 requires QuEL-3")
    options = options or F0G1AcquisitionOptions()
    port = exp.targets[target].channel.port
    if not isinstance(port.number, int):
        raise TypeError("An integer generator port is required")
    aliases = (
        (f"{target}-f0g1", _estimate_f0g1_frequency(exp, target, options)),
        (f"{target}-ef-reset", exp.qubits[target].control_frequency_ef),
    )
    # Validate both aliases before deploying either; deployment itself is not atomic.
    for label, _ in aliases:
        if label in exp.targets:
            existing = exp.targets[label]
            if (
                existing.channel.port != port
                or existing.qubit != target
                or existing.type != TargetType.CTRL_FH
            ):
                raise ValueError(f"Conflicting target: {label}")
    for label, frequency in aliases:
        if label not in exp.targets:
            exp.register_custom_target(
                label=label,
                frequency=frequency,
                box_id=port.box_id,
                port_number=port.number,
                qubit_label=target,
                target_type=TargetType.CTRL_FH,
            )


def _align_duration(exp: Experiment, value: float) -> float:
    """Round upward in samples after suppressing floating-point boundary noise."""
    period = float(exp.ctx.measurement.sampling_period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("Invalid sampling period")
    return float(np.ceil(np.round(value / period, 10)) * period)


def build_f0g1_reset(
    exp: Experiment,
    target: str,
    parameters: F0G1PulseParameters,
    *,
    options: F0G1AcquisitionOptions | None = None,
) -> PulseSchedule:
    """
    Build simultaneous F0G1 and ef tones on the registered carriers.

    The requested physical frequency is encoded as a baseband offset. This
    lets a calibrated schedule be composed with `PulseSchedule.call` on the
    same registered targets, including when the reset ef tone is Stark shifted.
    A zero-drive control occupies the same time as an active pulse.

    Parameters
    ----------
    exp : Experiment
        Configured experiment with reset target aliases registered.
    target : str
        Qubit on whose control port both reset tones are applied.
    parameters : F0G1PulseParameters
        Requested frequencies in GHz, duration in ns and command amplitudes.
    options : F0G1AcquisitionOptions | None
        Ramp duration and post-drive wait in ns.

    Returns
    -------
    PulseSchedule
        Reset followed by the configured wait, without preparation or readout.
    """
    options = options or F0G1AcquisitionOptions()
    dt = float(exp.ctx.measurement.sampling_period)
    duration = _align_duration(exp, parameters.duration)
    wait = _align_duration(exp, options.wait_duration)
    labels = (f"{target}-f0g1", f"{target}-ef-reset")
    drives = (
        (labels[0], parameters.f0g1_amplitude, parameters.f0g1_frequency),
        (labels[1], parameters.ef_amplitude, parameters.ef_frequency),
    )
    with PulseSchedule([target, *labels]) as pulse:
        pulse.add(target, Blank(duration, sampling_period=dt))
        for label, amplitude, frequency in drives:
            carrier = exp.targets[label].frequency
            pulse.set_frequency(label, carrier)
            if not duration or not amplitude:
                continue
            if duration < 2 * options.ramp_duration:
                raise ValueError("Tone duration must accommodate two ramps")
            offset = frequency - carrier
            if not np.isfinite(offset) or abs(offset) >= 0.5 / dt:
                raise ValueError(f"{label} frequency offset exceeds digital Nyquist")
            envelope = FlatTop(
                duration=duration,
                amplitude=amplitude,
                tau=options.ramp_duration,
                sampling_period=dt,
            )
            pulse.add(label, envelope.detuned(offset))
        pulse.barrier()
        pulse.add(target, Blank(wait, sampling_period=dt))
    return pulse


async def _measure_populations(
    exp: Experiment,
    target: str,
    parameters: tuple[F0G1PulseParameters, ...],
    options: F0G1AcquisitionOptions,
    states: tuple[State, ...] = ("g", "e", "f"),
    *,
    verification: bool = False,
) -> F0G1PopulationScan:
    """Measure ordered pulse coordinates in bounded, fixed-carrier shot batches."""
    if not isinstance(exp.backend_controller, Quel3BackendController):
        raise TypeError("F0G1 requires QuEL-3")
    labels = (target, f"{target}-ef", f"{target}-f0g1", f"{target}-ef-reset")
    for label in labels:
        if label not in exp.targets:
            raise ValueError(f"Missing target: {label}")
    classifier = exp.classifiers[target]
    if classifier.n_states != 3 or set(classifier.label_map.values()) != {0, 1, 2}:
        raise ValueError("A g/e/f classifier is required")
    ge_pi, ef_pi = (exp.get_pi_pulse(label) for label in labels[:2])
    carriers = {label: exp.targets[label].frequency for label in labels[2:]}
    shots = options.verification_shots if verification else options.n_shots
    parameters = tuple(
        replace(p, duration=_align_duration(exp, p.duration)) for p in parameters
    )
    counts = np.zeros((len(parameters), len(states), 3), dtype=np.int64)
    measurements = []
    for state_index, state in enumerate(states):
        with PulseSchedule(list(labels)) as prep:
            if state in ("e", "f"):
                prep.add(target, ge_pi)
                prep.barrier()
            if state == "f":
                prep.add(labels[1], ef_pi)
                prep.barrier()
        for start in range(0, len(parameters), 48):
            batch = parameters[start : start + 48]

            def schedule(
                index: int,
                batch: tuple[F0G1PulseParameters, ...] = batch,
                prep: PulseSchedule = prep,
            ) -> PulseSchedule:
                prepared = prep.copy()
                prepared.call(
                    build_f0g1_reset(exp, target, batch[int(index)], options=options)
                )
                prepared.set_frequencies(carriers)
                return prepared

            result = await exp.run_sweep_measurement(
                schedule,
                sweep_values=np.arange(len(batch)),
                frequencies={**carriers},
                n_shots=shots,
                shot_interval=options.shot_interval,
                time_integration=True,
                shot_averaging=False,
                state_classification=False,
                final_measurement=True,
                plot=False,
            )
            captures = result.data[target]
            if (
                len(captures) != 1
                or captures[0].shape != (len(batch), shots)
                or not np.isfinite(captures[0]).all()
            ):
                raise ValueError("Expected one finite single-shot capture per point")
            for i, iq in enumerate(captures[0]):
                assigned = np.asarray(classifier.predict(iq))
                if assigned.shape != (shots,) or not np.isin(assigned, (0, 1, 2)).all():
                    raise ValueError("Invalid classifier output")
                counts[start + i, state_index] = np.bincount(
                    assigned.astype(int), minlength=3
                )
            measurements.append(result)
    return F0G1PopulationScan(parameters, states, counts, tuple(measurements))
