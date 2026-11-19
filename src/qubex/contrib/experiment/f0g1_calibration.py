"""
Calibrate two-tone reset through measured qutrit populations.

The recipe is spectroscopy, joint drive selection, duration selection, then
independent verification. Each acquisition is also usable on its own.
Magnard et al., https://arxiv.org/abs/1801.07689 motivates measuring the ef
transition in the presence of the f0g1 tone. This empirical workflow does not
estimate coupling rates or correct assignment errors.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qubex.experiment import Experiment
from qubex.pulse import PulseSchedule

from ._f0g1_acquisition import (
    F0G1AcquisitionOptions,
    F0G1PopulationScan,
    F0G1PulseParameters,
    _align_duration,
    _estimate_f0g1_frequency,
    _measure_populations,
    build_f0g1_reset as build_f0g1_reset,
    register_f0g1_targets as register_f0g1_targets,
)


@dataclass(frozen=True)
class F0G1CalibrationOptions(F0G1AcquisitionOptions):
    """
    Configure the staged search and its empirical acceptance criterion.

    Parameters
    ----------
    ramp_duration : float
        Rise and fall time of each flat-top tone in ns; default 16.
    shot_interval : float
        Positive interval between shots in ns; default 500,000.
    search_duration : float
        Total tone duration, including ramps, for both drive scans in ns;
        default 1,000. A common duration makes the measured candidates comparable.
    durations : tuple[float, ...]
        Increasing duration grid in ns; default 100 through 10,000 in steps of
        100. Each value must accommodate two ramps. Acquisition rounds upward
        to the sampling grid and removes duplicate duration-scan points.
    frequency_offsets : tuple[float, ...]
        f0g1 offsets in GHz from `f_ge + f_ef - f_resonator`; default 21 points
        from -0.05 through +0.03.
    amplitudes : tuple[float, ...]
        f0g1 command amplitudes in (0, 1]; default seven points from 0.5 to 1.
    ef_frequency_offsets : tuple[float, ...]
        Offsets in GHz from the calibrated ef transition; default 21 points
        from -0.05 through +0.05. These are tested with f0g1 drive present.
    ef_amplitudes : tuple[float, ...]
        ef command amplitudes in (0, 1]; default 0.0025, 0.005, 0.0075, 0.01.
    target_excitation : float
        Maximum assigned `P(e) + P(f)` allowed for every initial state;
        default 0.01. Must be strictly between zero and one.
    n_shots : int
        Positive shots per point and preparation in baseline and search scans;
        default 1,024.
    verification_shots : int
        Positive shots per preparation for independent confirmation;
        default 4,096.
    wait_duration : float
        Nonnegative delay after the tones in ns; default zero. Included in
        both the measurements and returned schedule, but not in `duration`.
    reset_resonator_frequency : float | None
        Positive reset-resonator frequency in GHz. `None` uses
        `exp.resonators[target].frequency`.

    Notes
    -----
    All sweep grids must be nonempty, finite, and strictly increasing; a
    single-point grid is allowed. Defaults are starting ranges, not device
    calibrations. Amplitudes are command values, not physical Rabi rates.
    The threshold uses observed classification fractions without assignment
    correction or a finite-shot confidence bound.
    """

    search_duration: float = 1000.0
    durations: tuple[float, ...] = tuple(np.arange(100.0, 10001.0, 100.0))
    frequency_offsets: tuple[float, ...] = tuple(np.linspace(-0.05, 0.03, 21))
    amplitudes: tuple[float, ...] = tuple(np.linspace(0.5, 1.0, 7))
    ef_frequency_offsets: tuple[float, ...] = tuple(np.linspace(-0.05, 0.05, 21))
    ef_amplitudes: tuple[float, ...] = (0.0025, 0.005, 0.0075, 0.01)
    target_excitation: float = 0.01

    def __post_init__(self) -> None:
        """Reject invalid search specifications before acquisition."""
        super().__post_init__()
        if not np.isfinite(self.search_duration) or self.search_duration <= 0:
            raise ValueError("search_duration must be finite and positive")
        if not 0 < self.target_excitation < 1:
            raise ValueError("target_excitation must be between zero and one")
        for name in (
            "durations",
            "frequency_offsets",
            "amplitudes",
            "ef_frequency_offsets",
            "ef_amplitudes",
        ):
            grid = np.asarray(getattr(self, name), dtype=float)
            if (
                grid.ndim != 1
                or not grid.size
                or not np.all(np.isfinite(grid))
                or np.any(np.diff(grid) <= 0)
            ):
                raise ValueError(f"{name} must be a nonempty increasing finite grid")
        if (
            min(self.durations) < 2 * self.ramp_duration
            or self.search_duration < 2 * self.ramp_duration
        ):
            raise ValueError("Durations must accommodate both ramps")
        if any(not 0 < a <= 1 for a in (*self.amplitudes, *self.ef_amplitudes)):
            raise ValueError("Amplitudes must be in (0, 1]")


@dataclass(frozen=True)
class F0G1CalibrationResult:
    """
    Keep completed scans and expose a reset schedule only after verification.

    Attributes
    ----------
    success : bool
        Whether fresh measurements meet the threshold for every g/e/f input.
    message : str
        Outcome and, if unsuccessful, which acceptance stage failed.
    parameters : F0G1PulseParameters | None
        Selected candidate. If no duration qualifies, retains the drive-search
        candidate at `search_duration`; it is not a verified reset.
    pulse : PulseSchedule | None
        Reset and wait, without preparation or readout; `None` on failure.
    scans : dict[str, F0G1PopulationScan]
        `baseline`, `f0g1`, `ef`, `duration`, and, if a duration qualifies,
        `verification`. Acquisition errors propagate instead of returning a result.
    options : F0G1CalibrationOptions
        Settings used for this run.
    """

    success: bool
    message: str
    parameters: F0G1PulseParameters | None
    pulse: PulseSchedule | None
    scans: dict[str, F0G1PopulationScan] = field(default_factory=dict)
    options: F0G1CalibrationOptions = field(default_factory=F0G1CalibrationOptions)

    def save(self, directory: Path | str) -> Path:
        """
        Save metadata and raw measurements into a new unique run directory.

        Parameters
        ----------
        directory : Path | str
            Parent directory, created if needed.

        Returns
        -------
        Path
            Newly created `f0g1-*` directory. `summary.json` contains options,
            status, candidate coordinates, preparation order, and counts.
            Each `<stage>-<index>.nc` file holds a raw batch, in preparation and acquisition order.

        Notes
        -----
        Writes files without overwriting previous runs. Saving is not atomic;
        a serialization or filesystem error may leave partial output. Pulse
        objects and classifier models are not serialized, so this is a record
        of the measurements rather than a complete experiment checkpoint.
        """
        parent = Path(directory)
        parent.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="f0g1-", dir=parent))
        summary = {
            "options": asdict(self.options),
            "success": self.success,
            "message": self.message,
            "parameters": asdict(self.parameters) if self.parameters else None,
            "scans": {},
        }
        for name, scan in self.scans.items():
            summary["scans"][name] = {
                "parameters": [asdict(point) for point in scan.parameters],
                "initial_states": scan.initial_states,
                "counts": scan.counts.tolist(),
            }
            for index, measurement in enumerate(scan.measurements):
                measurement.save_netcdf(destination / f"{name}-{index:03d}.nc")
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return destination


async def scan_f0g1_drive(
    exp: Experiment, target: str, *, options: F0G1CalibrationOptions | None = None
) -> F0G1PopulationScan:
    """
    Measure f0g1-only transfer from f at a fixed duration.

    Parameters
    ----------
    exp : Experiment
        Prepared QuEL-3 experiment; see `calibrate_f0g1_pulse` for prerequisites.
    target : str
        Qubit label.
    options : F0G1CalibrationOptions | None
        f0g1 sweep grids, search duration, and shot settings; defaults if omitted.

    Returns
    -------
    F0G1PopulationScan
        Measured points ordered by amplitude then frequency offset, with only
        the f preparation. No amplitude is selected at this stage.

    Notes
    -----
    Acquires hardware data with ef reset disabled. Unlike the paper's constant
    amplitude-times-duration scan, every point uses `search_duration`.
    """
    options = options or F0G1CalibrationOptions()
    center = _estimate_f0g1_frequency(exp, target, options)
    points = tuple(
        F0G1PulseParameters(
            duration=options.search_duration,
            f0g1_frequency=center + offset,
            f0g1_amplitude=amplitude,
            ef_frequency=exp.qubits[target].control_frequency_ef,
            ef_amplitude=0.0,
        )
        for amplitude in options.amplitudes
        for offset in options.frequency_offsets
    )
    return await _measure_populations(exp, target, points, options, ("f",))


async def scan_f0g1_ef_drive(
    exp: Experiment,
    target: str,
    drive_scan: F0G1PopulationScan,
    *,
    options: F0G1CalibrationOptions | None = None,
) -> F0G1PopulationScan:
    """
    Test ef tones with the best f0g1 frequency at each amplitude.

    Parameters
    ----------
    exp : Experiment
        Prepared QuEL-3 experiment; see `calibrate_f0g1_pulse` for prerequisites.
    target : str
        Qubit label.
    drive_scan : F0G1PopulationScan
        Nonempty result of `scan_f0g1_drive` for this target. The first preparation
        axis is assumed to be f; manually constructed inputs must honor this.
    options : F0G1CalibrationOptions | None
        ef grids, search duration, and shot settings; defaults if omitted.

    Returns
    -------
    F0G1PopulationScan
        g/e/f measurements ordered by f0g1 amplitude, ef amplitude, then ef
        frequency offset. Each amplitude keeps the first maximum-P(g) frequency
        from `drive_scan`. All two-tone candidates are returned for comparison.

    Notes
    -----
    Acquires hardware data with f0g1 on so ef sees its induced Stark shift.
    """
    options = options or F0G1CalibrationOptions()
    ground_populations = drive_scan.populations[:, 0, 0]
    f0g1_amplitudes = sorted({point.f0g1_amplitude for point in drive_scan.parameters})
    points = []

    # Keep all amplitudes: f-only transfer does not determine two-tone performance.
    for f0g1_amplitude in f0g1_amplitudes:
        matching_indices = [
            index
            for index, point in enumerate(drive_scan.parameters)
            if point.f0g1_amplitude == f0g1_amplitude
        ]
        best_index = max(matching_indices, key=lambda index: ground_populations[index])
        f0g1_drive = drive_scan.parameters[best_index]

        for ef_amplitude in options.ef_amplitudes:
            if f0g1_amplitude + ef_amplitude > 1 + 1e-12:
                continue
            for ef_offset in options.ef_frequency_offsets:
                ef_frequency = exp.qubits[target].control_frequency_ef + ef_offset
                points.append(
                    replace(
                        f0g1_drive,
                        duration=options.search_duration,
                        ef_frequency=ef_frequency,
                        ef_amplitude=ef_amplitude,
                    )
                )
    return await _measure_populations(exp, target, tuple(points), options)


async def scan_f0g1_duration(
    exp: Experiment,
    target: str,
    parameters: F0G1PulseParameters,
    *,
    options: F0G1CalibrationOptions | None = None,
) -> F0G1PopulationScan:
    """
    Measure g/e/f reset versus duration with fixed drive frequencies and amplitudes.

    Parameters
    ----------
    exp : Experiment
        Prepared QuEL-3 experiment; see `calibrate_f0g1_pulse` for prerequisites.
    target : str
        Qubit label.
    parameters : F0G1PulseParameters
        Selected drive settings; its duration is replaced by the sweep grid.
    options : F0G1CalibrationOptions | None
        Duration grid and shot settings; defaults if omitted.

    Returns
    -------
    F0G1PopulationScan
        Hardware measurements ordered by increasing sampling-grid-aligned time.
        Duplicate effective times are removed. No fit or selection is performed.
    """
    options = options or F0G1CalibrationOptions()
    points = tuple(
        replace(parameters, duration=t)
        for t in sorted({_align_duration(exp, t) for t in options.durations})
    )
    return await _measure_populations(exp, target, points, options)


async def evaluate_f0g1_reset(
    exp: Experiment,
    target: str,
    parameters: F0G1PulseParameters,
    *,
    options: F0G1CalibrationOptions | None = None,
) -> F0G1PopulationScan:
    """
    Remeasure one reset candidate with fresh shots from g/e/f.

    Parameters
    ----------
    exp : Experiment
        Prepared QuEL-3 experiment; see `calibrate_f0g1_pulse` for prerequisites.
    target : str
        Qubit label.
    parameters : F0G1PulseParameters
        Candidate frequencies, amplitudes, and duration.
    options : F0G1CalibrationOptions | None
        Uses `verification_shots`; defaults if omitted.

    Returns
    -------
    F0G1PopulationScan
        One independently measured point. Compare `excitation.max()` with the
        target to decide acceptance. This function neither selects nor retries.
    """
    return await _measure_populations(
        exp,
        target,
        (parameters,),
        options or F0G1CalibrationOptions(),
        verification=True,
    )


def _select_duration(excitation: NDArray, threshold: float) -> int | None:
    """Return the first acceptable point; callers supply ascending durations."""
    worst_excitation = excitation.max(axis=1)
    acceptable_indices = np.flatnonzero(worst_excitation <= threshold)
    return int(acceptable_indices[0]) if acceptable_indices.size else None


def select_f0g1_drive(
    frequencies: ArrayLike, amplitudes: ArrayLike, ground_populations: ArrayLike
) -> tuple[float, float]:
    """
    Select coordinates of the largest entry in a ground-population map.

    Parameters
    ----------
    frequencies : ArrayLike
        One-dimensional frequency coordinates in GHz.
    amplitudes : ArrayLike
        One-dimensional command-amplitude coordinates.
    ground_populations : ArrayLike
        Finite nonempty map of shape `(n_frequencies, n_amplitudes)`.

    Returns
    -------
    tuple[float, float]
        Measured frequency and amplitude at the maximum population. Ties use
        frequency-major input order, then amplitude order.

    Raises
    ------
    ValueError
        If the population map is empty, nonfinite, or has the wrong shape.

    Notes
    -----
    This standalone array helper performs no measurement. It selects one point
    from a supplied map; the staged workflow instead retains a frequency for
    each f0g1 amplitude before comparing two-tone performance.
    """
    populations = np.asarray(ground_populations)
    if (
        populations.shape != (len(np.asarray(frequencies)), len(np.asarray(amplitudes)))
        or not populations.size
        or not np.all(np.isfinite(populations))
    ):
        raise ValueError("Invalid population grid")
    i, j = np.unravel_index(np.argmax(populations), populations.shape)
    return float(np.asarray(frequencies)[i]), float(np.asarray(amplitudes)[j])


async def calibrate_f0g1_pulse(
    exp: Experiment, target: str, *, options: F0G1CalibrationOptions | None = None
) -> F0G1CalibrationResult:
    """
    Select reset tones and duration, then independently verify residual excitation.

    Parameters
    ----------
    exp : Experiment
        Configured QuEL-3 experiment with ge/ef pi pulses, a classifier mapping
        0/1/2 to g/e/f, and registered qubit, ef, f0g1, and ef-reset targets.
    target : str
        Qubit label shared by those targets.
    options : F0G1CalibrationOptions | None
        Search ranges, acquisition settings, and empirical acceptance threshold.
        `None` uses defaults. Keep these settings consistent with registration.

    Returns
    -------
    F0G1CalibrationResult
        Candidate settings and completed scans. `success` is true and `pulse`
        is available only when independent verification meets the threshold
        for every preparation. Missing qualifying durations or failed
        verification return diagnostics with `success=False`.

    Raises
    ------
    TypeError
        If the experiment backend is not QuEL-3.
    ValueError
        If required targets, classifier labels, or measured capture payloads
        violate the workflow contract. Acquisition exceptions propagate.

    Notes
    -----
    Requires explicitly registered QuEL-3 reset targets and calibrated ge/ef
    pi pulses and classifier. Measures hardware without updating calibration
    settings. Success describes observed assignment fractions, not corrected
    populations or resonator vacuum. Search grids do not guarantee a global optimum.

    Examples
    --------
    >>> options = F0G1CalibrationOptions(target_excitation=0.01)
    >>> register_f0g1_targets(exp, "Q00", options=options)
    >>> result = await calibrate_f0g1_pulse(exp, "Q00", options=options)
    >>> if result.success:
    ...     reset = result.pulse

    Inspect the two-tone scan before choosing a duration manually:

    >>> drive = await scan_f0g1_drive(exp, "Q00", options=options)
    >>> joint = await scan_f0g1_ef_drive(exp, "Q00", drive, options=options)
    >>> joint.plot().show()
    >>> best = int(np.argmin(joint.excitation.max(axis=1)))
    >>> times = await scan_f0g1_duration(
    ...     exp, "Q00", joint.parameters[best], options=options
    ... )
    >>> times.plot().show()
    """
    options = options or F0G1CalibrationOptions()
    baseline = F0G1PulseParameters(
        duration=0.0,
        f0g1_frequency=_estimate_f0g1_frequency(exp, target, options),
        f0g1_amplitude=0.0,
        ef_frequency=exp.qubits[target].control_frequency_ef,
        ef_amplitude=0.0,
    )
    # Keep the baseline for diagnosis; acceptance uses uncorrected populations.
    scans = {"baseline": await _measure_populations(exp, target, (baseline,), options)}

    scans["f0g1"] = await scan_f0g1_drive(exp, target, options=options)
    drive_scan = await scan_f0g1_ef_drive(exp, target, scans["f0g1"], options=options)
    scans["ef"] = drive_scan
    if not drive_scan.parameters:
        return F0G1CalibrationResult(
            success=False,
            message="No two-tone point fits the shared amplitude budget",
            parameters=None,
            pulse=None,
            scans=scans,
            options=options,
        )
    worst_excitation = drive_scan.excitation.max(axis=1)
    best_index = int(worst_excitation.argmin())
    selected_drive = drive_scan.parameters[best_index]

    duration_scan = await scan_f0g1_duration(
        exp, target, selected_drive, options=options
    )
    scans["duration"] = duration_scan
    duration_index = _select_duration(
        duration_scan.excitation, options.target_excitation
    )
    if duration_index is None:
        return F0G1CalibrationResult(
            success=False,
            message="No measured duration meets the target",
            parameters=selected_drive,
            pulse=None,
            scans=scans,
            options=options,
        )
    selected_reset = duration_scan.parameters[duration_index]

    verification = await evaluate_f0g1_reset(
        exp, target, selected_reset, options=options
    )
    scans["verification"] = verification
    success = bool(verification.excitation.max() <= options.target_excitation)
    pulse = (
        build_f0g1_reset(exp, target, selected_reset, options=options)
        if success
        else None
    )
    return F0G1CalibrationResult(
        success=success,
        message="Verified" if success else "Independent verification failed",
        parameters=selected_reset,
        pulse=pulse,
        scans=scans,
        options=options,
    )
