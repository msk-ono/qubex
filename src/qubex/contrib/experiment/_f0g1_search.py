"""Empirical two-tone reset search using preparation-resolved measurements."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import betaincinv

from qubex.experiment import Experiment
from qubex.pulse import PulseSchedule

from ._f0g1_acquisition import (
    F0G1AcquisitionOptions,
    F0G1PopulationScan,
    F0G1PulseParameters,
    State,
    _align_duration,
    _estimate_f0g1_frequency,
    _measure_populations,
    build_f0g1_reset,
    register_f0g1_targets,
)


@dataclass(frozen=True)
class F0G1SearchOptions(F0G1AcquisitionOptions):
    """
    Set the measurement budget and acceptance criterion for empirical reset.

    Parameters
    ----------
    amplitudes, ef_amplitudes : tuple[float, ...]
        Increasing normalized drive amplitudes. Only pairs sharing at most one
        unit of port amplitude are measured. A F0G1 amplitude of one leaves no
        room for an ef tone and is skipped in this workflow.
    frequency_offsets, ef_frequency_offsets : tuple[float, ...]
        Increasing coarse grid offsets in GHz around the energy-balance seed
        and the calibrated ef frequency, respectively. Include both signs.
    max_extensions : int
        Maximum half-window extensions on either side in each survey.
    peaks_per_amplitude, finalists : int
        Maximum number of bracketed F0G1 peaks per amplitude and two-tone
        candidates carried into the duration comparison.
    probe_amplitude, probe_durations : float, tuple[float, ...]
        Weak ef probe and total pulse lengths in ns used for Stark spectroscopy.
        If headroom limits the probe, acquire OFF at that same reduced amplitude.
    search_duration, durations : float, tuple[float, ...]
        Initial two-tone screening duration and candidate reset lengths in ns.
        The final comparison checks 80%, 90% and 100% of each length.
    target_excitation, confidence : float
        Target for each preparation and duration. Independent ON counts must
        pass a pointwise, one-sided exact binomial upper bound at this confidence.
        Compare the worst bound against the largest idle lower bound from e/f
        controls at matching times. Idle bounds use a Bonferroni adjustment
        across the measured e/f states and times.
    wait_duration : float
        Delay after the tones in ns, including for matched idle controls;
        default 100 ns.
    """

    wait_duration: float = 100.0
    n_shots: int = 96
    verification_shots: int = 1024
    amplitudes: tuple[float, ...] = (0.4, 0.7, 0.9, 0.95, 0.98)
    ef_amplitudes: tuple[float, ...] = (0.003, 0.008, 0.02, 0.05)
    frequency_offsets: tuple[float, ...] = tuple(np.linspace(-0.1, 0.1, 51))
    ef_frequency_offsets: tuple[float, ...] = tuple(np.linspace(-0.064, 0.064, 33))
    max_extensions: int = 1
    peaks_per_amplitude: int = 2
    finalists: int = 4
    probe_amplitude: float = 0.02
    probe_durations: tuple[float, ...] = (32, 48, 72, 96, 122, 184, 260)
    search_duration: float = 1000
    durations: tuple[float, ...] = (250, 500, 1000, 2000, 4000, 8000)
    target_excitation: float = 0.01
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Validate grids before touching instruments."""
        super().__post_init__()
        for name in (
            "amplitudes",
            "ef_amplitudes",
            "frequency_offsets",
            "ef_frequency_offsets",
            "probe_durations",
            "durations",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if (
                values.ndim != 1
                or not len(values)
                or not np.isfinite(values).all()
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError(f"{name} must be a finite, increasing grid")
        for offsets in (self.frequency_offsets, self.ef_frequency_offsets):
            if len(offsets) < 3 or not offsets[0] < 0 < offsets[-1]:
                raise ValueError("Frequency offsets must span both sides of the seed")
        if not all(
            0 < value <= 1
            for value in (*self.amplitudes, *self.ef_amplitudes, self.probe_amplitude)
        ):
            raise ValueError("Drive amplitudes must be in (0, 1]")
        if self.amplitudes[0] == 1:
            raise ValueError("At least one F0G1 amplitude must leave headroom for ef")
        if len(self.probe_durations) < 6:
            raise ValueError(
                "At least six probe durations are needed for split-time validation"
            )
        if (
            not np.isfinite(self.search_duration)
            or min(
                self.probe_durations[0],
                0.8 * self.durations[0],
                self.search_duration / 2,
            )
            < 2 * self.ramp_duration
        ):
            raise ValueError("Probe and reset windows must accommodate both ramps")
        if not 0 < self.target_excitation < 1 or not 0.5 < self.confidence < 1:
            raise ValueError(
                "Require 0 < target_excitation < 1 and 0.5 < confidence < 1"
            )
        for name, lower in (
            ("max_extensions", 0),
            ("peaks_per_amplitude", 1),
            ("finalists", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")


@dataclass(frozen=True)
class F0G1TransferPeak:
    """
    Describe a bracketed F0G1-only transfer peak at a measured amplitude.

    Frequencies and optional displacements are in GHz. A displacement compares
    against a measured origin along a locally isolated sequence of peaks.
    No line can be measured at zero F0G1 amplitude.
    """

    amplitude: float
    frequency: float
    ground: float
    reference_amplitude: float | None = None
    reference_frequency: float | None = None

    @property
    def shift(self) -> float | None:
        """Return the measured displacement in GHz relative to the track origin."""
        return (
            None
            if self.reference_frequency is None
            else self.frequency - self.reference_frequency
        )


@dataclass(frozen=True)
class F0G1ProbeLine:
    """Describe one ef probe spectrum in GHz, with unresolved centers left empty."""

    frequency: float | None
    rabi_frequency: float | None
    heldout_rmse: float
    status: str
    frequencies_ghz: NDArray = field(repr=False, compare=False)
    durations_ns: NDArray = field(repr=False, compare=False)
    assigned_f: NDArray = field(repr=False, compare=False)


@dataclass(frozen=True)
class F0G1StarkEstimate:
    """Compare matched weak-probe ef measurements with F0G1 OFF and ON."""

    f0g1_amplitude: float
    f0g1_frequency: float
    probe_amplitude: float
    off: F0G1ProbeLine
    on: F0G1ProbeLine

    @property
    def shift(self) -> float | None:
        """Return the driven ef displacement in GHz when both centers resolve."""
        if self.off.frequency is None or self.on.frequency is None:
            return None
        return self.on.frequency - self.off.frequency


@dataclass(frozen=True)
class F0G1ResetSearchResult:
    """
    Retain the measured drive candidates, Stark diagnostics and verification.

    `parameters` identifies the best measured candidate even if it misses the
    target. `pulse` is populated only when fresh ON data pass the requested
    confidence bound and improve on matched OFF excitation. The `verification`
    and `verification_off` scans cover the same window with g/e/f preparation.
    All scans retain their effective coordinates, counts and raw measurements.
    """

    success: bool
    message: str
    parameters: F0G1PulseParameters | None
    pulse: PulseSchedule | None
    drive_peaks: tuple[F0G1TransferPeak, ...]
    ef_shifts: tuple[F0G1StarkEstimate, ...]
    scans: dict[str, F0G1PopulationScan]
    options: F0G1SearchOptions


def _rabi_response(detuning: NDArray, rate: float | NDArray, times: NDArray) -> NDArray:
    """Compute the driven e-to-f transition probability in GHz and ns."""
    generalized = np.hypot(detuning, rate)
    amplitude = (rate / generalized)[..., None]
    return (amplitude * np.sin(np.pi * generalized[..., None] * times)) ** 2


def _chevron_seeds(
    frequencies: NDArray, times: NDArray, observed: NDArray
) -> list[tuple[float, float, float, float, float]]:
    """Find separated resonance seeds while profiling out offset and contrast."""
    centers = np.sort(np.r_[frequencies, (frequencies[1:] + frequencies[:-1]) / 2])
    rates = np.linspace(0.0005, 0.1, max(60, min(300, int(0.8 * np.ptp(times)))))
    seeds = []
    for center in centers:
        templates = _rabi_response(frequencies[:, None] - center, rates, times)
        templates = templates.transpose(1, 0, 2).reshape(len(rates), -1)
        centered = templates - templates.mean(axis=1, keepdims=True)
        contrast = np.clip(
            np.sum(centered * (observed - observed.mean()), axis=1)
            / np.maximum(np.sum(centered**2, axis=1), 1e-12),
            0,
            1.3,
        )
        offset = np.clip(observed.mean() - contrast * templates.mean(axis=1), 0, 1)
        loss = np.mean(
            (offset[:, None] + contrast[:, None] * templates - observed) ** 2, axis=1
        )
        k = int(np.argmin(loss))
        seeds.append((loss[k], center, rates[k], contrast[k], offset[k]))
    minima, _ = find_peaks(-np.array([r[0] for r in seeds]))
    return sorted([seeds[i] for i in {0, len(seeds) - 1, *minima}])[:3]


def _probe_line(
    frequencies: NDArray, times: NDArray, data: NDArray, ramp: float, shots: int
) -> F0G1ProbeLine:
    """Estimate a single ef center from frequency and duration, reserving whole times."""
    held = np.arange(len(times)) % 4 == 2
    effective = np.maximum(0, times - ramp)
    observed = data[:, ~held].ravel()
    step = np.median(np.diff(frequencies))
    train_times = effective[~held]
    starts = _chevron_seeds(frequencies, train_times, observed)

    def residual(x: NDArray) -> NDArray:
        center, rate, contrast, offset = x
        return (
            offset
            + contrast * _rabi_response(frequencies - center, rate, train_times).ravel()
            - observed
        )

    bounds = ([frequencies[0], 0.0001, 0, 0], [frequencies[-1], 0.1, 1.3, 1])
    fits = sorted(
        (
            least_squares(
                residual, start[1:], bounds=bounds, x_scale="jac", max_nfev=120
            )
            for start in starts
        ),
        key=lambda fit: fit.cost,
    )
    center, rate, contrast, offset = fits[0].x
    prediction = offset + contrast * _rabi_response(
        frequencies - center, rate, effective[held]
    )
    rmse = float(np.sqrt(np.mean((prediction - data[:, held]) ** 2)))
    noise = float(np.sqrt(np.mean(data * (1 - data)) / max(shots - 1, 1)))
    null = np.sqrt(np.mean((observed.mean() - data[:, held]) ** 2))
    status = "resolved"
    if center < frequencies[0] + step / 2 or center > frequencies[-1] - step / 2:
        status = "edge"
    elif (
        not fits[0].success
        or fits[0].active_mask[1] != 0
        or rate * np.ptp(times) < 1
        or contrast < max(0.10, 3 * rmse)
        or rmse > max(0.04, 2 * noise)
        or rmse >= 0.9 * null
    ):
        status = "unresolved"
    else:
        for fit in fits[1:]:
            c, r, scale, floor = fit.x
            other = floor + scale * _rabi_response(frequencies - c, r, effective[held])
            other_rmse = np.sqrt(np.mean((other - data[:, held]) ** 2))
            if abs(c - center) > step and other_rmse <= rmse + noise / np.sqrt(
                prediction.size
            ):
                status = "ambiguous"
                break
    return F0G1ProbeLine(
        float(center) if status == "resolved" else None,
        float(rate) if status == "resolved" else None,
        rmse,
        status,
        frequencies,
        times,
        data,
    )


def _refined_peak_index(signal: NDArray) -> int | None:
    """Choose the most central equal maximum inside the independently measured bracket."""
    maxima = np.flatnonzero(signal == signal.max())
    index = int(maxima[np.argmin(abs(maxima - (len(signal) - 1) / 2))])
    return index if 0 < index < len(signal) - 1 else None


def _connect_peaks(
    previous: Sequence[F0G1TransferPeak],
    current: Sequence[F0G1TransferPeak],
    tolerance: float,
) -> list[F0G1TransferPeak]:
    """Connect only isolated one-to-one neighbours at adjacent measured amplitudes."""
    distances = np.array(
        [[abs(old.frequency - new.frequency) for old in previous] for new in current]
    )
    connected = []
    for peak, row in zip(current, distances, strict=True):
        close = np.flatnonzero(row <= tolerance)
        if (
            len(close) == 1
            and np.count_nonzero(distances[:, close[0]] <= tolerance) == 1
        ):
            old = previous[close[0]]
            peak = replace(
                peak,
                reference_frequency=old.reference_frequency or old.frequency,
                reference_amplitude=old.reference_amplitude or old.amplitude,
            )
        connected.append(peak)
    return connected


async def _spectrum(
    center: float,
    offsets: Sequence[float],
    extensions: int,
    noise: float,
    measure: Callable[[NDArray], Awaitable[NDArray]],
    summarize: Callable[[NDArray], NDArray],
) -> tuple[NDArray, NDArray]:
    """Expand active edges or a featureless initial window within a finite budget."""
    x = center + np.asarray(offsets)
    data = await measure(x)
    for _ in range(extensions):
        score = summarize(data)
        peaks, _ = find_peaks(score, prominence=noise)
        grow_left = score[0] > np.min(score) + noise
        grow_right = score[-1] > np.min(score) + noise
        if not len(peaks) and not (grow_left or grow_right):
            grow_left = grow_right = True
        if not (grow_left or grow_right):
            break
        half = len(offsets) // 2
        left = (
            x[0] - (np.arange(half, 0, -1) * (offsets[1] - offsets[0]))
            if grow_left
            else []
        )
        right = (
            x[-1] + (np.arange(1, half + 1) * (offsets[-1] - offsets[-2]))
            if grow_right
            else []
        )
        extra = np.r_[left, right]
        more = await measure(extra)
        order = np.argsort(np.r_[x, extra])
        x, data = np.r_[x, extra][order], np.concatenate([data, more])[order]
    return x, data


class _Search:
    """Keep acquisitions and derived quantities together for a single target."""

    def __init__(self, exp: Experiment, target: str, options: F0G1SearchOptions):
        self.exp, self.target, self.options = exp, target, options
        self.scans: dict[str, F0G1PopulationScan] = {}
        self.base = F0G1PulseParameters(
            options.search_duration,
            _estimate_f0g1_frequency(exp, target, options),
            0,
            exp.qubits[target].control_frequency_ef,
            0,
        )
        self.noise = max(0.04, 0.85 / np.sqrt(options.n_shots))
        self.probe_times = np.unique(
            [_align_duration(exp, t) for t in options.probe_durations]
        )
        if len(self.probe_times) < 6:
            raise ValueError(
                "Probe durations collapse to fewer than six hardware samples"
            )

    async def acquire(
        self,
        name: str,
        points: Sequence[F0G1PulseParameters],
        states: tuple[State, ...],
        *,
        verification: bool = False,
    ) -> F0G1PopulationScan:
        scan = await _measure_populations(
            self.exp,
            self.target,
            tuple(points),
            self.options,
            states,
            verification=verification,
        )
        self.scans[name] = scan
        return scan

    async def drive_peaks(self) -> tuple[F0G1TransferPeak, ...]:
        opt = self.options
        times = (self.base.duration / 2, self.base.duration)
        baseline = await self.acquire(
            "baseline", [replace(self.base, duration=t) for t in times], ("g", "e", "f")
        )
        background = np.mean(baseline.populations[:, 2, 0])
        step = float(np.median(np.diff(opt.frequency_offsets)))
        all_peaks: list[F0G1TransferPeak] = []
        previous: list[F0G1TransferPeak] = []
        for i, a in enumerate(opt.amplitudes):
            if a >= 1:
                continue

            async def measure(freqs: NDArray, a: float = a, i: int = i) -> NDArray:
                points = [
                    replace(
                        self.base, f0g1_frequency=float(f), f0g1_amplitude=a, duration=t
                    )
                    for f, t in product(freqs, times)
                ]
                scan = await self.acquire(
                    f"transfer_{i}_{len(self.scans)}", points, ("f",)
                )
                return scan.populations[:, 0, 0].reshape(len(freqs), len(times))

            x, y = await _spectrum(
                self.base.f0g1_frequency,
                opt.frequency_offsets,
                opt.max_extensions,
                self.noise,
                measure,
                lambda data: data.mean(axis=1),
            )
            score = y.mean(axis=1)
            candidates, _ = find_peaks(
                score, prominence=self.noise, height=background + self.noise, distance=2
            )
            top = sorted(candidates, key=lambda k: -score[k])[: opt.peaks_per_amplitude]
            current = []
            for k in top:
                local = np.linspace(x[k - 1], x[k + 1], 7)
                ground = (await measure(local)).mean(axis=1)
                index = _refined_peak_index(ground)
                if index is not None and ground[index] > background + self.noise:
                    current.append(
                        F0G1TransferPeak(a, float(local[index]), float(ground[index]))
                    )
            previous = _connect_peaks(previous, current, 3 * step)
            all_peaks.extend(previous)
        return tuple(all_peaks)

    async def probe(
        self, point: F0G1PulseParameters
    ) -> tuple[F0G1ProbeLine, list[float]]:
        times = self.probe_times

        async def measure(freqs: NDArray) -> NDArray:
            points = [
                replace(point, ef_frequency=float(f), duration=float(t))
                for f, t in product(freqs, times)
            ]
            scan = await self.acquire(f"ef_probe_{len(self.scans)}", points, ("e",))
            return scan.populations[:, 0, 2].reshape(len(freqs), len(times))

        x, data = await _spectrum(
            self.base.ef_frequency,
            self.options.ef_frequency_offsets,
            self.options.max_extensions,
            self.noise,
            measure,
            lambda data: np.ptp(data, axis=1),
        )
        line = _probe_line(
            x, times, data, self.options.ramp_duration, self.options.n_shots
        )
        contrast = np.ptp(data, axis=1)
        peaks, _ = find_peaks(contrast, prominence=self.noise)
        candidates = [
            float(x[k]) for k in sorted(peaks, key=lambda k: -contrast[k])[:2]
        ]
        if line.frequency is not None:
            candidates.insert(0, line.frequency)
        return line, candidates

    async def joint(
        self, peaks: Sequence[F0G1TransferPeak]
    ) -> tuple[list[F0G1StarkEstimate], F0G1PopulationScan]:
        opts = self.options
        references: dict[float, F0G1ProbeLine] = {}
        estimates = []
        points = []
        for peak in peaks:
            budget = 1 - peak.amplitude
            b = min(budget, opts.probe_amplitude)
            if b not in references:
                references[b], _ = await self.probe(replace(self.base, ef_amplitude=b))
            point = replace(
                self.base,
                f0g1_amplitude=peak.amplitude,
                f0g1_frequency=peak.frequency,
                ef_amplitude=b,
            )
            line, frequencies = await self.probe(point)
            estimates.append(
                F0G1StarkEstimate(
                    peak.amplitude, peak.frequency, b, references[b], line
                )
            )
            frequencies.append(references[b].frequency or self.base.ef_frequency)
            # Keep one command when a rounded amplitude coincides with headroom.
            requested = np.unique(np.minimum(budget, (*opts.ef_amplitudes, b)))
            amplitudes = requested[np.r_[True, np.diff(requested) > 1e-12]]
            for amp in amplitudes:
                centers = list(frequencies)
                if line.frequency is None:
                    # A weak, short probe may miss a line that resets over a microsecond.
                    centers.extend(line.frequencies_ghz)
                elif amp < 0.67 * b or amp > 1.5 * b:
                    # The ef tone itself can shift the best operating frequency.
                    centers.extend(
                        line.frequency + np.asarray(opts.ef_frequency_offsets)
                    )
                unique = sorted({round(float(fe), 10) for fe in centers})
                points.extend(
                    replace(point, ef_frequency=fe, ef_amplitude=amp) for fe in unique
                )
        return estimates, await self.acquire("joint", points, ("e", "f"))

    async def finalists(self, joint: F0G1PopulationScan) -> list[F0G1PulseParameters]:
        options = self.options
        # Refine a winner for each driving line before pruning on the all-state objective.
        best_by_line: dict[tuple[float, float], F0G1PulseParameters] = {}
        for k in np.argsort(joint.excitation.max(axis=1), kind="stable"):
            p = joint.parameters[k]
            best_by_line.setdefault((p.f0g1_amplitude, p.f0g1_frequency), p)
        coarse = list(best_by_line.values())[: options.finalists]
        points = []
        half_step = float(np.median(np.diff(options.ef_frequency_offsets))) / 2
        for point in coarse:
            amplitudes = sorted(
                {
                    p.ef_amplitude
                    for p in joint.parameters
                    if p.f0g1_amplitude == point.f0g1_amplitude
                }
            )
            i = amplitudes.index(point.ef_amplitude)
            lo = amplitudes[i - 1] if i else 0.0
            hi = (
                amplitudes[i + 1]
                if i + 1 < len(amplitudes)
                else min(1 - point.f0g1_amplitude, 2 * point.ef_amplitude)
            )
            for amp, delta in product(
                np.unique(
                    [
                        (lo + point.ef_amplitude) / 2,
                        point.ef_amplitude,
                        (hi + point.ef_amplitude) / 2,
                    ]
                ),
                (-half_step, 0, half_step),
            ):
                points.append(
                    replace(
                        point,
                        ef_amplitude=float(amp),
                        ef_frequency=point.ef_frequency + delta,
                    )
                )
        fine = await self.acquire("refinement", points, ("g", "e", "f"))
        rank = np.argsort(fine.excitation.max(axis=1), kind="stable")
        # Retain the best waveform from each carrier before filling remaining slots.
        keep: dict[tuple[float, float], F0G1PulseParameters] = {}
        for k in rank:
            p = fine.parameters[k]
            keep.setdefault((p.f0g1_amplitude, p.f0g1_frequency), p)
        return list(
            dict.fromkeys([*keep.values(), *(fine.parameters[i] for i in rank)])
        )[: options.finalists]


def _upper_excitation(
    excited: NDArray | float, ground: NDArray | float, confidence: float
) -> NDArray:
    """Return exact one-sided upper bounds for assigned excited fractions."""
    return np.where(ground == 0, 1, betaincinv(excited + 1, ground, confidence))


async def search_f0g1_reset(
    exp: Experiment, target: str, *, options: F0G1SearchOptions | None = None
) -> F0G1ResetSearchResult:
    """
    Find and independently verify a shared-port reset without CKP parameters.

    Requires a configured QuEL-3 experiment, calibrated ge/ef preparation pulses,
    an assigned g/e/f classifier and a readout frequency (or an explicit
    `reset_resonator_frequency`). Compatible reset aliases are registered
    automatically. Acquisition errors propagate.

    Bracket F0G1-only transfer maxima at each amplitude, then measure the ef
    spectrum both with and without each retained drive. Spectra with unresolved
    Stark shifts still contribute empirical two-tone candidates. Screen both
    excited preparations, refine the ef controls and compare time windows from
    all three preparations. Select the shortest window projected to meet the
    target at verification precision, or the best observed window.

    Parameters
    ----------
    exp : Experiment
        Connected and configured experiment with calibrated state preparation.
    target : str
        Qubit label.
    options : F0G1SearchOptions | None
        Acquisition settings, survey grids and verification target.

    Returns
    -------
    F0G1ResetSearchResult
        Acquired scans, matched Stark probes and the best candidate settings.
        A schedule is supplied only after independent ON/OFF verification.
        Without any confirmed F0G1 line, candidate settings are also absent.
    """
    opt = options or F0G1SearchOptions()
    run = _Search(exp, target, opt)
    register_f0g1_targets(exp, target, options=opt)
    peaks = await run.drive_peaks()
    if not peaks:
        return F0G1ResetSearchResult(
            False, "No confirmed F0G1 transfer peak", None, None, (), (), run.scans, opt
        )
    shifts, joint = await run.joint(peaks)
    finalists = await run.finalists(joint)
    windows = {
        tuple(sorted({_align_duration(exp, t * f) for f in (0.8, 0.9, 1.0)}))
        for t in opt.durations
    }
    unique_times = sorted({t for w in windows for t in w})
    points = [replace(p, duration=t) for p, t in product(finalists, unique_times)]
    duration = await run.acquire("duration", points, ("g", "e", "f"))
    excitations = dict(
        zip(duration.parameters, duration.excitation.max(axis=1), strict=True)
    )
    scored = []
    for p, window in product(finalists, sorted(windows)):
        worst = max(excitations[replace(p, duration=t)] for t in window)
        # Optimize time once feasible; otherwise minimize measured excitation.
        expected_count = np.ceil(worst * opt.verification_shots)
        qualifies = (
            _upper_excitation(
                expected_count, opt.verification_shots - expected_count, opt.confidence
            )
            <= opt.target_excitation
        )
        key = (0, window[-1], worst) if qualifies else (1, worst, window[-1])
        scored.append((key, p, window))
    _, selected, times = min(scored, key=lambda r: r[0])
    selected = replace(selected, duration=times[-1])
    fresh = await run.acquire(
        "verification",
        [replace(selected, duration=t) for t in times],
        ("g", "e", "f"),
        verification=True,
    )
    off = await run.acquire(
        "verification_off",
        [replace(run.base, duration=t) for t in times],
        ("g", "e", "f"),
        verification=True,
    )
    # Exact one-sided binomial bounds. Each tested state/time must pass.
    on_k = fresh.counts[..., 1:].sum(axis=-1)
    on_g = fresh.counts[..., 0]
    upper = _upper_excitation(on_k, on_g, opt.confidence)
    off_k = off.counts[..., 1:].sum(axis=-1)
    # The idle comparison selects the worst of two excited preparations over
    # the window. Simultaneous lower bounds account for that selection.
    tail = (1 - opt.confidence) / off_k[:, 1:].size
    lower = np.where(off_k == 0, 0, betaincinv(off_k, off.counts[..., 0] + 1, tail))
    success = False
    if upper.max() > opt.target_excitation:
        message = "Independent window exceeds target excitation"
    elif upper.max() >= lower[:, 1:].max():
        message = "Active reset improvement over idle is unresolved"
    else:
        message = "Verified"
        success = True
    pulse = build_f0g1_reset(exp, target, selected, options=opt) if success else None
    return F0G1ResetSearchResult(
        success, message, selected, pulse, peaks, tuple(shifts), run.scans, opt
    )
