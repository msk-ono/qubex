"""Measure F0G1 transfer and identify its coupling to a local CKP mode."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from plotly.colors import qualitative
from scipy.optimize import least_squares

from qubex.experiment import Experiment

from ._f0g1_acquisition import (
    F0G1AcquisitionOptions,
    F0G1PopulationScan,
    F0G1PulseParameters,
    _align_duration,
    _measure_populations,
    register_f0g1_targets,
)
from ._f0g1_physics import (
    fit_transfer,
    heldout_time_mask,
    mode_parameters,
    normalized_reset_rate,
    survey_transfer,
)


@dataclass(frozen=True)
class F0G1CKP:
    """
    Specify the lossy resonator/filter model in ordinary frequency units.

    All four fields are in MHz. `linewidth_mhz` is the filter's photon
    population decay rate divided by 2π. Pole indices follow ascending hybrid
    frequency. The input parameters and derived linewidths are treated as fixed.
    """

    resonator_mhz: float
    filter_mhz: float
    coupling_mhz: float
    linewidth_mhz: float

    def __post_init__(self) -> None:
        """Require positive frequencies, a coupled filter and finite loss."""
        if not np.all(np.isfinite(self.parameters)) or min(self.parameters) <= 0:
            raise ValueError(
                "CKP frequencies, coupling and linewidth must be finite and positive"
            )

    @property
    def parameters(self) -> tuple[float, float, float, float]:
        """Return the resonator, filter, exchange and filter linewidth in MHz."""
        return (
            self.resonator_mhz,
            self.filter_mhz,
            self.coupling_mhz,
            self.linewidth_mhz,
        )

    @property
    def modes(self) -> tuple[NDArray, NDArray]:
        """Return hybrid frequencies and their photon population linewidths in MHz."""
        return mode_parameters(self.parameters)


@dataclass(frozen=True)
class F0G1PhysicsOptions:
    """
    Configure one broadband transfer survey and local time sweeps per amplitude.

    `amplitudes` are normalized F0G1 commands, including 1 by default.
    The ef reset tone stays off. Frequencies are sampled at up to half the
    narrowest CKP linewidth, limited by `max_frequency_points` per initial
    survey. An active edge can extend the survey once. The result retains the
    actual spacing so an unresolved narrow line is visible in the diagnostics.
    `survey_duration_ns` sets the fixed-duration search, while `durations_ns`
    include the ramps and zero-drive reference for the time-domain fits.
    `acquisition.n_shots` is the shot count for fine peak confirmation and
    time-domain measurements; `survey_shots` is the count for coarse surveys.
    """

    amplitudes: tuple[float, ...] = (0.35, 0.65, 0.85, 1.0)
    survey_duration_ns: float = 1000
    durations_ns: tuple[float, ...] = (0.0, *np.geomspace(32.0, 3600.0, 20))
    survey_shots: int = 64
    max_frequency_points: int = 129
    acquisition: F0G1AcquisitionOptions = field(
        default_factory=lambda: F0G1AcquisitionOptions(n_shots=192, wait_duration=100)
    )

    def __post_init__(self) -> None:
        """Reject invalid sweeps before instrument registration."""
        a = np.asarray(self.amplitudes)
        if (
            a.ndim != 1
            or not len(a)
            or not np.isfinite(a).all()
            or np.any((a <= 0) | (a > 1))
            or np.any(np.diff(a) <= 0)
        ):
            raise ValueError("amplitudes must increase strictly in (0, 1]")
        times = np.asarray(self.durations_ns)
        minimum = 2 * self.acquisition.ramp_duration
        if (
            times.ndim != 1
            or len(times) < 8
            or times[0] != 0
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or times[1] < minimum
        ):
            raise ValueError(
                "durations_ns must begin at zero and contain at least seven increasing two-ramp durations"
            )
        if (
            not np.isfinite(self.survey_duration_ns)
            or self.survey_duration_ns < minimum
        ):
            raise ValueError("survey_duration_ns must fit two ramps")
        if type(self.survey_shots) is not int or self.survey_shots <= 0:
            raise ValueError("survey_shots must be a positive integer")
        if type(self.max_frequency_points) is not int or self.max_frequency_points < 5:
            raise ValueError("max_frequency_points must be at least five")


@dataclass(frozen=True)
class F0G1FitWindow:
    """Two-pole fit from one CKP seed, including predictions at excluded times."""

    seed_shift_mhz: float
    shift_mhz: float
    bare_coupling_mhz: float
    train_rmse: float
    heldout_rmse: float
    valid: bool
    prediction: NDArray


@dataclass(frozen=True)
class F0G1CouplingPoint:
    """
    Store a local time-domain measurement and its conditional mode assignment.

    The middle drive frequency is the measured spectral peak. `mode` is the
    zero-based CKP pole selected by the two-pole fit. `mode_coupling_mhz` and
    `g_over_kappa` require a second, identifiable single-pole fit to the same
    data. The local uncertainty in the ratio holds the CKP linewidth fixed.
    """

    amplitude: float
    frequencies_mhz: NDArray
    durations_ns: NDArray
    assigned_f: NDArray
    windows: tuple[F0G1FitWindow, ...]
    selected_window: int
    status: str
    mode: int | None = None
    mode_coupling_mhz: float | None = None
    g_over_kappa: float | None = None
    g_over_kappa_error: float | None = None
    single_detuning_mhz: float | None = None
    single_heldout_rmse: float | None = None

    @property
    def frequency_mhz(self) -> float:
        """Return the drive frequency of the confirmed transfer peak."""
        return float(self.frequencies_mhz[len(self.frequencies_mhz) // 2])

    @property
    def heldout_rmse(self) -> float:
        """Return held-out population RMSE of the training-selected two-pole fit."""
        return self.windows[self.selected_window].heldout_rmse

    @property
    def bare_coupling_mhz(self) -> float | None:
        """Return the identifiable coupling to the bare resonator in MHz."""
        return (
            self.windows[self.selected_window].bare_coupling_mhz
            if self.mode is not None
            else None
        )

    def plot(self) -> go.Figure:
        """Overlay two-pole predictions, training assignments and held-out times."""
        fig = go.Figure()
        held = heldout_time_mask(self.durations_ns)
        traces = zip(
            self.frequencies_mhz,
            self.assigned_f,
            self.windows[self.selected_window].prediction,
            strict=True,
        )
        for index, (freq, values, prediction) in enumerate(traces):
            color = qualitative.Plotly[index % len(qualitative.Plotly)]
            label = f"{freq:.2f} MHz"
            fig.add_scatter(
                x=self.durations_ns,
                y=prediction,
                mode="lines",
                name=label,
                legendgroup=label,
                line_color=color,
            )
            fig.add_scatter(
                x=self.durations_ns,
                y=values,
                mode="markers",
                marker=dict(
                    symbol=np.where(held, "circle-open", "circle"), color=color
                ),
                name=label,
                legendgroup=label,
                showlegend=False,
            )
        fig.update_layout(
            xaxis_title="Duration (ns) · open circles are held-out times",
            yaxis_title="Assigned f",
            title=f"A = {self.amplitude:g} · {self.status} · held-out RMSE {self.heldout_rmse:.3f}",
        )
        return fig


def analyze_f0g1_transfer(
    frequencies_mhz: ArrayLike,
    durations_ns: ArrayLike,
    assigned_f: ArrayLike,
    *,
    amplitude: float,
    transition_sum_mhz: float,
    ckp: F0G1CKP,
    off_decay_per_ns: float,
    ramp_duration_ns: float = 16,
) -> F0G1CouplingPoint:
    """
    Identify an effective coupling from f-survival traces at detuned drives.

    Frequencies increase around the central candidate. Assigned populations
    have shape (frequency, duration), with distinct increasing pulse durations
    in ns. The effective interaction time is max(duration − ramp, 0).
    `transition_sum_mhz` is the sum of the undriven ge and ef frequencies.

    Both shift windows use the same lossy two-pole Hamiltonian and OFF decay.
    The lower training error selects the window. For distinct minima, require
    the mean paired improvement in held-out squared error to exceed one
    standard error over times, averaging over drives at each time. This
    small-sample diagnostic is a model-selection heuristic. Only a single
    pole whose half-linewidth contains the inferred photon frequency can
    supply a linewidth for normalization. The single-pole refit must support
    the same nearest pole and agree with the two-pole center within that pole's
    half-linewidth; the two models have independent coupling and shift fits.
    """
    f, t, y = (
        np.asarray(a, dtype=float) for a in (frequencies_mhz, durations_ns, assigned_f)
    )
    if (
        f.ndim != 1
        or len(f) < 3
        or len(f) % 2 == 0
        or not np.isfinite(f).all()
        or np.any(f <= 0)
        or np.any(np.diff(f) <= 0)
    ):
        raise ValueError(
            "Provide an odd number of at least three increasing positive frequencies"
        )
    if (
        t.ndim != 1
        or len(t) < 8
        or not np.isfinite(t).all()
        or t[0] < 0
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError(
            "Provide at least eight distinct increasing nonnegative pulse durations"
        )
    if (
        y.shape != (len(f), len(t))
        or not np.isfinite(y).all()
        or np.any((y < 0) | (y > 1))
    ):
        raise ValueError(
            "assigned_f must be a frequency-by-duration array of probabilities"
        )
    if not (
        0 < amplitude <= 1
        and np.isfinite(transition_sum_mhz)
        and transition_sum_mhz > 0
        and np.isfinite(off_decay_per_ns)
        and off_decay_per_ns >= 0
        and np.isfinite(ramp_duration_ns)
        and ramp_duration_ns >= 0
    ):
        raise ValueError("Invalid amplitude, transition sum, OFF decay or ramp")
    te = np.maximum(0, t - ramp_duration_ns)
    held = heldout_time_mask(t)
    if np.unique(te[~held]).size < 4 or np.unique(te[held]).size < 2:
        raise ValueError(
            "Provide distinct effective interaction times for training and validation"
        )
    poles, widths = ckp.modes
    center = f[len(f) // 2]
    windows = []
    for pole in poles:
        guess = center - transition_sum_mhz + pole
        fit = fit_transfer(
            f, te, y, transition_sum_mhz, ckp.parameters, guess, off_decay_per_ns
        )
        pred = fit["prediction"]
        windows.append(
            F0G1FitWindow(
                guess,
                fit["shift"],
                fit["g"],
                float(np.sqrt(np.mean((pred[:, ~held] - y[:, ~held]) ** 2))),
                fit["heldout_rmse"],
                fit["valid"],
                pred,
            )
        )
    best = min(range(len(windows)), key=lambda idx: windows[idx].train_rmse)
    winner, rival = windows[best], windows[1 - best]
    same = np.allclose(
        [winner.bare_coupling_mhz, winner.shift_mhz],
        [rival.bare_coupling_mhz, rival.shift_mhz],
        rtol=1e-5,
        atol=1e-3,
    ) and np.allclose(winner.prediction, rival.prediction, atol=1e-4)
    if same and not winner.valid and rival.valid:
        best = 1 - best
        winner, rival = rival, winner
    point = F0G1CouplingPoint(
        amplitude, f, t, y, tuple(windows), best, "two-pole fit unresolved"
    )
    if not winner.valid:
        return point
    errors = (rival.prediction[:, held] - y[:, held]) ** 2 - (
        winner.prediction[:, held] - y[:, held]
    ) ** 2
    paired = errors.mean(axis=0)
    if not same and paired.mean() <= paired.std(ddof=1) / np.sqrt(paired.size):
        return replace(point, status="shift windows indistinguishable")
    inferred = transition_sum_mhz + winner.shift_mhz - center
    near = np.flatnonzero((widths > 0) & (abs(poles - inferred) <= widths / 2))
    if len(near) != 1:
        return replace(point, status="no unique pole within half-linewidth")
    mode = int(near[0])
    single = fit_transfer(
        f,
        te,
        y,
        transition_sum_mhz,
        ckp.parameters,
        winner.shift_mhz,
        off_decay_per_ns,
        mode=mode,
    )
    single_inferred = transition_sum_mhz + single["shift"] - center
    point = replace(
        point,
        mode=mode,
        single_detuning_mhz=single_inferred - poles[mode],
        single_heldout_rmse=single["heldout_rmse"],
    )
    if not single["valid"]:
        return replace(point, status="single-pole fit unresolved")
    if (
        int(np.argmin(abs(poles - single_inferred))) != mode
        or abs(single_inferred - inferred) > widths[mode] / 2
    ):
        return replace(point, status="single- and two-pole centers disagree")
    return replace(
        point,
        status="identified",
        mode_coupling_mhz=single["g"],
        g_over_kappa=single["g"] / widths[mode],
        g_over_kappa_error=single["g_se"] / widths[mode],
    )


@dataclass(frozen=True)
class F0G1TransferSpectrum:
    """Coarse f-to-g transfer and independently confirmed peak carriers at one amplitude."""

    amplitude: float
    frequencies_mhz: NDArray
    assigned_g: NDArray
    confirmed_frequencies_mhz: tuple[float, ...]
    active_edges: tuple[bool, bool]
    undersampled_modes: tuple[int, ...]


@dataclass(frozen=True)
class F0G1PhysicsResult:
    """
    Retain the spectroscopy, local fits and acquisition records for one qubit.

    `off_decay_per_ns` is the population decay rate from the drive-off
    reference, with the long-time f assignment anchored by g preparation.
    It is absent when the reference contrast or residual is inadequate.
    `at_full_amplitude` returns all traced peaks at exactly A=1, including
    those for which no linewidth or coupling could be assigned.
    """

    target: str
    ckp: F0G1CKP
    off_decay_per_ns: float | None
    spectra: tuple[F0G1TransferSpectrum, ...]
    points: tuple[F0G1CouplingPoint, ...]
    scans: tuple[F0G1PopulationScan, ...]

    @property
    def at_full_amplitude(self) -> tuple[F0G1CouplingPoint, ...]:
        """Return the observed A=1 transfer points, retaining indeterminate fits."""
        return tuple(point for point in self.points if point.amplitude == 1)

    @property
    def full_amplitude_status(self) -> str:
        """Explain the observation available at A=1 without extrapolation."""
        if not any(s.amplitude == 1 for s in self.spectra):
            return "not surveyed"
        if self.off_decay_per_ns is None:
            return "unresolved OFF decay"
        if not self.at_full_amplitude:
            return "no confirmed peak"
        if any(p.g_over_kappa is not None for p in self.at_full_amplitude):
            return "identified"
        return "coupling unresolved"

    def plot(self) -> go.Figure:
        """Plot measured g/κ with local fit errors for each identifiable CKP pole."""
        figure = go.Figure()
        for mode in range(2):
            points = [
                p for p in self.points if p.mode == mode and p.g_over_kappa is not None
            ]
            if points:
                figure.add_scatter(
                    x=[p.amplitude for p in points],
                    y=[p.g_over_kappa for p in points],
                    customdata=[p.frequency_mhz for p in points],
                    hovertemplate="A=%{x:g}, g/κ=%{y:.3g}, drive=%{customdata:.2f} MHz",
                    error_y=dict(
                        type="data", array=[p.g_over_kappa_error for p in points]
                    ),
                    mode="markers",
                    name=f"CKP pole {mode + 1}",
                )
        figure.update_layout(
            xaxis_title="F0G1 amplitude",
            yaxis_title="Effective g / κ",
            title=f"{self.target} · full amplitude: {self.full_amplitude_status}",
        )
        return figure

    def plot_reset_map(self) -> go.Figure:
        """
        Place measured full-amplitude coupling on the ideal resonant reset map.

        The color is the asymptotic population decay rate Γ/κ of the resonant
        single-mode model. Ω is the ef Hamiltonian matrix element, equal to half
        the resonant population Rabi angular frequency. Dashed vertical lines
        mark identified A=1 values of g/κ. Their heights do not indicate an ef
        measurement: the shared-port amplitude budget requires Ω=0 at A=1.
        """
        identified = [p for p in self.at_full_amplitude if p.g_over_kappa is not None]
        xmax = max(
            [
                1.0,
                *[
                    1.1 * p.g_over_kappa
                    for p in identified
                    if p.g_over_kappa is not None
                ],
            ]
        )
        x = xmax * np.linspace(0, 1, 151) ** 1.5
        y = max(1.0, xmax) * np.linspace(0, 1, 151) ** 1.5
        figure = go.Figure(
            go.Heatmap(
                x=x,
                y=y,
                z=normalized_reset_rate(x[None, :], y[:, None]),
                zmin=0,
                zmax=1 / 3,
                colorscale="Viridis",
                colorbar=dict(title="Γ / κ"),
            )
        )
        for point in identified:
            figure.add_scatter(
                x=[point.g_over_kappa] * 2,
                y=[0, y[-1]],
                mode="lines",
                line=dict(color="white", dash="dash"),
                name=f"A=1 · {point.frequency_mhz:.2f} MHz · g/κ={point.g_over_kappa:.3g}",
                hoverinfo="name",
            )
        figure.update_layout(
            title=f"{self.target} · resonant single-mode model · full amplitude: {self.full_amplitude_status}",
            xaxis_title="F0G1 coupling g / κ",
            yaxis_title="EF coupling Ω / κ",
            legend=dict(orientation="h", y=-0.25),
        )
        return figure

    def plot_spectra(self) -> go.Figure:
        """Show measured transfer versus frequency at every surveyed amplitude."""
        figure = go.Figure()
        for spectrum in self.spectra:
            figure.add_scatter(
                x=spectrum.frequencies_mhz,
                y=spectrum.assigned_g,
                mode="lines+markers",
                name=f"A={spectrum.amplitude:g}",
            )
        figure.update_layout(
            xaxis_title="F0G1 drive (MHz)",
            yaxis_title="Assigned g from f preparation",
            title=f"{self.target} · transfer spectroscopy",
        )
        return figure


def _off_decay(scan: F0G1PopulationScan) -> float | None:
    """Fit the f-prepared survival with an independently measured g-prepared floor."""
    times = np.array([p.duration for p in scan.parameters])
    floor = float(scan.populations[:, scan.initial_states.index("g"), 2].mean())
    pf = scan.populations[:, scan.initial_states.index("f"), 2]
    contrast = pf[0] - floor
    if contrast <= 0.14:
        return None
    fit = least_squares(
        lambda x: floor + x[0] * np.exp(-x[1] * times) - pf,
        [contrast, 0.0001],
        bounds=([0, 0], [1, 0.05]),
        x_scale="jac",
    )
    if not fit.success or fit.x[0] <= 0.14 or np.sqrt(np.mean(fit.fun**2)) >= 0.085:
        return None
    return float(fit.x[1])


async def characterize_f0g1(
    exp: Experiment,
    target: str,
    *,
    ckp: F0G1CKP,
    options: F0G1PhysicsOptions | None = None,
) -> F0G1PhysicsResult:
    """
    Measure drive-induced couplings and the accessible g/κ at full amplitude.

    Parameters
    ----------
    exp : Experiment
        Configured QuEL-3 experiment with calibrated ge/ef π pulses and a
        three-state classifier. Compatible reset aliases are registered on demand.
    target : str
        Qubit label.
    ckp : F0G1CKP
        Independently calibrated bare resonator and Purcell-filter parameters.
    options : F0G1PhysicsOptions, optional
        Amplitudes, pulse times, resolution budget and acquisition settings.

    Returns
    -------
    F0G1PhysicsResult
        Measured OFF reference, spectra, peak-specific fits and raw scans.
        Each identified ratio applies at its measured amplitude and carrier.

    Notes
    -----
    The survey covers the CKP poles and both flanks, extending an active edge
    once and remeasuring up to two peaks at higher resolution. Lines narrower
    than the coarse grid or weak at the survey duration may be missed.
    Ratios at A=1 characterize transfer with ef off. On a shared port that
    amplitude leaves no headroom for a simultaneous ef reset tone.
    """
    config = options or F0G1PhysicsOptions()
    acquisition = config.acquisition
    qubit = exp.qubits[target]
    fef = qubit.control_frequency_ef
    total = (qubit.control_frequency_ge + fef) * 1000
    poles, widths = ckp.modes
    nominal = total - poles
    # Numerically coincident widths give a single detuning scale.
    narrow, broad = np.maximum(np.sort(widths / 2), 1e-6)
    scales = [broad] if np.isclose(narrow, broad) else [narrow, broad]
    offsets = np.r_[-np.asarray(scales)[::-1], 0.0, scales]
    if np.min(nominal) <= 0 or not np.isfinite(total):
        raise ValueError("CKP poles imply a nonpositive or nonfinite F0G1 carrier")
    times = np.unique([_align_duration(exp, t) for t in config.durations_ns])
    if len(times) < 8:
        raise ValueError(
            "The hardware sample grid must support at least eight time points"
        )
    duration = _align_duration(exp, config.survey_duration_ns)
    off_times = np.unique(
        [
            0,
            duration,
            *[
                _align_duration(exp, t)
                for t in np.geomspace(
                    max(32, 2 * acquisition.ramp_duration),
                    max(16000, times[-1] * 4),
                    13,
                )
            ],
        ]
    )
    register_f0g1_targets(exp, target, options=acquisition)
    scans = []

    async def measure(
        amplitude: float,
        frequencies: Sequence[float] | NDArray,
        durations: Sequence[float] | NDArray,
        shots: int,
        *,
        reference: bool = False,
    ) -> F0G1PopulationScan:
        parameters = tuple(
            F0G1PulseParameters(float(t), float(f) / 1000, amplitude, fef, 0)
            for f in frequencies
            for t in durations
        )
        scan = await _measure_populations(
            exp,
            target,
            parameters,
            replace(acquisition, n_shots=shots),
            ("g", "f") if reference else ("f",),
        )
        scans.append(scan)
        return scan

    off = await measure(0, nominal[:1], off_times, acquisition.n_shots, reference=True)
    gamma = _off_decay(off)
    off_ground = float(off.populations[np.flatnonzero(off_times == duration)[0], 1, 0])
    points = []
    spectra = []
    for amplitude in config.amplitudes:

        async def survey(
            frequencies: NDArray, shots: int, amplitude: float = amplitude
        ) -> NDArray:
            scan = await measure(amplitude, frequencies, [duration], shots)
            return scan.populations[:, 0, 0]

        found = await survey_transfer(
            survey,
            nominal,
            widths,
            (),
            off_ground,
            config.survey_shots,
            acquisition.n_shots,
            frequency_step_mhz=float(np.clip(widths.min() / 2, 0.1, 5)),
            max_frequency_points=config.max_frequency_points,
        )
        frequencies = found["frequencies"]
        confirmed = tuple(float(peak["frequency"]) for peak in found["confirmed"])
        step = np.max(np.diff(frequencies))
        spectra.append(
            F0G1TransferSpectrum(
                amplitude,
                frequencies,
                found["ground"],
                confirmed,
                found["edges"],
                tuple(int(i) for i in np.flatnonzero(widths < 2 * step - 1e-8)),
            )
        )
        if gamma is None:
            continue
        # Measure both linewidth scales before assigning a physical pole.
        for center in confirmed:
            drives = center + offsets
            trace = await measure(amplitude, drives, times, acquisition.n_shots)
            assigned_f = trace.populations[:, 0, 2].reshape(len(drives), len(times))
            points.append(
                analyze_f0g1_transfer(
                    drives,
                    times,
                    assigned_f,
                    amplitude=amplitude,
                    transition_sum_mhz=total,
                    ckp=ckp,
                    off_decay_per_ns=gamma,
                    ramp_duration_ns=acquisition.ramp_duration,
                )
            )

    return F0G1PhysicsResult(
        target, ckp, gamma, tuple(spectra), tuple(points), tuple(scans)
    )
