"""Numerical analysis for the adaptive F0G1 notebook (MHz, ns, population rates)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from qubex.pulse import Blank, FlatTop, PulseSchedule, Waveform

RAD = 2 * np.pi / 1000  # MHz to rad/ns


def mode_parameters(ckp: Sequence[float]) -> tuple[NDArray, NDArray]:
    """Return hybrid pole frequencies and energy linewidths in ascending frequency."""
    fr, fp, j, kp = ckp
    poles = np.linalg.eigvals([[fr, j], [j, fp - 0.5j * kp]])
    poles = poles[np.argsort(poles.real)]
    return poles.real, -2 * poles.imag


def peaks(
    x: NDArray, y: NDArray, prominence: float
) -> tuple[NDArray, tuple[bool, bool]]:
    """Find resolved interior maxima and strong signals cut off at either edge."""
    y = np.asarray(y)
    indices, _ = find_peaks(y, prominence=prominence)
    background = np.quantile(y, 0.25)
    edge = y[[0, -1]]
    extend = (edge > background + prominence) & (edge >= y[[1, -2]] - prominence / 2)
    return np.asarray(x)[indices], (bool(extend[0]), bool(extend[1]))


async def survey_transfer(
    measure: Callable[[NDArray, int], Awaitable[NDArray]],
    nominal: Sequence[float] | NDArray,
    widths: Sequence[float] | NDArray,
    known: Sequence[float],
    off_ground: float,
    shots_search: int = 64,
    shots_fit: int = 192,
    *,
    frequency_step_mhz: float | None = None,
    max_frequency_points: int = 65,
) -> dict[str, Any]:
    """
    Discover disconnected transfer peaks at a fixed amplitude and duration.

    Frequencies and linewidths are in MHz. The callback acquires fresh ground
    assignment fractions at the requested frequencies and shot count. An initial
    coarse grid covers both CKP poles and their flanks. Extend an active edge
    once, then independently resolve up to two peaks away from known lines.
    Each fine window contains eleven local samples and up to two remeasured
    shoulders from the coarse grid. The initial window uses at most
    `max_frequency_points`; extending both edges at most doubles that budget.
    """
    reach = float(np.clip(max(2 * np.ptp(nominal), 3 * np.max(widths)), 80, 120))
    low, high = np.min(nominal) - reach, np.max(nominal) + reach
    spacing = (
        float(np.clip(0.65 * np.min(widths), 2, 5))
        if frequency_step_mhz is None
        else frequency_step_mhz
    )
    if not np.isfinite(spacing) or spacing <= 0 or max_frequency_points < 5:
        raise ValueError(
            "Survey resolution requires a positive spacing and at least five points"
        )
    x = np.linspace(
        low, high, min(max_frequency_points, int(np.ceil((high - low) / spacing)) + 1)
    )
    step = x[1] - x[0]
    y = await measure(x, shots_search)
    prominence = max(0.08, 1 / np.sqrt(shots_search))
    _, edges = peaks(x, y, prominence)
    if any(edges):
        offsets = step * np.arange(1, int(np.ceil(reach / step)) + 1)
        extra = np.r_[
            x[0] - offsets if edges[0] else [], x[-1] + offsets if edges[1] else []
        ]
        x = np.r_[x, extra]
        y = np.r_[y, await measure(extra, shots_search)]
        order = np.argsort(x)
        x, y = x[order], y[order]

    candidates, edges = peaks(x, y, prominence)
    known_f = np.asarray(known, dtype=float)
    remote = [
        f
        for f in candidates
        if np.all(abs(known_f - f) > 2 * step)
        and y[np.searchsorted(x, f)] > off_ground + prominence
    ]
    remote.sort(key=lambda f: y[np.searchsorted(x, f)], reverse=True)
    confirmed = []
    if remote:
        windows = []
        for center in remote[:2]:
            # A saturated transfer band can be flat across the fine grid.
            # Remeasure a coarse shoulder on either side to establish prominence.
            fine = center + step * np.linspace(-2, 2, 11)
            shoulder_level = y[np.searchsorted(x, center)] - 1.5 * prominence
            lower = x[(x < fine[0]) & (y <= shoulder_level)]
            upper = x[(x > fine[-1]) & (y <= shoulder_level)]
            windows.append(np.r_[lower[-1:], fine, upper[:1]])
        frequencies = np.unique(np.concatenate([known_f, *windows]))
        ground = await measure(frequencies, shots_fit)
        reference = max(
            ground[np.searchsorted(frequencies, known_f)], default=off_ground
        )
        margin = max(0.08, 2.5 * np.sqrt(0.5 / shots_fit))
        for window in windows:
            profile = ground[np.searchsorted(frequencies, window)]
            maxima, _ = find_peaks(profile, prominence=margin)
            if len(maxima):
                k = maxima[np.argmax(profile[maxima])]
                f, score = window[k], profile[k]
                if (
                    score > off_ground + margin
                    and score >= reference - margin
                    and np.all(abs(known_f - f) > 2 * step)
                    and all(abs(line["frequency"] - f) > step for line in confirmed)
                ):
                    confirmed.append(dict(frequency=float(f), ground=float(score)))
    return dict(frequencies=x, ground=y, confirmed=confirmed, edges=edges)


def chevron(
    f: NDArray,
    t: NDArray,
    center: float,
    rate: float,
    contrast: float,
    offset: float,
    delay: float,
    decay: float,
) -> NDArray:
    """Predict assigned f population on a frequency-by-time grid."""
    effective = np.hypot(np.asarray(f)[:, None] - center, rate)
    time = np.maximum(0, np.asarray(t) - delay)
    return (
        offset
        + contrast
        * np.exp(-decay * time)
        * (rate / effective * np.sin(0.5 * RAD * effective * time)) ** 2
    )


def fit_chevron(f: NDArray, t: NDArray, data: NDArray) -> dict[str, Any]:
    """Estimate a local oscillation and reject unresolved or poorly modeled maps."""
    f, t, data = np.asarray(f), np.asarray(t), np.asarray(data)
    row = int(np.argmax(np.std(data, axis=1)))
    trace = data[row] - data[row].mean()
    nyquist = 500 / np.max(np.diff(t))
    rates = np.linspace(0.25e3 / np.ptp(t), nyquist * 0.98, 300)
    power = abs(np.exp(1j * RAD * rates[:, None] * t) @ trace)
    maxima, _ = find_peaks(power)
    seeds = (
        rates[maxima[np.argsort(power[maxima])[-3:]]]
        if len(maxima)
        else [rates[np.argmax(power)]]
    )
    lower = [f[0], 0.01, 0, 0, 0, 0]
    upper = [f[-1], nyquist, 1.5, 1, max(32, t.min() + 1), 0.05]
    fits = [
        least_squares(
            lambda x: (chevron(f, t, *x) - data).ravel(),
            [
                f[row],
                seed,
                max(0.1, np.ptp(trace)),
                np.clip(data.min(), 0.001, 0.99),
                8,
                0.0005,
            ],
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=220,
        )
        for seed in seeds
    ]
    best = min(fits, key=lambda r: r.cost)
    center, rate, contrast, *_ = best.x
    rmse = np.sqrt(2 * best.cost / data.size)
    valid = (
        best.success
        and f[0] + np.min(np.diff(f)) / 2 < center < f[-1] - np.min(np.diff(f)) / 2
        and rate * np.ptp(t) / 1000 >= 1.4
        and rate < 0.9 * nyquist
        and contrast > max(0.12, 4 * rmse)
    )
    return dict(
        center=center,
        rate=rate,
        valid=bool(valid),
        rmse=rmse,
        prediction=chevron(f, t, *best.x),
    )


def _population(h: NDArray, t: NDArray, start: int) -> NDArray:
    """Propagate a constant Hamiltonian and return surviving basis populations."""
    eig, vectors = np.linalg.eig(h)
    basis = np.eye(len(h))[:, start]
    if np.linalg.cond(vectors) > 1e7:
        return np.array([abs(expm(-1j * h * time) @ basis) ** 2 for time in t])
    weights = np.linalg.solve(vectors, basis)
    return abs((vectors @ (weights[:, None] * np.exp(-1j * eig[:, None] * t))).T) ** 2


def transfer(
    f: NDArray,
    t: NDArray,
    fsum: float,
    ckp: Sequence[float],
    g: float,
    shift: float,
    gamma: float,
) -> NDArray:
    """Predict f survival with both photon modes and an OFF population decay rate."""
    fr, fp, j, kp = ckp
    values = []
    for freq in f:
        z = fsum + shift - freq
        h = RAD * np.array(
            [[-0.5j * gamma / RAD, g, 0], [g, fr - z, j], [0, j, fp - z - 0.5j * kp]]
        )
        values.append(_population(h, t, 0)[:, 0])
    return np.array(values)


def heldout_time_mask(times: NDArray) -> NDArray:
    """Withhold every fourth time sample across all drives and shift windows."""
    return np.arange(len(times)) % 4 == 2


def fit_transfer(
    f: NDArray,
    t: NDArray,
    pf: NDArray,
    fsum: float,
    ckp: Sequence[float],
    shift_guess: float,
    gamma: float,
    *,
    mode: int | None = None,
) -> dict[str, Any]:
    """
    Fit coupling in MHz using detuned traces and held-out times.

    The two-pole model estimates bare resonator coupling. Selecting a CKP
    pole by index instead fits the effective coupling to that pole alone,
    with its photon population linewidth fixed to the CKP value.
    """
    t, pf = np.asarray(t), np.asarray(pf)
    train = ~heldout_time_mask(t)
    lo = [0.001, shift_guess - 30, 0, 0]
    hi = [80, shift_guess + 30, 1.3, 0.9]
    poles, widths = mode_parameters(ckp)

    def predict(x: NDArray, times: NDArray) -> NDArray:
        g, shift, scale, offset = x
        if mode is None:
            survival = transfer(f, times, fsum, ckp, g, shift, gamma)
        else:
            survival = []
            for detuning in np.asarray(f) + poles[mode] - fsum - shift:
                h = np.array(
                    [
                        [-0.5j * gamma, RAD * g],
                        [RAD * g, RAD * (detuning - 0.5j * widths[mode])],
                    ]
                )
                survival.append(_population(h, times, 0)[:, 0])
            survival = np.array(survival)
        return offset + scale * survival

    guesses = [
        least_squares(
            lambda x: (predict(x, t[train]) - pf[:, train]).ravel(),
            [g, shift_guess, 0.65, 0.1],
            bounds=(lo, hi),
            x_scale="jac",
        )
        for g in (0.6, 3.0, 14.0)
    ]
    fit = min(guesses, key=lambda r: r.cost)
    g, shift, scale, offset = fit.x
    error = np.sqrt(np.mean((predict(fit.x, t[~train]) - pf[:, ~train]) ** 2))
    jac = fit.jac
    norm = np.linalg.norm(jac, axis=0)
    cond = np.linalg.cond(jac / np.maximum(norm, 1e-15))
    se = np.sqrt(
        max(
            0.0,
            np.linalg.pinv(jac.T @ jac)[0, 0] * 2 * fit.cost / max(1, jac.shape[0] - 4),
        )
    )
    valid = (
        fit.success
        and not fit.active_mask[:2].any()
        and cond < 1e4
        and se < 0.5 * g
        and np.ptp(pf) > 0.14
        and error < 0.085
        and scale > 0.12
    )
    return dict(
        g=g,
        shift=shift,
        valid=bool(valid),
        heldout_rmse=error,
        g_se=se,
        condition=cond,
        prediction=predict(fit.x, t),
        offset=offset,
    )


def normalized_reset_rate(g_over_kappa: NDArray, omega_over_kappa: NDArray) -> NDArray:
    """Return the resonant single-mode population decay rate in units of kappa."""
    g, omega = np.broadcast_arrays(g_over_kappa, omega_over_kappa)
    h = np.zeros((*g.shape, 3, 3), dtype=complex)
    h[..., 0, 1] = h[..., 1, 0] = omega
    h[..., 1, 2] = h[..., 2, 1] = g
    h[..., 2, 2] = -0.5j
    return np.maximum(0.0, -2 * np.linalg.eigvals(h).imag.max(axis=-1))


def reset_map_points(
    mode_fits: Sequence[dict],
    ef_lines: Sequence[dict],
    widths: Sequence[float] | NDArray,
    selected: dict | None,
) -> list[dict]:
    """Locate identifiable driven chevrons and the selected waveform on the single-mode map."""
    fits = {(r["j"], r["A"], r["fd"]): r for r in mode_fits if r["valid"]}

    def waveform(r: dict) -> tuple:
        return r["j"], r["A"], r["fd"], round(r["b"], 12), round(r["fe"], 6)

    measured = {waveform(r): r for r in ef_lines if r["valid"] and not r["edge"]}
    chosen = waveform(selected) if selected is not None else None
    lines = dict(measured)
    if selected is not None and "ideal_worst" in selected:
        lines.setdefault(waveform(selected), selected)

    points = []
    for key, line in lines.items():
        fit = fits.get(key[:3])
        rate = line["rate"]
        if fit is None or rate is None:
            continue
        width = widths[line["j"]]
        if (
            not np.isfinite([width, fit["g"], rate]).all()
            or min(width, fit["g"], rate) <= 0
        ):
            continue
        points.append(
            dict(
                j=line["j"],
                A=line["A"],
                fd=line["fd"],
                b=line["b"],
                fe=line["fe"],
                g_MHz=fit["g"],
                linewidth_MHz=width,
                rabi_MHz=rate,
                g_over_kappa=fit["g"] / width,
                omega_over_kappa=rate / (2 * width),
                source="chevron" if key in measured else "interpolated",
                selected=key == chosen,
            )
        )
    return points


def reset_prediction(
    ckp: Sequence[float],
    fsum: float,
    fd: float,
    shift: float,
    g: float,
    rate: float,
    times: NDArray,
    ef_detuning: float = 0.0,
) -> tuple[NDArray, float]:
    """Compute ideal finite-time qubit excitation and asymptotic population rate."""
    fr, fp, j, kp = ckp
    z = fsum + shift - fd
    h = RAD * np.array(
        [
            [ef_detuning, rate / 2, 0, 0],
            [rate / 2, 0, g, 0],
            [0, g, fr - z, j],
            [0, 0, j, fp - z - 0.5j * kp],
        ]
    )
    excited = np.array([_population(h, times, i)[:, :2].sum(axis=1) for i in (0, 1)])
    return excited, float(-2 * np.max(np.linalg.eigvals(h).imag))


def build_schedule(
    state: str,
    labels: tuple[str, str, str, str],
    preparation: tuple[Waveform, Waveform],
    amplitudes: tuple[float, float],
    frequencies: tuple[float, float],
    carriers: Sequence[float],
    duration: float,
    ramp: float,
    wait: float,
    dt: float,
) -> PulseSchedule:
    """Build a common-envelope two-tone experiment using MHz and ns."""
    if (
        state not in {"g", "e", "f"}
        or min(amplitudes) < 0
        or sum(amplitudes) > 1 + 1e-10
    ):
        raise ValueError("Invalid preparation or shared-port amplitude budget.")
    with PulseSchedule(list(labels)) as schedule:
        for i in range("gef".index(state)):
            schedule.add(labels[i], preparation[i])
            schedule.barrier()
        if duration:
            for label, amp, frequency, carrier in zip(
                labels[2:], amplitudes, frequencies, carriers, strict=True
            ):
                pulse = FlatTop(
                    duration=duration, amplitude=amp, tau=ramp, sampling_period=dt
                )
                schedule.add(label, pulse.detuned((frequency - carrier) / 1000))
        schedule.barrier()
        schedule.add(
            labels[0], Blank(float(np.ceil(wait / dt) * dt), sampling_period=dt)
        )
        schedule.barrier()
    for label, carrier in zip(labels[2:], carriers, strict=True):
        schedule.set_frequency(label, carrier / 1000)
    return schedule
