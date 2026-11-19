"""
Calibrate a two-tone reset and characterize the driven F0G1 transition.

Use `search_f0g1_reset` for a measured reset with a configured QuEL-3
experiment, ge/ef preparation pulses and a g/e/f classifier. It compares
candidate waveforms from all three preparations and reports empirical Stark
shifts. No CKP parameters are required.

Use `characterize_f0g1` to measure transfer versus drive amplitude and
frequency, including full F0G1 amplitude with ef off. A CKP resonator/filter
model and detuned time-domain measurements determine when a mode-specific
coupling and linewidth ratio can be identified.

For deliberately fixed experimental grids, the separate
`f0g1_calibration.calibrate_f0g1_pulse` recipe exposes each sweep stage.

Pulse parameters use GHz and ns. CKP parameters and transfer-model fits
explicitly use MHz. Population measurements retain preparation and readout
errors. Neither workflow changes stored qubit calibrations.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._f0g1_acquisition import (
    F0G1AcquisitionOptions,
    F0G1PopulationScan,
    F0G1PulseParameters,
    build_f0g1_reset,
    register_f0g1_targets,
)
from ._f0g1_characterization import (
    F0G1CKP,
    F0G1CouplingPoint,
    F0G1PhysicsOptions,
    F0G1PhysicsResult,
    analyze_f0g1_transfer,
    characterize_f0g1,
)
from ._f0g1_search import (
    F0G1ProbeLine,
    F0G1ResetSearchResult,
    F0G1SearchOptions,
    F0G1StarkEstimate,
    F0G1TransferPeak,
    search_f0g1_reset,
)

__all__ = [
    "F0G1CKP",
    "F0G1AcquisitionOptions",
    "F0G1CouplingPoint",
    "F0G1PhysicsOptions",
    "F0G1PhysicsResult",
    "F0G1PopulationScan",
    "F0G1ProbeLine",
    "F0G1PulseParameters",
    "F0G1ResetSearchResult",
    "F0G1SearchOptions",
    "F0G1StarkEstimate",
    "F0G1TransferPeak",
    "analyze_f0g1_transfer",
    "build_f0g1_reset",
    "characterize_f0g1",
    "plot_f0g1_ef_probe",
    "plot_f0g1_stark_shifts",
    "plot_f0g1_verification",
    "register_f0g1_targets",
    "search_f0g1_reset",
]


def plot_f0g1_ef_probe(estimate: F0G1StarkEstimate) -> go.Figure:
    """Compare matched ef frequency-by-duration maps with the F0G1 tone OFF and ON."""
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, subplot_titles=("F0G1 OFF", "F0G1 ON")
    )
    for col, line in enumerate((estimate.off, estimate.on), 1):
        fig.add_trace(
            go.Heatmap(
                x=1000 * line.frequencies_ghz,
                y=line.durations_ns,
                z=line.assigned_f.T,
                coloraxis="coloraxis",
                hovertemplate="%{x:.2f} MHz · %{y:.1f} ns<br>Assigned f = %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        if line.frequency is not None:
            fig.add_trace(
                go.Scatter(
                    x=[1000 * line.frequency] * 2,
                    y=line.durations_ns[[0, -1]],
                    mode="lines",
                    line=dict(color="white", dash="dot"),
                    showlegend=False,
                    name="Resolved ef center",
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="ef drive (MHz)")
    fig.update_yaxes(title_text="Duration (ns)", row=1, col=1)
    fig.update_layout(
        title=f"ef probe b={estimate.probe_amplitude:g} · F0G1 {estimate.f0g1_frequency * 1000:.2f} MHz, A={estimate.f0g1_amplitude:g}",
        coloraxis=dict(
            cmin=0, cmax=1, colorscale="Viridis", colorbar_title="Assigned f"
        ),
    )
    return fig


def plot_f0g1_stark_shifts(result: F0G1ResetSearchResult) -> go.Figure:
    """Show F0G1 displacements from a measured track origin and matched ef ON−OFF shifts."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("F0G1 relative to track origin", "ef relative to matched OFF"),
    )
    origins = sorted(
        {
            (p.reference_amplitude, p.reference_frequency)
            for p in result.drive_peaks
            if p.reference_amplitude is not None and p.reference_frequency is not None
        }
    )
    for a0, f0 in origins:
        peaks = [
            p
            for p in result.drive_peaks
            if (p.reference_amplitude, p.reference_frequency) == (a0, f0)
        ]
        fig.add_scatter(
            x=[a0**2, *[p.amplitude**2 for p in peaks]],
            y=[0, *[1000 * (p.frequency - f0) for p in peaks]],
            mode="markers",
            name=f"Origin {1000 * f0:.2f} MHz, A={a0:g}",
            row=1,
            col=1,
        )
    resolved = [(e, shift) for e in result.ef_shifts if (shift := e.shift) is not None]
    fig.add_scatter(
        x=[e.f0g1_amplitude**2 for e, _ in resolved],
        y=[1000 * shift for _, shift in resolved],
        customdata=[1000 * e.f0g1_frequency for e, _ in resolved],
        mode="markers",
        name="ef ON − OFF",
        hovertemplate="A² = %{x:.3f}<br>F0G1 %{customdata:.2f} MHz<br>ef shift %{y:.2f} MHz<extra></extra>",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="F0G1 command amplitude squared")
    fig.update_yaxes(title_text="Frequency change (MHz)", row=1, col=1)
    fig.update_layout(
        title=f"Measured frequency shifts · {len(resolved)}/{len(result.ef_shifts)} ef pairs resolved"
    )
    return fig


def plot_f0g1_verification(result: F0G1ResetSearchResult) -> go.Figure:
    """Plot independent assigned excitation and matched idle controls across the verification window."""
    fig = go.Figure()
    for name, stage in (("ON", "verification"), ("OFF", "verification_off")):
        scan = result.scans.get(stage)
        if scan is None:
            continue
        times = [p.duration for p in scan.parameters]
        for i, state in enumerate(scan.initial_states):
            fig.add_scatter(
                x=times,
                y=scan.excitation[:, i],
                mode="lines+markers",
                name=f"{state} prepared · {name}",
                line=dict(
                    color={"g": "#33867b", "e": "#b65f20", "f": "#75569b"}[state],
                    dash="solid" if name == "ON" else "dot",
                ),
            )
    fig.add_hline(
        y=result.options.target_excitation, line_dash="dash", annotation_text="Target"
    )
    fig.update_layout(
        xaxis_title="Reset duration (ns)",
        yaxis_title="Assigned P(e) + P(f)",
        yaxis_range=[0, 1],
        title=result.message,
    )
    return fig
