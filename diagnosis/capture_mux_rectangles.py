"""Drive all qubits in selected muxes with rectangles and capture them together."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from qxpulse import PulseSchedule, Rect


def _mux_label(value: str) -> str | int:
    """Parse a mux label from a command-line argument."""
    return int(value) if value.isdecimal() else value


def _configure_logging(log_file: Path | None) -> None:
    """Configure console and optional file logging for the run."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Drive every active qubit in selected muxes with the same rectangle pulse "
            "and capture all corresponding readout channels over the full schedule."
        )
    )
    parser.add_argument(
        "--system-id",
        default=os.getenv("QUBEX_SYSTEM_ID", "YOUR_SYSTEM_ID"),
        help="Qubex system id. Defaults to QUBEX_SYSTEM_ID or YOUR_SYSTEM_ID.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional path to the qubex config directory.",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=None,
        help="Optional path to the qubex params directory for the system.",
    )
    parser.add_argument(
        "--muxes",
        nargs="+",
        type=_mux_label,
        default=None,
        help="Mux indices or labels to diagnose together. Defaults to 0.",
    )
    parser.add_argument(
        "--exclude-qubits",
        nargs="+",
        default=None,
        help="Optional qubit labels to exclude from the selected muxes.",
    )
    parser.add_argument(
        "--drive-duration-ns",
        type=float,
        default=1024.0,
        help="Rectangle drive duration in ns.",
    )
    parser.add_argument(
        "--drive-amplitude",
        type=float,
        default=0.05,
        help="Rectangle drive amplitude applied to every mux qubit.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=1,
        help="Number of shots.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of repeats.",
    )
    parser.add_argument(
        "--shot-interval-ns",
        type=float,
        default=None,
        help="Optional shot interval in ns. Uses the Measurement default when omitted.",
    )
    parser.add_argument(
        "--shot-averaging",
        action="store_true",
        default=None,
        help="Enable hardware shot averaging.",
    )
    parser.add_argument(
        "--time-integration",
        action="store_true",
        default=False,
        help="Enable time integration. By default raw capture waveforms are returned.",
    )
    parser.add_argument(
        "--state-classification",
        action="store_true",
        default=False,
        help="Enable state classification.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the generated schedule before execution.",
    )
    parser.add_argument(
        "--timing-diagnostics",
        action="store_true",
        help="Enable QuEL-1 timing diagnostic logs when the backend supports them.",
    )
    parser.add_argument(
        "--e7awghal-timing-slow-ms",
        type=float,
        default=None,
        help="Optional slow threshold for E7AWGHAL_TIMING_SLOW_MS.",
    )
    parser.add_argument(
        "--timing-log-emit-slow-ms",
        type=float,
        default=None,
        help="Optional slow threshold for QUBEX_TIMING_LOG_EMIT_SLOW_MS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the schedule and print targets without connecting or executing.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Optional path to save captured arrays as a compressed .npz file.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/capture_mux_rectangles.log",
        help="Log file path. Use --log-file '' to disable file logging.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command-line values before touching hardware."""
    if not math.isfinite(args.drive_duration_ns) or args.drive_duration_ns <= 0.0:
        raise ValueError("--drive-duration-ns must be a positive finite value")
    if not math.isfinite(args.drive_amplitude):
        raise ValueError("--drive-amplitude must be finite")
    if args.n_shots <= 0:
        raise ValueError("--n-shots must be positive")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be positive")
    if args.shot_interval_ns is not None and args.shot_interval_ns <= 0.0:
        raise ValueError("--shot-interval-ns must be positive when provided")
    if args.e7awghal_timing_slow_ms is not None and args.e7awghal_timing_slow_ms < 0.0:
        raise ValueError("--e7awghal-timing-slow-ms must be non-negative")
    if args.timing_log_emit_slow_ms is not None and args.timing_log_emit_slow_ms < 0.0:
        raise ValueError("--timing-log-emit-slow-ms must be non-negative")


def _enable_timing_diagnostics(args: argparse.Namespace) -> None:
    """Enable supported QuEL-1 timing diagnostic log paths."""
    os.environ["QUBEX_QUEL1_TIMING_DIAGNOSTICS"] = "1"
    os.environ["QXDRIVER_QUEL1_TIMING_DIAGNOSTICS"] = "1"
    os.environ["QUEL_IC_CONFIG_TIMING_DIAGNOSTICS"] = "1"
    os.environ["E7AWGHAL_TIMING_DIAGNOSTICS"] = "1"
    if args.e7awghal_timing_slow_ms is not None:
        os.environ["E7AWGHAL_TIMING_SLOW_MS"] = str(args.e7awghal_timing_slow_ms)
    if args.timing_log_emit_slow_ms is not None:
        os.environ["QUBEX_TIMING_LOG_EMIT_SLOW_MS"] = str(args.timing_log_emit_slow_ms)


def _resolve_muxes(args: argparse.Namespace) -> list[str | int]:
    """Return mux labels selected by the command line."""
    if args.muxes is not None:
        muxes = list(args.muxes)
    else:
        muxes = []

    if len(muxes) == 0:
        raise ValueError("at least one mux must be selected")
    return list(dict.fromkeys(muxes))


def _make_experiment(args: argparse.Namespace) -> Any:
    """Create an Experiment for the requested muxes."""
    import qubex as qx

    kwargs: dict[str, Any] = {
        "system_id": args.system_id,
        "muxes": _resolve_muxes(args),
        "exclude_qubits": args.exclude_qubits,
    }
    if args.config_dir is not None:
        kwargs["config_dir"] = args.config_dir
    if args.params_dir is not None:
        kwargs["params_dir"] = args.params_dir
    return qx.Experiment(**kwargs)


def _build_drive_schedule(
    *,
    qubits: list[str],
    duration_ns: float,
    amplitude: float,
) -> PulseSchedule:
    """Build one simultaneous rectangle-drive schedule."""
    with PulseSchedule(qubits) as schedule:
        for qubit in qubits:
            schedule.add(
                qubit,
                Rect(
                    duration=duration_ns,
                    amplitude=amplitude,
                ),
            )
    return schedule


def _capture_targets(exp: Any, qubits: list[str]) -> list[str]:
    """Return readout capture target labels for the qubits."""
    return [exp.ctx.resolve_read_label(qubit) for qubit in qubits]


def _log_box_summary(exp: Any, qubits: list[str], logger: logging.Logger) -> None:
    """Log boxes involved in the selected qubits when available."""
    try:
        boxes = exp.ctx.experiment_system.get_boxes_for_qubits(qubits)
    except Exception as exc:  # pragma: no cover - best-effort script logging
        logger.info("could not resolve boxes for qubits: %s", exc)
        return
    logger.info("boxes=%s", [box.id for box in boxes])


def _save_npz(path: Path, result: Any) -> None:
    """Save captured arrays and metadata to a compressed npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {"targets": list(result.data)}
    for target, captures in result.data.items():
        metadata[target] = []
        for index, capture in enumerate(captures):
            key = f"{target}_{index}"
            array = np.asarray(capture.data)
            payload[key] = array
            metadata[target].append(
                {
                    "key": key,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )
    payload["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **payload)


async def _run(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Run the mux capture diagnosis."""
    muxes = _resolve_muxes(args)
    exp = _make_experiment(args)
    qubits = list(exp.qubit_labels)
    if not qubits:
        raise RuntimeError(f"no active qubits were resolved for muxes {muxes!r}")
    capture_targets = _capture_targets(exp, qubits)
    schedule = _build_drive_schedule(
        qubits=qubits,
        duration_ns=args.drive_duration_ns,
        amplitude=args.drive_amplitude,
    )
    measurement_schedule = exp.measurement_service.build_measurement_schedule(
        pulse_schedule=schedule,
        final_measurement=False,
        capture_placement="entire_schedule",
        capture_targets=capture_targets,
        plot=args.plot,
    )

    logger.info("muxes=%s", muxes)
    logger.info("qubits=%s", qubits)
    logger.info("capture_targets=%s", capture_targets)
    logger.info(
        "drive rectangle duration_ns=%.3f amplitude=%.6g",
        args.drive_duration_ns,
        args.drive_amplitude,
    )
    logger.info("schedule_duration_ns=%.3f", schedule.duration)
    _log_box_summary(exp, qubits, logger)

    if args.dry_run:
        logger.info("dry run: not connecting or executing")
        return 0

    logger.info("connecting")
    exp.connect()
    try:
        logger.info("running measurement")
        results = []
        for _ in range(args.n_repeats):
            result = await exp.run_measurement(
                measurement_schedule,
                n_shots=args.n_shots,
                shot_interval=args.shot_interval_ns,
                shot_averaging=args.shot_averaging,
                time_integration=args.time_integration,
                state_classification=args.state_classification,
            )
            results.append(result)
    finally:
        logger.info("disconnecting")
        exp.disconnect()

    logger.info("#results=%r", len(results))
    logger.info("results[0]=%r", results[0])
    for target, captures in results[0].data.items():
        for index, capture in enumerate(captures):
            data = np.asarray(capture.data)
            logger.info(
                "capture target=%s index=%s shape=%s dtype=%s",
                target,
                index,
                data.shape,
                data.dtype,
            )

    if args.output_npz is not None:
        _save_npz(args.output_npz, results[0])
        logger.info("saved output_npz=%s", args.output_npz)

    logger.info("done")
    return 0


def main() -> int:
    """Run the command-line entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    log_file = Path(args.log_file) if args.log_file else None
    _configure_logging(log_file)
    logger = logging.getLogger("capture_mux_rectangles")
    if args.timing_diagnostics:
        _enable_timing_diagnostics(args)
        logger.info("enabled QuEL-1 timing diagnostics")
    return asyncio.run(_run(args, logger))


if __name__ == "__main__":
    raise SystemExit(main())
