"""Run repeated multi-mux T1 measurements on a QuEL-1 setup."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


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
            "Run repeated T1 measurements across multiple muxes/qubits. "
            "The T1 sequence uses only single-qubit excitation pulses."
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
        default=[0, 1],
        help="Mux labels to include when --qubits is omitted.",
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        default=None,
        help="Optional explicit qubit labels, for example Q00 Q01 Q08 Q09.",
    )
    parser.add_argument(
        "--exclude-qubits",
        nargs="+",
        default=None,
        help="Optional qubit labels to exclude from mux-selected targets.",
    )
    parser.add_argument(
        "--min-delay-ns",
        type=float,
        default=100.0,
        help="Minimum T1 wait in ns.",
    )
    parser.add_argument(
        "--max-delay-ns",
        type=float,
        default=200_000.0,
        help="Maximum T1 wait in ns.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=31,
        help="Number of logarithmic T1 wait points.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=1024,
        help="Number of shots per sweep point.",
    )
    parser.add_argument(
        "--shot-interval-ns",
        type=float,
        default=None,
        help="Optional shot interval in ns. Uses the Experiment default when omitted.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of full T1 sweeps to run.",
    )
    parser.add_argument(
        "--skip-pulse-calibration",
        action="store_true",
        help="Skip Rabi and HPI calibration when the calibration note is already valid.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render T1 fit plots.",
    )
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="Save T1 fit images.",
    )
    parser.add_argument(
        "--timing-diagnostics",
        action="store_true",
        help="Enable qxdriver_quel1 timing diagnostics from the optional patch.",
    )
    parser.add_argument(
        "--dummy-box-threads",
        type=int,
        default=0,
        help=(
            "Extra no-op worker threads to launch on each per-box parallel dispatch. "
            "Use this to stress Python scheduling as an analogue for more boxes."
        ),
    )
    parser.add_argument(
        "--dummy-thread-spin-ms",
        type=float,
        default=0.0,
        help="Optional busy-spin duration for each dummy worker after dispatch starts.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/multi_mux_t1_stress.log",
        help="Log file path. Use --log-file '' to disable file logging.",
    )
    return parser


def _make_experiment(args: argparse.Namespace) -> Any:
    """Create an Experiment for the requested muxes or qubits."""
    import qubex as qx

    kwargs: dict[str, Any] = {
        "system_id": args.system_id,
        "muxes": args.muxes if args.qubits is None else None,
        "qubits": args.qubits,
        "exclude_qubits": args.exclude_qubits,
    }
    if args.config_dir is not None:
        kwargs["config_dir"] = args.config_dir
    if args.params_dir is not None:
        kwargs["params_dir"] = args.params_dir
    return qx.Experiment(**kwargs)


@contextmanager
def _inject_dummy_box_threads(
    *,
    extra_threads: int,
    spin_ms: float,
) -> Iterator[None]:
    """Add dummy workers to each qubex per-box dispatch while in the context."""
    if extra_threads < 0:
        raise ValueError("--dummy-box-threads must be non-negative")
    if spin_ms < 0:
        raise ValueError("--dummy-thread-spin-ms must be non-negative")
    if extra_threads == 0:
        yield
        return

    from qubex.backend.quel1.compat import parallel_action_builder

    logger = logging.getLogger("multi_mux_t1_stress")
    original_run_per_box_parallel = parallel_action_builder._run_per_box_parallel  # noqa: SLF001

    def _dummy_worker(barrier: threading.Barrier) -> int:
        barrier.wait()
        deadline = time.perf_counter() + spin_ms / 1000.0
        iterations = 0
        while time.perf_counter() < deadline:
            iterations += 1
        return iterations

    def _wrapped_run_per_box_parallel(
        items: Sequence[tuple[str, Any]],
        runner: Callable[[str, Any], Any],
    ) -> dict[str, Any]:
        if not items:
            return {}

        worker_count = len(items) + extra_threads
        barrier = threading.Barrier(worker_count)

        def _invoke(item: tuple[str, Any]) -> tuple[str, Any]:
            name, payload = item
            barrier.wait()
            return name, runner(name, payload)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            dummy_futures = [
                executor.submit(_dummy_worker, barrier) for _ in range(extra_threads)
            ]
            box_futures = [executor.submit(_invoke, item) for item in items]
            results = dict(future.result() for future in box_futures)
            for future in dummy_futures:
                future.result()
        return results

    logger.info(
        "enabled dummy per-box dispatch threads extra_threads=%s spin_ms=%.3f",
        extra_threads,
        spin_ms,
    )
    parallel_action_builder._run_per_box_parallel = _wrapped_run_per_box_parallel  # noqa: SLF001
    try:
        yield
    finally:
        parallel_action_builder._run_per_box_parallel = original_run_per_box_parallel  # noqa: SLF001
        logger.info("disabled dummy per-box dispatch threads")


def _time_range(args: argparse.Namespace) -> np.ndarray:
    """Return logarithmic T1 wait points in ns."""
    if args.points < 2:
        raise ValueError("--points must be at least 2")
    if args.min_delay_ns <= 0 or args.max_delay_ns <= args.min_delay_ns:
        raise ValueError("--min-delay-ns must be positive and below --max-delay-ns")
    return np.geomspace(args.min_delay_ns, args.max_delay_ns, args.points)


def main() -> int:
    """Run the multi-mux T1 stress measurement."""
    parser = _build_parser()
    args = parser.parse_args()

    log_file = Path(args.log_file) if args.log_file else None
    _configure_logging(log_file)

    if args.timing_diagnostics:
        os.environ["QXDRIVER_QUEL1_TIMING_DIAGNOSTICS"] = "1"

    logger = logging.getLogger("multi_mux_t1_stress")
    logger.info(
        "creating experiment system_id=%s muxes=%s qubits=%s",
        args.system_id,
        args.muxes,
        args.qubits,
    )
    exp = _make_experiment(args)
    targets = list(args.qubits) if args.qubits is not None else list(exp.qubit_labels)
    if not targets:
        raise RuntimeError("no target qubits were resolved")

    waits = _time_range(args)
    logger.info("resolved targets=%s", targets)
    logger.info(
        "wait_range_ns first=%s last=%s points=%s",
        waits[0],
        waits[-1],
        len(waits),
    )
    logger.info("connecting")
    exp.connect()

    try:
        if args.skip_pulse_calibration:
            logger.info("skipping Rabi and HPI calibration")
        else:
            logger.info("running Rabi and HPI calibration")
            exp.obtain_rabi_params(plot=False)
            exp.calibrate_hpi_pulse(plot=False)
            exp.calib_note.save()

        with _inject_dummy_box_threads(
            extra_threads=args.dummy_box_threads,
            spin_ms=args.dummy_thread_spin_ms,
        ):
            for repeat_idx in range(args.repeats):
                started = time.perf_counter()
                logger.info(
                    "starting T1 sweep repeat=%s/%s",
                    repeat_idx + 1,
                    args.repeats,
                )
                result = exp.t1_experiment(
                    targets,
                    time_range=waits,
                    n_shots=args.n_shots,
                    shot_interval=args.shot_interval_ns,
                    # plot=args.plot,
                    # save_image=args.save_image,
                    plot=False,
                    save_image=False,
                    xaxis_type="log",
                )
                elapsed_s = time.perf_counter() - started
                logger.info(
                    "finished T1 sweep repeat=%s/%s elapsed_s=%.3f result_targets=%s",
                    repeat_idx + 1,
                    args.repeats,
                    elapsed_s,
                    list(result.data),
                )
    finally:
        logger.info("disconnecting")
        exp.disconnect()

    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
