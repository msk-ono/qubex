"""Analyze QuEL-1 timing diagnostic logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<level>\S+)\s+"
    r"(?P<logger>\S+)\s+"
    r"(?P<message>.*)$"
)
FIELD_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>.*?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)"
)
TIMING_MARKERS = {
    "qxdriver_quel1_timing": "qxdriver",
    "quel_ic_config_timing": "quel_ic_config",
    "qubex_timing": "qubex",
}
DEFAULT_SCHEDULED_OFFSET_MS = 150.0


@dataclass(frozen=True)
class TimingRecord:
    """Represent one parsed timing diagnostic record."""

    path: str
    line_number: int
    timestamp: datetime
    level: str
    logger: str
    source: str
    fields: dict[str, Any]

    @property
    def event(self) -> str:
        """Return the event field."""
        return str(self.fields.get("event", "unknown"))

    @property
    def phase(self) -> str:
        """Return the phase field."""
        return str(self.fields.get("phase", "unknown"))

    @property
    def box(self) -> str | None:
        """Return the box field when present."""
        value = self.fields.get("box")
        return str(value) if value is not None else None


@dataclass(frozen=True)
class MetricStats:
    """Represent summary statistics for one metric."""

    count: int
    minimum: float
    median: float
    maximum: float
    mean: float


@dataclass(frozen=True)
class ActionSummary:
    """Represent one Qubex multi-action timing summary."""

    index: int
    path: str
    line_start: int
    line_end: int
    status: str
    awg_count: int | None
    capture_count: int | None
    scheduled_offset_ms: float | None
    action_elapsed_ms: float | None
    build_scheduled_times_elapsed_ms: float | None
    capture_start_all_elapsed_ms: float | None
    capture_start_remaining_ms: float | None
    min_margin_ms: float | None
    margin_threshold_ms: float | None
    margin_spare_ms: float | None
    deadline_consumed_to_capture_start_end_ms: float | None
    deadline_consumed_to_margin_ms: float | None
    max_add_awg_start_elapsed_ms: float | None
    capture_stop_all_elapsed_ms: float | None
    max_post_trigger_task_elapsed_ms: float | None
    max_gen_task_result_elapsed_ms: float | None
    queue_wait_by_box: dict[str, float]
    capture_start_elapsed_by_box: dict[str, float]
    too_late_count: int
    error_count: int


@dataclass(frozen=True)
class WorstRecord:
    """Represent one high-cost timing record."""

    category: str
    action_index: int | None
    value_ms: float
    path: str
    line_number: int
    event: str
    phase: str
    box: str | None
    detail: str


@dataclass(frozen=True)
class AnalysisReport:
    """Represent a timing-log analysis report."""

    paths: list[str]
    records_count: int
    actions: list[ActionSummary]
    condition_lines: list[str]
    repeat_lines: list[str]
    too_late_count: int
    error_count: int
    metric_stats: dict[str, MetricStats]
    worst_records: dict[str, list[WorstRecord]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze qubex_timing, qxdriver_quel1_timing, and "
            "quel_ic_config_timing logs by Qubex multi-action."
        )
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="Timing log files to analyze.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the full report as JSON instead of text.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of worst records to show per category in text output.",
    )
    return parser


def _parse_timestamp(value: str) -> datetime:
    """Parse one log timestamp."""
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")


def _parse_value(value: str) -> Any:
    """Parse a scalar timing field value."""
    stripped = value.strip()
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        stripped,
    ):
        return float(stripped)
    return stripped


def _split_timing_message(message: str) -> tuple[str, str] | None:
    """Return source and key-value field text from a timing message."""
    for marker, source in TIMING_MARKERS.items():
        index = message.find(marker)
        if index >= 0:
            return source, message[index + len(marker) :].strip()
    return None


def _parse_fields(text: str) -> dict[str, Any]:
    """Parse one key-value field string."""
    return {
        match.group("key"): _parse_value(match.group("value"))
        for match in FIELD_RE.finditer(text)
    }


def parse_log_file(path: Path) -> list[TimingRecord]:
    """Parse timing diagnostic records from one log file."""
    records: list[TimingRecord] = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            match = LOG_LINE_RE.match(line.rstrip("\n"))
            if match is None:
                continue
            split = _split_timing_message(match.group("message"))
            if split is None:
                continue
            source, field_text = split
            records.append(
                TimingRecord(
                    path=str(path),
                    line_number=line_number,
                    timestamp=_parse_timestamp(match.group("timestamp")),
                    level=match.group("level"),
                    logger=match.group("logger"),
                    source=source,
                    fields=_parse_fields(field_text),
                )
            )
    return records


def _read_context_lines(paths: Sequence[Path]) -> tuple[list[str], list[str]]:
    """Read non-diagnostic context lines that describe run conditions."""
    condition_lines: list[str] = []
    repeat_lines: list[str] = []
    condition_patterns = (
        "enabled dummy per-box dispatch threads",
        "creating experiment",
        "wait_range_ns",
    )
    repeat_patterns = (
        "starting T1 sweep",
        "finished T1 sweep",
    )
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.rstrip("\n")
                prefix = f"{path}:{line_number}: "
                if any(pattern in stripped for pattern in condition_patterns):
                    condition_lines.append(prefix + stripped)
                if any(pattern in stripped for pattern in repeat_patterns):
                    repeat_lines.append(prefix + stripped)
    return condition_lines, repeat_lines


def _to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _records_matching(
    records: Iterable[TimingRecord],
    event: str,
    phase: str | None = None,
) -> list[TimingRecord]:
    """Return records matching one event and optional phase."""
    return [
        record
        for record in records
        if record.event == event and (phase is None or record.phase == phase)
    ]


def _first_record(
    records: Iterable[TimingRecord],
    event: str,
    phase: str | None = None,
) -> TimingRecord | None:
    """Return the first matching record."""
    for record in records:
        if record.event == event and (phase is None or record.phase == phase):
            return record
    return None


def _float_field(record: TimingRecord | None, field: str) -> float | None:
    """Read one float field from a record."""
    if record is None:
        return None
    return _to_float(record.fields.get(field))


def _int_field(record: TimingRecord | None, field: str) -> int | None:
    """Read one int field from a record."""
    if record is None:
        return None
    return _to_int(record.fields.get(field))


def _field_values(
    records: Iterable[TimingRecord],
    event: str,
    phase: str | None,
    field: str,
) -> list[float]:
    """Return numeric field values matching one event and phase."""
    values: list[float] = []
    for record in _records_matching(records, event, phase):
        value = _to_float(record.fields.get(field))
        if value is not None:
            values.append(value)
    return values


def _terminal_field_values(
    records: Iterable[TimingRecord],
    event: str,
    field: str,
) -> list[float]:
    """Return numeric field values for end or error phases."""
    values: list[float] = []
    for record in records:
        if record.event != event or record.phase not in {"end", "error"}:
            continue
        value = _to_float(record.fields.get(field))
        if value is not None:
            values.append(value)
    return values


def _max_or_none(values: Sequence[float]) -> float | None:
    """Return max(values) or None for an empty sequence."""
    return max(values) if values else None


def _min_or_none(values: Sequence[float]) -> float | None:
    """Return min(values) or None for an empty sequence."""
    return min(values) if values else None


def _derive_scheduled_offset_ms(build_record: TimingRecord | None) -> float | None:
    """Return the scheduled offset in milliseconds for one action."""
    remaining = _float_field(build_record, "remaining_ms")
    if remaining is not None:
        return remaining
    min_time_offset = _float_field(build_record, "min_time_offset")
    if min_time_offset is not None:
        return min_time_offset * 8e-6
    return DEFAULT_SCHEDULED_OFFSET_MS


def _action_segments(records: Sequence[TimingRecord]) -> list[list[TimingRecord]]:
    """Split timing records into Qubex multi-action ranges."""
    segments: list[list[TimingRecord]] = []
    current: list[TimingRecord] | None = None
    for record in records:
        if record.event == "qubex.multi.action" and record.phase == "start":
            current = [record]
            continue
        if current is None:
            continue
        current.append(record)
        if record.event == "qubex.multi.action" and record.phase in {"end", "error"}:
            segments.append(current)
            current = None
    if current:
        segments.append(current)
    return segments


def _box_float_map(
    records: Iterable[TimingRecord],
    event: str,
    phase: str,
    field: str,
) -> dict[str, float]:
    """Return a box-keyed float field map."""
    values: dict[str, float] = {}
    for record in _records_matching(records, event, phase):
        box = record.box
        value = _to_float(record.fields.get(field))
        if box is not None and value is not None:
            values[box] = value
    return values


def _action_counts(records: Sequence[TimingRecord]) -> tuple[int | None, int | None]:
    """Return awg and capture counts for one action."""
    record = _first_record(records, "qubex.multi.capture_start.box", "start")
    if record is None:
        record = _first_record(records, "qubex.multi.capture_start.box", "end")
    return _int_field(record, "awg_count"), _int_field(record, "capture_count")


def _summarize_action(index: int, records: Sequence[TimingRecord]) -> ActionSummary:
    """Build a summary for one Qubex multi-action segment."""
    action_end = _first_record(records, "qubex.multi.action", "end")
    action_error = _first_record(records, "qubex.multi.action", "error")
    action_terminal = action_end or action_error
    build_end = _first_record(records, "qubex.multi.build_scheduled_times", "end")
    capture_start_end = _first_record(records, "qubex.multi.capture_start.all", "end")
    capture_stop_end = _first_record(records, "qubex.multi.capture_stop.all", "end")
    capture_stop_error = _first_record(records, "qubex.multi.capture_stop.all", "error")
    capture_stop_terminal = capture_stop_end or capture_stop_error
    min_margin = _min_or_none(
        _field_values(
            records,
            "wave.awgunits_timed.activate",
            "margin_check",
            "remaining_ms_at_check",
        )
    )
    margin_threshold = _min_or_none(
        _field_values(
            records,
            "wave.awgunits_timed.activate",
            "margin_check",
            "margin_ms",
        )
    )
    scheduled_offset = _derive_scheduled_offset_ms(build_end)
    capture_start_remaining = _float_field(capture_start_end, "remaining_ms")
    awg_count, capture_count = _action_counts(records)
    return ActionSummary(
        index=index,
        path=records[0].path if records else "",
        line_start=records[0].line_number if records else 0,
        line_end=records[-1].line_number if records else 0,
        status=action_terminal.phase if action_terminal is not None else "open",
        awg_count=awg_count,
        capture_count=capture_count,
        scheduled_offset_ms=scheduled_offset,
        action_elapsed_ms=_float_field(action_terminal, "elapsed_ms"),
        build_scheduled_times_elapsed_ms=_float_field(build_end, "elapsed_ms"),
        capture_start_all_elapsed_ms=_float_field(capture_start_end, "elapsed_ms"),
        capture_start_remaining_ms=capture_start_remaining,
        min_margin_ms=min_margin,
        margin_threshold_ms=margin_threshold,
        margin_spare_ms=(
            min_margin - margin_threshold
            if min_margin is not None and margin_threshold is not None
            else None
        ),
        deadline_consumed_to_capture_start_end_ms=(
            scheduled_offset - capture_start_remaining
            if scheduled_offset is not None and capture_start_remaining is not None
            else None
        ),
        deadline_consumed_to_margin_ms=(
            scheduled_offset - min_margin
            if scheduled_offset is not None and min_margin is not None
            else None
        ),
        max_add_awg_start_elapsed_ms=_max_or_none(
            _field_values(
                records,
                "wave.awgunits_timed.add_awg_start",
                "end",
                "elapsed_ms",
            )
        ),
        capture_stop_all_elapsed_ms=_float_field(
            capture_stop_terminal,
            "elapsed_ms",
        ),
        max_post_trigger_task_elapsed_ms=_max_or_none(
            _field_values(records, "wave.awgunits_task.body", "end", "elapsed_ms")
        ),
        max_gen_task_result_elapsed_ms=_max_or_none(
            _field_values(
                records,
                "single.capture_stop.gen_task_result",
                "end",
                "elapsed_ms",
            )
        ),
        queue_wait_by_box=_box_float_map(
            records,
            "qubex.multi.capture_start.box",
            "start",
            "queue_wait_ms",
        ),
        capture_start_elapsed_by_box=_box_float_map(
            records,
            "qubex.multi.capture_start.box",
            "end",
            "elapsed_ms",
        ),
        too_late_count=sum(1 for record in records if record.phase == "too_late"),
        error_count=sum(1 for record in records if record.phase == "error"),
    )


def _stats(values: Sequence[float]) -> MetricStats | None:
    """Return summary statistics for numeric values."""
    if not values:
        return None
    return MetricStats(
        count=len(values),
        minimum=min(values),
        median=statistics.median(values),
        maximum=max(values),
        mean=statistics.mean(values),
    )


def _metric_stats(records: Sequence[TimingRecord]) -> dict[str, MetricStats]:
    """Build metric statistics used by the text report."""
    metrics = {
        "build_scheduled_times elapsed": _field_values(
            records,
            "qubex.multi.build_scheduled_times",
            "end",
            "elapsed_ms",
        ),
        "capture_start.all elapsed": _field_values(
            records,
            "qubex.multi.capture_start.all",
            "end",
            "elapsed_ms",
        ),
        "capture_start.all remaining": _field_values(
            records,
            "qubex.multi.capture_start.all",
            "end",
            "remaining_ms",
        ),
        "capture_start.box queue_wait": _field_values(
            records,
            "qubex.multi.capture_start.box",
            "start",
            "queue_wait_ms",
        ),
        "capture_start.box elapsed": _field_values(
            records,
            "qubex.multi.capture_start.box",
            "end",
            "elapsed_ms",
        ),
        "single.start_capture elapsed": _field_values(
            records,
            "single.start_capture_by_awg_trigger",
            "end",
            "elapsed_ms",
        ),
        "wave.start_awgunits_timed elapsed": _field_values(
            records,
            "wave.start_awgunits_timed",
            "end",
            "elapsed_ms",
        ),
        "margin_check remaining": _field_values(
            records,
            "wave.awgunits_timed.activate",
            "margin_check",
            "remaining_ms_at_check",
        ),
        "add_awg_start elapsed": _field_values(
            records,
            "wave.awgunits_timed.add_awg_start",
            "end",
            "elapsed_ms",
        ),
        "capture_stop.all elapsed": _terminal_field_values(
            records,
            "qubex.multi.capture_stop.all",
            "elapsed_ms",
        ),
        "capture_stop.box elapsed": _terminal_field_values(
            records,
            "qubex.multi.capture_stop.box",
            "elapsed_ms",
        ),
        "wave task body elapsed": _field_values(
            records,
            "wave.awgunits_task.body",
            "end",
            "elapsed_ms",
        ),
        "gen task result elapsed": _field_values(
            records,
            "single.capture_stop.gen_task_result",
            "end",
            "elapsed_ms",
        ),
        "action elapsed": _terminal_field_values(
            records,
            "qubex.multi.action",
            "elapsed_ms",
        ),
    }
    return {
        name: stats
        for name, values in metrics.items()
        if (stats := _stats(values)) is not None
    }


def _record_detail(record: TimingRecord) -> str:
    """Return a compact detail string for a timing record."""
    keys = (
        "remaining_ms",
        "remaining_ms_at_check",
        "margin_ms",
        "queue_wait_ms",
        "awg_count",
        "capture_count",
        "awgunit_count",
        "awgunits",
        "task",
        "error",
    )
    parts = []
    for key in keys:
        value = record.fields.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _worst_records(
    action_segments: Sequence[Sequence[TimingRecord]],
    limit: int = 10,
) -> dict[str, list[WorstRecord]]:
    """Return high-cost records grouped by bottleneck class."""
    categories: dict[str, list[WorstRecord]] = {
        "capture_start queue_wait": [],
        "capture_start box elapsed": [],
        "wave task body elapsed": [],
        "gen task result elapsed": [],
        "low margin_check remaining": [],
    }

    def add(
        category: str,
        action_index: int | None,
        record: TimingRecord,
        field: str,
    ) -> None:
        value = _to_float(record.fields.get(field))
        if value is None:
            return
        categories[category].append(
            WorstRecord(
                category=category,
                action_index=action_index,
                value_ms=value,
                path=record.path,
                line_number=record.line_number,
                event=record.event,
                phase=record.phase,
                box=record.box,
                detail=_record_detail(record),
            )
        )

    for action_index, segment in enumerate(action_segments, start=1):
        for record in _records_matching(
            segment,
            "qubex.multi.capture_start.box",
            "start",
        ):
            add("capture_start queue_wait", action_index, record, "queue_wait_ms")
        for record in _records_matching(
            segment,
            "qubex.multi.capture_start.box",
            "end",
        ):
            add("capture_start box elapsed", action_index, record, "elapsed_ms")
        for record in _records_matching(segment, "wave.awgunits_task.body", "end"):
            add("wave task body elapsed", action_index, record, "elapsed_ms")
        for record in _records_matching(
            segment,
            "single.capture_stop.gen_task_result",
            "end",
        ):
            add("gen task result elapsed", action_index, record, "elapsed_ms")
        for record in _records_matching(
            segment,
            "wave.awgunits_timed.activate",
            "margin_check",
        ):
            add(
                "low margin_check remaining",
                action_index,
                record,
                "remaining_ms_at_check",
            )

    for category, records in categories.items():
        reverse = category != "low margin_check remaining"
        records.sort(key=lambda record: record.value_ms, reverse=reverse)
        categories[category] = records[:limit]
    return categories


def analyze_log_files(paths: Sequence[Path]) -> AnalysisReport:
    """Analyze one or more timing log files."""
    records: list[TimingRecord] = []
    for path in paths:
        records.extend(parse_log_file(path))
    records.sort(key=lambda record: (record.timestamp, record.path, record.line_number))
    action_segments = _action_segments(records)
    condition_lines, repeat_lines = _read_context_lines(paths)
    return AnalysisReport(
        paths=[str(path) for path in paths],
        records_count=len(records),
        actions=[
            _summarize_action(index, segment)
            for index, segment in enumerate(action_segments, start=1)
        ],
        condition_lines=condition_lines,
        repeat_lines=repeat_lines,
        too_late_count=sum(1 for record in records if record.phase == "too_late"),
        error_count=sum(1 for record in records if record.phase == "error"),
        metric_stats=_metric_stats(records),
        worst_records=_worst_records(action_segments),
    )


def _fmt_ms(value: float | None) -> str:
    """Format a millisecond value."""
    return "-" if value is None else f"{value:.3f}"


def _format_box_map(values: dict[str, float]) -> str:
    """Format one box-keyed metric map."""
    if not values:
        return "-"
    return ",".join(
        f"{box_name}:{value:.3f}" for box_name, value in sorted(values.items())
    )


def _render_metric_table(
    report: AnalysisReport,
    title: str,
    names: Sequence[str],
) -> Table:
    """Render metric statistics as a rich table."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        box=box.HEAVY_HEAD,
    )
    table.add_column("Metric")
    table.add_column("count", justify="right")
    table.add_column("min[ms]", justify="right")
    table.add_column("median[ms]", justify="right")
    table.add_column("max[ms]", justify="right")
    table.add_column("mean[ms]", justify="right")
    for name in names:
        stats = report.metric_stats.get(name)
        if stats is None:
            table.add_row(name, "0", "-", "-", "-", "-")
            continue
        table.add_row(
            name,
            str(stats.count),
            f"{stats.minimum:.3f}",
            f"{stats.median:.3f}",
            f"{stats.maximum:.3f}",
            f"{stats.mean:.3f}",
        )
    return table


def _render_action_summary_table(report: AnalysisReport) -> Table:
    """Render per-action timing summary as a rich table."""
    table = Table(
        title="Per-action Summary",
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
    )
    table.add_column("idx", justify="right", no_wrap=True)
    table.add_column("status", justify="center", no_wrap=True)
    table.add_column("lines")
    table.add_column("awg/cap", justify="right", no_wrap=True)
    table.add_column("action_ms", justify="right")
    table.add_column("cap_start_ms", justify="right")
    table.add_column("cap_rem_ms", justify="right")
    table.add_column("min_margin", justify="right")
    table.add_column("spare", justify="right")
    table.add_column("consumed_to_margin", justify="right")
    table.add_column("cap_stop_ms", justify="right")
    table.add_column("wave_task_max", justify="right")
    table.add_column("gen_wait_max", justify="right")
    table.add_column("qwait_by_box", overflow="fold")
    table.add_column("cap_elapsed_by_box", overflow="fold")
    for action in report.actions:
        table.add_row(
            str(action.index),
            action.status,
            f"{action.path}:{action.line_start}-{action.line_end}",
            f"{action.awg_count or '-'}/{action.capture_count or '-'}",
            _fmt_ms(action.action_elapsed_ms),
            _fmt_ms(action.capture_start_all_elapsed_ms),
            _fmt_ms(action.capture_start_remaining_ms),
            _fmt_ms(action.min_margin_ms),
            _fmt_ms(action.margin_spare_ms),
            _fmt_ms(action.deadline_consumed_to_margin_ms),
            _fmt_ms(action.capture_stop_all_elapsed_ms),
            _fmt_ms(action.max_post_trigger_task_elapsed_ms),
            _fmt_ms(action.max_gen_task_result_elapsed_ms),
            _format_box_map(action.queue_wait_by_box),
            _format_box_map(action.capture_start_elapsed_by_box),
        )
    if not report.actions:
        table.add_row(
            "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"
        )
    return table


def _render_worst_records_table(
    report: AnalysisReport,
    *,
    top: int,
) -> list[Table]:
    """Render bottleneck-heavy records as rich tables."""
    tables: list[Table] = []
    for category, records in report.worst_records.items():
        table = Table(
            title=f"Worst: {category}",
            show_header=True,
            header_style="bold",
            box=box.MINIMAL_HEAVY_HEAD,
        )
        table.add_column("action", justify="right", no_wrap=True)
        table.add_column("file:line", ratio=2, overflow="fold")
        table.add_column("value[ms]", justify="right")
        table.add_column("event")
        table.add_column("phase")
        table.add_column("box")
        table.add_column("detail", ratio=3, overflow="fold")
        if not records:
            table.add_row("-", "-", "-", "-", "-", "-", "-")
            tables.append(table)
            continue
        for record in records[:top]:
            table.add_row(
                "-" if record.action_index is None else str(record.action_index),
                f"{record.path}:{record.line_number}",
                f"{record.value_ms:.3f}",
                record.event,
                record.phase,
                record.box or "-",
                record.detail,
            )
        tables.append(table)
    return tables


def _render_lines_panel(title: str, lines: Sequence[str]) -> Panel:
    """Render line-based context messages."""
    return Panel(
        "\n".join(lines),
        title=title,
        box=box.ROUNDED,
        expand=False,
    )


def _render_overview_panel(report: AnalysisReport) -> Panel:
    """Render overview fields as a compact panel."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(ratio=3)
    table.add_row("Files", ", ".join(report.paths))
    table.add_row(
        "Records",
        f"{report.records_count} actions={len(report.actions)}",
    )
    table.add_row(
        "Issues",
        f"too_late={report.too_late_count} errors={report.error_count}",
    )
    return Panel(table, title="Timing Log Analysis", box=box.ROUNDED, expand=False)


def render_text_report(report: AnalysisReport, *, top: int = 5) -> Group:
    """Render an analysis report using rich renderables."""
    sections = [
        _render_overview_panel(report),
        _render_metric_table(
            report,
            "Deadline-critical path",
            (
                "build_scheduled_times elapsed",
                "capture_start.all elapsed",
                "capture_start.all remaining",
                "capture_start.box queue_wait",
                "capture_start.box elapsed",
                "single.start_capture elapsed",
                "wave.start_awgunits_timed elapsed",
                "margin_check remaining",
                "add_awg_start elapsed",
            ),
        ),
        _render_metric_table(
            report,
            "Post-trigger wait",
            (
                "capture_stop.all elapsed",
                "capture_stop.box elapsed",
                "wave task body elapsed",
                "gen task result elapsed",
                "action elapsed",
            ),
        ),
        _render_action_summary_table(report),
    ]
    if report.condition_lines:
        sections.append(_render_lines_panel("Conditions", report.condition_lines))
    if report.repeat_lines:
        sections.append(_render_lines_panel("Sweep Lines", report.repeat_lines))
    sections.extend(_render_worst_records_table(report, top=top))
    return Group(*sections)


def print_text_report(report: AnalysisReport, *, top: int = 5) -> None:
    """Print an analysis report using rich."""
    console = Console()
    console.print(render_text_report(report, top=top))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the timing-log analyzer CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = analyze_log_files(args.log_files)
    if args.json:
        json.dump(
            report.to_dict(), sys.stdout, ensure_ascii=False, indent=2, default=str
        )
        sys.stdout.write("\n")
    else:
        print_text_report(report, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
