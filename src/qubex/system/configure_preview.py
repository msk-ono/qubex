"""Preview models and QuEL-1 comparison logic for `configure()`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rich.console import Console
from rich.table import Table

from qubex.backend.backend_controller import BackendKind
from qubex.system.control_system import BoxType, PortType
from qubex.typing import ConfigurationMode

if TYPE_CHECKING:
    from qubex.system.experiment_system import ExperimentSystem
    from qubex.system.system_manager import BackendSettings


@dataclass(frozen=True)
class ConfigureStateChange:
    """One field-level state comparison that `configure()` would evaluate."""

    box_id: str
    component: str
    field: str
    before: object
    after: object
    unit: str | None = None
    is_frequency: bool = False

    @property
    def has_change(self) -> bool:
        """Return whether the compared field would change."""
        return self.before != self.after


@dataclass(frozen=True)
class ConfigurePreview:
    """Structured preview of device state changes from `configure()`."""

    backend_kind: BackendKind
    box_ids: tuple[str, ...]
    mode: ConfigurationMode | None
    entries: tuple[ConfigureStateChange, ...] = ()
    missing_box_ids: tuple[str, ...] = ()

    @property
    def changes(self) -> tuple[ConfigureStateChange, ...]:
        """Return field-level comparisons that would change."""
        return tuple(entry for entry in self.entries if entry.has_change)

    @property
    def has_changes(self) -> bool:
        """Return whether `configure()` would change any tracked fields."""
        return len(self.changes) > 0

    @property
    def has_frequency_changes(self) -> bool:
        """Return whether `configure()` would change frequency-related fields."""
        return any(change.is_frequency for change in self.changes)

    @property
    def is_complete(self) -> bool:
        """Return whether all requested boxes were fetched for comparison."""
        return len(self.missing_box_ids) == 0

    def print_summary(self, console: Console | None = None) -> None:
        """Print field-level changes that `configure()` would apply."""
        if console is None:
            console = Console()

        table = Table(
            show_header=True,
            header_style="bold",
            title="Configure Preview Changes",
        )
        table.add_column("BOX", justify="left")
        table.add_column("COMPONENT", justify="left")
        table.add_column("FIELD", justify="left")
        table.add_column("BEFORE", justify="right")
        table.add_column("AFTER", justify="right")
        table.add_column("UNIT", justify="left")
        table.add_column("FREQ", justify="center")

        self._add_rows(table, self.changes, include_change=False)
        console.print(table)

    def print_full(self, console: Console | None = None) -> None:
        """Print all previewed field-level comparisons."""
        if console is None:
            console = Console()

        table = Table(
            show_header=True,
            header_style="bold",
            title="Configure Preview Full",
        )
        table.add_column("BOX", justify="left")
        table.add_column("COMPONENT", justify="left")
        table.add_column("FIELD", justify="left")
        table.add_column("BEFORE", justify="right")
        table.add_column("AFTER", justify="right")
        table.add_column("UNIT", justify="left")
        table.add_column("FREQ", justify="center")
        table.add_column("CHANGE", justify="center")

        self._add_rows(table, self.entries, include_change=True)
        console.print(table)

    def _add_rows(
        self,
        table: Table,
        entries: Sequence[ConfigureStateChange],
        *,
        include_change: bool,
    ) -> None:
        """Add preview rows to `table`."""
        for entry in entries:
            row = [
                entry.box_id,
                entry.component,
                entry.field,
                _format_value(entry.before),
                _format_value(entry.after),
                entry.unit or "",
                "yes" if entry.is_frequency else "",
            ]
            if include_change:
                row.append("yes" if entry.has_change else "no")
            table.add_row(*row)
        for box_id in self.missing_box_ids:
            row = [box_id, "box", "fetch", "failed", "", "", ""]
            if include_change:
                row.append("")
            table.add_row(*row)
        if not entries and not self.missing_box_ids:
            row = ["-", "-", "-", "no changes", "", "", ""]
            if include_change:
                row.append("")
            table.add_row(*row)


class ConfigurePreviewProvider(Protocol):
    """Backend-specific provider for `configure()` previews."""

    def build_preview(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: BackendSettings,
        box_ids: Sequence[str],
        mode: ConfigurationMode | None,
    ) -> ConfigurePreview:
        """Build one configure preview from hardware and planned software state."""
        ...


class Quel1ConfigurePreviewProvider:
    """Build configure previews for QuEL-1-family backends."""

    def build_preview(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: BackendSettings,
        box_ids: Sequence[str],
        mode: ConfigurationMode | None,
    ) -> ConfigurePreview:
        """Build a QuEL-1 configure preview."""
        entries: list[ConfigureStateChange] = []
        missing_box_ids: list[str] = []

        for box_id in box_ids:
            box_config = backend_settings.get(box_id)
            if not isinstance(box_config, Mapping):
                missing_box_ids.append(box_id)
                continue
            box = experiment_system.get_box(box_id)
            ports_config = box_config.get("ports", {})
            if not isinstance(ports_config, Mapping):
                missing_box_ids.append(box_id)
                continue
            entries.extend(
                self._compare_box(
                    box_id=box_id,
                    box=box,
                    ports_config=ports_config,
                )
            )

        return ConfigurePreview(
            backend_kind="quel1",
            box_ids=tuple(box_ids),
            mode=mode,
            entries=tuple(entries),
            missing_box_ids=tuple(missing_box_ids),
        )

    def _compare_box(
        self,
        *,
        box_id: str,
        box: object,
        ports_config: Mapping[object, object],
    ) -> list[ConfigureStateChange]:
        entries: list[ConfigureStateChange] = []
        for port in getattr(box, "ports", ()):
            port_type = getattr(port, "type", None)
            if port_type in (PortType.NOT_AVAILABLE, PortType.MNTR_OUT):
                continue
            if port_type in (PortType.CTRL, PortType.READ_OUT, PortType.PUMP):
                entries.extend(
                    self._compare_generator_port(
                        box_id=box_id,
                        box=box,
                        port=port,
                        port_config=_get_mapping_item(
                            ports_config,
                            getattr(port, "number", None),
                        ),
                    )
                )
            elif port_type in (PortType.READ_IN, PortType.MNTR_IN):
                entries.extend(
                    self._compare_capture_port(
                        box_id=box_id,
                        port=port,
                        port_config=_get_mapping_item(
                            ports_config,
                            getattr(port, "number", None),
                        ),
                    )
                )
        return entries

    def _compare_generator_port(
        self,
        *,
        box_id: str,
        box: object,
        port: object,
        port_config: object,
    ) -> list[ConfigureStateChange]:
        port_number = getattr(port, "number", None)
        component = f"port {port_number}"
        config = port_config if isinstance(port_config, Mapping) else {}
        entries: list[ConfigureStateChange] = []

        entries.extend(
            _field_entry(
                box_id=box_id,
                component=component,
                field=field,
                before=_normalize_value(config.get(field)),
                after=_normalize_value(
                    self._effective_generator_port_value(
                        box=box,
                        port=port,
                        field=field,
                    )
                ),
                unit="Hz" if field in FREQUENCY_FIELDS else None,
                is_frequency=field in FREQUENCY_FIELDS,
            )
            for field in (
                "lo_freq",
                "cnco_freq",
                "vatt",
                "sideband",
                "fullscale_current",
                "rfswitch",
            )
        )

        channels_config = config.get("channels", {})
        if not isinstance(channels_config, Mapping):
            channels_config = {}
        for channel in getattr(port, "channels", ()):
            channel_number = getattr(channel, "number", None)
            channel_config = _get_mapping_item(channels_config, channel_number)
            if not isinstance(channel_config, Mapping):
                channel_config = {}
            planned_fnco = _normalize_value(getattr(channel, "fnco_freq", None))
            if planned_fnco is None:
                continue
            entries.append(
                _field_entry(
                    box_id=box_id,
                    component=f"port {port_number} channel {channel_number}",
                    field="fnco_freq",
                    before=_normalize_value(channel_config.get("fnco_freq")),
                    after=planned_fnco,
                    unit="Hz",
                    is_frequency=True,
                )
            )
        return entries

    @staticmethod
    def _effective_generator_port_value(
        *,
        box: object,
        port: object,
        field: str,
    ) -> object:
        """Return the value that QuEL-1 backend configuration will actually use."""
        value = getattr(port, field, None)
        if getattr(box, "type", None) != BoxType.QUEL1SE_R8:
            return value

        port_number = getattr(port, "number", None)
        is_mixer_port = port_number in QUEL1SE_R8_MIXER_PORTS
        if field == "lo_freq" and not is_mixer_port:
            return None
        if field in ("vatt", "sideband") and not is_mixer_port:
            return None
        return value

    def _compare_capture_port(
        self,
        *,
        box_id: str,
        port: object,
        port_config: object,
    ) -> list[ConfigureStateChange]:
        port_number = getattr(port, "number", None)
        component = f"port {port_number}"
        config = port_config if isinstance(port_config, Mapping) else {}
        entries: list[ConfigureStateChange] = []

        entries.extend(
            _field_entry(
                box_id=box_id,
                component=component,
                field=field,
                before=_normalize_value(config.get(field)),
                after=_normalize_value(getattr(port, field, None)),
                unit="Hz" if field in FREQUENCY_FIELDS else None,
                is_frequency=field in FREQUENCY_FIELDS,
            )
            for field in ("lo_freq", "cnco_freq", "rfswitch")
        )

        runits_config = config.get("runits", {})
        if not isinstance(runits_config, Mapping):
            runits_config = {}
        for channel in getattr(port, "channels", ()):
            channel_number = getattr(channel, "number", None)
            channel_config = _get_mapping_item(runits_config, channel_number)
            if not isinstance(channel_config, Mapping):
                channel_config = {}
            planned_fnco = _normalize_value(getattr(channel, "fnco_freq", None))
            if planned_fnco is None:
                continue
            entries.append(
                _field_entry(
                    box_id=box_id,
                    component=f"port {port_number} runit {channel_number}",
                    field="fnco_freq",
                    before=_normalize_value(channel_config.get("fnco_freq")),
                    after=planned_fnco,
                    unit="Hz",
                    is_frequency=True,
                )
            )
        return entries


FREQUENCY_FIELDS = frozenset({"lo_freq", "cnco_freq", "fnco_freq"})
QUEL1SE_R8_MIXER_PORTS = frozenset({1, 2})


def _field_entry(
    *,
    box_id: str,
    component: str,
    field: str,
    before: object,
    after: object,
    unit: str | None,
    is_frequency: bool,
) -> ConfigureStateChange:
    return ConfigureStateChange(
        box_id=box_id,
        component=component,
        field=field,
        before=before,
        after=after,
        unit=unit,
        is_frequency=is_frequency,
    )


def _get_mapping_item(mapping: Mapping[object, object], key: object) -> object:
    if key in mapping:
        return mapping[key]
    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]
    return None


def _normalize_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:_}"
    return str(value)
