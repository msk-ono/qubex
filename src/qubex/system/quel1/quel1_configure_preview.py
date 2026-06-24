"""QuEL-1 configure-preview builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1
from qubex.system.configure_preview import (
    ConfigurePreview,
    ConfigurePreviewProvider,
    ConfigureStateChange,
)
from qubex.system.control_system import BoxType, PortType
from qubex.typing import ConfigurationMode

if TYPE_CHECKING:
    from qubex.system.experiment_system import ExperimentSystem


class Quel1ConfigurePreviewProvider(ConfigurePreviewProvider):
    """Build configure previews for QuEL-1-family backends."""

    def build_preview(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: Mapping[str, dict],
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
            backend_kind=BACKEND_KIND_QUEL1,
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
        """Return preview rows for one QuEL-1-family box."""
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
        """Return preview rows for one output-like QuEL-1 port."""
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
        """Return preview rows for one input-like QuEL-1 port."""
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
