"""QuEL-1 configure-preview builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

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
        effective_port_values = self._resolve_effective_port_values(box=box)
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
                        effective_port_values=effective_port_values,
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
                        effective_port_values=effective_port_values,
                    )
                )
        return entries

    def _resolve_effective_port_values(
        self,
        *,
        box: object,
    ) -> dict[tuple[object, str], object]:
        """Return final port values after applying shared-resource writes."""
        ports = tuple(getattr(box, "ports", ()))
        box_type = getattr(box, "type", None)
        final_resource_values: dict[tuple[str, tuple[object, ...]], object] = {}

        for port in _ports_in_configure_order(ports):
            port_number = getattr(port, "number", None)
            port_type = getattr(port, "type", None)
            for field in SHARED_RESOURCE_FIELDS:
                value = _normalize_value(
                    self._effective_port_value(
                        box=box,
                        port=port,
                        field=field,
                    )
                )
                if value is None:
                    continue
                resource_key = _resource_key(
                    box_type=box_type,
                    field=field,
                    port_number=port_number,
                )
                final_resource_values[resource_key] = _encode_resource_value(
                    field=field,
                    port_type=port_type,
                    value=value,
                )

        effective_port_values: dict[tuple[object, str], object] = {}
        for port in ports:
            port_number = getattr(port, "number", None)
            port_type = getattr(port, "type", None)
            if port_type not in CONFIGURED_PORT_TYPES:
                continue
            for field in SHARED_RESOURCE_FIELDS:
                resource_key = _resource_key(
                    box_type=box_type,
                    field=field,
                    port_number=port_number,
                )
                if resource_key not in final_resource_values:
                    continue
                effective_port_values[port_number, field] = _decode_resource_value(
                    field=field,
                    port_type=port_type,
                    value=final_resource_values[resource_key],
                )
        return effective_port_values

    def _effective_port_value(
        self,
        *,
        box: object,
        port: object,
        field: str,
    ) -> object:
        """Return one port value that the backend will attempt to write."""
        if getattr(port, "type", None) in GENERATOR_PORT_TYPES:
            return self._effective_generator_port_value(
                box=box,
                port=port,
                field=field,
            )
        return getattr(port, field, None)

    def _compare_generator_port(
        self,
        *,
        box_id: str,
        box: object,
        port: object,
        port_config: object,
        effective_port_values: Mapping[tuple[object, str], object],
    ) -> list[ConfigureStateChange]:
        """Return preview rows for one output-like QuEL-1 port."""
        port_number = getattr(port, "number", None)
        component = f"port {port_number}"
        config = port_config if isinstance(port_config, Mapping) else {}
        entries: list[ConfigureStateChange] = []

        for field in (
            "lo_freq",
            "cnco_freq",
            "vatt",
            "sideband",
            "fullscale_current",
            "rfswitch",
        ):
            if field == "rfswitch" and field not in config:
                continue
            planned_value = _normalize_value(
                effective_port_values.get(
                    (port_number, field),
                    self._effective_generator_port_value(
                        box=box,
                        port=port,
                        field=field,
                    ),
                )
            )
            if planned_value is None:
                continue
            entries.append(
                _field_entry(
                    box_id=box_id,
                    component=component,
                    field=field,
                    before=_normalize_value(config.get(field)),
                    after=planned_value,
                    unit="Hz" if field in FREQUENCY_FIELDS else None,
                    is_frequency=field in FREQUENCY_FIELDS,
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
        effective_port_values: Mapping[tuple[object, str], object],
    ) -> list[ConfigureStateChange]:
        """Return preview rows for one input-like QuEL-1 port."""
        port_number = getattr(port, "number", None)
        component = f"port {port_number}"
        config = port_config if isinstance(port_config, Mapping) else {}
        entries: list[ConfigureStateChange] = []

        for field in ("lo_freq", "cnco_freq", "rfswitch"):
            if field == "rfswitch" and field not in config:
                continue
            planned_value = _normalize_value(
                effective_port_values.get(
                    (port_number, field),
                    getattr(port, field, None),
                )
            )
            if planned_value is None:
                continue
            entries.append(
                _field_entry(
                    box_id=box_id,
                    component=component,
                    field=field,
                    before=_normalize_value(config.get(field)),
                    after=planned_value,
                    unit="Hz" if field in FREQUENCY_FIELDS else None,
                    is_frequency=field in FREQUENCY_FIELDS,
                )
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
CAPTURE_PORT_TYPES: Final = frozenset({PortType.READ_IN, PortType.MNTR_IN})
GENERATOR_PORT_TYPES: Final = frozenset(
    {PortType.CTRL, PortType.READ_OUT, PortType.PUMP}
)
CONFIGURED_PORT_TYPES: Final = CAPTURE_PORT_TYPES | GENERATOR_PORT_TYPES
SHARED_RESOURCE_FIELDS: Final = ("lo_freq", "rfswitch")

# Port groups are derived from quel_ic_config's port-to-line, LO, and RF-switch
# maps. Keep them local so importing the system model does not load optional
# QuEL-1 driver dependencies.
SHARED_RESOURCE_PORT_GROUPS: Final[
    dict[BoxType, dict[str, tuple[tuple[int, ...], ...]]]
] = {
    BoxType.QUEL1SE_A: {
        "lo_freq": ((0, 1), (3, 5), (7, 8), (10, 12)),
        "rfswitch": ((0, 1), (7, 8)),
    },
    BoxType.QUEL1SE_B: {
        "lo_freq": ((3, 5), (10, 12)),
    },
    BoxType.QUEL1SE_R8: {
        "lo_freq": ((0, 1), (4, 10)),
        "rfswitch": ((0, 1),),
    },
    BoxType.QUEL1_A: {
        "lo_freq": ((0, 1), (3, 5), (7, 8), (10, 12)),
        "rfswitch": ((0, 1), (7, 8)),
    },
    BoxType.QUEL1_B: {
        "lo_freq": ((2, 5), (9, 12)),
    },
    BoxType.QUBE_RIKEN_A: {
        "lo_freq": ((0, 1), (2, 4), (12, 13), (9, 11)),
        "rfswitch": ((0, 1), (12, 13)),
    },
    BoxType.QUBE_RIKEN_B: {
        "lo_freq": ((2, 4), (9, 11)),
    },
    BoxType.QUBE_OU_A: {
        "lo_freq": ((0, 1), (12, 13)),
    },
}


def _ports_in_configure_order(ports: Sequence[object]) -> tuple[object, ...]:
    """Return ports in the order used by QuEL-1 hardware configuration."""
    capture_ports = tuple(
        port for port in ports if getattr(port, "type", None) in CAPTURE_PORT_TYPES
    )
    generator_ports = tuple(
        port for port in ports if getattr(port, "type", None) in GENERATOR_PORT_TYPES
    )
    return capture_ports + generator_ports


def _resource_key(
    *,
    box_type: object,
    field: str,
    port_number: object,
) -> tuple[str, tuple[object, ...]]:
    """Return the physical-resource key for one logical port field."""
    if isinstance(box_type, BoxType):
        field_groups = SHARED_RESOURCE_PORT_GROUPS.get(box_type, {}).get(field, ())
        for port_group in field_groups:
            if port_number in port_group:
                return field, port_group
    return field, (port_number,)


def _encode_resource_value(
    *,
    field: str,
    port_type: object,
    value: object,
) -> object:
    """Encode a logical port value into its physical-resource value."""
    del port_type
    if field != "rfswitch":
        return value
    if value in ("open", "pass"):
        return False
    if value in ("loop", "block"):
        return True
    return value


def _decode_resource_value(
    *,
    field: str,
    port_type: object,
    value: object,
) -> object:
    """Decode a physical-resource value for one logical port."""
    if field != "rfswitch" or not isinstance(value, bool):
        return value
    if port_type in CAPTURE_PORT_TYPES:
        return "loop" if value else "open"
    return "block" if value else "pass"


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
