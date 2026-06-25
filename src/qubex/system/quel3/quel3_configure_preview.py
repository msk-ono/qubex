"""QuEL-3 configure-preview builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from qubex.backend.backend_controller import BACKEND_KIND_QUEL3
from qubex.backend.quel3.models import InstrumentDeployRequest
from qubex.system.configure_preview import (
    ConfigurePreview,
    ConfigurePreviewProvider,
    ConfigureStateChange,
)
from qubex.typing import ConfigurationMode

if TYPE_CHECKING:
    from qubex.system.experiment_system import ExperimentSystem


class Quel3ConfigurePreviewProvider(ConfigurePreviewProvider):
    """Build configure previews for QuEL-3 instrument deployment."""

    def build_preview(
        self,
        *,
        experiment_system: ExperimentSystem,
        backend_settings: Mapping[str, dict],
        box_ids: Sequence[str],
        mode: ConfigurationMode | None,
        requests: Sequence[InstrumentDeployRequest],
    ) -> ConfigurePreview:
        """Build a QuEL-3 configure preview."""
        box_id_by_unit_label = {
            experiment_system.get_box(box_id).name: box_id for box_id in box_ids
        }
        entries: list[ConfigureStateChange] = []
        missing_box_ids: list[str] = []
        missing_box_id_set: set[str] = set()

        for request in requests:
            box_id = _resolve_box_id(
                request=request,
                box_id_by_unit_label=box_id_by_unit_label,
            )
            if box_id is None:
                continue
            box_config = backend_settings.get(box_id)
            if not isinstance(box_config, Mapping):
                if box_id not in missing_box_id_set:
                    missing_box_ids.append(box_id)
                    missing_box_id_set.add(box_id)
                continue
            instruments_config = box_config.get("instruments", {})
            if not isinstance(instruments_config, Mapping):
                if box_id not in missing_box_id_set:
                    missing_box_ids.append(box_id)
                    missing_box_id_set.add(box_id)
                continue
            instrument_config = _find_instrument_config(
                request=request,
                instruments_config=instruments_config,
            )
            entries.extend(
                _compare_request(
                    box_id=box_id,
                    request=request,
                    instrument_config=instrument_config,
                )
            )

        return ConfigurePreview(
            backend_kind=BACKEND_KIND_QUEL3,
            box_ids=tuple(box_ids),
            mode=mode,
            entries=tuple(entries),
            missing_box_ids=tuple(missing_box_ids),
        )


def _resolve_box_id(
    *,
    request: InstrumentDeployRequest,
    box_id_by_unit_label: Mapping[str, str],
) -> str | None:
    unit_label = request.port_id.split(":", maxsplit=1)[0]
    return box_id_by_unit_label.get(unit_label)


def _compare_request(
    *,
    box_id: str,
    request: InstrumentDeployRequest,
    instrument_config: Mapping[str, object] | None,
) -> list[ConfigureStateChange]:
    definition_config = _extract_definition_config(instrument_config)
    profile_config = _extract_profile_config(definition_config)

    component = _component_name(request)
    return [
        _field_entry(
            box_id=box_id,
            component=component,
            field="alias",
            before=_normalize_value(definition_config.get("alias")),
            after=_planned_definition_alias(request),
        ),
        _field_entry(
            box_id=box_id,
            component=component,
            field="role",
            before=_normalize_role(definition_config.get("role")),
            after=request.role,
        ),
        _field_entry(
            box_id=box_id,
            component=component,
            field="mode",
            before=_normalize_value(definition_config.get("mode")),
            after="FIXED_TIMELINE",
        ),
        _field_entry(
            box_id=box_id,
            component=component,
            field="frequency_range_min",
            before=_normalize_value(profile_config.get("frequency_range_min")),
            after=_normalize_value(request.frequency_range_min_hz),
            unit="Hz",
            is_frequency=True,
        ),
        _field_entry(
            box_id=box_id,
            component=component,
            field="frequency_range_max",
            before=_normalize_value(profile_config.get("frequency_range_max")),
            after=_normalize_value(request.frequency_range_max_hz),
            unit="Hz",
            is_frequency=True,
        ),
    ]


def _find_instrument_config(
    *,
    request: InstrumentDeployRequest,
    instruments_config: Mapping[object, object],
) -> Mapping[str, object] | None:
    for alias in (request.alias, _planned_definition_alias(request)):
        exact_config = instruments_config.get(alias)
        if isinstance(exact_config, Mapping):
            return exact_config

    definition_alias_configs = [
        instrument_config
        for instrument_config in instruments_config.values()
        if isinstance(instrument_config, Mapping)
        and _definition_alias_matches_request(
            instrument_config=instrument_config,
            request=request,
        )
    ]
    if len(definition_alias_configs) == 1:
        return definition_alias_configs[0]

    matched_configs = [
        instrument_config
        for instrument_config in instruments_config.values()
        if isinstance(instrument_config, Mapping)
        and _instrument_matches_request(
            instrument_config=instrument_config,
            request=request,
        )
    ]
    if len(matched_configs) != 1:
        return None
    return matched_configs[0]


def _instrument_matches_request(
    *,
    instrument_config: Mapping[str, object],
    request: InstrumentDeployRequest,
) -> bool:
    return (
        instrument_config.get("port_id") == request.port_id
        and _normalize_role(instrument_config.get("role")) == request.role
    )


def _definition_alias_matches_request(
    *,
    instrument_config: Mapping[str, object],
    request: InstrumentDeployRequest,
) -> bool:
    definition_config = _extract_definition_config(instrument_config)
    definition_alias = definition_config.get("alias")
    return definition_alias in {request.alias, _planned_definition_alias(request)}


def _extract_definition_config(
    instrument_config: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if instrument_config is None:
        return {}
    definition_config = instrument_config.get("definition", {})
    if not isinstance(definition_config, Mapping):
        return {}
    return definition_config


def _extract_profile_config(
    definition_config: Mapping[str, object],
) -> Mapping[str, object]:
    profile_config = definition_config.get("profile", {})
    if not isinstance(profile_config, Mapping):
        return {}
    return profile_config


def _planned_definition_alias(request: InstrumentDeployRequest) -> str:
    if ":" in request.alias:
        return request.alias
    unit_label = request.port_id.split(":", maxsplit=1)[0]
    return f"{unit_label}:{request.alias}"


def _component_name(request: InstrumentDeployRequest) -> str:
    return f"{_strip_resource_prefix(request.port_id)} {_strip_resource_prefix(request.alias)}"


def _strip_resource_prefix(resource_id: str) -> str:
    return resource_id.split(":", maxsplit=1)[-1]


def _field_entry(
    *,
    box_id: str,
    component: str,
    field: str,
    before: object,
    after: object,
    unit: str | None = None,
    is_frequency: bool = False,
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


def _normalize_role(value: object) -> object:
    if value is None:
        return None
    role_name = getattr(value, "name", value)
    return role_name if isinstance(role_name, str) else str(role_name)


def _normalize_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
