"""Tests for QuEL-3 configure preview behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from qubex.backend.backend_controller import BACKEND_KIND_QUEL3
from qubex.backend.quel3.models import InstrumentDeployRequest
from qubex.system import ConfigurePreview, ConfigureStateChange
from qubex.system.quel3 import Quel3ConfigurePreviewProvider, Quel3SystemSynchronizer


class _ExperimentSystemStub:
    def __init__(self) -> None:
        self._boxes = {"BOX1": SimpleNamespace(id="BOX1", name="quel3-02-a01")}

    def get_box(self, box_id: str) -> Any:
        return self._boxes[box_id]

    @property
    def hash(self) -> int:
        return 0


def _request(
    *,
    alias: str = "Q00",
    port_id: str = "quel3-02-a01:tx_p02",
    role: str = "TRANSMITTER",
    frequency_range_min_hz: float = 4.1e9,
    frequency_range_max_hz: float = 4.3e9,
) -> InstrumentDeployRequest:
    return InstrumentDeployRequest(
        port_id=port_id,
        role=cast(Any, role),
        frequency_range_min_hz=frequency_range_min_hz,
        frequency_range_max_hz=frequency_range_max_hz,
        alias=alias,
        target_labels=(alias,),
    )


def _backend_settings(
    *,
    alias: str = "Q00",
    definition_alias: str | None = None,
    port_id: str = "quel3-02-a01:tx_p02",
    role: str = "TRANSMITTER",
    definition_role: str | None = None,
    mode: str = "FIXED_TIMELINE",
    frequency_range_min_hz: float = 4.1e9,
    frequency_range_max_hz: float = 4.3e9,
) -> dict[str, dict]:
    if definition_alias is None:
        definition_alias = f"{port_id.split(':', maxsplit=1)[0]}:{alias}"
    if definition_role is None:
        definition_role = role
    return {
        "BOX1": {
            "instruments": {
                alias: {
                    "resource_id": "inst-q00",
                    "port_id": port_id,
                    "role": role,
                    "definition": {
                        "alias": definition_alias,
                        "role": definition_role,
                        "mode": mode,
                        "profile": {
                            "frequency_range_min": frequency_range_min_hz,
                            "frequency_range_max": frequency_range_max_hz,
                        },
                    },
                }
            }
        }
    }


def _preview(
    *,
    backend_settings: dict[str, dict],
    requests: tuple[InstrumentDeployRequest, ...],
) -> ConfigurePreview:
    return Quel3ConfigurePreviewProvider().build_preview(
        experiment_system=cast(Any, _ExperimentSystemStub()),
        backend_settings=backend_settings,
        box_ids=["BOX1"],
        mode="ge-cr-cr",
        requests=requests,
    )


def test_preview_configure_reports_no_changes() -> None:
    """Given matching instruments and deploy plan, preview should report no changes."""
    preview = _preview(
        backend_settings=_backend_settings(),
        requests=(_request(),),
    )

    assert preview.backend_kind == BACKEND_KIND_QUEL3
    assert preview.is_complete is True
    assert preview.has_changes is False
    assert preview.changes == ()
    assert len(preview.entries) == 5
    assert [entry.component for entry in preview.entries] == [
        "tx_p02 Q00",
        "tx_p02 Q00",
        "tx_p02 Q00",
        "tx_p02 Q00",
        "tx_p02 Q00",
    ]
    assert [entry.field for entry in preview.entries] == [
        "alias",
        "role",
        "mode",
        "frequency_range_min",
        "frequency_range_max",
    ]


def test_preview_configure_ignores_immutable_port_and_role() -> None:
    """Given immutable fields differ, preview should not report port or role changes."""
    preview = _preview(
        backend_settings=_backend_settings(
            definition_alias="quel3-02-a01:Q00",
            port_id="quel3-02-a01:tx_p04",
            role="TRANSCEIVER",
            definition_role="TRANSMITTER",
        ),
        requests=(_request(),),
    )

    assert preview.changes == ()


def test_preview_configure_detects_definition_role_and_mode_changes() -> None:
    """Given changed definition fields, preview should report role and mode changes."""
    preview = _preview(
        backend_settings=_backend_settings(
            definition_role="TRANSCEIVER",
            mode="OTHER_MODE",
        ),
        requests=(_request(),),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="role",
            before="TRANSCEIVER",
            after="TRANSMITTER",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="mode",
            before="OTHER_MODE",
            after="FIXED_TIMELINE",
        ),
    )


def test_preview_configure_matches_unique_current_instrument_by_port_and_role() -> None:
    """Given alias mismatch with one port match, preview should use current frequency."""
    preview = _preview(
        backend_settings=_backend_settings(
            alias="legacy-Q00",
            frequency_range_min_hz=4.0e9,
        ),
        requests=(_request(alias="Q00", frequency_range_min_hz=4.1e9),),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="alias",
            before="quel3-02-a01:legacy-Q00",
            after="quel3-02-a01:Q00",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_min",
            before=4_000_000_000,
            after=4_100_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_detects_frequency_range_changes() -> None:
    """Given changed frequency range, preview should mark frequency changes."""
    preview = _preview(
        backend_settings=_backend_settings(frequency_range_min_hz=4.0e9),
        requests=(_request(frequency_range_min_hz=4.1e9),),
    )

    assert preview.has_frequency_changes is True
    assert preview.changes == (
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_min",
            before=4_000_000_000,
            after=4_100_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_keeps_before_unknown_for_missing_instrument() -> None:
    """Given missing instrument snapshot, preview should show planned frequencies."""
    preview = _preview(
        backend_settings={"BOX1": {"instruments": {}}},
        requests=(_request(),),
    )

    assert preview.is_complete is True
    assert preview.changes == (
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="alias",
            before=None,
            after="quel3-02-a01:Q00",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="role",
            before=None,
            after="TRANSMITTER",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="mode",
            before=None,
            after="FIXED_TIMELINE",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_min",
            before=None,
            after=4_100_000_000,
            unit="Hz",
            is_frequency=True,
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_max",
            before=None,
            after=4_300_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_keeps_before_unknown_for_ambiguous_port_match() -> None:
    """Given ambiguous port matches, preview should keep current frequency unknown."""
    backend_settings = _backend_settings(alias="legacy-Q00")
    backend_settings["BOX1"]["instruments"]["legacy-Q01"] = _backend_settings(
        alias="legacy-Q01",
    )["BOX1"]["instruments"]["legacy-Q01"]

    preview = _preview(
        backend_settings=backend_settings,
        requests=(_request(alias="Q00"),),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="alias",
            before=None,
            after="quel3-02-a01:Q00",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="role",
            before=None,
            after="TRANSMITTER",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="mode",
            before=None,
            after="FIXED_TIMELINE",
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_min",
            before=None,
            after=4_100_000_000,
            unit="Hz",
            is_frequency=True,
        ),
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_max",
            before=None,
            after=4_300_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_synchronizer_preview_configure_fetches_and_plans() -> None:
    """Given QuEL-3 synchronizer, preview should compare fetched settings to deploy plan."""
    planner_calls: list[dict[str, object]] = []
    fetch_calls: list[dict[str, object]] = []
    request = _request(frequency_range_max_hz=4.4e9)

    class _FakeDeployPlanner:
        def build_deploy_requests(
            self,
            *,
            experiment_system: object,
            box_ids: list[str],
            target_labels: list[str] | None = None,
        ) -> tuple[InstrumentDeployRequest, ...]:
            planner_calls.append(
                {
                    "experiment_system": experiment_system,
                    "box_ids": box_ids,
                    "target_labels": target_labels,
                }
            )
            return (request,)

    class _FakeConfigurationManager:
        def fetch_backend_settings_from_hardware(
            self,
            *,
            unit_labels_by_box_id: dict[str, str],
            parallel: bool | None = None,
        ) -> dict[str, dict]:
            fetch_calls.append(
                {
                    "unit_labels_by_box_id": unit_labels_by_box_id,
                    "parallel": parallel,
                }
            )
            return _backend_settings(frequency_range_max_hz=4.3e9)

    backend_controller = SimpleNamespace(
        configuration_manager=_FakeConfigurationManager()
    )
    synchronizer = Quel3SystemSynchronizer(
        backend_controller=cast(Any, backend_controller),
        deploy_planner=cast(Any, _FakeDeployPlanner()),
    )
    experiment_system = _ExperimentSystemStub()

    preview = synchronizer.preview_configure(
        experiment_system=cast(Any, experiment_system),
        box_ids=["BOX1"],
        mode="ge-cr-cr",
        parallel=False,
        target_labels=["Q00"],
    )

    assert planner_calls == [
        {
            "experiment_system": experiment_system,
            "box_ids": ["BOX1"],
            "target_labels": ["Q00"],
        }
    ]
    assert fetch_calls == [
        {
            "unit_labels_by_box_id": {"BOX1": "quel3-02-a01"},
            "parallel": False,
        }
    ]
    assert preview.changes == (
        ConfigureStateChange(
            box_id="BOX1",
            component="tx_p02 Q00",
            field="frequency_range_max",
            before=4_300_000_000,
            after=4_400_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )
