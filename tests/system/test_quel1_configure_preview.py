"""Tests for QuEL-1 configure preview behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.system import ConfigurePreview, ConfigureStateChange
from qubex.system.control_system import BoxType, PortType
from qubex.system.quel1 import Quel1ConfigurePreviewProvider


class _ExperimentSystemStub:
    def __init__(self, boxes: list[Any]) -> None:
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> Any:
        return self._boxes[box_id]

    @property
    def hash(self) -> int:
        return 0


def _make_system(
    *,
    box_type: BoxType = BoxType.QUEL1SE_R8,
    port_number: int = 1,
    port_type: PortType = PortType.CTRL,
    lo_freq: int | None = 10_000_000_000,
    cnco_freq: int = 1_500_000_000,
    fnco_freq: int | None = 100_000_000,
    rfswitch: str = "pass",
    vatt: int | None = 2048,
    sideband: str | None = "L",
    fullscale_current: int | None = 40527,
) -> _ExperimentSystemStub:
    channel = SimpleNamespace(number=0, fnco_freq=fnco_freq)
    port = SimpleNamespace(
        number=port_number,
        type=port_type,
        lo_freq=lo_freq,
        cnco_freq=cnco_freq,
        vatt=vatt,
        sideband=sideband,
        fullscale_current=fullscale_current,
        rfswitch=rfswitch,
        channels=(channel,),
    )
    return _ExperimentSystemStub(
        [SimpleNamespace(id="A", name="Alpha", type=box_type, ports=(port,))]
    )


def _backend_settings(
    *,
    port_number: int = 1,
    lo_freq: int | None = 10_000_000_000,
    cnco_freq: int = 1_500_000_000,
    fnco_freq: int = 100_000_000,
    rfswitch: str = "pass",
    vatt: int | None = 2048,
    sideband: str | None = "L",
    fullscale_current: int | None = 40527,
) -> dict[str, dict]:
    return {
        "A": {
            "ports": {
                port_number: {
                    "direction": "out",
                    "lo_freq": lo_freq,
                    "cnco_freq": cnco_freq,
                    "vatt": vatt,
                    "sideband": sideband,
                    "fullscale_current": fullscale_current,
                    "rfswitch": rfswitch,
                    "channels": {0: {"fnco_freq": fnco_freq}},
                }
            }
        }
    }


def _make_port(
    *,
    number: int,
    port_type: PortType,
    lo_freq: int | None = None,
    rfswitch: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        number=number,
        type=port_type,
        lo_freq=lo_freq,
        cnco_freq=None,
        vatt=None,
        sideband=None,
        fullscale_current=None,
        rfswitch=rfswitch,
        channels=(),
    )


def _make_shared_system(
    *,
    box_type: BoxType,
    ports: list[SimpleNamespace],
) -> _ExperimentSystemStub:
    return _ExperimentSystemStub(
        [
            SimpleNamespace(
                id="A",
                name="Alpha",
                type=box_type,
                ports=tuple(sorted(ports, key=lambda port: port.number)),
            )
        ]
    )


def _shared_backend_settings(
    *port_configs: tuple[int, str, int | None, str | None],
    include_rfswitch: bool = True,
) -> dict[str, dict]:
    ports: dict[int, dict[str, object]] = {}
    for port_number, direction, lo_freq, rfswitch in port_configs:
        config: dict[str, object] = {
            "direction": direction,
            "lo_freq": lo_freq,
        }
        if include_rfswitch:
            config["rfswitch"] = rfswitch
        config["channels" if direction == "out" else "runits"] = {}
        ports[port_number] = config
    return {"A": {"ports": ports}}


def _preview(
    *,
    experiment_system: _ExperimentSystemStub,
    backend_settings: dict[str, dict],
) -> ConfigurePreview:
    return Quel1ConfigurePreviewProvider().build_preview(
        experiment_system=cast(Any, experiment_system),
        backend_settings=backend_settings,
        box_ids=["A"],
        mode="ge-cr-cr",
    )


def test_preview_configure_reports_no_changes() -> None:
    """Given matching hardware and config, preview should report no changes."""
    preview = _preview(
        experiment_system=_make_system(),
        backend_settings=_backend_settings(),
    )

    assert preview.is_complete is True
    assert preview.has_changes is False
    assert preview.has_frequency_changes is False
    assert preview.changes == ()
    assert len(preview.entries) > 0
    assert all(not entry.has_change for entry in preview.entries)


def test_preview_configure_detects_frequency_changes() -> None:
    """Given changed LO frequency, preview should mark frequency changes."""
    preview = _preview(
        experiment_system=_make_system(lo_freq=11_000_000_000),
        backend_settings=_backend_settings(lo_freq=10_000_000_000),
    )

    assert preview.has_changes is True
    assert preview.has_frequency_changes is True
    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="lo_freq",
            before=10_000_000_000,
            after=11_000_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_before_uses_fetched_dump() -> None:
    """Given changed CNCO frequency, preview before should use fetched hardware state."""
    preview = _preview(
        experiment_system=_make_system(cnco_freq=2_109_375_000),
        backend_settings=_backend_settings(cnco_freq=2_320_312_500),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="cnco_freq",
            before=2_320_312_500,
            after=2_109_375_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_detects_non_frequency_changes() -> None:
    """Given changed RF switch, preview should not mark frequency changes."""
    preview = _preview(
        experiment_system=_make_system(rfswitch="pass"),
        backend_settings=_backend_settings(rfswitch="block"),
    )

    assert preview.has_changes is True
    assert preview.has_frequency_changes is False
    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="rfswitch",
            before="block",
            after="pass",
            unit=None,
            is_frequency=False,
        ),
    )


def test_preview_configure_uses_effective_r8_generator_port_values() -> None:
    """Given R8 non-mixer CTRL port VATT, preview should not report backend-ignored values."""
    preview = _preview(
        experiment_system=_make_system(
            port_number=6,
            lo_freq=10_000_000_000,
            vatt=3072,
            sideband="L",
            fnco_freq=0,
        ),
        backend_settings=_backend_settings(
            port_number=6,
            lo_freq=None,
            vatt=None,
            sideband=None,
            fnco_freq=0,
        ),
    )

    assert preview.changes == ()


def test_preview_configure_ignores_unspecified_fnco() -> None:
    """Given planned FNCO is unspecified, preview should not show zero-to-blank changes."""
    preview = _preview(
        experiment_system=_make_system(fnco_freq=None),
        backend_settings=_backend_settings(fnco_freq=0),
    )

    assert preview.changes == ()


def test_preview_configure_ignores_unspecified_port_fields() -> None:
    """Given a planned port field is unspecified, preview should preserve hardware."""
    preview = _preview(
        experiment_system=_make_system(
            box_type=BoxType.QUEL1SE_A,
            lo_freq=None,
        ),
        backend_settings=_backend_settings(lo_freq=10_000_000_000),
    )

    assert preview.changes == ()


def test_preview_configure_marks_missing_fetch_incomplete() -> None:
    """Given missing hardware fetch result, preview should be incomplete."""
    preview = _preview(
        experiment_system=_make_system(),
        backend_settings={},
    )

    assert preview.is_complete is False
    assert preview.missing_box_ids == ("A",)
    assert preview.changes == ()


@pytest.mark.parametrize(
    ("box_type", "capture_port_number", "generator_port_number"),
    [
        (BoxType.QUEL1SE_A, 0, 1),
        (BoxType.QUEL1SE_A, 5, 3),
        (BoxType.QUEL1SE_A, 7, 8),
        (BoxType.QUEL1SE_A, 12, 10),
        (BoxType.QUEL1SE_B, 5, 3),
        (BoxType.QUEL1SE_B, 12, 10),
        (BoxType.QUEL1_A, 0, 1),
        (BoxType.QUEL1_A, 5, 3),
        (BoxType.QUEL1_A, 7, 8),
        (BoxType.QUEL1_A, 12, 10),
        (BoxType.QUEL1_B, 5, 2),
        (BoxType.QUEL1_B, 12, 9),
        (BoxType.QUEL1SE_R8, 0, 1),
        (BoxType.QUBE_RIKEN_A, 1, 0),
        (BoxType.QUBE_RIKEN_A, 4, 2),
        (BoxType.QUBE_RIKEN_A, 12, 13),
        (BoxType.QUBE_RIKEN_A, 9, 11),
        (BoxType.QUBE_RIKEN_B, 4, 2),
        (BoxType.QUBE_RIKEN_B, 9, 11),
        (BoxType.QUBE_OU_A, 1, 0),
        (BoxType.QUBE_OU_A, 12, 13),
    ],
)
def test_preview_configure_uses_final_generator_lo_for_shared_resource(
    box_type: BoxType,
    capture_port_number: int,
    generator_port_number: int,
) -> None:
    """Given a shared LO restored by a generator, preview should use its final value."""
    system = _make_shared_system(
        box_type=box_type,
        ports=[
            _make_port(
                number=capture_port_number,
                port_type=PortType.READ_IN,
                lo_freq=9_000_000_000,
            ),
            _make_port(
                number=generator_port_number,
                port_type=PortType.CTRL,
                lo_freq=11_000_000_000,
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (capture_port_number, "in", 11_000_000_000, None),
            (generator_port_number, "out", 11_000_000_000, None),
        ),
    )

    assert preview.changes == ()
    assert preview.has_frequency_changes is False


def test_preview_configure_uses_last_capture_lo_for_shared_r8_resource() -> None:
    """Given R8 monitor ports sharing an LO, preview should use the last capture write."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_R8,
        ports=[
            _make_port(
                number=4,
                port_type=PortType.MNTR_IN,
                lo_freq=9_000_000_000,
            ),
            _make_port(
                number=10,
                port_type=PortType.MNTR_IN,
                lo_freq=11_000_000_000,
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (4, "in", 11_000_000_000, None),
            (10, "in", 11_000_000_000, None),
        ),
    )

    assert preview.changes == ()


def test_preview_configure_reports_final_shared_lo_for_each_port() -> None:
    """Given a changing shared LO, preview should retain rows for every logical port."""
    system = _make_shared_system(
        box_type=BoxType.QUBE_RIKEN_B,
        ports=[
            _make_port(
                number=2,
                port_type=PortType.CTRL,
                lo_freq=11_000_000_000,
            ),
            _make_port(
                number=4,
                port_type=PortType.MNTR_IN,
                lo_freq=None,
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (2, "out", 10_000_000_000, None),
            (4, "in", 10_000_000_000, None),
        ),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 2",
            field="lo_freq",
            before=10_000_000_000,
            after=11_000_000_000,
            unit="Hz",
            is_frequency=True,
        ),
        ConfigureStateChange(
            box_id="A",
            component="port 4",
            field="lo_freq",
            before=10_000_000_000,
            after=11_000_000_000,
            unit="Hz",
            is_frequency=True,
        ),
    )


def test_preview_configure_normalizes_shared_rfswitch_values() -> None:
    """Given a restored shared RF switch, preview should compare its final state."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_A,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="loop",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="pass",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, "open"),
            (1, "out", None, "pass"),
        ),
    )

    assert preview.changes == ()


def test_preview_configure_reports_final_shared_rfswitch_for_each_port() -> None:
    """Given a changing shared RF switch, preview should decode each logical port."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1SE_A,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="open",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="block",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, "open"),
            (1, "out", None, "pass"),
        ),
    )

    assert preview.changes == (
        ConfigureStateChange(
            box_id="A",
            component="port 0",
            field="rfswitch",
            before="open",
            after="loop",
            unit=None,
            is_frequency=False,
        ),
        ConfigureStateChange(
            box_id="A",
            component="port 1",
            field="rfswitch",
            before="pass",
            after="block",
            unit=None,
            is_frequency=False,
        ),
    )


def test_preview_configure_ignores_unreported_rfswitch_state() -> None:
    """Given a dump without RF switches, preview should not infer their changes."""
    system = _make_shared_system(
        box_type=BoxType.QUEL1_A,
        ports=[
            _make_port(
                number=0,
                port_type=PortType.READ_IN,
                rfswitch="open",
            ),
            _make_port(
                number=1,
                port_type=PortType.READ_OUT,
                rfswitch="pass",
            ),
        ],
    )

    preview = _preview(
        experiment_system=system,
        backend_settings=_shared_backend_settings(
            (0, "in", None, None),
            (1, "out", None, None),
            include_rfswitch=False,
        ),
    )

    assert preview.changes == ()
