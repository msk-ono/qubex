"""Tests for QuEL-1 configure preview behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

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


def test_preview_configure_marks_missing_fetch_incomplete() -> None:
    """Given missing hardware fetch result, preview should be incomplete."""
    preview = _preview(
        experiment_system=_make_system(),
        backend_settings={},
    )

    assert preview.is_complete is False
    assert preview.missing_box_ids == ("A",)
    assert preview.changes == ()
