"""Tests for configure preview behavior."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from qubex.backend.backend_controller import BACKEND_KIND_QUEL1, BACKEND_KIND_QUEL3
from qubex.system import ConfigurePreview, ConfigureStateChange
from qubex.system.control_system import BoxType, PortType
from qubex.system.system_manager import SystemManager


class _ExperimentSystemStub:
    def __init__(self, boxes: list[Any]) -> None:
        self._boxes = {box.id: box for box in boxes}

    def get_box(self, box_id: str) -> Any:
        return self._boxes[box_id]

    @property
    def hash(self) -> int:
        return 0


class _SynchronizerStub:
    def __init__(self, backend_settings: dict[str, dict]) -> None:
        self.backend_settings = backend_settings
        self.calls: list[dict[str, object]] = []

    def fetch_backend_settings_from_hardware(
        self,
        *,
        experiment_system: object,
        box_ids: list[str],
        parallel: bool | None = None,
    ) -> dict[str, dict]:
        self.calls.append(
            {
                "experiment_system": experiment_system,
                "box_ids": box_ids,
                "parallel": parallel,
            }
        )
        return {
            box_id: self.backend_settings[box_id]
            for box_id in box_ids
            if box_id in self.backend_settings
        }


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
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_system: _ExperimentSystemStub,
    backend_settings: dict[str, dict],
    backend_kind: str = BACKEND_KIND_QUEL1,
) -> ConfigurePreview:
    manager = SystemManager.shared()
    synchronizer = _SynchronizerStub(backend_settings)
    monkeypatch.setattr(manager, "_system_synchronizer", synchronizer)
    monkeypatch.setattr(manager, "_backend_controller", SimpleNamespace(hash=0))
    monkeypatch.setattr(
        manager,
        "_load_preview_experiment_system",
        lambda **_: (experiment_system, backend_kind),
        raising=False,
    )

    return manager.preview_configure(
        chip_id="chip",
        system_id="system",
        config_dir="config",
        params_dir="params",
        targets_to_exclude=None,
        configuration_mode="ge-cr-cr",
        box_ids=["A"],
    )


def test_preview_configure_reports_no_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given matching hardware and config, preview should report no changes."""
    preview = _preview(
        monkeypatch,
        experiment_system=_make_system(),
        backend_settings=_backend_settings(),
    )

    assert preview.is_complete is True
    assert preview.has_changes is False
    assert preview.has_frequency_changes is False
    assert preview.changes == ()
    assert len(preview.entries) > 0
    assert all(not entry.has_change for entry in preview.entries)


def test_preview_configure_detects_frequency_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given changed LO frequency, preview should mark frequency changes."""
    preview = _preview(
        monkeypatch,
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


def test_preview_configure_before_uses_fetched_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given changed CNCO frequency, preview before should use fetched hardware state."""
    preview = _preview(
        monkeypatch,
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


def test_preview_configure_detects_non_frequency_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given changed RF switch, preview should not mark frequency changes."""
    preview = _preview(
        monkeypatch,
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


def test_preview_configure_uses_effective_r8_generator_port_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given R8 non-mixer CTRL port VATT, preview should not report backend-ignored values."""
    preview = _preview(
        monkeypatch,
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


def test_preview_configure_ignores_unspecified_fnco(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given planned FNCO is unspecified, preview should not show zero-to-blank changes."""
    preview = _preview(
        monkeypatch,
        experiment_system=_make_system(fnco_freq=None),
        backend_settings=_backend_settings(fnco_freq=0),
    )

    assert preview.changes == ()


def test_preview_configure_marks_missing_fetch_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given missing hardware fetch result, preview should be incomplete."""
    preview = _preview(
        monkeypatch,
        experiment_system=_make_system(),
        backend_settings={},
    )

    assert preview.is_complete is False
    assert preview.missing_box_ids == ("A",)
    assert preview.changes == ()


def test_preview_configure_rejects_unsupported_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given QuEL-3 backend, preview should fail until a provider exists."""
    with pytest.raises(NotImplementedError, match="QuEL-3"):
        _preview(
            monkeypatch,
            experiment_system=_make_system(),
            backend_settings={},
            backend_kind=BACKEND_KIND_QUEL3,
        )


def test_configure_preview_prints_summary_and_full_tables() -> None:
    """Given preview entries, summary should show changes while full shows all entries."""
    preview = ConfigurePreview(
        backend_kind=BACKEND_KIND_QUEL1,
        box_ids=("A",),
        mode="ge-cr-cr",
        entries=(
            ConfigureStateChange(
                box_id="A",
                component="port 1",
                field="lo_freq",
                before=10_000_000_000,
                after=11_000_000_000,
                unit="Hz",
                is_frequency=True,
            ),
            ConfigureStateChange(
                box_id="A",
                component="port 1",
                field="rfswitch",
                before="pass",
                after="pass",
            ),
        ),
    )
    summary_io = StringIO()
    full_io = StringIO()

    preview.print_summary(Console(file=summary_io, force_terminal=False, width=120))
    preview.print_full(Console(file=full_io, force_terminal=False, width=120))

    summary = summary_io.getvalue()
    full = full_io.getvalue()
    assert "Configure Preview Changes" in summary
    assert "A" in summary
    assert "lo_freq" in summary
    assert "rfswitch" not in summary
    assert "Configure Preview Full" in full
    assert "CHANGE" in full
    assert "lo_freq" in full
    assert "rfswitch" in full
    assert "yes" in full
    assert "no" in full
