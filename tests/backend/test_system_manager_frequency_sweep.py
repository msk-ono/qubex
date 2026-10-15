"""Frequency sweep hardware preparation contracts."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qubex.system.system_manager import SystemManager


@pytest.fixture
def manager(monkeypatch):
    """Provide isolated hardware configuration and restoration recording."""
    instance = object.__new__(SystemManager)
    target = SimpleNamespace(
        frequency=6.1,
        fine_frequency=6.0,
        sideband="U",
        is_read=True,
        channel=SimpleNamespace(port=SimpleNamespace(box_id="BOX0")),
    )
    backend = SimpleNamespace(initialize_awg_and_capunits=Mock())
    synchronizer = SimpleNamespace(supports_mutable_backend_settings_cache=True)
    box = SimpleNamespace(
        traits=SimpleNamespace(
            readout_ssb="U",
            readout_cnco_center=1_500_000_000,
            ctrl_ssb="L",
        )
    )
    instance.__dict__.update(
        _backend_controller=backend,
        _system_synchronizer=synchronizer,
        _experiment_system=SimpleNamespace(
            get_target=lambda _: target,
            resolve_qubit_label=lambda _: "Q00",
            get_readout_box_for_qubit=lambda _: box,
            get_control_box_for_qubit=lambda _: box,
        ),
    )
    events = []

    @contextmanager
    def settings(**kwargs):
        events.append(("enter", kwargs))
        try:
            yield
        finally:
            events.append(("restore", kwargs))

    monkeypatch.setattr(instance, "modified_backend_settings", settings)
    mixing = Mock(return_value=(8_000_000_000, 1_500_000_000, 0))
    monkeypatch.setattr("qubex.system.system_manager.MixingUtil.calc_lo_cnco", mixing)
    return instance, target, backend, synchronizer, events, mixing


@pytest.mark.parametrize("is_read", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_sweep_settings_restore_after_measurement(manager, is_read, fail):
    """Sweep settings are restored after successful and failed measurements."""
    instance, target, backend, _, events, mixing = manager
    target.is_read = is_read
    label = "R00" if is_read else "Q00"

    def measure():
        with instance.prepare_frequency_sweep(label, center_frequency=6.5) as frequency:
            assert frequency == 6.5
            assert [event[0] for event in events] == ["enter"]
            if fail:
                raise RuntimeError("measurement failed")

    if fail:
        with pytest.raises(RuntimeError, match="measurement failed"):
            measure()
    else:
        measure()
    assert [event[0] for event in events] == ["enter", "restore"]
    assert events[0][1]["label"] == label
    mixing.assert_called_once_with(
        6.5e9,
        ssb="U" if is_read else "L",
        cnco_center=1_500_000_000 if is_read else 2_250_000_000,
    )
    backend.initialize_awg_and_capunits.assert_not_called()


@pytest.mark.parametrize("frequency", [None, 6.1])
def test_nearby_frequency_resets_once_without_retuning(manager, frequency):
    """A nearby electrical-delay sweep resets once without changing oscillators."""
    instance, _, backend, _, events, mixing = manager
    with instance.prepare_frequency_sweep(
        "R00",
        center_frequency=frequency,
        retune_threshold=0.2,
    ) as resolved:
        assert resolved == (6.0 if frequency is None else frequency)
    backend.initialize_awg_and_capunits.assert_called_once_with(["BOX0"])
    assert events == []
    mixing.assert_not_called()


@pytest.mark.parametrize("frequency", [None, 9.0])
def test_direct_frequency_sweep_needs_no_mutable_configuration(
    manager, monkeypatch, frequency
):
    """Direct-frequency execution neither retunes nor resets nor asks for confirmation."""
    instance, target, backend, synchronizer, events, mixing = manager
    synchronizer.supports_mutable_backend_settings_cache = False
    del target.fine_frequency
    ask = Mock(side_effect=AssertionError("unexpected confirmation"))
    monkeypatch.setattr("qubex.system.system_manager.Confirm.ask", ask)
    with instance.prepare_frequency_sweep(
        "R00",
        center_frequency=frequency,
        retune_threshold=0.2,
        confirm=True,
    ) as resolved:
        assert resolved == (6.1 if frequency is None else frequency)
    assert events == []
    mixing.assert_not_called()
    backend.initialize_awg_and_capunits.assert_not_called()


def test_declined_retune_leaves_hardware_unchanged(manager, monkeypatch):
    """Declining a retune cancels preparation before touching hardware."""
    instance, _, backend, _, events, mixing = manager
    ask = Mock(return_value=False)
    monkeypatch.setattr("qubex.system.system_manager.Confirm.ask", ask)
    with instance.prepare_frequency_sweep(
        "R00", center_frequency=9.0, confirm=True
    ) as resolved:
        assert resolved is None
    ask.assert_called_once()
    assert events == []
    mixing.assert_not_called()
    backend.initialize_awg_and_capunits.assert_not_called()
