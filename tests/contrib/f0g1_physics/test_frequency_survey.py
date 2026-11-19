"""Regress wideband discovery for the adaptive F0G1 experiment."""

import asyncio

import numpy as np
import pytest

from qubex.contrib.experiment._f0g1_physics import survey_transfer


@pytest.mark.parametrize("remote", [1799.0, 1957.0])
def test_distant_peak_survives_local_maximum(remote):
    """A stronger disconnected line should reach independent fine confirmation on either side."""
    calls = []

    async def measure(f, shots):
        calls.append((f, shots))
        return (
            0.15
            + 0.42 * np.exp(-(((f - 1842) / 4) ** 2))
            + 0.62 * np.exp(-(((f - remote) / 5) ** 2))
        )

    result = asyncio.run(survey_transfer(measure, [1857, 1900], [8, 22], [1842], 0.15))
    assert len(result["confirmed"]) == 1
    assert result["confirmed"][0]["frequency"] == pytest.approx(remote, abs=1.5)
    assert [n for _, n in calls] == [64, 192]
    assert max(map(len, (f for f, _ in calls))) <= 65


def test_survey_only_known_peak():
    """A resolved known line should incur just the coarse survey."""
    calls = []

    async def measure(f, shots):
        calls.append((f, shots))
        return 0.1 + 0.8 * np.exp(-(((f - 1900) / 5) ** 2))

    result = asyncio.run(survey_transfer(measure, [1857, 1900], [8, 22], [1900], 0.1))
    assert result["confirmed"] == []
    assert len(calls) == 1


def test_noisy_survey_does_not_promote_peak():
    """A one-shot-batch fluctuation should disappear on independent confirmation."""

    async def measure(f, shots):
        if shots == 64:
            return 0.2 + 0.5 * np.exp(-(((f - 1799) / 5) ** 2))
        return np.full_like(f, 0.2)

    result = asyncio.run(survey_transfer(measure, [1857, 1900], [8, 22], [], 0.2))
    assert result["confirmed"] == []


def test_survey_expands_active_edge():
    """An edge rising towards a distant line should trigger another frequency window."""
    calls = []

    async def measure(f, shots):
        if not calls:
            calls.append(f.min())
        return 0.08 + 0.65 * np.exp(-(((f - (calls[0] - 4)) / 7) ** 2))

    result = asyncio.run(survey_transfer(measure, [1857, 1900], [8, 22], [], 0.08))
    assert result["frequencies"][0] < calls[0] - 10
    assert result["confirmed"][0]["frequency"] == pytest.approx(calls[0] - 4, abs=1.5)
    assert not any(result["edges"])


def test_survey_line_enters_hardware_finalists():
    """A confirmed unassigned line should reach two-tone verification within the amplitude budget."""
    import ast
    import json
    from pathlib import Path
    from typing import Any
    from unittest.mock import AsyncMock

    path = Path(__file__).resolve().parents[3] / "f0g1_physics_adaptive.ipynb"
    cells = {
        c["id"]: "".join(c["source"]) for c in json.loads(path.read_text())["cells"]
    }
    probe_lines = AsyncMock(return_value=[dict(fe=3900.0, score=0.7)])
    measurement = AsyncMock(
        side_effect=lambda *args: np.tile([0.85, 0.1, 0.05], (len(args[-2]), 1))
    )
    ns: dict[str, Any] = dict(
        np=np,
        discoveries=[dict(A=0.95, fd=1799.0)],
        branches=[[], []],
        probe=0.02,
        ref=dict(fe=3950.0, rate=8.0),
        probe_lines=probe_lines,
        save_records=lambda *_: None,
        empirical=[],
        predictions=[],
        measure=measurement,
        decision_time=1000.0,
        shots_search=64,
        shots_fit=192,
        time_window=np.linspace(800, 1000, 9),
        dt=1,
        aligned=np.unique,
    )
    for name in ("survey-ef", "screen-finalists"):
        code = compile(cells[name], name, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        asyncio.run(eval(code, ns))  # noqa: S307
    assert {(r["A"], r["fd"]) for r in ns["results"]} == {(0.95, 1799.0)}
    assert all(r["A"] + r["b"] <= 1.0 for r in ns["results"])
    assert {args.args[0] for args in measurement.await_args_list} == set("gef")


def test_survey_rejects_reproducible_but_weak_remote_line():
    """A repeatable remote peak well below known transfer should not add an ef scan."""

    async def measure(f, shots):
        return (
            0.1
            + 0.75 * np.exp(-(((f - 1857) / 5) ** 2))
            + 0.35 * np.exp(-(((f - 1799) / 5) ** 2))
        )

    result = asyncio.run(survey_transfer(measure, [1857, 1900], [8, 22], [1857], 0.1))
    assert result["confirmed"] == []
