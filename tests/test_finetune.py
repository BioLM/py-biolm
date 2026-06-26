"""Unit tests for the Finetune SDK client.

These tests exercise payload construction, endpoint routing, error mapping, and
the wait() polling loop without any network access — the async HTTP helpers are
monkeypatched to capture calls. (Live behavior is covered by the biolm_web API
tests against the Modal QA deploys.)
"""
from types import SimpleNamespace

import pytest

import biolmai.finetune as ft
from biolmai import Finetune


@pytest.fixture
def captured(monkeypatch):
    """Capture calls to the module-level HTTP helpers; return canned responses."""
    calls = []

    async def fake_post(endpoint, payload, api_key, base_url, action):
        calls.append(("POST", endpoint, payload))
        return {"run_id": "ALY_test", "dag": "x", "status": "scheduled"}

    async def fake_get(endpoint, api_key, base_url, action):
        calls.append(("GET", endpoint, None))
        return {"run_id": "ALY_test", "status": "scheduled"}

    async def fake_delete(endpoint, api_key, base_url, action):
        calls.append(("DELETE", endpoint, None))
        return {"run_id": "ALY_test", "status": "cancelled"}

    monkeypatch.setattr(ft, "_post", fake_post)
    monkeypatch.setattr(ft, "_get", fake_get)
    monkeypatch.setattr(ft, "_delete", fake_delete)
    return calls


# -- endpoint routing -------------------------------------------------------


def test_xgboost_routes_and_builds_payload(captured):
    Finetune.xgboost(
        train_data=[{"sequence": "EVQLVESGGG", "label": 0}],
        embedding_models=["esm2-8m"],
        task_type="classification",
        hyperopt=True,
        hyperopt_n_trials=10,
    )
    method, endpoint, payload = captured[0]
    assert (method, endpoint) == ("POST", "xgboost/runs/")
    assert payload["embedding_models"] == ["esm2-8m"]
    assert payload["hyperopt"] is True
    assert payload["hyperopt_n_trials"] == 10
    assert payload["train_data"] == [{"sequence": "EVQLVESGGG", "label": 0}]


def test_dsm_stage1_routes(captured):
    Finetune.dsm_stage1(train_data=[{"sequence": "AAA"}], max_steps=5)
    method, endpoint, payload = captured[0]
    assert (method, endpoint) == ("POST", "dsm/stage1/runs/")
    assert payload["max_steps"] == 5
    assert payload["sequence_col"] == "sequence"


def test_dsm_stage2_routes(captured):
    Finetune.dsm_stage2(
        stage1_checkpoint="finetune/dsm-mab/u/1/ckpt",
        paired_data=[{"heavy": "AAA", "light": "BBB"}],
    )
    method, endpoint, payload = captured[0]
    assert (method, endpoint) == ("POST", "dsm/stage2/runs/")
    assert payload["stage1_checkpoint"] == "finetune/dsm-mab/u/1/ckpt"
    assert payload["heavy_col"] == "heavy"


def test_dsm_rl_routes(captured):
    Finetune.dsm_rl(seed_sequences=["EVQLVESGGG"], num_episodes=2, oracle_type="esmc")
    method, endpoint, payload = captured[0]
    assert (method, endpoint) == ("POST", "dsm/rl/runs/")
    assert payload["seed_sequences"] == ["EVQLVESGGG"]
    assert payload["num_episodes"] == 2


def test_get_progress_cancel_routes(captured):
    Finetune.get_run("ALY_1")
    Finetune.progress("ALY_1")
    Finetune.cancel("ALY_1")
    assert captured[0] == ("GET", "runs/ALY_1/", None)
    assert captured[1] == ("GET", "runs/ALY_1/progress/", None)
    assert captured[2] == ("DELETE", "runs/ALY_1/", None)


def test_list_runs_query_params(captured):
    Finetune.list_runs(dag="finetune_xgboost", status="running", page=2, page_size=50)
    method, endpoint, _ = captured[0]
    assert method == "GET"
    assert endpoint.startswith("runs/?")
    assert "dag=finetune_xgboost" in endpoint
    assert "status=running" in endpoint
    assert "page=2" in endpoint
    assert "page_size=50" in endpoint


# -- payload hygiene --------------------------------------------------------


def test_drop_none_omits_unset_optionals(captured):
    # environment_id / test_data left unset → must not appear in payload so the
    # server applies its own defaults.
    Finetune.xgboost(train_data=[{"sequence": "A", "label": 1}], embedding_models=["esm2-8m"])
    _, _, payload = captured[0]
    assert "environment_id" not in payload
    assert "test_data" not in payload
    assert "validation_data" not in payload


def test_drop_none_helper():
    assert ft._drop_none({"a": 1, "b": None, "c": 0}) == {"a": 1, "c": 0}


# -- error mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    "code,exc",
    [
        (401, PermissionError),
        (402, PermissionError),
        (404, FileNotFoundError),
        (400, ValueError),
        (500, ValueError),
    ],
)
def test_raise_for_status_maps_errors(code, exc):
    resp = SimpleNamespace(
        status_code=code,
        text="boom",
        json=lambda: {"error": "boom"},
    )
    with pytest.raises(exc):
        ft._raise_for_status(resp, "do thing")


def test_raise_for_status_ok_passes():
    resp = SimpleNamespace(status_code=201, text="", json=lambda: {})
    assert ft._raise_for_status(resp, "do thing") is None


# -- wait() polling ---------------------------------------------------------


def test_wait_returns_detail_when_terminal(monkeypatch):
    statuses = iter(["running", "running", "succeeded"])

    monkeypatch.setattr(
        Finetune, "progress", classmethod(lambda cls, rid, **kw: {"status": next(statuses)})
    )
    monkeypatch.setattr(
        Finetune,
        "get_run",
        classmethod(lambda cls, rid, **kw: {"run_id": rid, "status": "succeeded", "results": {}}),
    )
    monkeypatch.setattr(ft.time, "sleep", lambda *_: None)

    out = Finetune.wait("ALY_1", poll_interval=0)
    assert out["status"] == "succeeded"
    assert out["run_id"] == "ALY_1"


def test_wait_times_out(monkeypatch):
    monkeypatch.setattr(
        Finetune, "progress", classmethod(lambda cls, rid, **kw: {"status": "running"})
    )
    monkeypatch.setattr(ft.time, "sleep", lambda *_: None)
    # monotonic jumps past the deadline on the second read.
    ticks = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(ft.time, "monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError):
        Finetune.wait("ALY_1", poll_interval=0, timeout=10)
