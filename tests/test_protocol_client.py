"""Tests for ProtocolClient, ProtocolRun, and the top-level run_protocol() function.

HTTP calls are mocked entirely via unittest.mock.patch on httpx.Client so no real
network requests are made. Each _get / _post / _delete helper opens httpx.Client as
a context manager, so the mock hierarchy is:

    mock_client_cls.return_value.__enter__.return_value  →  mock_http (the "session")
    mock_http.get.return_value   / .post.return_value / .delete.return_value
        .status_code, .is_success, .text, .json(), .content

Tests are grouped into three classes:
    TestProtocolClient  — auth + list / get / submit / get_run / run_and_wait
    TestProtocolRun     — wait / progress / results / download_files / cancel / refresh
    TestTopLevel        — biolmai.run_protocol()
"""

import io
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from biolmai.protocol_runs import (
    ProtocolClient,
    ProtocolNotFoundError,
    ProtocolRun,
    ProtocolRunError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = "ALY_abc123"
_SLUG = "my-proto"
_TOKEN = "test-token-xyz"

# Minimal ProtocolRun data dict
_RUN_DATA = {
    "run_id": _RUN_ID,
    "protocol_slug": _SLUG,
    "protocol_version": 1,
    "status": "scheduled",
}


def _make_client(token: str = _TOKEN, base_url: str = "https://biolm.ai") -> ProtocolClient:
    """Return a ProtocolClient with a known token, no env dependency."""
    return ProtocolClient(api_key=token, base_url=base_url)


def _mock_response(
    status_code: int = 200,
    json_data: object = None,
    text: str = "",
    content: bytes = b"",
) -> MagicMock:
    """Build a fake httpx.Response-like mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.text = text or (str(json_data) if json_data is not None else "")
    resp.json.return_value = json_data if json_data is not None else {}
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def _patch_httpx_client(method: str, response: MagicMock):
    """Context-manager factory: patches httpx.Client so that ``method`` (get/post/delete)
    returns *response*.  Returns the patch object (use as context manager)."""
    mock_cls = MagicMock()
    mock_http = MagicMock()
    mock_cls.return_value.__enter__.return_value = mock_http
    mock_cls.return_value.__exit__.return_value = False
    getattr(mock_http, method).return_value = response
    return mock_cls, mock_http


# ---------------------------------------------------------------------------
# TestProtocolClient
# ---------------------------------------------------------------------------


class TestProtocolClient:
    """Tests for ProtocolClient auth, list, get, submit, get_run, run_and_wait."""

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def test_no_token_raises(self, monkeypatch):
        """ProtocolClient with no token source raises ValueError with helpful message."""
        monkeypatch.delenv("BIOLMAI_TOKEN", raising=False)
        monkeypatch.delenv("BIOLM_TOKEN", raising=False)
        client = ProtocolClient()  # no api_key= kwarg

        mock_cls, mock_http = _patch_httpx_client("get", _mock_response(200, {}))
        with patch("httpx.Client", mock_cls):
            with pytest.raises(ValueError, match="BIOLMAI_TOKEN"):
                client._get("")

    def test_token_from_env(self, monkeypatch):
        """BIOLMAI_TOKEN env var is used in the Authorization header."""
        monkeypatch.setenv("BIOLMAI_TOKEN", "env-token-123")
        client = ProtocolClient()

        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)
        with patch("httpx.Client", mock_cls):
            client._get("")

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Token env-token-123"

    def test_token_from_arg(self, monkeypatch):
        """api_key= kwarg takes precedence over BIOLMAI_TOKEN env var."""
        monkeypatch.setenv("BIOLMAI_TOKEN", "env-token-999")
        client = ProtocolClient(api_key="kwarg-token-456")

        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)
        with patch("httpx.Client", mock_cls):
            client._get("")

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Token kwarg-token-456"

    # ------------------------------------------------------------------
    # list()
    # ------------------------------------------------------------------

    def test_list_success(self):
        """list() returns paginated dict from the GET response."""
        payload = {"count": 2, "next": None, "previous": None, "results": [{"slug": "a"}, {"slug": "b"}]}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = client.list()

        assert result["count"] == 2
        assert len(result["results"]) == 2

    def test_list_search_param(self):
        """list(search='antibody') includes ?search=antibody in the request params."""
        client = _make_client()
        resp = _mock_response(200, {"count": 0, "results": []})
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            client.list(search="antibody")

        call_kwargs = mock_http.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params.get("search") == "antibody"

    def test_list_http_error(self):
        """list() raises ProtocolRunError when the server returns 500."""
        client = _make_client()
        resp = _mock_response(500, text="internal server error")
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="500"):
                client.list()

    # ------------------------------------------------------------------
    # get()
    # ------------------------------------------------------------------

    def test_get_success(self):
        """get() returns protocol detail with inputs_schema present."""
        payload = {
            "slug": _SLUG,
            "version": 1,
            "inputs_schema": {"sequence": {"type": "text", "required": True}},
        }
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = client.get(_SLUG)

        assert "inputs_schema" in result
        assert result["slug"] == _SLUG

    def test_get_with_version(self):
        """get(slug, version=2) passes ?version=2 in the request params."""
        client = _make_client()
        resp = _mock_response(200, {"slug": _SLUG, "version": 2, "inputs_schema": {}})
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            client.get(_SLUG, version=2)

        call_kwargs = mock_http.get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params.get("version") == 2

    def test_get_404(self):
        """get() raises ProtocolNotFoundError on 404."""
        client = _make_client()
        resp = _mock_response(404, text="not found")
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolNotFoundError):
                client.get("nonexistent-slug")

    # ------------------------------------------------------------------
    # submit()
    # ------------------------------------------------------------------

    def test_submit_returns_protocol_run(self):
        """submit() returns a ProtocolRun with run_id, protocol_slug, and status."""
        payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("post", resp)

        with patch("httpx.Client", mock_cls):
            run = client.submit(_SLUG, inputs={"sequence": "MKTAY"})

        assert isinstance(run, ProtocolRun)
        assert run.run_id == _RUN_ID
        assert run.protocol_slug == _SLUG
        assert run.status == "scheduled"

    def test_submit_json_body(self):
        """submit() sends JSON body containing 'inputs' (and optional run_name)."""
        import json as _json

        payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, mock_http = _patch_httpx_client("post", resp)

        with patch("httpx.Client", mock_cls):
            client.submit(_SLUG, inputs={"sequence": "MKTAY"}, run_name="my-run")

        call_kwargs = mock_http.post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
        assert "inputs" in sent_json
        assert sent_json["inputs"] == {"sequence": "MKTAY"}
        assert sent_json.get("run_name") == "my-run"

    def test_submit_with_files(self):
        """submit(files=...) sends a multipart request (files= kwarg, not json=)."""
        payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, mock_http = _patch_httpx_client("post", resp)

        fake_file = io.BytesIO(b"ATOM 1 ...")
        with patch("httpx.Client", mock_cls):
            client.submit(_SLUG, inputs={"sequence": "MKTAY"}, files={"pdb_file": fake_file})

        call_kwargs = mock_http.post.call_args
        # Multipart uses files= not json=
        assert call_kwargs.kwargs.get("json") is None
        assert call_kwargs.kwargs.get("files") is not None or call_kwargs[1].get("files") is not None

    def test_submit_error_response(self):
        """submit() raises ProtocolRunError when the server returns 400."""
        client = _make_client()
        resp = _mock_response(400, text="invalid inputs")
        mock_cls, _ = _patch_httpx_client("post", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="400"):
                client.submit(_SLUG, inputs={"bad": "field"})

    # ------------------------------------------------------------------
    # get_run()
    # ------------------------------------------------------------------

    def test_get_run_reconnects(self):
        """get_run() returns a ProtocolRun with correct run_id and status."""
        payload = {
            "run_id": _RUN_ID,
            "protocol_slug": _SLUG,
            "protocol_version": 2,
            "status": "succeeded",
        }
        client = _make_client()
        resp = _mock_response(200, payload)
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            run = client.get_run(_RUN_ID)

        assert isinstance(run, ProtocolRun)
        assert run.run_id == _RUN_ID
        assert run.status == "succeeded"

    # ------------------------------------------------------------------
    # run_and_wait()
    # ------------------------------------------------------------------

    def test_run_and_wait_end_to_end(self):
        """run_and_wait() submits, polls progress until succeeded, then returns results."""
        client = _make_client()

        submit_payload = {"run_id": _RUN_ID, "protocol_slug": _SLUG, "status": "scheduled"}
        results_payload = {
            "run_id": _RUN_ID,
            "status": "succeeded",
            "results": {"designed_sequences": ["MKTAY"]},
        }

        # We mock at the ProtocolRun level to avoid multi-layer HTTP mock complexity
        with patch.object(ProtocolClient, "submit") as mock_submit:
            mock_run = MagicMock(spec=ProtocolRun)
            mock_run.wait.return_value = mock_run
            mock_run.results.return_value = results_payload
            mock_submit.return_value = mock_run

            result = client.run_and_wait(_SLUG, inputs={"sequence": "MKTAY"}, show_progress=False)

        mock_submit.assert_called_once_with(_SLUG, {"sequence": "MKTAY"}, run_name=None)
        mock_run.wait.assert_called_once()
        mock_run.results.assert_called_once()
        assert result == results_payload.get("results", {})


# ---------------------------------------------------------------------------
# TestProtocolRun
# ---------------------------------------------------------------------------


class TestProtocolRun:
    """Tests for ProtocolRun methods: wait, progress, results, download_files, cancel, refresh."""

    def _make_run(self, status: str = "scheduled") -> tuple[ProtocolRun, ProtocolClient]:
        client = _make_client()
        data = dict(_RUN_DATA, status=status)
        run = ProtocolRun(data, client=client)
        return run, client

    # ------------------------------------------------------------------
    # progress()
    # ------------------------------------------------------------------

    def test_progress_returns_dict(self):
        """progress() calls GET /runs/{run_id}/progress/ and returns the dict."""
        run, client = self._make_run()
        progress_payload = {
            "status": "running",
            "progress_pct": 42,
            "tasks": [],
            "log_lines": ["starting..."],
            "result_row_count": None,
        }
        resp = _mock_response(200, progress_payload)
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = run.progress()

        assert result["status"] == "running"
        assert result["progress_pct"] == 42
        # URL should contain 'progress'
        called_url = mock_http.get.call_args[0][0]
        assert "progress" in called_url

    # ------------------------------------------------------------------
    # results()
    # ------------------------------------------------------------------

    def test_results_returns_detail(self):
        """results() calls GET /runs/{run_id}/ and returns the full detail dict."""
        run, client = self._make_run(status="succeeded")
        detail_payload = {
            "run_id": _RUN_ID,
            "status": "succeeded",
            "results": {"output": "data"},
            "failure_error": None,
        }
        resp = _mock_response(200, detail_payload)
        mock_cls, mock_http = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            result = run.results()

        assert result["run_id"] == _RUN_ID
        assert result["results"] == {"output": "data"}
        # URL must NOT contain 'progress' — it's the plain run detail endpoint
        called_url = mock_http.get.call_args[0][0]
        assert "progress" not in called_url

    # ------------------------------------------------------------------
    # refresh()
    # ------------------------------------------------------------------

    def test_refresh_updates_status(self):
        """refresh() fetches run detail and updates self.status in place."""
        run, _ = self._make_run(status="scheduled")
        assert run.status == "scheduled"

        resp = _mock_response(200, {"run_id": _RUN_ID, "status": "running"})
        mock_cls, _ = _patch_httpx_client("get", resp)

        with patch("httpx.Client", mock_cls):
            returned = run.refresh()

        assert run.status == "running"
        assert returned is run  # returns self

    # ------------------------------------------------------------------
    # cancel()
    # ------------------------------------------------------------------

    def test_cancel_sets_status(self):
        """cancel() calls DELETE and sets run.status = 'cancelled'."""
        run, _ = self._make_run(status="running")
        resp = _mock_response(200, {"detail": "cancelled"})
        mock_cls, mock_http = _patch_httpx_client("delete", resp)

        with patch("httpx.Client", mock_cls):
            run.cancel()

        assert run.status == "cancelled"
        assert mock_http.delete.called

    def test_cancel_http_error(self):
        """cancel() raises ProtocolRunError when the server returns 400."""
        run, _ = self._make_run(status="succeeded")
        resp = _mock_response(400, text="already in terminal state")
        mock_cls, _ = _patch_httpx_client("delete", resp)

        with patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="400"):
                run.cancel()

    # ------------------------------------------------------------------
    # wait()
    # ------------------------------------------------------------------

    def test_wait_polls_until_succeeded(self):
        """wait() keeps polling progress() until status is 'succeeded', then returns self."""
        run, _ = self._make_run()

        progress_calls = [
            {"status": "running", "progress_pct": 50, "tasks": []},
            {"status": "succeeded", "progress_pct": 100, "tasks": []},
        ]

        with patch.object(run, "progress", side_effect=progress_calls), \
             patch("time.sleep"):
            result = run.wait(poll_interval=0.001, show_progress=False)

        assert result is run
        assert run.status == "succeeded"

    def test_wait_raises_on_failed(self):
        """wait() raises ProtocolRunError when the run transitions to 'failed'."""
        run, _ = self._make_run()

        progress_snap = {"status": "failed", "progress_pct": None, "tasks": []}
        detail_resp = _mock_response(200, {"run_id": _RUN_ID, "status": "failed", "failure_error": "OOM"})
        mock_cls, _ = _patch_httpx_client("get", detail_resp)

        with patch.object(run, "progress", return_value=progress_snap), \
             patch("time.sleep"), \
             patch("httpx.Client", mock_cls):
            with pytest.raises(ProtocolRunError, match="failed"):
                run.wait(poll_interval=0.001, show_progress=False)

    def test_wait_raises_on_cancelled(self):
        """wait() raises ProtocolRunError when the run is cancelled."""
        run, _ = self._make_run()

        progress_snap = {"status": "cancelled", "progress_pct": None, "tasks": []}

        with patch.object(run, "progress", return_value=progress_snap), \
             patch("time.sleep"):
            with pytest.raises(ProtocolRunError, match="cancelled"):
                run.wait(poll_interval=0.001, show_progress=False)

    def test_wait_timeout(self):
        """wait() raises TimeoutError when the deadline is reached before completion."""
        run, _ = self._make_run()

        # Always return 'running'
        progress_snap = {"status": "running", "progress_pct": 10, "tasks": []}

        # Use a real short timeout; patch time.sleep so the loop doesn't actually wait,
        # but time.monotonic() must advance naturally for the deadline to trigger.
        call_count = {"n": 0}

        def fake_sleep(_):
            call_count["n"] += 1
            if call_count["n"] > 50:
                # Safety valve: force time forward via monkeypatching if loop is stuck
                raise RuntimeError("sleep called too many times in test")

        with patch.object(run, "progress", return_value=progress_snap), \
             patch("time.sleep", fake_sleep):
            with pytest.raises(TimeoutError):
                run.wait(poll_interval=0.0, timeout=0.0, show_progress=False)

    def test_wait_no_progress_output(self, capsys):
        """wait(show_progress=False) prints nothing to stdout."""
        run, _ = self._make_run()

        progress_calls = [
            {"status": "running", "progress_pct": 30, "tasks": []},
            {"status": "succeeded", "progress_pct": 100, "tasks": []},
        ]

        with patch.object(run, "progress", side_effect=progress_calls), \
             patch("time.sleep"):
            run.wait(poll_interval=0.001, show_progress=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_wait_prints_progress(self, capsys):
        """wait(show_progress=True) prints the run_id and progress percentage."""
        run, _ = self._make_run()

        progress_calls = [
            {"status": "running", "progress_pct": 25, "tasks": [{"name": "step1", "status": "running"}]},
            {"status": "succeeded", "progress_pct": 100, "tasks": []},
        ]

        with patch.object(run, "progress", side_effect=progress_calls), \
             patch("time.sleep"):
            run.wait(poll_interval=0.001, show_progress=True)

        captured = capsys.readouterr()
        assert _RUN_ID in captured.out
        assert "25" in captured.out

    # ------------------------------------------------------------------
    # download_files()
    # ------------------------------------------------------------------

    def test_download_files_fetches_and_saves(self, tmp_path):
        """download_files() fetches presigned URLs and writes files to tmp_path."""
        run, _ = self._make_run(status="succeeded")

        url_payload = {
            "urls": {
                "1": {
                    "designed_pdb": {"url": "https://storage.example.com/row1_pdb.pdb"}
                }
            }
        }

        # First GET: download endpoint → url_payload
        # Second GET: presigned URL → file content
        url_resp = _mock_response(200, url_payload)
        file_content = b"ATOM 1 CA ALA A 1 ..."
        file_resp = _mock_response(200, content=file_content)

        mock_cls = MagicMock()
        mock_http = MagicMock()
        mock_cls.return_value.__enter__.return_value = mock_http
        mock_cls.return_value.__exit__.return_value = False

        # _get() opens one httpx.Client; download_files() opens another httpx.Client
        # Both are patched via the same mock_cls but we need sequential return values.
        mock_http.get.side_effect = [url_resp, file_resp]

        with patch("httpx.Client", mock_cls):
            paths = run.download_files(output_dir=str(tmp_path), columns=["designed_pdb"])

        assert len(paths) == 1
        dest = tmp_path / "1_designed_pdb.pdb"
        assert dest.exists()
        assert dest.read_bytes() == file_content

    def test_download_files_skips_existing(self, tmp_path):
        """download_files(overwrite=False) does not re-download an existing file."""
        run, _ = self._make_run(status="succeeded")

        # Pre-create the file that would be downloaded
        existing = tmp_path / "1_designed_pdb.pdb"
        existing.write_bytes(b"already-here")

        url_payload = {
            "urls": {
                "1": {
                    "designed_pdb": {"url": "https://storage.example.com/row1_pdb.pdb"}
                }
            }
        }
        url_resp = _mock_response(200, url_payload)

        mock_cls = MagicMock()
        mock_http = MagicMock()
        mock_cls.return_value.__enter__.return_value = mock_http
        mock_cls.return_value.__exit__.return_value = False
        mock_http.get.return_value = url_resp

        with patch("httpx.Client", mock_cls):
            paths = run.download_files(output_dir=str(tmp_path), overwrite=False)

        # File not re-downloaded: get called once (for URL list) not twice
        assert mock_http.get.call_count == 1
        assert existing in paths
        assert existing.read_bytes() == b"already-here"

    def test_download_files_empty_urls(self, tmp_path):
        """download_files() returns [] when the API returns an empty urls dict."""
        run, _ = self._make_run(status="succeeded")

        url_payload = {"urls": {}}
        url_resp = _mock_response(200, url_payload)

        mock_cls = MagicMock()
        mock_http = MagicMock()
        mock_cls.return_value.__enter__.return_value = mock_http
        mock_cls.return_value.__exit__.return_value = False
        mock_http.get.return_value = url_resp

        with patch("httpx.Client", mock_cls):
            paths = run.download_files(output_dir=str(tmp_path))

        assert paths == []


# ---------------------------------------------------------------------------
# TestTopLevel
# ---------------------------------------------------------------------------


class TestTopLevel:
    """Tests for the top-level biolmai.run_protocol() convenience function."""

    def test_top_level_convenience(self, monkeypatch):
        """run_protocol() builds a ProtocolClient, calls run_and_wait, returns results."""
        import biolmai

        monkeypatch.setenv("BIOLMAI_TOKEN", _TOKEN)

        expected_results = {"designed_sequences": ["MKTAY", "MKLAY"]}

        with patch.object(ProtocolClient, "run_and_wait", return_value=expected_results) as mock_raw:
            result = biolmai.run_protocol(
                _SLUG,
                inputs={"sequence": "MKTAY"},
                run_name="top-level-test",
                api_key=_TOKEN,
                show_progress=False,
            )

        assert result == expected_results
        mock_raw.assert_called_once_with(
            _SLUG,
            {"sequence": "MKTAY"},
            run_name="top-level-test",
            poll_interval=5.0,
            timeout=3600.0,
            show_progress=False,
        )
