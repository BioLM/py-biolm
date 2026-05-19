"""Protocol Submission API client — programmatic run submission, progress tracking, and results retrieval.

Usage::

    from biolmai import ProtocolClient

    client = ProtocolClient()

    # Discover available protocols
    client.list(search="antibody")

    # Inspect what inputs a protocol expects before submitting
    schema = client.get("antibody-optimization")
    print(schema["inputs_schema"])

    # Submit a run
    run = client.submit(
        "antibody-optimization",
        inputs={"sequence": "MKTAYIAKQRQ", "n_rounds": 3},
        run_name="my first run",
    )

    # Option A — block until complete, then download results
    run.wait()
    path = run.download(output_dir="./results")  # saves ALY_xxx_results.csv.zip

    # Option B — one-liner (returns run detail dict)
    detail = client.run_and_wait("antibody-optimization", inputs={"sequence": "MKTAYIAKQRQ"})

    # Download the compressed results file
    path = run.download(output_dir="./results", file_type="jsonl")
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from biolmai.core.const import BIOLMAI_BASE_DOMAIN

_DEFAULT_TIMEOUT = 30
_UPLOAD_TIMEOUT = 120
_DOWNLOAD_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    token = api_key or os.environ.get("BIOLMAI_TOKEN") or os.environ.get("BIOLM_TOKEN")
    if not token:
        raise ValueError(
            "No API key found. Set the BIOLMAI_TOKEN environment variable or pass api_key= to ProtocolClient().\n"
            "Get a token at https://biolm.ai/console/user/api-keys/"
        )
    return {"Authorization": f"Token {token}"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ProtocolRunError(Exception):
    """A protocol run failed, was cancelled, or the API returned an error."""


class ProtocolNotFoundError(ProtocolRunError):
    """The requested protocol slug/version does not exist or is not accessible."""


# ---------------------------------------------------------------------------
# ProtocolRun
# ---------------------------------------------------------------------------


class ProtocolRun:
    """A submitted protocol run returned by :meth:`ProtocolClient.submit`.

    Attributes:
        run_id: The ``ALY_...`` run identifier.
        protocol_slug: The protocol slug this run belongs to.
        protocol_version: The protocol version used.
        status: Current status string (``scheduled`` / ``running`` / ``succeeded`` /
            ``failed`` / ``cancelled``).
    """

    def __init__(self, data: Dict[str, Any], client: "ProtocolClient") -> None:
        self.run_id: str = data["run_id"]
        self.protocol_slug: str = data.get("protocol_slug", "")
        self.protocol_version: int = data.get("protocol_version", 1)
        self.status: str = data.get("status", "scheduled")
        self._client = client

    def __repr__(self) -> str:
        return (
            f"ProtocolRun(run_id={self.run_id!r}, "
            f"slug={self.protocol_slug!r}, status={self.status!r})"
        )

    # ------------------------------------------------------------------
    # Status & progress
    # ------------------------------------------------------------------

    def refresh(self) -> "ProtocolRun":
        """Fetch current status from the API and update ``self.status`` in place.

        Returns:
            ``self`` for chaining.
        """
        data = self._client._get(f"runs/{self.run_id}/")
        self.status = data.get("status", self.status)
        return self

    def progress(self) -> Dict[str, Any]:
        """Return the current status and WebSocket channel for this run.

        For live progress events (log lines, task statuses, per-step updates),
        subscribe to the WebSocket channel returned in ``channel_id`` — the
        same channel the browser console uses.

        Returns:
            Dict with keys:

            - ``status`` — current run status
            - ``channel_id`` — WebSocket pub/sub channel for live events
              (e.g. ``telemetry_ALY_...``). Connect to ``/ws/`` and subscribe.
        """
        return self._client._get(f"runs/{self.run_id}/progress/")

    # ------------------------------------------------------------------
    # Blocking wait
    # ------------------------------------------------------------------

    def wait(
        self,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
        show_progress: bool = True,
    ) -> "ProtocolRun":
        """Block until this run reaches a terminal state.

        Args:
            poll_interval: Seconds between progress polls. Default ``5``.
            timeout: Maximum seconds to wait before raising
                :class:`TimeoutError`. Default ``3600`` (1 hour).
            show_progress: Print progress lines to stdout. Default ``True``.

        Returns:
            ``self`` for chaining (e.g. ``run.wait().results()``).

        Raises:
            :class:`ProtocolRunError`: If the run fails or is cancelled.
            :class:`TimeoutError`: If ``timeout`` seconds elapse before completion.

        Example::

            run = client.submit("my-protocol", inputs={"sequence": "MKLL..."})
            run.wait(poll_interval=10, timeout=7200)
            print(run.results())
        """
        deadline = time.monotonic() + timeout
        last_status: Optional[str] = None

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Protocol run {self.run_id} did not complete within {timeout:.0f}s. "
                    "Increase timeout= or poll manually with run.progress()."
                )

            try:
                snap = self.progress()
                self.status = snap.get("status", self.status)
            except Exception:
                # Network blip — fall back to detail endpoint for status
                try:
                    self.refresh()
                except Exception:
                    pass
                time.sleep(poll_interval)
                continue

            if show_progress and self.status != last_status:
                print(f"  [{self.run_id}] {self.status}")
                last_status = self.status

            if self.status in ("succeeded", "failed", "cancelled"):
                break

            time.sleep(poll_interval)

        if self.status == "failed":
            try:
                detail = self._client._get(f"runs/{self.run_id}/").get("failure_error") or "unknown error"
            except Exception:
                detail = "unknown error"
            raise ProtocolRunError(f"Protocol run {self.run_id} failed: {detail}")

        if self.status == "cancelled":
            raise ProtocolRunError(f"Protocol run {self.run_id} was cancelled.")

        if show_progress:
            print(f"  [{self.run_id}] complete ✓")

        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self) -> Dict[str, Any]:
        """Fetch the run detail (status, timestamps, failure info).

        The actual result data (CSV/JSONL rows) is in the download stream —
        use :meth:`download` to save the compressed results file locally.

        Returns:
            Dict with keys:

            - ``run_id``, ``protocol_slug``, ``protocol_version``
            - ``status``
            - ``results`` — mapped output fields (may be empty if no response_mapping)
            - ``failure_error`` — error message if failed, else ``None``
            - ``workflow_started_at``, ``workflow_ended_at``
        """
        return self._client._get(f"runs/{self.run_id}/")

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def download(
        self,
        output_dir: Union[str, Path] = ".",
        file_type: str = "csv",
        overwrite: bool = False,
    ) -> Path:
        """Download the compressed results file for this run.

        Streams the results zip (CSV or JSONL) directly from the server and
        saves it to ``output_dir``.

        Args:
            output_dir: Directory to save the file. Created if it does not exist.
                Default ``"."`` (current directory).
            file_type: ``"csv"`` (default) or ``"jsonl"``. Selects which
                compressed results file to download.
            overwrite: Re-download if the file already exists. Default ``False``.

        Returns:
            :class:`pathlib.Path` to the downloaded zip file.

        Raises:
            :class:`ProtocolRunError`: If the run has not yet succeeded or the
                download fails.

        Example::

            path = run.download(output_dir="./results")
            # path → PosixPath('./results/ALY_xxx_results.csv.zip')
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"{self.run_id}_results.{file_type}.zip"

        if dest.exists() and not overwrite:
            return dest

        url = self._client._url(f"runs/{self.run_id}/download/")
        params = {"file_type": file_type}
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT) as http:
            with http.stream(
                "GET", url, headers=self._client._headers(), params=params
            ) as resp:
                if resp.status_code == 400:
                    raise ProtocolRunError(
                        f"Run {self.run_id} is not complete — cannot download yet."
                    )
                if not resp.is_success:
                    raise ProtocolRunError(
                        f"Download failed: {resp.status_code} {resp.text[:200]}"
                    )
                with dest.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        return dest

    def download_files(
        self,
        output_dir: Union[str, Path] = ".",
        file_type: str = "csv",
        overwrite: bool = False,
    ) -> Path:
        """Alias for :meth:`download` — downloads the compressed results file.

        Args:
            output_dir: Directory to save the file. Default ``"."``.
            file_type: ``"csv"`` (default) or ``"jsonl"``.
            overwrite: Re-download if already exists. Default ``False``.

        Returns:
            :class:`pathlib.Path` to the downloaded zip file.
        """
        return self.download(output_dir=output_dir, file_type=file_type, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel this run if it is still ``scheduled`` or ``running``.

        Raises:
            :class:`ProtocolRunError`: If the API rejects the cancellation
                (e.g. run is already in a terminal state).
        """
        self._client._delete(f"runs/{self.run_id}/cancel/")
        self.status = "cancelled"


# ---------------------------------------------------------------------------
# ProtocolClient
# ---------------------------------------------------------------------------


class ProtocolClient:
    """Client for the BioLM Protocol Submission API.

    Provides a clean interface for discovering, submitting, and retrieving
    results from BioLM hosted protocols.

    Args:
        api_key: BioLM API token. Reads ``BIOLMAI_TOKEN`` env var if not provided.
        base_url: Override the API base domain (default: ``https://biolm.ai``).
            Useful for pointing at a local dev server::

                client = ProtocolClient(base_url="http://localhost:7777")

    Example::

        import biolmai

        client = biolmai.ProtocolClient()

        # Browse protocols
        page = client.list(search="antibody")
        for p in page["results"]:
            print(p["slug"], p["name"])

        # Inspect required inputs before submitting
        detail = client.get("antibody-optimization")
        print(detail["inputs_schema"])

        # Submit and wait (blocking)
        results = client.run_and_wait(
            "antibody-optimization",
            inputs={"sequence": "MKTAYIAKQRQ", "n_rounds": 3},
        )
        print(results)

        # Submit async, then download structure files
        run = client.submit("structure-prediction", inputs={"sequence": "MKLL..."})
        run.wait()
        run.download_files("./structures", columns=["pdb"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        domain = (base_url or BIOLMAI_BASE_DOMAIN).rstrip("/")
        self._base = f"{domain}/api/protocols"

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return _auth_headers(self._api_key)

    def _url(self, path: str) -> str:
        url = f"{self._base}/{path.lstrip('/')}"
        if not url.endswith("/"):
            url += "/"
        return url

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = self._url(path)
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as http:
            resp = http.get(url, headers=self._headers(), params=params)
        if resp.status_code == 404:
            raise ProtocolNotFoundError(f"Not found: {url}")
        if not resp.is_success:
            raise ProtocolRunError(
                f"GET {url} returned {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()

    def _post(
        self,
        path: str,
        json_body: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        url = self._url(path)
        with httpx.Client(timeout=_UPLOAD_TIMEOUT) as http:
            if files:
                resp = http.post(url, headers=self._headers(), files=files, data=data)
            else:
                resp = http.post(url, headers=self._headers(), json=json_body)
        if not resp.is_success:
            raise ProtocolRunError(
                f"POST {url} returned {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()

    def _delete(self, path: str) -> Dict[str, Any]:
        url = self._url(path)
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as http:
            resp = http.delete(url, headers=self._headers())
        if not resp.is_success:
            raise ProtocolRunError(
                f"DELETE {url} returned {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Protocol discovery
    # ------------------------------------------------------------------

    def list(
        self,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List protocols accessible to your account.

        Args:
            search: Substring match on name or slug.
            page: Page number (default 1).
            page_size: Results per page (default 20, max 100).

        Returns:
            Paginated dict::

                {
                    "count": 42,
                    "next": "/api/protocols/?page=2",
                    "previous": None,
                    "results": [
                        {
                            "slug": "antibody-optimization",
                            "version": 2,
                            "name": "Antibody Optimization Pipeline",
                            "description": "...",
                            "is_public": True,
                            "input_fields": ["sequence", "n_rounds"],
                        },
                        ...
                    ]
                }
        """
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if search:
            params["search"] = search
        return self._get("", params=params)

    def get(self, slug: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get protocol detail and full ``inputs_schema`` for SDK introspection.

        Call this before :meth:`submit` to discover what input fields are required
        and what types/constraints they have.

        Args:
            slug: Protocol slug.
            version: Specific version to fetch. Defaults to the latest published version.

        Returns:
            Dict::

                {
                    "slug": "antibody-optimization",
                    "version": 2,
                    "name": "Antibody Optimization Pipeline",
                    "description": "...",
                    "inputs_schema": {
                        "sequence": {
                            "type": "text",
                            "required": True,
                            "label": "Input Sequence",
                            "help_text": "Protein sequence in single-letter code",
                        },
                        "n_rounds": {
                            "type": "integer",
                            "required": False,
                            "initial": 3,
                            "min": 1,
                            "max": 10,
                        },
                    },
                    "example_inputs": {"sequence": "MKLL...", "n_rounds": 3},
                }

        Raises:
            :class:`ProtocolNotFoundError`: If the slug does not exist or is not accessible.
        """
        params = {"version": version} if version is not None else None
        return self._get(f"{slug}/", params=params)

    # ------------------------------------------------------------------
    # Run submission
    # ------------------------------------------------------------------

    def submit(
        self,
        slug: str,
        inputs: Dict[str, Any],
        version: Optional[int] = None,
        run_name: Optional[str] = None,
        environment_id: Optional[int] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> ProtocolRun:
        """Submit a protocol run.

        Args:
            slug: Protocol slug.
            inputs: Dict mapping input field names to values, matching the
                protocol's ``inputs_schema`` (use :meth:`get` to inspect it first).
            version: Specific protocol version to run. Defaults to latest.
            run_name: Optional human-readable label for this run.
            environment_id: Optional environment ID to scope billing and API key usage.
            files: Optional ``{field_name: file-like-object}`` for file-type inputs.
                Triggers multipart form submission.

        Returns:
            :class:`ProtocolRun` — the run is immediately ``scheduled``. Call
            ``.wait()`` to block until complete, or poll with ``.progress()``.

        Raises:
            :class:`ProtocolNotFoundError`: If the protocol is not accessible.
            :class:`ProtocolRunError`: If the server rejects the submission
                (e.g. invalid inputs or budget exceeded).

        Example::

            run = client.submit(
                "antibody-optimization",
                inputs={"sequence": "MKTAYIAKQRQ", "n_rounds": 3},
                run_name="experiment v7",
            )
            print(run.run_id)  # ALY_...
            run.wait()
        """
        if files:
            form_data: Dict[str, str] = {"inputs": json.dumps(inputs)}
            if version is not None:
                form_data["version"] = str(version)
            if run_name:
                form_data["run_name"] = run_name
            if environment_id is not None:
                form_data["environment_id"] = str(environment_id)
            multipart = {k: (None, v) for k, v in files.items()}
            data = self._post(f"{slug}/runs/", files=multipart, data=form_data)
        else:
            body: Dict[str, Any] = {"inputs": inputs}
            if version is not None:
                body["version"] = version
            if run_name:
                body["run_name"] = run_name
            if environment_id is not None:
                body["environment_id"] = environment_id
            data = self._post(f"{slug}/runs/", json_body=body)

        return ProtocolRun(data, client=self)

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def runs(
        self,
        protocol_slug: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """List your protocol runs.

        Args:
            protocol_slug: Filter to runs for a specific protocol.
            status: Filter by status. One of ``scheduled``, ``running``,
                ``succeeded``, ``failed``, ``cancelled``.
            page: Page number (default 1).
            page_size: Results per page (default 20, max 100).

        Returns:
            Paginated dict with ``results`` list of run summaries.
        """
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if protocol_slug:
            params["protocol_slug"] = protocol_slug
        if status:
            params["status"] = status
        return self._get("runs/", params=params)

    def get_run(self, run_id: str) -> ProtocolRun:
        """Reconnect to a previously submitted run by its ID.

        Args:
            run_id: The ``ALY_...`` run ID from a prior :meth:`submit` call.

        Returns:
            :class:`ProtocolRun` with current status.

        Example::

            # In a later session:
            run = client.get_run("ALY_507f1f77bcf86cd799439011")
            if run.status != "succeeded":
                run.wait()
            print(run.results())
        """
        data = self._get(f"runs/{run_id}/")
        return ProtocolRun(
            {
                "run_id": data["run_id"],
                "protocol_slug": data.get("protocol_slug", ""),
                "protocol_version": data.get("protocol_version", 1),
                "status": data.get("status", "unknown"),
            },
            client=self,
        )

    # ------------------------------------------------------------------
    # One-liner convenience
    # ------------------------------------------------------------------

    def run_and_wait(
        self,
        slug: str,
        inputs: Dict[str, Any],
        run_name: Optional[str] = None,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Submit a run, wait for completion, and return results.

        Combines :meth:`submit` + :meth:`ProtocolRun.wait` +
        :meth:`ProtocolRun.results` into a single blocking call.

        Args:
            slug: Protocol slug.
            inputs: Input field values.
            run_name: Optional run label.
            poll_interval: Seconds between polls (default 5).
            timeout: Max seconds to wait (default 3600).
            show_progress: Print progress updates (default True).

        Returns:
            Run detail dict (``run_id``, ``status``, timestamps, etc.).
            For the actual result rows, call ``run.download()`` after this returns.

        Raises:
            :class:`ProtocolRunError`: If the run fails or is cancelled.
            :class:`TimeoutError`: If ``timeout`` is exceeded.

        Example::

            run = client.submit("antibody-optimization", inputs={"sequence": "MKTAYIAKQRQ"})
            run.wait()
            path = run.download(output_dir="./results")  # saves the CSV zip
        """
        run = self.submit(slug, inputs, run_name=run_name)
        if show_progress:
            print(f"Submitted: {run.run_id}  [{slug}]")
        run.wait(poll_interval=poll_interval, timeout=timeout, show_progress=show_progress)
        return run.results()
