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

    # Option A — block until complete
    run.wait()
    print(run.results())

    # Option B — one-liner
    results = client.run_and_wait("antibody-optimization", inputs={"sequence": "MKTAYIAKQRQ"})

    # Download structure files after completion
    paths = run.download_files(output_dir="./structures", columns=["designed_pdb"])
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
        """Return a live progress snapshot for this run.

        Reads from the same event log as the browser console's live view.

        Returns:
            Dict with keys:

            - ``status`` — current run status
            - ``progress_pct`` — 0–100 integer, or ``None`` if not yet available
            - ``tasks`` — list of per-task dicts (``name``, ``status``,
              ``completed_subtasks``, ``total_subtasks``, ``failed_subtasks``)
            - ``log_lines`` — last 50 log lines (newest last)
            - ``result_row_count`` — rows emitted so far, or ``None``
        """
        return self._client._get(f"runs/{self.run_id}/progress/")

    def log_lines(self) -> List[str]:
        """Convenience wrapper — returns the latest log lines from :meth:`progress`."""
        return self.progress().get("log_lines", [])

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
        last_pct: Optional[int] = None

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Protocol run {self.run_id} did not complete within {timeout:.0f}s. "
                    "Increase timeout= or poll manually with run.progress()."
                )

            try:
                snap = self.progress()
            except Exception:
                # Network blip — refresh plain status and retry
                try:
                    self.refresh()
                except Exception:
                    pass
                time.sleep(poll_interval)
                continue

            self.status = snap.get("status", self.status)
            pct: Optional[int] = snap.get("progress_pct")
            tasks: List[Dict] = snap.get("tasks", [])

            if show_progress and pct != last_pct:
                running = [t["name"] for t in tasks if t.get("status") == "running"]
                if pct is not None:
                    suffix = f" — {running[0]}" if running else ""
                    print(f"  [{self.run_id}] {pct:>3}%{suffix}")
                elif running:
                    print(f"  [{self.run_id}] running: {', '.join(running)}")
                last_pct = pct

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
        """Fetch the full run detail including the results payload.

        Returns:
            Dict with keys:

            - ``run_id``, ``protocol_slug``, ``protocol_version``
            - ``status``
            - ``results`` — the ``return_json`` from the server (may be large)
            - ``failure_error`` — error message if failed, else ``None``
            - ``workflow_started_at``, ``workflow_ended_at``, ``expires_at``
        """
        return self._client._get(f"runs/{self.run_id}/")

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def download(self, format: str = "json") -> Dict[str, Any]:
        """Fetch the download response for this run.

        Args:
            format: ``"json"`` (default) returns ``results`` inline.
                    ``"urls"`` returns presigned R2 URLs for structure files.

        Returns:
            Download response dict from the API.
        """
        return self._client._get(
            f"runs/{self.run_id}/download/", params={"format": format}
        )

    def download_files(
        self,
        output_dir: Union[str, Path] = ".",
        columns: Optional[List[str]] = None,
        rows: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> List[Path]:
        """Download structure files (PDB/CIF) to a local directory.

        Fetches presigned R2 download URLs for the specified columns and rows,
        then saves each file as ``{output_dir}/{row}_{column}.{ext}``.

        Args:
            output_dir: Directory to save files. Created if it does not exist.
                Default ``"."`` (current directory).
            columns: Structure column names to download, e.g. ``["pdb", "designed_pdb"]``.
                Defaults to auto-detecting structure columns from the results.
            rows: 1-based row numbers to download. Defaults to all rows up to 500.
            overwrite: Re-download files that already exist. Default ``False``.

        Returns:
            List of :class:`pathlib.Path` objects for every file that was written
            (or already existed when ``overwrite=False``).

        Example::

            paths = run.download_files(
                output_dir="./structures",
                columns=["designed_pdb"],
                rows=list(range(1, 11)),  # first 10 rows
            )
        """
        params: Dict[str, Any] = {"format": "urls"}
        if columns:
            params["columns"] = ",".join(columns)
        if rows:
            params["rows"] = ",".join(str(r) for r in rows)

        url_data = self._client._get(f"runs/{self.run_id}/download/", params=params)
        urls: Dict[str, Dict] = url_data.get("urls", {})
        if not urls:
            return []

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        downloaded: List[Path] = []
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT) as http:
            for row_str, col_map in urls.items():
                for col, info in col_map.items():
                    ext = "cif" if "cif" in col else "pdb"
                    dest = out / f"{row_str}_{col}.{ext}"
                    if dest.exists() and not overwrite:
                        downloaded.append(dest)
                        continue
                    resp = http.get(info["url"])
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    downloaded.append(dest)

        return downloaded

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
            The ``results`` payload (``return_json``) from the completed run.

        Raises:
            :class:`ProtocolRunError`: If the run fails or is cancelled.
            :class:`TimeoutError`: If ``timeout`` is exceeded.

        Example::

            results = client.run_and_wait(
                "antibody-optimization",
                inputs={"sequence": "MKTAYIAKQRQ", "n_rounds": 5},
            )
            print(results["designed_sequences"])
        """
        run = self.submit(slug, inputs, run_name=run_name)
        if show_progress:
            print(f"Submitted: {run.run_id}  [{slug}]")
        run.wait(poll_interval=poll_interval, timeout=timeout, show_progress=show_progress)
        return run.results().get("results", {})
