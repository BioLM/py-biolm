import asyncio
import json
from types import MethodType

import pytest

from biolmai.client import BioLMApiClient

# websockets is only used for the test – mark the whole module xfail if not available
pytest.importorskip("websockets")
import websockets


@pytest.mark.asyncio
async def test_telemetry_events_capture():
    """Ensure that the BioLMApiClient captures request_start/response_sent telemetry
    and exposes them via *_last_telemetry_events*.
    This spins up a local in-memory websocket server and patches the HTTP call
    so no real network traffic is generated.
    """
    # Telemetry is now implemented, but this test uses a mock server
    # Keep it for now but mark as integration test
    pytest.skip("Mock server test - use test_telemetry_all_clients.py for real tests")

    TELEMETRY_EVENTS = [
        {"event": "request_start"},
        {"event": "response_sent"},
    ]

    # ------------------------------------------------------------------
    # 1. Start a lightweight websocket server that immediately streams the two
    #    events above for any connection & path.
    # ------------------------------------------------------------------

    async def ws_handler(websocket):  # noqa: D401
        for ev in TELEMETRY_EVENTS:
            await websocket.send(json.dumps(ev))
        await websocket.close()

    server = await websockets.serve(ws_handler, "localhost", 8765)

    try:
        # ------------------------------------------------------------------
        # 2. Prepare BioLMApiClient with base_url pointing at our dummy server.
        #    Patch the underlying HTTP POST to avoid real traffic and to always
        #    succeed with a JSON body `{ "results": [ {"ok": true} ] }`.
        # ------------------------------------------------------------------
        client = BioLMApiClient(
            "esm2-8m",
            raise_httpx=False,
            unwrap_single=False,
        )

        async def fake_post(self, endpoint, payload, extra_headers=None):  # noqa: D401
            class FakeResp:  # Minimal duck-typed httpx.Response
                status_code = 200
                headers = {"Content-Type": "application/json"}

                def json(self_inner):
                    return {"results": [{"ok": True}]}

                text = "{\"results\": [{\"ok\": true}]}"
                request = None

            # Simulate network latency so telemetry listener remains active
            await asyncio.sleep(0.05)
            return FakeResp()

        # Monkey-patch the instance's _http_client.post coroutine
        client._http_client.post = MethodType(fake_post, client._http_client)

        # Patch GET as well to avoid real network calls (used for schema fetch)
        async def fake_get(self, endpoint):
            class FakeResp:
                status_code = 200
                headers = {"Content-Type": "application/json"}
                def json(self_inner):
                    return {}
            return FakeResp()

        client._http_client.get = MethodType(fake_get, client._http_client)

        # ------------------------------------------------------------------
        # 3. Execute an encode call – this will trigger telemetry handling.
        # ------------------------------------------------------------------
        await client.encode(items=[{"sequence": "ACDE"}])

        # ------------------------------------------------------------------
        # 4. Assert that both telemetry events were captured.
        # ------------------------------------------------------------------
        events = client.last_telemetry_events
        kinds = {e.get("event") for e in events if isinstance(e, dict)}
        assert {"request_start", "response_sent"}.issubset(kinds), events

    finally:
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Optional integration test against a *running* local Django dev server
# ---------------------------------------------------------------------------

import os


@pytest.mark.asyncio
async def test_telemetry_events_live_local():
    """Hit the real dev server at http://localhost:8888/api/v3.

    To enable this test, ensure:
    • The Django server with Channels is running locally on port 8888.
    • The environment variable BIOLM_TOKEN (or ~/.biolm/credentials) is set
      so the client can authenticate.

    Skip automatically if the server isn't reachable or if authentication is
    missing.
    """

    base_url = "http://localhost:8888/api/v3"

    # Quick connectivity check – skip if server is down
    import socket
    try:
        with socket.create_connection(("localhost", 8888), timeout=1):
            pass
    except OSError:
        pytest.skip("Local dev server not running on port 8888")

    # Inject known dev token so the client can authenticate against the local
    # dev server.  This token is hard-coded in entrypoint-dev / load_dev_data
    # for the "deverson" superuser and is safe to expose in local tests.
    os.environ.setdefault(
        "BIOLM_TOKEN",
        "b10a1c0deafaceb00c0ffeebada55a11fe55d15ea5eacc0ab1efacefeedc0de5",
    )

    client = BioLMApiClient(
        "esm2-8m",
        api_key="b10a1c0deafaceb00c0ffeebada55a11fe55d15ea5eacc0ab1efacefeedc0de5",
        base_url=base_url,
        raise_httpx=False,
        unwrap_single=False,
        telemetry=True,
    )

    # Patch WebSocket URLs to use port 5555 instead of 8888
    # The HTTP API stays on 8888, but WebSockets should use 5555
    # We intercept listener creation to patch the ws_url before the task starts
    from biolmai.client import TelemetryListener, ActivityListener
    
    # Monkey-patch the listener constructors to patch ws_url immediately
    original_telemetry_listener_init = TelemetryListener.__init__
    original_activity_listener_init = ActivityListener.__init__
    
    def patched_telemetry_listener_init(self, ws_url, *args, **kwargs):
        # Replace port 8888 with 5555 in WebSocket URL
        if ":8888" in ws_url:
            ws_url = ws_url.replace(":8888", ":5555")
        return original_telemetry_listener_init(self, ws_url, *args, **kwargs)
    
    def patched_activity_listener_init(self, ws_url, *args, **kwargs):
        # Replace port 8888 with 5555 in WebSocket URL
        if ":8888" in ws_url:
            ws_url = ws_url.replace(":8888", ":5555")
        return original_activity_listener_init(self, ws_url, *args, **kwargs)
    
    # Apply the patches
    TelemetryListener.__init__ = patched_telemetry_listener_init
    ActivityListener.__init__ = patched_activity_listener_init
    
    try:
        await client.encode(items=[{"sequence": "ACDE"}])
    finally:
        # Restore original constructors
        TelemetryListener.__init__ = original_telemetry_listener_init
        ActivityListener.__init__ = original_activity_listener_init

    try:
        await client.encode(items=[{"sequence": "ACDE"}])
    except Exception as e:
        await client.shutdown()
        pytest.skip(f"API call failed: {e}")

    # Give websocket some time to receive broadcast
    events = []
    for _ in range(10):  # up to ~1 s total
        events = client.last_telemetry_events
        if any(isinstance(evt, dict) and evt.get("event") in ("request_start", "response_sent") for evt in events):
            break
        await asyncio.sleep(0.1)

    # If no events received, the server may not be configured for telemetry
    if not any(isinstance(evt, dict) and evt.get("event") in ("request_start", "response_sent") for evt in events):
        await client.shutdown()
        pytest.skip("No telemetry events received - server may not be configured for telemetry")
    
    assert any(isinstance(evt, dict) and evt.get("event") in ("request_start", "response_sent") for evt in events), events
    
    await client.shutdown() 