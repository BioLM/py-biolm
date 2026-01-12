import asyncio
import functools
import json
import os
import time
from collections import namedtuple, OrderedDict
from contextlib import asynccontextmanager
from itertools import chain
from itertools import tee, islice
from json import dumps as json_dumps
from typing import Callable
from typing import Optional, Union, List, Any, Dict, Tuple

import aiofiles
import httpx
import httpx._content
from async_lru import alru_cache
from httpx import AsyncHTTPTransport
from httpx import ByteStream
from synchronicity import Synchronizer

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

# Telemetry imports
import uuid
import json as _json
from urllib.parse import urlparse
import inspect

# Progress manager import (optional)
try:
    from biolmai.telemetry_progress import TelemetryProgressManager
except ImportError:  # pragma: no cover
    TelemetryProgressManager = None  # type: ignore

# "websockets" is a lightweight dependency (~50 KB) used for realtime telemetry
# It is optional at runtime; if unavailable, telemetry is silently disabled.
try:
    import websockets  # type: ignore
except ImportError:  # pragma: no cover – runtime fallback when websockets not installed
    websockets = None  # type: ignore

import contextlib

def custom_httpx_encode_json(json: Any) -> Tuple[Dict[str, str], ByteStream]:
    # disable ascii for json_dumps
    body = json_dumps(json, ensure_ascii=False).encode("utf-8")
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, ByteStream(body)

# fix encoding utf-8 bug
httpx._content.encode_json = custom_httpx_encode_json

import sys

def debug(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

import logging

# Turn this on to dev lots of logs
if os.environ.get("DEBUG", '').upper().strip() in ('TRUE', '1'):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,  # Python 3.8+
    )

USER_BIOLM_DIR = os.path.join(os.path.expanduser("~"), ".biolmai")
ACCESS_TOK_PATH = os.path.join(USER_BIOLM_DIR, "credentials")
TIMEOUT_MINS = 20  # Match API server's keep-alive/timeout
DEFAULT_TIMEOUT = httpx.Timeout(TIMEOUT_MINS * 60, connect=10.0)
DEFAULT_BASE_URL = "https://biolm.ai/api/v3"

LookupResult = namedtuple("LookupResult", ["data", "raw"])

_synchronizer = Synchronizer()

if not hasattr(_synchronizer, "sync"):
    if hasattr(_synchronizer, "wrap"):
        _synchronizer.sync = _synchronizer.wrap
    if hasattr(_synchronizer, "create_blocking"):
        _synchronizer.sync = _synchronizer.create_blocking
    else:
        raise ImportError(f"Your version of 'synchronicity' ({version('synchronicity')}) is incompatible.")

def type_check(param_types: Dict[str, Any]):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for param, expected_type in param_types.items():
                value = kwargs.get(param)
                if value is None and len(args) > 0:
                    arg_names = func.__code__.co_varnames
                    if param in arg_names:
                        idx = arg_names.index(param)
                        if idx < len(args):
                            value = args[idx]
                if value is not None:
                    # Allow tuple of types or single type
                    if not isinstance(expected_type, tuple):
                        expected_types = (expected_type,)
                    else:
                        expected_types = expected_type
                    if not isinstance(value, expected_types):
                        type_names = ", ".join([t.__name__ for t in expected_types])
                        raise TypeError(
                            f"Parameter '{param}' must be of type {type_names}, got {type(value).__name__}"
                        )
                    # Check for empty list/tuple
                    # if isinstance(value, (list, tuple)) and len(value) == 0:
                    #     raise ValueError(
                    #         f"Parameter '{param}' must not be an empty {type(value).__name__}"
                    #     )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class AsyncRateLimiter:
    def __init__(self, max_calls: int, period: float):
        self._max_calls = max_calls
        self._period = period
        self._lock = asyncio.Lock()
        self._calls = []

    @asynccontextmanager
    async def limit(self):
        async with self._lock:
            now = time.monotonic()
            # Remove calls outside the window
            self._calls = [t for t in self._calls if now - t < self._period]
            if len(self._calls) >= self._max_calls:
                sleep_time = self._period - (now - self._calls[0])
                await asyncio.sleep(max(0, sleep_time))
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self._period]
            self._calls.append(time.monotonic())
        yield

def parse_rate_limit(rate: str):
    # e.g. "1000/second", "60/minute"
    if not rate:
        return None
    num, per = rate.strip().split("/")
    num = int(num)
    per = per.strip().lower()
    if per == "second":
        return num, 1.0
    elif per == "minute":
        return num, 60.0
    else:
        raise ValueError(f"Unknown rate period: {per}")

class CredentialsProvider:
    @staticmethod
    def get_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
        if api_key:
            return {"Authorization": f"Token {api_key}"}
        api_token = os.environ.get("BIOLMAI_TOKEN")
        if api_token:
            return {"Authorization": f"Token {api_token}"}
        if os.path.exists(ACCESS_TOK_PATH):
            with open(ACCESS_TOK_PATH) as f:
                creds = json.load(f)
            access = creds.get("access")
            refresh = creds.get("refresh")
            return {
                "Cookie": f"access={access};refresh={refresh}",
                "Content-Type": "application/json",
            }
        raise AssertionError("No credentials found. Set BIOLMAI_TOKEN or run `biolmai login`.")


class HttpClient:

    def __init__(self, base_url: str, headers: Dict[str, str], timeout: httpx.Timeout):
        self._base_url = base_url.rstrip("/") + "/"
        self._headers = headers
        self._timeout = timeout
        self._async_client: Optional[httpx.AsyncClient] = None
        self._transport = None
        # Removed AsyncResolver, use default resolver
        self._transport = AsyncHTTPTransport()

    async def get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or getattr(self._async_client, 'is_closed', False):
            if self._transport:
                self._async_client = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers=self._headers,
                    timeout=self._timeout,
                    transport=self._transport,
                )
            else:
                self._async_client = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers=self._headers,
                    timeout=self._timeout,
                )
        return self._async_client

    async def post(self, endpoint: str, payload: dict, extra_headers: Optional[dict] = None) -> httpx.Response:
        """POST with optional *extra_headers* added just for this request."""
        client = await self.get_async_client()
        # Remove leading slash, ensure trailing slash
        endpoint = endpoint.lstrip("/")
        if not endpoint.endswith("/"):
            endpoint += "/"

        headers = None
        if extra_headers:
            headers = {**client.headers, **extra_headers}

        if "Content-Type" not in (headers or client.headers):
            # Ensure JSON content-type
            (headers or client.headers)["Content-Type"] = "application/json"

        r = await client.post(endpoint, json=payload, headers=headers)
        return r

    async def get(self, endpoint: str) -> httpx.Response:
        client = await self.get_async_client()
        endpoint = endpoint.lstrip("/")
        if not endpoint.endswith("/"):
            endpoint += "/"
        return await client.get(endpoint)


    async def close(self):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None


def is_list_of_lists(items, check_n=10):
    # Accepts any iterable, checks first N items for list/tuple-ness
    # Returns (is_list_of_lists, first_n_items, rest_iter)
    if isinstance(items, (list, tuple)):
        if not items:
            return False, [], iter(())
        first_n = items[:check_n]
        is_lol = all(isinstance(x, (list, tuple)) for x in first_n)
        return is_lol, first_n, iter(items[check_n:])
    # For iterators/generators
    items, items_copy = tee(items)
    first_n = list(islice(items_copy, check_n))
    is_lol = all(isinstance(x, (list, tuple)) for x in first_n) and bool(first_n)
    return is_lol, first_n, items

def batch_iterable(iterable, batch_size):
    # Yields lists of up to batch_size from any iterable, deleting as we go
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

class TelemetryListener:
    """Simple async helper that connects to the server-side websocket channel
    and prints each message (JSON decoded if possible).  It exits automatically
    when the websocket connection is closed by the server.  It is designed to
    stay alive across multiple API calls so the same channel can be reused.
    
    The listener captures all events in ``self.events`` so tests can assert on
    them without scraping stdout.
    """

    def __init__(self, ws_url: str, handler: Optional[Callable[[Any], Any]] = None, progress_manager: Optional[Any] = None, headers: Optional[dict] = None):
        self.ws_url = ws_url
        self.headers = headers or {}
        self.events: List[Any] = []
        self._connected = asyncio.Event()
        self._handler = handler  # user-supplied callback (sync or async)
        self._progress_manager = progress_manager  # progress manager for tqdm bars

    async def _notify(self, data):
        """Invoke the user callback (if any) and progress manager with *data*.

        If the callback returns an awaitable we await it to preserve ordering;
        otherwise we swallow any exceptions and keep listening."""
        # Call user handler first
        if self._handler is not None:
            try:
                res = self._handler(data)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                debug(f"[Telemetry] handler error: {e}")

        # Call progress manager if available
        if self._progress_manager is not None:
            try:
                evt = data.get("event")
                request_id = data.get("request_id") or data.get("rid")

                if evt == "request_start" and request_id:
                    model = data.get("model", "?")
                    action = data.get("action", "?")
                    n_items = data.get("n_items", 0)
                    self._progress_manager.start_request(request_id, model, action, n_items)
                elif evt == "call_submitted" and request_id:
                    backend_items = data.get("backend_items", 0)
                    self._progress_manager.submit_request(request_id, backend_items)
                elif evt == "cache_hit" and request_id:
                    self._progress_manager.cache_hit(request_id)
                elif evt == "call_finished" and request_id:
                    elapsed = data.get("elapsed", 0.0)
                    try:
                        elapsed = float(elapsed)
                    except (ValueError, TypeError):
                        elapsed = 0.0
                    self._progress_manager.finish_request(request_id, elapsed)
                elif evt == "error" and request_id:
                    status_code = data.get("status_code", 400)
                    self._progress_manager.error_request(request_id, status_code)
                elif evt == "response_sent" and request_id:
                    self._progress_manager.complete_request(request_id)
            except Exception as e:
                debug(f"[Telemetry] progress manager error: {e}")

    async def listen(self):  # pragma: no cover – executed in asyncio task
        if websockets is None:
            return  # websockets library not present – noop
        
        # Convert headers dict to list of tuples for websockets
        # Note: WebSocket connections need headers in tuple format
        # Cookie headers work the same way as Authorization headers
        extra_headers = [(k, v) for k, v in self.headers.items()] if self.headers else []
        if extra_headers:
            debug(f"[Telemetry] using auth headers: {[k for k, v in extra_headers]}")
        
        # Retry logic: up to 3 attempts (initial + 2 retries) with 15s timeout each
        max_attempts = 3
        ws_timeout = 15.0
        
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    debug(f"[Telemetry] retry attempt {attempt - 1}/{max_attempts - 1}")
                else:
                    debug(f"[Telemetry] connecting to {self.ws_url}")
                
                # Use longer timeout (15s) since we may connect before request is made
                async with websockets.connect(
                    self.ws_url, 
                    additional_headers=extra_headers,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=ws_timeout,
                    compression="deflate",
                ) as ws:  # type: ignore
                    debug("[Telemetry] websocket connection established")
                    self._connected.set()
                    message_count = 0
                    # Log that we're entering the message loop
                    debug("[Telemetry] Entering message receive loop, waiting for messages...")
                    async for raw in ws:
                        message_count += 1
                        if message_count == 1:
                            debug(f"[Telemetry] received first message (total: {message_count})")
                        elif message_count <= 5:
                            debug(f"[Telemetry] received message #{message_count}")
                        try:
                            data = _json.loads(raw)
                            debug(f"[Telemetry] event: {data.get('event', 'unknown')} for request {data.get('request_id', 'unknown')[:8]}")
                        except Exception:
                            data = raw
                            debug(f"[Telemetry] received non-JSON message: {str(raw)[:100]}")
                        self.events.append(data)
                        await self._notify(data)

                        # Pretty-print: include request id and basic event name
                        rid_full = data.get("request_id") or data.get("rid") or "—"
                        rid_short = str(rid_full)[:6]

                        tag_prefix = rid_short
                        if "model" in data and "action" in data:
                            tag_prefix = f"{data['model']}.{data['action']}-{rid_short}"

                        evt = data.get("event", "?")

                        # Build a concise suffix depending on event type
                        suffix = ""
                        if evt == "cache_hit":
                            suffix = f" {data.get('n_items', '?')} hit"
                        elif evt == "request_start" and "n_items" in data:
                            suffix = f" {data['n_items']} recvd"
                        elif evt == "call_submitted":
                            suffix = f" {data.get('backend_items', '?')} sent"
                        elif evt == "error":
                            suffix = f" {data.get('status_code', '?')}"
                        elif evt == "call_finished" and "elapsed" in data:
                            try:
                                suffix = f" {float(data['elapsed']):.2f}s"
                            except Exception:
                                suffix = f" {data['elapsed']}"
                        # No extra suffix for response_sent (model.action already
                        # present in the tag_prefix)

                        debug(f"[Telemetry-{tag_prefix}] {evt}{suffix}")
                    
                    # If we exit the loop normally, connection was closed by server
                    break
                    
            except Exception as e:
                if attempt < max_attempts:
                    debug(f"[Telemetry] connection attempt {attempt} failed: {e}, retrying immediately...")
                    # No delay between retries - continue immediately
                    continue
                else:
                    # Last attempt failed
                    debug(f"[Telemetry] Listener error after {max_attempts} attempts: {e}")
                    break


class ActivityListener:
    """Listens to account-level activity updates via Activity WebSocket.
    
    Connects to /ws/activity/ endpoint and receives activity_update,
    billing_update, and activity_hint events.
    """

    def __init__(self, ws_url: str, headers: Dict[str, str], handler: Optional[Callable[[Any], Any]] = None):
        self.ws_url = ws_url
        self.headers = headers
        self.events: List[Any] = []
        self._connected = asyncio.Event()
        self._handler = handler  # user-supplied callback (sync or async)

    async def _notify(self, data):
        """Invoke the user callback (if any) with *data*.

        If the callback returns an awaitable we await it to preserve ordering;
        otherwise we swallow any exceptions and keep listening."""
        if self._handler is None:
            return
        try:
            res = self._handler(data)
            if inspect.isawaitable(res):
                await res
        except Exception as e:
            debug(f"[Activity] handler error: {e}")

    async def listen(self):  # pragma: no cover – executed in asyncio task
        if websockets is None:
            return  # websockets library not present – noop
        
        # Convert headers dict to list of tuples for websockets
        # Note: WebSocket connections need headers in tuple format
        # Cookie headers work the same way as Authorization headers
        extra_headers = [(k, v) for k, v in self.headers.items()]
        if extra_headers:
            debug(f"[Activity] using auth headers: {[k for k, v in extra_headers]}")
        
        # Retry logic: up to 3 attempts (initial + 2 retries) with 15s timeout each
        max_attempts = 3
        ws_timeout = 15.0
        
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    debug(f"[Activity] retry attempt {attempt - 1}/{max_attempts - 1}")
                else:
                    debug(f"[Activity] connecting to {self.ws_url}")
                
                # Use longer timeout (15s) for activity WebSocket
                # Don't use context manager - keep connection open explicitly
                ws = await websockets.connect(
                    self.ws_url, 
                    additional_headers=extra_headers,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=ws_timeout,
                    compression="deflate",
                )  # type: ignore
                try:
                    debug("[Activity] websocket connection established")
                    self._connected.set()
                    message_count = 0
                    debug("[Activity] Entering message receive loop, waiting for messages...")
                    async for raw in ws:
                        message_count += 1
                        if message_count == 1:
                            debug(f"[Activity] received first message (total: {message_count})")
                        elif message_count <= 10:
                            debug(f"[Activity] received message #{message_count}")
                        try:
                            data = _json.loads(raw)
                            event_type = data.get('type', 'unknown')
                            debug(f"[Activity] event type: {event_type}")
                            # Log more details for activity_update events
                            if event_type == "activity_update":
                                activity_data = data.get("data", {})
                                algorithms = activity_data.get("algorithms", {})
                                totals = activity_data.get("totals", {})
                                debug(f"[Activity] activity_update: {len(algorithms)} algorithms, totals: {totals}")
                        except Exception as e:
                            data = raw
                            debug(f"[Activity] received non-JSON message: {str(raw)[:100]}, error: {e}")
                        self.events.append(data)
                        await self._notify(data)
                    
                    # If we exit the loop normally, connection was closed by server
                    debug(f"[Activity] Exited message loop (received {message_count} total messages)")
                    try:
                        close_code = ws.close_code if hasattr(ws, 'close_code') else None
                        close_reason = ws.close_reason if hasattr(ws, 'close_reason') else None
                        debug(
                            f"[Activity] websocket closed by server "
                            f"(code={close_code}, reason={close_reason})"
                        )
                    except Exception:
                        debug("[Activity] websocket closed by server (unable to get close details)")
                finally:
                    # Explicitly close the connection
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                    
            except Exception as e:
                if attempt < max_attempts:
                    debug(f"[Activity] connection attempt {attempt} failed: {e}, retrying immediately...")
                    # No delay between retries - continue immediately
                    continue
                else:
                    # Last attempt failed
                    debug(f"[Activity] Listener error after {max_attempts} attempts: {e}")
                    break


class BioLMApiClient:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        raise_httpx: bool = True,
        unwrap_single: bool = False,
        semaphore: 'Optional[Union[int, asyncio.Semaphore]]' = None,
        rate_limit: 'Optional[str]' = None,
        retry_error_batches: bool = False,
        *,
        telemetry: bool = False,
        telemetry_handler: Optional[Callable[[Any], Any]] = None,
        progress: bool = True,
    ):
        self.model_name = model_name
        # Resolve base_url with precedence: explicit param > env vars > default
        if base_url is None:
            base_url = os.getenv("BIOLM_BASE_URL") or os.getenv("BIOLMAI_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = base_url.rstrip("/") + "/"  # Ensure trailing slash
        self.timeout = timeout
        self.raise_httpx = raise_httpx
        self.unwrap_single = unwrap_single
        self._headers = CredentialsProvider.get_auth_headers(api_key)
        self._http_client = HttpClient(self.base_url, self._headers, self.timeout)
        self._semaphore = None
        self._rate_limiter = None
        self._rate_limit_lock = None
        self._rate_limit_initialized = False
        self.retry_error_batches = retry_error_batches
        self.telemetry_enabled = telemetry and websockets is not None
        self._telemetry_handler = telemetry_handler
        self.progress_enabled = progress and (TelemetryProgressManager is not None)

        # -- Telemetry state -------------------------------------------------
        self._telemetry_channel: Optional[str] = None
        self._telemetry_listener: Optional[TelemetryListener] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._telemetry_event_cursor: int = 0  # Index into events list
        # Stores events from the most-recent request (for tests / debugging)
        self._last_telemetry_events: List[Any] = []

        # -- Progress manager ------------------------------------------------
        self._progress_manager: Optional[Any] = None
        if self.progress_enabled:
            self._progress_manager = TelemetryProgressManager(enable=True)
            # Enable telemetry automatically if progress is enabled
            if not self.telemetry_enabled:
                self.telemetry_enabled = websockets is not None

        # -- Activity WebSocket state -----------------------------------------
        self._activity_listener: Optional[ActivityListener] = None
        self._activity_task: Optional[asyncio.Task] = None

        if self.telemetry_enabled:
            # Generate a single channel for the lifetime of this client
            self._telemetry_channel = f"telemetry_{uuid.uuid4().hex}"
            # Inject header so every request uses the same channel
            self._http_client._headers["X-Telemetry-Channel"] = self._telemetry_channel

        # Concurrency limit
        if isinstance(semaphore, asyncio.Semaphore):
            self._semaphore = semaphore
        elif isinstance(semaphore, int):
            self._semaphore = asyncio.Semaphore(semaphore)

        # RPS limit
        if rate_limit:
            max_calls, period = parse_rate_limit(rate_limit)
            self._rate_limiter = AsyncRateLimiter(max_calls, period)
            self._rate_limit_initialized = True

        # Listener will be started lazily on first _api_call when an event loop is present

    async def _ensure_rate_limit(self):
            if self._rate_limit_lock is None:
                self._rate_limit_lock = asyncio.Lock()
            if self._rate_limit_initialized:
                return
            async with self._rate_limit_lock:
                if self._rate_limit_initialized:
                    return
                if self._rate_limiter is None:
                    schema = await self.schema(self.model_name, "encode")
                    throttle_rate = schema.get("throttle_rate") if schema else None
                    if throttle_rate:
                        max_calls, period = parse_rate_limit(throttle_rate)
                        self._rate_limiter = AsyncRateLimiter(max_calls, period)
                self._rate_limit_initialized = True

    @asynccontextmanager
    async def _limit(self):
        """
         Usage:
            # No throttling: BioLMApiClient(...)
            # Concurrency limit: BioLMApiClient(..., semaphore=5)
            # User's own semaphore: BioLMApiClient(..., semaphore=my_semaphore)
            # RPS limit: BioLMApiClient(..., rate_limit="1000/second")
            # Both: BioLMApiClient(..., semaphore=5, rate_limit="1000/second")
        """
        if self._semaphore:
            async with self._semaphore:
                if self._rate_limiter:
                    async with self._rate_limiter.limit():
                        yield
                else:
                    yield
        elif self._rate_limiter:
            async with self._rate_limiter.limit():
                yield
        else:
            yield

    @alru_cache(maxsize=8)
    async def schema(
        self,
        model: str,
        action: str,
    ) -> Optional[dict]:
        """
        Fetch the JSON schema for a given model and action, with caching.
        Returns the schema dict if successful, else None.
        """
        endpoint = f"schema/{model}/{action}/"
        try:
            resp = await self._http_client.get(endpoint)
            if resp.status_code == 200:
                schema = resp.json()
                return schema
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def extract_max_items(schema: dict) -> Optional[int]:
        """
        Extracts the 'maxItems' value for the 'items' key from the schema.
        Returns the integer value if found, else None.
        """
        try:
            props = schema.get('properties', {})
            items_schema = props.get('items', {})
            max_items = items_schema.get('maxItems')
            if isinstance(max_items, int):
                return max_items
        except Exception:
            pass
        return None

    async def _get_max_batch_size(self, model: str, action: str) -> Optional[int]:
        schema = await self.schema(model, action)
        if schema:
            return self.extract_max_items(schema)
        return None

    async def _fetch_rps_limit_async(self) -> Optional[int]:
        return None
        # Not implemented yet
        try:
            async with httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=5.0) as client:
                resp = await client.get(f"/{self.model_name}/")
                if resp.status_code == 200:
                    meta = resp.json()
                    return meta.get("rps_limit") or meta.get("max_rps") or meta.get("requests_per_second")
        except Exception:
            pass
        return None

    async def _api_call(
        self, endpoint: str, payload: dict, raw: bool = False
    ) -> Union[dict, Tuple[Any, httpx.Response]]:
        await self._ensure_rate_limit()
        # --------------------------- Telemetry listener -----------------------
        if self.telemetry_enabled and self._listener_task is None:
            parsed = urlparse(self.base_url)
            ws_scheme = "wss" if parsed.scheme == "https" else "ws"
            host = parsed.netloc
            ws_url = f"{ws_scheme}://{host}/ws/telemetry/{self._telemetry_channel}/"
            debug(f"[Telemetry] Setting up listener for channel: {self._telemetry_channel}")
            debug(f"[Telemetry] WebSocket URL: {ws_url}")
            debug(f"[Telemetry] Channel will be sent in header: X-Telemetry-Channel = {self._telemetry_channel}")
            self._telemetry_listener = TelemetryListener(
                ws_url,
                handler=self._telemetry_handler,
                progress_manager=self._progress_manager,
                headers=self._headers,  # Pass authentication headers
            )
            self._listener_task = asyncio.create_task(self._telemetry_listener.listen())
            # wait until websocket connected (avoid race)
            # Use longer timeout (15s) since we may connect to channel before request is made
            try:
                await asyncio.wait_for(self._telemetry_listener._connected.wait(), timeout=15.0)
                debug("[Telemetry] websocket connected successfully")
            except asyncio.TimeoutError:
                debug("[Telemetry] websocket connection timeout (15s)")

        # --------------------------- Activity listener -------------------------
        if self.progress_enabled and self._activity_task is None:
            parsed = urlparse(self.base_url)
            ws_scheme = "wss" if parsed.scheme == "https" else "ws"
            host = parsed.netloc
            activity_ws_url = f"{ws_scheme}://{host}/ws/activity/"

            async def activity_handler(data):
                """Handle activity WebSocket events."""
                debug(f"[Activity] handler called with event type: {data.get('type', 'unknown')}")
                if self._progress_manager is None:
                    debug("[Activity] handler: progress_manager is None, skipping")
                    return
                try:
                    event_type = data.get("type")
                    debug(f"[Activity] handler processing event type: {event_type}")
                    if event_type == "activity_update":
                        activity_data = data.get("data", {})
                        algorithms = activity_data.get("algorithms", {})
                        totals = activity_data.get("totals", {})
                        debug(f"[Activity] handler: calling update_resources with {len(algorithms)} algorithms, totals: {totals}")
                        self._progress_manager.update_resources(activity_data)
                    elif event_type == "billing_update":
                        billing_data = data.get("data", {})
                        debug(f"[Activity] handler: calling update_billing")
                        self._progress_manager.update_billing(billing_data)
                    elif event_type == "activity_hint":
                        hint_data = data.get("data", {})
                        debug(f"[Activity] handler: calling update_hint")
                        self._progress_manager.update_hint(hint_data)
                    else:
                        debug(f"[Activity] handler: unknown event type: {event_type}")
                except Exception as e:
                    debug(f"[Activity] handler error: {e}")

            self._activity_listener = ActivityListener(
                activity_ws_url,
                headers=self._headers,
                handler=activity_handler,
            )
            self._activity_task = asyncio.create_task(self._activity_listener.listen())
            # wait until websocket connected (avoid race)
            # Use longer timeout (15s) for activity WebSocket
            try:
                await asyncio.wait_for(self._activity_listener._connected.wait(), timeout=15.0)
                debug("[Activity] websocket connected successfully")
            except asyncio.TimeoutError:
                debug("[Activity] websocket connection timeout (15s)")

        # --------------------------- Make HTTP call ---------------------------
        async with self._limit():
            # ------------------------------------------------------------------
            # Tag this request with a unique ID so server will echo it back via
            # telemetry, letting us correlate events belonging to a single HTTP
            # call while still reusing the persistent websocket channel.
            # ------------------------------------------------------------------
            request_id = uuid.uuid4().hex
            extra_headers = {"X-Request-Id": request_id}
            
            # Log the request ID for debugging
            debug(f"[API] Making request with ID: {request_id[:8]} to {endpoint}")
            if self.telemetry_enabled:
                debug(f"[API] Request will include header: X-Request-Id = {request_id}")
                debug(f"[API] Request will include header: X-Telemetry-Channel = {self._telemetry_channel}")

            resp = await self._http_client.post(endpoint, payload, extra_headers=extra_headers)

        content_type = resp.headers.get("Content-Type", "")

        assert hasattr(resp, 'status_code') or hasattr(resp, 'status') or 'status' in resp or 'status_code' in resp

        try:
            resp_json = resp.json()
        except Exception:
            resp_json = ''

        assert resp.status_code
        if resp.status_code >= 400 or 'error' in resp_json:
            if 'application/json' in content_type:
                try:
                    error_json = resp_json
                    # If the API already returns a dict with "error" or similar, just return it
                    if isinstance(error_json, (dict, list)):
                        DEFAULT_STATUS_CODE = 502
                        stat = error_json.get('status', DEFAULT_STATUS_CODE)
                        error_json['status_code'] = resp.status_code or error_json.get('status_code', stat)
                        if raw:
                            return (error_json, resp)
                        if self.raise_httpx:
                            raise httpx.HTTPStatusError(message=resp.text, request=resp.request, response=resp)
                        return error_json
                    else:
                        # If the JSON is not a dict or list, wrap it
                        error_info = {'error': error_json, 'status_code': resp.status_code}
                except Exception:
                    error_info = {'error': resp.text, 'status_code': resp.status_code}
            else:
                error_info = {'error': resp.text, 'status_code': resp.status_code}
            if raw:
                return (error_info, resp)
            if self.raise_httpx:
                raise httpx.HTTPStatusError(message=resp.text, request=resp.request, response=resp)
            return error_info

        data = resp.json() if 'application/json' in content_type else {"error": resp.text, "status_code": resp.status_code}

        # ------------------------------------------------------------------
        # Telemetry: give the background listener a window to pick up
        # the "response_sent" event then capture only the *new* events since
        # the previous call that match this request_id.
        # ------------------------------------------------------------------
        if self.telemetry_enabled and self._telemetry_listener is not None:
            # Wait for events to arrive, with polling to check if they've arrived
            # This is especially important for sync wrappers where the event loop
            # might close before events arrive. We poll up to 5 seconds for events.
            max_wait_time = 5.0
            poll_interval = 0.1
            waited = 0.0
            new_events = []
            
            while waited < max_wait_time:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                
                # Filter events by request_id to only get events for this specific request
                # Events may use either "request_id" or "rid" field
                all_new_events = self._telemetry_listener.events[self._telemetry_event_cursor :]
                new_events = [
                    e for e in all_new_events
                    if isinstance(e, dict) and (
                        e.get("request_id") == request_id or 
                        e.get("rid") == request_id
                    )
                ]
                
                # If we got a response_sent event, we're done (request completed)
                if any(e.get("event") == "response_sent" for e in new_events):
                    break
                
                # If we got any events and waited at least 0.5s, break early
                # (gives time for initial events to arrive)
                if new_events and waited >= 0.5:
                    break
            
            # Update cursor to mark all new events as processed (even if not for this request)
            # This prevents re-processing events on subsequent calls
            all_new_events = self._telemetry_listener.events[self._telemetry_event_cursor :]
            self._telemetry_event_cursor = len(self._telemetry_listener.events)
            
            # Store only events for this request
            self._last_telemetry_events = list(new_events)
            
            # Check total events received (for debugging)
            total_events = len(self._telemetry_listener.events)
            debug(f"[API] Total telemetry events so far: {total_events}, new events for request {request_id[:8]}: {len(new_events)}")
            
            if new_events:
                debug(f"[API] Received {len(new_events)} telemetry events for request {request_id[:8]}")
                # Log event types received
                event_types = [e.get("event", "unknown") if isinstance(e, dict) else "non-dict" for e in new_events]
                debug(f"[API] Event types: {event_types}")
            else:
                # Log all new events (not filtered) for debugging
                all_event_types = [e.get("event", "unknown") if isinstance(e, dict) else "non-dict" for e in all_new_events]
                if all_new_events:
                    debug(f"[API] No telemetry events received for request {request_id[:8]} (total events in listener: {total_events}, all new events: {len(all_new_events)}, event types: {all_event_types})")
                else:
                    debug(f"[API] No telemetry events received for request {request_id[:8]} (total events in listener: {total_events})")

        return (data, resp) if raw else data

    async def call(self, func: str, items: List[dict], params: Optional[dict] = None, raw: bool = False):
        if not items:
            return items

        endpoint = f"{self.model_name}/{func}/"
        endpoint = endpoint.lstrip("/")
        payload = {'items': items} if func != 'lookup' else {'query': items}
        # Always include params, even if empty, as API requires it
        payload['params'] = params if params is not None else {}
        try:
            res = await self._api_call(endpoint, payload, raw=raw if func == 'lookup' else False)
        except Exception as e:
            if self.raise_httpx:
                raise
            res = self._format_exception(e, 0)
        res = self._format_result(res)
        if isinstance(res, dict) and ('error' in res or 'status_code' in res):
            return res
        elif isinstance(res, (list, tuple)):
            return list(res)
        else:
            return res

    async def _batch_call_autoschema_or_manual(
        self,
        func: str,
        items,
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        raw: bool = False,
    ):
        if not items:
            return items

        is_lol, first_n, rest_iter = is_list_of_lists(items)
        results = []

        async def retry_batch_individually(batch):
            out = []
            for item in batch:
                single_result = await self.call(func, [item], params=params, raw=raw)
                if isinstance(single_result, list) and len(single_result) == 1:
                    out.append(single_result[0])
                else:
                    out.append(single_result)
            return out

        if is_lol:
            all_batches = chain(first_n, rest_iter)
            if output == 'disk':
                path = file_path or f"{self.model_name}_{func}_output.jsonl"
                async with aiofiles.open(path, 'w', encoding='utf-8') as file_handle:
                    for batch in all_batches:
                        batch_results = await self.call(func, batch, params=params, raw=raw)
                        if (
                            self.retry_error_batches and
                            isinstance(batch_results, dict) and
                            ('error' in batch_results or 'status_code' in batch_results)
                        ):
                            batch_results = await retry_batch_individually(batch)

                        if isinstance(batch_results, list):
                            assert len(batch_results) == len(batch), (
                                f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                "This is a contract violation."
                            )
                            for res in batch_results:
                                await file_handle.write(json.dumps(res) + '\n')
                        else:
                            for _ in batch:
                                await file_handle.write(json.dumps(batch_results) + '\n')
                        await file_handle.flush()

                        if stop_on_error and (
                            (isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results)) or
                            (isinstance(batch_results, list) and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results))
                        ):
                            break
                return
            else:
                for batch in all_batches:
                    batch_results = await self.call(func, batch, params=params, raw=raw)
                    if (
                        self.retry_error_batches and
                        isinstance(batch_results, dict) and
                        ('error' in batch_results or 'status_code' in batch_results)
                    ):
                        batch_results = await retry_batch_individually(batch)
                    if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                        results.extend([batch_results] * len(batch))
                        if stop_on_error:
                            break
                    elif isinstance(batch_results, list):
                        assert len(batch_results) == len(batch), (
                            f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                            "This is a contract violation."
                        )
                        results.extend(batch_results)
                        if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                            break
                    else:
                        results.append(batch_results)
                return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

        all_items = chain(first_n, rest_iter)
        max_batch = await self._get_max_batch_size(self.model_name, func) or 1

        if output == 'disk':
            path = file_path or f"{self.model_name}_{func}_output.jsonl"
            async with aiofiles.open(path, 'w', encoding='utf-8') as file_handle:
                for batch in batch_iterable(all_items, max_batch):
                    batch_results = await self.call(func, batch, params=params, raw=raw)

                    if (
                        self.retry_error_batches and
                        isinstance(batch_results, dict) and
                        ('error' in batch_results or 'status_code' in batch_results)
                    ):
                        batch_results = await retry_batch_individually(batch)
                        # After retry, always treat as list
                        for res in batch_results:
                            to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                            await file_handle.write(json.dumps(to_dump) + '\n')
                        await file_handle.flush()
                        if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                            break
                        continue  # move to next batch

                    if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                        for _ in batch:
                            await file_handle.write(json.dumps(batch_results) + '\n')
                        await file_handle.flush()
                        if stop_on_error:
                            break
                    else:
                        if not isinstance(batch_results, list):
                            batch_results = [batch_results]
                        assert len(batch_results) == len(batch), (
                            f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                            "This is a contract violation."
                        )
                        for res in batch_results:
                            to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                            await file_handle.write(json.dumps(to_dump) + '\n')
                        await file_handle.flush()
                        if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                            break

            return
        else:
            for batch in batch_iterable(all_items, max_batch):
                batch_results = await self.call(func, batch, params=params, raw=raw)

                if (
                    self.retry_error_batches and
                    isinstance(batch_results, dict) and
                    ('error' in batch_results or 'status_code' in batch_results)
                ):
                    batch_results = await retry_batch_individually(batch)
                    results.extend(batch_results)
                    if stop_on_error and any(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                        break
                    continue  # move to next batch


                if isinstance(batch_results, dict) and ('error' in batch_results or 'status_code' in batch_results):
                    results.extend([batch_results] * len(batch))
                    if stop_on_error:
                        break
                else:
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    assert len(batch_results) == len(batch), (
                        f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                        "This is a contract violation."
                    )
                    results.extend(batch_results)
                    if stop_on_error and all(isinstance(r, dict) and ('error' in r or 'status_code' in r) for r in batch_results):
                        break

            return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

    @staticmethod
    def _format_result(res: Union[dict, List[dict], Tuple[dict, int]]) -> Union[dict, List[dict], Tuple[dict, int]]:
        if isinstance(res, dict) and 'results' in res:
            return res['results']
        elif isinstance(res, list):
            if all(isinstance(x, dict) for x in res):
                return res
            raise ValueError("Unexpected response format")
        elif isinstance(res, dict) and ('error' in res or 'status_code' in res):
            return res
        return res


    def _format_exception(self, exc: Exception, index: int) -> dict:
        return {"error": str(exc), "index": index}

    @staticmethod
    def _unwrap_single(result):
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        return result

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def generate(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
    ):
        return await self._batch_call_autoschema_or_manual(
            "generate", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def predict(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
    ):
        return await self._batch_call_autoschema_or_manual(
            "predict", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def encode(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
    ):
        return await self._batch_call_autoschema_or_manual(
            "encode", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path
        )

    async def lookup(
        self,
        query: Union[dict, List[dict]],
        *,
        raw: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
    ):
        items = query if isinstance(query, list) else [query]
        res = await self.call("lookup", items, params=None, raw=raw)
        if raw:
            single = len(items) == 1
            if single:
                data, resp = res
                return LookupResult(data, resp)
            return [LookupResult(r[0], r[1]) for r in res]
        return res

    async def shutdown(self):
        await self._http_client.close()
        if self._listener_task is not None:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        if self._activity_task is not None:
            self._activity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._activity_task
            self._activity_task = None
        if self._progress_manager is not None:
            self._progress_manager.close_all()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    # ------------------------------------------------------------------
    # Fallback: best-effort cleanup if the user forgets to call shutdown()
    # ------------------------------------------------------------------
    def __del__(self):
        """Cancel background telemetry task and close the httpx client.

        This runs in the garbage-collector context, so we cannot *await*.
        We try to schedule the coroutine on a running loop or fall back to a
        short best-effort close when no loop is available.  Any exception is
        swallowed – this is purely hygiene to avoid noisy warnings during
        pytest shutdown.
        """
        try:
            if self._listener_task and not self._listener_task.done():
                self._listener_task.cancel()
            if self._activity_task and not self._activity_task.done():
                self._activity_task.cancel()
        except Exception:
            pass

        try:
            if self._http_client and not getattr(self._http_client, "is_closed", False):
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass  # no running loop

                if loop and not loop.is_closed():
                    loop.create_task(self._http_client.aclose())
                else:
                    # synchronous close – opens its own loop internally
                    try:
                        asyncio.run(self._http_client.aclose())
                    except RuntimeError:
                        # event-loop already closed; give up silently
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public helper: set or replace telemetry callback at runtime
    # ------------------------------------------------------------------
    def set_telemetry_handler(self, handler: Optional[Callable[[Any], Any]]):
        """Register a callback that will be invoked for every telemetry event.

        Can be called after the client is created; takes effect immediately for
        any active listener."""
        self._telemetry_handler = handler
        if self._telemetry_listener is not None:
            self._telemetry_listener._handler = handler

    @property
    def last_telemetry_events(self) -> List[Any]:
        """Get the last telemetry events captured for the most recent request.
        
        Returns a list of telemetry events (dicts) from the most recent API call.
        This is useful for testing and debugging. Events may be empty if:
        - Telemetry is disabled
        - No events were received yet
        - The websocket connection hasn't been established
        """
        return self._last_telemetry_events

# Synchronous wrapper for compatibility
@_synchronizer.sync
class BioLMApi(BioLMApiClient):
    pass

