import asyncio
import functools
import gzip
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

from biolmai.const import BIOLMAI_BASE_API_URL

USER_BIOLM_DIR = os.path.join(os.path.expanduser("~"), ".biolmai")
ACCESS_TOK_PATH = os.path.join(USER_BIOLM_DIR, "credentials")
TIMEOUT_MINS = 20  # Match API server's keep-alive/timeout
DEFAULT_TIMEOUT = httpx.Timeout(TIMEOUT_MINS * 60, connect=10.0)

# Connection pool limits
DEFAULT_LIMITS = httpx.Limits(
    max_connections=100,              # Total concurrent connections
    max_keepalive_connections=20,    # Idle connections to keep alive (default)
    keepalive_expiry=30.0            # Keep idle connections for 30s
)

# Retry configuration for network errors
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 4.0  # Start with 1 second

# Mapping of httpx exception types to user-friendly error messages
HTTPX_EXCEPTION_MESSAGES = {
    httpx.ReadError: "Connection read error: The server closed the connection unexpectedly or data could not be read",
    httpx.ConnectError: "Connection error: Failed to establish a connection to the server",
    httpx.NetworkError: "Network error: A network-related error occurred",
    httpx.WriteError: "Connection write error: Failed to send data to the server",
    httpx.CloseError: "Connection close error: Failed to close the connection properly",
    httpx.TimeoutException: "Request timeout: The request took too long to complete",
    httpx.ConnectTimeout: "Connection timeout: Failed to establish a connection within the timeout period",
    httpx.ReadTimeout: "Read timeout: The server did not send data within the timeout period",
    httpx.WriteTimeout: "Write timeout: Failed to send data within the timeout period",
    httpx.PoolTimeout: "Connection pool timeout: No connections available in the pool",
    httpx.HTTPStatusError: "HTTP error: The server returned an error status code",
    httpx.RequestError: "Request error: An error occurred while making the request",
    httpx.TransportError: "Transport error: An error occurred at the transport layer",
}

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

    def __init__(
        self, 
        base_url: str, 
        headers: Dict[str, str], 
        timeout: httpx.Timeout,
        compress_requests: bool = True,
        compress_threshold: int = 256,
        limits: Optional[httpx.Limits] = None,
        http2: bool = True
    ):
        self._base_url = base_url.rstrip("/") + "/"
        self._headers = headers
        self._timeout = timeout
        self._compress_requests = compress_requests
        self._compress_threshold = compress_threshold
        self._async_client: Optional[httpx.AsyncClient] = None
        self._limits = limits or DEFAULT_LIMITS
        self._http2 = http2
        self._transport = None
        # Create transport with HTTP/2 enabled
        self._transport = AsyncHTTPTransport(http2=http2)

    async def get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or getattr(self._async_client, 'is_closed', False):
            if self._transport:
                self._async_client = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers=self._headers,
                    timeout=self._timeout,
                    transport=self._transport,
                    limits=self._limits,
                )
            else:
                self._async_client = httpx.AsyncClient(
                    base_url=self._base_url,
                    headers=self._headers,
                    timeout=self._timeout,
                    limits=self._limits,
                )
        return self._async_client

    async def _post_with_retry(
        self, 
        client: httpx.AsyncClient, 
        endpoint: str, 
        **request_kwargs
    ) -> httpx.Response:
        """POST with retry logic for network errors (ReadError, ConnectError, NetworkError)."""
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return await client.post(endpoint, **request_kwargs)
            except (httpx.ReadError, httpx.ConnectError, httpx.NetworkError) as e:
                last_exception = e
                # Don't retry on the last attempt
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                # Last attempt failed, re-raise
                raise
            # Don't retry on HTTP status errors (4xx, 5xx) - these are not network errors
            except httpx.HTTPStatusError:
                raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise httpx.NetworkError("Failed to make request after retries")

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

        # Check if we should compress
        if self._compress_requests:
            # Use the same JSON encoding as httpx to get accurate size
            json_bytes = json_dumps(payload, ensure_ascii=False).encode("utf-8")
            
            if len(json_bytes) > self._compress_threshold:
                compressed_body = gzip.compress(json_bytes)
                if headers is None:
                    # Start with client's default headers but ensure we override Content-Type
                    headers = dict(client.headers)
                else:
                    # Merge extra_headers with client headers, then override for compression
                    headers = {**client.headers, **headers}
                # Set headers for compressed request - these must override any defaults
                # When using content= (raw bytes), we need to explicitly set Content-Type
                # Important: Set these headers explicitly to ensure they override any defaults
                headers["Content-Type"] = "application/json"
                headers["Content-Encoding"] = "gzip"
                headers["Content-Length"] = str(len(compressed_body))
                # Use content= for raw bytes (httpx recommends this over data= for raw content)
                # The headers we provide should override client defaults
                r = await self._post_with_retry(
                    client, endpoint, content=compressed_body, headers=headers
                )
                return r
        
        # Default: send uncompressed JSON
        r = await self._post_with_retry(client, endpoint, json=payload, headers=headers)
        return r

    async def get(self, endpoint: str) -> httpx.Response:
        """GET with retry logic for network errors."""
        client = await self.get_async_client()
        endpoint = endpoint.lstrip("/")
        if not endpoint.endswith("/"):
            endpoint += "/"
        
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                return await client.get(endpoint)
            except (httpx.ReadError, httpx.ConnectError, httpx.NetworkError) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                raise
            except httpx.HTTPStatusError:
                raise
        
        if last_exception:
            raise last_exception
        raise httpx.NetworkError("Failed to make request after retries")


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

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
        compress_requests: bool = True,
        compress_threshold: int = 256,
    ):
        # Use base_url parameter if provided, otherwise use default from const
        final_base_url = base_url if base_url is not None else BIOLMAI_BASE_API_URL
        
        self.model_name = model_name
        self.base_url = final_base_url.rstrip("/") + "/"  # Ensure trailing slash
        self.timeout = timeout
        self.raise_httpx = raise_httpx
        self.unwrap_single = unwrap_single
        self._headers = CredentialsProvider.get_auth_headers(api_key)
        self._http_client = HttpClient(
            self.base_url, 
            self._headers, 
            self.timeout,
            compress_requests=compress_requests,
            compress_threshold=compress_threshold
        )
        self._semaphore = None
        self._rate_limiter = None
        self._rate_limit_lock = None
        self._rate_limit_initialized = False
        self.retry_error_batches = retry_error_batches


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
            # Reuse existing http_client instead of creating a new client
            resp = await self._http_client.get(f"/{self.model_name}/")
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
        async with self._limit():
            resp = await self._http_client.post(endpoint, payload)
        content_type = resp.headers.get("Content-Type", "")

        assert hasattr(resp, 'status_code') or hasattr(resp, 'status') or 'status' in resp or 'status_code' in resp

        # Read response content once and parse as JSON or use as text
        # Note: resp.json() and resp.text both consume the body stream, so we read text first, then parse
        resp_json = None
        resp_text = None
        try:
            # Read as text first (this consumes the body, but we can parse it multiple times)
            resp_text = resp.text
            # Try to parse the text as JSON
            if resp_text:
                try:
                    resp_json = json.loads(resp_text)
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, use as text
                    resp_json = resp_text
            else:
                resp_json = ''
        except Exception:
            # If reading text fails, try json() as fallback (may also fail if body already consumed)
            try:
                resp_json = resp.json()
            except Exception:
                resp_text = ''
                resp_json = ''

        assert resp.status_code
        # Check for errors: non-200 status OR top-level error/detail/description keys
        # Note: v3 can return validation errors (200 status) with item-specific errors
        has_error_key = isinstance(resp_json, dict) and ('error' in resp_json or 'detail' in resp_json or 'description' in resp_json)
        is_error_status = resp.status_code >= 400
        
        # Check if response has both error and results (partial success scenario)
        # In this case, we return the response as-is so batch processing can handle item-level errors
        has_results = isinstance(resp_json, dict) and 'results' in resp_json
        
        if is_error_status or (has_error_key and not has_results):
            if 'application/json' in content_type:
                try:
                    # Copy to avoid mutating original response
                    if isinstance(resp_json, dict):
                        error_json = resp_json.copy()
                    else:
                        error_json = resp_json
                    # If the API already returns a dict with "error" or similar, normalize it
                    if isinstance(error_json, (dict, list)):
                        # Normalize error keys - ensure "error" key exists
                        if isinstance(error_json, dict) and 'error' not in error_json:
                            if 'detail' in error_json:
                                error_json['error'] = error_json['detail']
                            elif 'description' in error_json:
                                error_json['error'] = error_json['description']
                        # Handle empty error strings - provide fallback message
                        elif isinstance(error_json, dict) and 'error' in error_json:
                            error_value = error_json.get('error')
                            if not error_value or (isinstance(error_value, str) and not error_value.strip()):
                                # Empty or whitespace-only error string - use fallback
                                error_json['error'] = (
                                    error_json.get('detail') or 
                                    error_json.get('description') or 
                                    f"API returned error status {resp.status_code}"
                                )
                        DEFAULT_STATUS_CODE = 502
                        stat = error_json.get('status', DEFAULT_STATUS_CODE)
                        error_json['status_code'] = resp.status_code or error_json.get('status_code', stat)
                        if raw:
                            return (error_json, resp)
                        if self.raise_httpx:
                            # Use stored resp_text if available, otherwise use error_json as string
                            error_msg = resp_text if resp_text is not None else str(resp_json)
                            raise httpx.HTTPStatusError(message=error_msg, request=resp.request, response=resp)
                        return error_json
                    else:
                        # If the JSON is not a dict or list, wrap it
                        error_info = {'error': error_json, 'status_code': resp.status_code}
                except Exception:
                    # Use stored resp_text if available, otherwise use fallback message
                    error_text = resp_text if resp_text is not None else f"API returned error status {resp.status_code}"
                    error_info = {'error': error_text, 'status_code': resp.status_code}
            else:
                # Use stored resp_text if available, otherwise use fallback message
                error_text = resp_text if resp_text is not None else f"API returned error status {resp.status_code}"
                error_info = {'error': error_text, 'status_code': resp.status_code}
            if raw:
                return (error_info, resp)
            if self.raise_httpx:
                # Use stored resp_text if available, otherwise use error_info error as string
                error_msg = resp_text if resp_text is not None else str(error_info.get('error', ''))
                raise httpx.HTTPStatusError(message=error_msg, request=resp.request, response=resp)
            return error_info

        # Success response (200 status, no error keys)
        # Use the already-parsed resp_json instead of calling resp.json() again
        if isinstance(resp_json, dict):
            data = resp_json
        elif 'application/json' in content_type:
            # If resp_json is not a dict but content-type says JSON, something went wrong
            # Use stored resp_text if available, otherwise use fallback message
            error_msg = resp_text if resp_text is not None else f"Failed to parse JSON response (status {resp.status_code})"
            data = {"error": error_msg, "status_code": resp.status_code}
        else:
            # Use stored resp_text if available, otherwise use fallback message
            error_msg = resp_text if resp_text is not None else f"Failed to read response body (status {resp.status_code})"
            data = {"error": error_msg, "status_code": resp.status_code}
        
        # If response has both error and results, it's a partial success scenario
        # Add status_code and return as-is so batch processing can handle item-level errors
        if isinstance(data, dict) and 'error' in data and 'results' in data:
            # Partial success: some items have errors, some have results
            if 'status_code' not in data:
                data['status_code'] = resp.status_code
            return (data, resp) if raw else data
        
        # Pure success response - return as-is (will be formatted by _format_result)
        return (data, resp) if raw else data

    async def call(self, func: str, items: List[dict], params: Optional[dict] = None, raw: bool = False):
        if not items:
            return items

        endpoint = f"{self.model_name}/{func}/"
        endpoint = endpoint.lstrip("/")
        payload = {'items': items} if func != 'lookup' else {'query': items}
        if params:
            payload['params'] = params
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
        overwrite: bool = False,
    ):
        if not items:
            return items

        is_lol, first_n, rest_iter = is_list_of_lists(items)

        # Check if file exists and overwrite is False
        if output == 'disk' and not overwrite:
            path = file_path or f"{self.model_name}_{func}_output.jsonl"
            if os.path.exists(path):
                # Read existing file and return its contents
                results = []
                async with aiofiles.open(path, 'r', encoding='utf-8') as file_handle:
                    async for line in file_handle:
                        line = line.strip()
                        if line:
                            try:
                                results.append(json.loads(line))
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
                # Return in the same format as memory output would
                if is_lol:
                    return results
                return self._unwrap_single(results) if self.unwrap_single and len(results) == 1 else results

        results = []

        def parse_validation_errors(batch_results, batch_size):
            """Parse validation errors with items__N__sequence keys and distribute to specific items.
            
            Returns a list of results, one per item in the batch.
            - Items with validation errors get item-specific error dicts
            - Items without errors get the batch error dict (batch-level failure)
            - If results array exists, valid items get their results
            """
            error_dict = batch_results.get('error', {})
            status_code = batch_results.get('status_code', 0)
            
            # Check if this is a validation error with item-specific errors
            is_validation_error = (
                status_code == 200 and 
                isinstance(error_dict, dict) and 
                any(key.startswith('items__') and '__sequence' in key for key in error_dict.keys())
            )
            
            if not is_validation_error:
                # Batch-level error (4xx/5xx): apply to all items
                return [batch_results] * batch_size
            
            # Parse which items have validation errors
            import re
            error_item_indices = set()
            for key in error_dict.keys():
                match = re.match(r'items__(\d+)__', key)
                if match:
                    error_item_indices.add(int(match.group(1)))
            
            # Check if response also has results (partial success scenario)
            if 'results' in batch_results:
                # Partial success: some items have errors, some have results
                results_list = batch_results.get('results', [])
                item_results = []
                for idx in range(batch_size):
                    if idx in error_item_indices:
                        # Item has validation error
                        item_error = {
                            'error': {k: v for k, v in error_dict.items() if f'items__{idx}__' in k},
                            'status_code': status_code
                        }
                        item_results.append(item_error)
                    elif idx < len(results_list):
                        # Item has successful result
                        item_results.append(results_list[idx])
                    else:
                        # Item index out of range - apply batch error as fallback
                        item_results.append(batch_results)
                return item_results
            else:
                # Full batch validation error: all items get error, but preserve item-specific error info
                item_results = []
                for idx in range(batch_size):
                    if idx in error_item_indices:
                        # Item has specific validation error
                        item_error = {
                            'error': {k: v for k, v in error_dict.items() if f'items__{idx}__' in k},
                            'status_code': status_code
                        }
                        item_results.append(item_error)
                    else:
                        # Item doesn't have specific error, but batch failed validation
                        # Apply the full error dict (batch-level validation failure)
                        item_results.append(batch_results)
                return item_results

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
                            # For 'generate' actions, models may return multiple results per item
                            # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                            if func != "generate":
                                assert len(batch_results) == len(batch), (
                                    f"API returned {len(batch_results)} results for a batch of {len(batch)} items. "
                                    "This is a contract violation."
                                )
                            for res in batch_results:
                                await file_handle.write(json.dumps(res) + '\n')
                        else:
                            # Parse validation errors to distribute to specific items
                            item_results = parse_validation_errors(batch_results, len(batch))
                            for res in item_results:
                                await file_handle.write(json.dumps(res) + '\n')
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
                        # Parse validation errors to distribute to specific items
                        item_results = parse_validation_errors(batch_results, len(batch))
                        results.extend(item_results)
                        if stop_on_error:
                            break
                    elif isinstance(batch_results, list):
                        # For 'generate' actions, models may return multiple results per item
                        # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                        if func != "generate":
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
                        # For 'generate' actions, models may return multiple results per item
                        # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                        if func != "generate":
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
            for batch_idx, batch in enumerate(batch_iterable(all_items, max_batch)):
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
                    # Parse validation errors to distribute to specific items
                    item_results = parse_validation_errors(batch_results, len(batch))
                    results.extend(item_results)
                    if stop_on_error:
                        break
                elif isinstance(batch_results, list):
                    # Successful batch - results is already a list
                    results.extend(batch_results)
                else:
                    if not isinstance(batch_results, list):
                        batch_results = [batch_results]
                    # For 'generate' actions, models may return multiple results per item
                    # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
                    if func != "generate":
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
        """Format an exception as an error dict, with fallback for empty error messages."""
        error_msg = str(exc)
        
        # Handle empty or whitespace-only error messages
        if not error_msg or not error_msg.strip():
            # Check if it's a known httpx exception type
            exc_type = type(exc)
            if exc_type in HTTPX_EXCEPTION_MESSAGES:
                error_msg = HTTPX_EXCEPTION_MESSAGES[exc_type]
            else:
                # Fallback: use exception class name
                error_msg = f"{exc_type.__name__}: Network or connection error occurred"
        
        return {"error": error_msg, "index": index}

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
        overwrite: bool = False,
    ):
        return await self._batch_call_autoschema_or_manual(
            "generate", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
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
        overwrite: bool = False,
    ):
        return await self._batch_call_autoschema_or_manual(
            "predict", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
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
        overwrite: bool = False,
    ):
        return await self._batch_call_autoschema_or_manual(
            "encode", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def search(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        return await self._batch_call_autoschema_or_manual(
            "search", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
        )

    @type_check({'items': (list, tuple), 'params': (dict, OrderedDict, None)})
    async def score(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        return await self._batch_call_autoschema_or_manual(
            "score", items, params=params, stop_on_error=stop_on_error, output=output, file_path=file_path, overwrite=overwrite
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

# Synchronous wrapper for compatibility
@_synchronizer.sync
class BioLMApi(BioLMApiClient):
    pass
