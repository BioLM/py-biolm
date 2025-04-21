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

    async def post(self, endpoint: str, payload: dict) -> httpx.Response:
        client = await self.get_async_client()
        # Remove leading slash, ensure trailing slash
        endpoint = endpoint.lstrip("/")
        if not endpoint.endswith("/"):
            endpoint += "/"
        if "Content-Type" not in client.headers:
            client.headers["Content-Type"] = "application/json"
        r = await client.post(endpoint, json=payload)
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

class BioLMApiClient:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://biolm.ai/api/v3",
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        raise_httpx: bool = True,
        unwrap_single: bool = False,
        semaphore: 'Optional[Union[int, asyncio.Semaphore]]' = None,
        rate_limit: 'Optional[str]' = None,
        retry_error_batches: bool = False,

    ):
        self.model_name = model_name
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
        async with self._limit():
            resp = await self._http_client.post(endpoint, payload)
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

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

# Synchronous wrapper for compatibility
@_synchronizer.sync
class BioLMApi(BioLMApiClient):
    pass
