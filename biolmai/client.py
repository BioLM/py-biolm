import functools
from typing import Callable
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version
import asyncio
import httpx
import os
import json
from typing import Optional, Union, List, Any, Dict, Tuple
from synchronicity import Synchronizer
from collections import namedtuple, OrderedDict
from contextlib import asynccontextmanager

import sys

def debug(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,  # Python 3.8+
)

BASE_DOMAIN = "https://biolm.ai"
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
                    if isinstance(value, (list, tuple)) and len(value) == 0:
                        raise ValueError(
                            f"Parameter '{param}' must not be an empty {type(value).__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator


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

class Throttle:
    def __init__(self, max_rps: Optional[int]):
        self._sem = asyncio.Semaphore(max_rps) if max_rps else None

    @asynccontextmanager
    async def limit(self):
        if self._sem:
            async with self._sem:
                yield
        else:
            yield

class HttpClient:

    def __init__(self, base_url: str, headers: Dict[str, str], timeout: httpx.Timeout):
        self._base_url = base_url.rstrip("/") + "/"
        self._headers = headers
        self._timeout = timeout
        self._async_client: Optional[httpx.AsyncClient] = None
        self._transport = None
        # Try to use aiodns if available
        try:
            from httpx import AsyncHTTPTransport
            from httpx._transports.default import AsyncResolver
            self._transport = AsyncHTTPTransport(resolver=AsyncResolver())
        except ImportError:
            self._transport = None

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
        debug("DEBUG async post: endpoint =" + endpoint)
        if "Content-Type" not in client.headers:
            client.headers["Content-Type"] = "application/json"
        return await client.post(endpoint, json=payload)

    async def close(self):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None


class BioLMApiClient:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://biolm.ai/api/v3",
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        raise_httpx: bool = True,
        unwrap_single: bool = False,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") + "/"  # Ensure trailing slash
        self.timeout = timeout
        self.raise_httpx = raise_httpx
        self.unwrap_single = unwrap_single
        self._headers = CredentialsProvider.get_auth_headers(api_key)
        self._http_client = HttpClient(self.base_url, self._headers, self.timeout)
        self._rps_limit = None
        self._throttle = None
        self._rps_limit_lock = asyncio.Lock()  # Only locks on initial GET of throttle info, then throttle cached

    async def _ensure_throttle(self):
        if self._throttle is not None:
            return
        async with self._rps_limit_lock:
            if self._throttle is not None:
                return
            self._rps_limit = await self._fetch_rps_limit_async()
            self._throttle = Throttle(self._rps_limit)

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
        await self._ensure_throttle()
        async with self._throttle.limit():
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

    async def _batch_call(
        self,
        func: str,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
        raw: bool = False,
    ):
        endpoint = f"{self.model_name}/{func}/"
        endpoint = endpoint.lstrip("/")  # Make sure no starting slash, it's in `base_url``
        debug("DEBUG async _batch_call: base_url =" + self.base_url)
        debug("DEBUG async _batch_call: full URL =" + self.base_url + endpoint)
        results = []
        single = len(items) == 1
        file_handle = None
        if output == 'disk':
            path = file_path or f"{self.model_name}_{func}_output.jsonl"
            file_handle = open(path, 'w', encoding='utf-8')
        for idx, item in enumerate(items):
            payload = {'items': [item]} if func != 'lookup' else {'query': item}
            if params:
                payload['params'] = params
            try:
                res = await self._api_call(endpoint, payload, raw=raw if func == 'lookup' else False)
                debug(f"DEBUG async _api_call: endpoint={endpoint}, status={res.get('status_code')}")
            except Exception as e:
                if self.raise_httpx:
                    raise
                res = self._format_exception(e, idx)
            res = self._format_result(res)
            results.append(res)
            if file_handle:
                to_dump = res[0] if (raw and isinstance(res, tuple)) else res
                file_handle.write(json.dumps(to_dump) + '\n')
                file_handle.flush()
            if stop_on_error and isinstance(res, dict) and ('error' in res or 'status_code' in res):
                break
        if file_handle:
            file_handle.close()
        return self._unwrap_single(results) if self.unwrap_single is True and single else results

    @staticmethod
    def _format_result(res: Union[dict, List[dict], Tuple[dict, int]]) -> Union[dict, List[dict], Tuple[dict, int]]:
        if isinstance(res, dict) and 'results' in res:
            first_result = res['results'][0]  # Sending one at a time
            return first_result
        elif isinstance(res, list):
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

    async def generate(
        self,
        *,
        items: List[dict],
        params: Optional[dict] = None,
        stop_on_error: bool = False,
        output: str = 'memory',
        file_path: Optional[str] = None,
    ):
        return await self._batch_call(
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
        return await self._batch_call(
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
        return await self._batch_call(
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
        res = await self._batch_call(
            "lookup", items, params=None, stop_on_error=False, output=output, file_path=file_path, raw=raw
        )
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
        # Optionally, you could initialize resources here
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()

# Synchronous wrapper for compatibility
@_synchronizer.sync
class BioLMApi(BioLMApiClient):
    pass
