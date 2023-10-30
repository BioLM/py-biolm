import aiohttp.resolver

from biolmai.const import BASE_API_URL, MULTIPROCESS_THREADS

aiohttp.resolver.DefaultResolver = aiohttp.resolver.AsyncResolver
from aiohttp import ClientSession, TCPConnector
from typing import List
import json
import asyncio

from asyncio import create_task, gather, run, sleep



async def get_one(session: ClientSession, slug: str, action: str,
                  payload: dict, response_key: str):
    pass


from aiohttp import ClientSession


async def get_one(session: ClientSession, url: str) -> None:
    print("Requesting", url)
    async with session.get(url) as resp:
        text = await resp.text()
        # await sleep(2)  # for demo purposes
        text_resp = text.strip().split("\n", 1)[0]
        print("Got response from", url, text_resp)
        return text_resp


async def get_one_biolm(session: ClientSession,
                        url: str,
                        pload: dict,
                        headers: dict,
                        response_key: str = None) -> None:
    print("Requesting", url)
    pload_batch = pload.pop('batch')
    pload_batch_size = pload.pop('batch_size')
    t = aiohttp.ClientTimeout(
        total=1600,  # 27 mins
        # total timeout (time consists connection establishment for a new connection or waiting for a free connection from a pool if pool connection limits are exceeded) default value is 5 minutes, set to `None` or `0` for unlimited timeout
        sock_connect=None,
        # Maximal number of seconds for connecting to a peer for a new connection, not given from a pool. See also connect.
        sock_read=None
        # Maximal number of seconds for reading a portion of data from a peer
    )
    async with session.post(url, headers=headers, json=pload, timeout=t) as resp:
        resp_json = await resp.json()
        resp_json['batch'] = pload_batch
        status_code = resp.status
        expected_root_key = response_key
        to_ret = []
        if status_code and status_code == 200:
            list_of_individual_seq_results = resp_json[expected_root_key]
        # elif local_err:
        #     list_of_individual_seq_results = [{'error': resp_json}]
        elif status_code and status_code != 200 and isinstance(resp_json, dict):
            list_of_individual_seq_results = [resp_json] * pload_batch_size
        else:
            raise ValueError("Unexpected response in parser")
        for idx, item in enumerate(list_of_individual_seq_results):
            d = {'status_code': status_code,
                 'batch_id': pload_batch,
                 'batch_item': idx}
            if not status_code or status_code != 200:
                d.update(item)  # Put all resp keys at root there
            else:
                # We just append one item, mimicking a single seq in POST req/resp
                d[expected_root_key] = []
                d[expected_root_key].append(item)
            to_ret.append(d)
        return to_ret

        # text = await resp.text()
        # await sleep(2)  # for demo purposes
        # text_resp = text.strip().split("\n", 1)[0]
        # print("Got response from", url, text_resp)
        return j


async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)


async def get_all(urls: List[str], num_concurrent: int) -> List:
    url_iterator = iter(urls)
    keep_going = True
    results = []
    async with ClientSession() as session:
        while keep_going:
            tasks = []
            for _ in range(num_concurrent):
                try:
                    url = next(url_iterator)
                except StopIteration:
                    keep_going = False
                    break
                new_task = create_task(get_one(session, url))
                tasks.append(new_task)
            res = await gather(*tasks)
            results.extend(res)
    return results


async def get_all_biolm(url: str,
                        ploads: List[dict],
                        headers: dict,
                        num_concurrent: int,
                        response_key: str = None) -> List:
    ploads_iterator = iter(ploads)
    keep_going = True
    results = []
    connector = aiohttp.TCPConnector(limit=100,
                                     limit_per_host=50,
                                     ttl_dns_cache=60)
    ov_tout = aiohttp.ClientTimeout(
        total=None,
        # total timeout (time consists connection establishment for a new connection or waiting for a free connection from a pool if pool connection limits are exceeded) default value is 5 minutes, set to `None` or `0` for unlimited timeout
        sock_connect=None,
        # Maximal number of seconds for connecting to a peer for a new connection, not given from a pool. See also connect.
        sock_read=None
        # Maximal number of seconds for reading a portion of data from a peer
    )
    async with ClientSession(connector=connector, timeout=ov_tout) as session:
        while keep_going:
            tasks = []
            for _ in range(num_concurrent):
                try:
                    pload = next(ploads_iterator)
                except StopIteration:
                    keep_going = False
                    break
                new_task = create_task(get_one_biolm(session, url, pload,
                                                     headers, response_key))
                tasks.append(new_task)
            res = await gather(*tasks)
            results.extend(res)
    return results


async def async_main(urls, concurrency) -> List:
    return await get_all(urls, concurrency)


async def async_api_calls(model_name,
                          action,
                          headers,
                          payloads,
                          response_key=None):
    """Hit an arbitrary BioLM model inference API."""
    # Normally would POST multiple sequences at once for greater efficiency,
    # but for simplicity sake will do one at at time right now
    url = f'{BASE_API_URL}/models/{model_name}/{action}/'

    if not isinstance(payloads, (list, dict)):
        err = "API request payload must be a list or dict, got {}"
        raise AssertionError(err.format(type(payloads)))

    concurrency = int(MULTIPROCESS_THREADS)
    return await get_all_biolm(url, payloads, headers, concurrency,
                               response_key)

    # payload = json.dumps(payload)
    # session = requests_retry_session()
    # tout = urllib3.util.Timeout(total=180, read=180)
    # response = retry_minutes(session, url, headers, payload, tout, mins=10)
    # # If token expired / invalid, attempt to refresh.
    # if response.status_code == 401 and os.path.exists(ACCESS_TOK_PATH):
    #     # Add jitter to slow down in case we're multiprocessing so all threads
    #     # don't try to re-authenticate at once
    #     time.sleep(random.random() * 4)
    #     with open(ACCESS_TOK_PATH, 'r') as f:
    #         access_refresh_dict = json.load(f)
    #     refresh = access_refresh_dict.get('refresh')
    #     if not refresh_access_token(refresh):
    #         err = "Unauthenticated! Please run `biolmai status` to debug or " \
    #               "`biolmai login`."
    #         raise AssertionError(err)
    #     headers = get_user_auth_header()  # Need to re-get these now
    #     response = retry_minutes(session, url, headers, payload, tout, mins=10)
    # return response
