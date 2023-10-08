import aiohttp.resolver
aiohttp.resolver.DefaultResolver = aiohttp.resolver.AsyncResolver
from aiohttp import ClientSession, TCPConnector
from typing import List
import asyncio

connector = aiohttp.TCPConnector(limit=2,
                                 resolver=aiohttp.resolver.AsyncResolver,
                                 ttl_dns_cache=60)

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


async def async_range(count):
    for i in range(count):
        yield(i)
        await asyncio.sleep(0.0)


async def get_all(urls: list[str], num_concurrent: int) -> List:
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


async def async_main(urls, concurrency) -> List:
    return await get_all(urls, concurrency)

