import pytest
import asyncio
import random
import string
from biolmai.client import BioLMApiClient
from biolmai.biolmai import BioLM

def random_sequence(length=5):
    return ''.join(random.choices('ACDEFGHIKLMNPQRSTVWYB', k=length))

@pytest.fixture(scope='function')
def model():
    return BioLMApiClient("esm2-8m", raise_httpx=False, unwrap_single=False)

@pytest.mark.asyncio
async def test_large_batch_encode_consistency(model):
    items = [{"sequence": random_sequence()} for _ in range(3)]
    # 1. New async method (schema-batched, concurrent)
    results_async = await model.encode(items=items)
    # 2. Old method: _batch_call (single-item batching, sequential)
    results_old = await model._batch_call("encode", items)
    # 3. Universal client (sync, run in thread)
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, raise_httpx=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    # All should be lists of the same length
    assert isinstance(results_async, list)
    assert isinstance(results_old, list)
    assert isinstance(results_biolm, list)
    assert len(results_async) == len(items)
    assert len(results_old) == len(items)
    assert len(results_biolm) == len(items)
    # All results should be dicts with "embeddings"
    for r1, r2, r3 in zip(results_async, results_old, results_biolm):
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        assert isinstance(r3, dict)
        assert "embeddings" in r1
        assert "embeddings" in r2
        assert "embeddings" in r3
        # Optionally, check values are equal (if deterministic)
        # assert r1 == r2 == r3

@pytest.mark.asyncio
async def test_large_batch_encode_with_errors(model):
    items = [{"sequence": random_sequence()} for _ in range(20)] + \
            [{"sequence": "BAD::BAD"} for _ in range(5)] + \
            [{"sequence": random_sequence()} for _ in range(15)]
    results_async = await model.encode(items=items, stop_on_error=False)
    results_old = await model._batch_call("encode", items, stop_on_error=False)
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=False, raise_httpx=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    assert len(results_async) == len(items)
    assert len(results_old) == len(items)
    assert len(results_biolm) == len(items)
    for i, (r1, r2, r3) in enumerate(zip(results_async, results_old, results_biolm)):
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        assert isinstance(r3, dict)
        if 20 <= i < 25:
            assert "error" in r1 and "error" in r2 and "error" in r3
        else:
            assert "embeddings" in r1
            assert "embeddings" in r2
            assert "embeddings" in r3

@pytest.mark.asyncio
async def test_large_batch_encode_stop_on_error(model):
    items = [{"sequence": random_sequence()} for _ in range(10)] + \
            [{"sequence": "BAD::BAD"}] + \
            [{"sequence": random_sequence()} for _ in range(10)]
    results_async = await model.encode(items=items, stop_on_error=True)
    results_old = await model._batch_call("encode", items, stop_on_error=True)
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=True, raise_httpx=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    # Should stop after the first error (11 results)
    assert len(results_async) == 11
    assert len(results_old) == 11
    assert len(results_biolm) == 11
    assert "error" in results_async[-1]
    assert "error" in results_old[-1]
    assert "error" in results_biolm[-1]
    for r1, r2, r3 in zip(results_async[:-1], results_old[:-1], results_biolm[:-1]):
        assert "embeddings" in r1
        assert "embeddings" in r2
        assert "embeddings" in r3


@pytest.mark.asyncio
async def test_biolm_stop_on_error_shorter_results(model):
    # 10 valid, 1 invalid, 10 valid
    items = [{"sequence": random_sequence()} for _ in range(10)] + \
            [{"sequence": "BAD::BAD"}] + \
            [{"sequence": random_sequence()} for _ in range(10)]
    # Async client and old method: continue on error
    results_async = await model.encode(items=items, stop_on_error=False)
    results_old = await model._batch_call("encode", items, stop_on_error=False)
    # BioLM: stop on error
    loop = asyncio.get_event_loop()
    def run_biolm():
        return BioLM(entity="esm2-8m", action="encode", items=items, stop_on_error=True, raise_httpx=False)
    results_biolm = await loop.run_in_executor(None, run_biolm)
    # BioLM should have fewer results than the others
    assert len(results_biolm) < len(results_async)
    assert len(results_biolm) < len(results_old)
    # The others should be full length
    assert len(results_async) == len(items)
    assert len(results_old) == len(items)
    # The last result in BioLM should be an error
    assert "error" in results_biolm[-1]
    # The first 10 should be valid
    for r in results_biolm[:10]:
        assert "embeddings" in r

@pytest.mark.asyncio
async def test_predict_stop_on_error_vs_continue(model):
    # 1 valid, 1 invalid, 1 valid
    items = [
        {"sequence": "MDN<mask>LE"},
        {"sequence": "MENDELSEMYEFFF<mask>EFMLYRRTELSYYYUPPPPPU::"},
        {"sequence": "MD<mask>ELE"}
    ]
    # stop_on_error=False: should return all results
    results_continue = await model.predict(items=items, stop_on_error=False)
    assert isinstance(results_continue, list)
    assert len(results_continue) == 3
    # stop_on_error=True: should stop at first error (so only 2 results)
    results_stop = await model.predict(items=items, stop_on_error=True)
    assert isinstance(results_stop, list)
    assert len(results_stop) < len(results_continue)
    assert len(results_stop) == 2
    # The last result should be an error
    assert "error" in results_stop[-1]
    # The first should be a valid prediction
    assert "logits" in results_stop[0]
