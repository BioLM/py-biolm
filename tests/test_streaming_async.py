import pytest
from biolmai.client import BioLMApiClient

@pytest.fixture(scope='function')
def model():
    """Provide a BioLMApiClient instance for streaming tests."""
    return BioLMApiClient("esmfold", raise_httpx=False, unwrap_single=False)

@pytest.mark.asyncio
async def test_predict_stream_async_yields_indexed_results(model):
    """
    Ensure predict_stream_async yields (index, result) pairs with correct indices and result dicts.
    """
    items = [{"sequence": "MDNELE"}, {"sequence": "MUUUUDANLEPY"}]
    seen = set()
    async for idx, res in model.predict_stream_async(items=items):
        assert isinstance(idx, int)
        assert idx in (0, 1)
        assert isinstance(res, dict)
        seen.add(idx)
    assert seen == {0, 1}

@pytest.mark.asyncio
async def test_predict_stream_ordered_async_returns_ordered_results(model):
    """
    Ensure predict_stream_ordered_async yields results in the same order as input items.
    """
    items = [{"sequence": "MDNELE"}, {"sequence": "MUUUUDANLEPY"}]
    results = []
    async for res in model.predict_stream_ordered_async(items=items):
        assert isinstance(res, dict)
        results.append(res)
    assert len(results) == 2
    # Check no errors and correct keys
    for res in results:
        assert "mean_plddt" in res
        assert "pdb" in res

@pytest.mark.asyncio
async def test_predict_stream_ordered_async_handles_errors(model):
    """
    Check that predict_stream_ordered_async yields errors in order and continues.
    """
    items = [{"sequence": "BAD::BAD"}, {"sequence": "MDNELE"}]
    results = []
    async for res in model.predict_stream_ordered_async(items=items):
        results.append(res)
    assert len(results) == 2
    # First result should be an error
    assert isinstance(results[0], dict)
    assert "error" in results[0]
    # Second result should be a valid prediction
    assert "mean_plddt" in results[1]
    assert "pdb" in results[1] 