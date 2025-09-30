import json
import logging

from biolmai.client import BioLMApi

LOGGER = logging.getLogger(__name__)

import pytest
from biolmai.client import BioLMApiClient

@pytest.fixture(scope='function')
def model():
    return BioLMApiClient("esmfold", raise_httpx=False, unwrap_single=False)

@pytest.mark.asyncio
async def test_valid_sequence(model):
    result = await model.predict(items=[{"sequence": "MDNELE"}])
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert isinstance(res, dict)
    assert "mean_plddt" in res
    assert "pdb" in res
    assert not "status_code" in res
    assert not "error" in res

@pytest.mark.asyncio
async def test_valid_sequences(model):
    result = await model.predict(items=[{"sequence": "MDNELE"},
                                         {"sequence": "MUUUUDANLEPY"}])
    assert isinstance(result, list)
    assert len(result) == 2
    for res in result:
        assert isinstance(res, dict)
        assert "mean_plddt" in res
        assert "pdb" in res
        assert not "status_code" in res
        assert not "error" in res
        assert isinstance(result, dict) or isinstance(result, list)

@pytest.mark.asyncio
async def test_invalid_sequence_single(model):
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    result = await model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert "error" in res
    assert "status_code" in res
    assert "Consecutive occurrences of ':' " in res['error']['items__0__sequence'][0]
    assert res["status_code"] in (400, 422)

@pytest.mark.asyncio
async def test_mixed_sequence_batch(model):
    items = [{"sequence": "MENDELSEMYEFF:FEEFMLYRRTELSYYYUPPPPPU"},
             {"sequence": "MDNELE"}]
    result = await model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "mean_plddt" in result[0]
    assert "pdb" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

@pytest.mark.asyncio
async def test_mixed_valid_invalid_sequence_batch_continue_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = await model.predict(items=items, stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "error" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

@pytest.mark.asyncio
async def test_mixed_valid_invalid_sequence_batch_stop_on_error(model):
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"},
             {"sequence": "MDNELE"}]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result) == 2
    assert "error" in result[0]
    assert "error" in result[1]

@pytest.mark.asyncio
async def test_mixed_valid_invalid_sequence_nonbatch_stop_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result) == 1
    assert "error" in result[0]

@pytest.mark.asyncio
async def test_stop_on_error_with_previous_success(model):
    items = [
        [{"sequence": "MDNELE"}],
        [{"sequence": "MDNELE"}],
        [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
        [{"sequence": "MDNELE"}]
    ]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert "pdb" in result[0]
    assert "pdb" in result[1]
    assert "error" in result[2]
    assert 'items__0__sequence' in result[2].get('error')

@pytest.mark.asyncio
async def test_stop_on_error_with_previous_success_reordered(model):
    items = [
        {"sequence": "MDNELE"},
        {"sequence": "MDNELE"},  # ESMFold allow two per req, so first two is one batch
        {"sequence": "MDNELE"},  # The entire second batch is bad
        {"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"},  # Second batch is bad
    ]
    result = await model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 4
    assert "pdb" in result[0]
    assert "pdb" in result[1]
    assert "error" in result[2]
    assert 'items__1__sequence' in result[2].get('error')
    assert 'items__1__sequence' in result[3].get('error')

@pytest.mark.asyncio
async def test_raise_httpx():
    model = BioLMApi("esmfold", raise_httpx=True)
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    with pytest.raises(Exception):
        await model.predict(items=items)

@pytest.mark.asyncio
async def test_no_double_error_key(model):
    items = [{"sequence": "BAD::BAD"}]
    result = await model.predict(items=items)
    res = result[0] if isinstance(result, list) else result
    keys = list(res.keys())
    assert keys.count("error") == 1
    if isinstance(res["error"], dict):
        assert "error" not in res["error"]

@pytest.mark.asyncio
async def test_single_predict_to_disk(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    await model.predict(items=[{"sequence": "MDNELE"}], output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "mean_plddt" in data

@pytest.mark.asyncio
async def test_single_predict_to_disk_with_error(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    await model.predict(items=[{"sequence": "MD::NELE"}], output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "error" in data

@pytest.mark.asyncio
async def test_batch_predict_to_disk(tmp_path, model):
    items = [{"sequence": "MDNELE"}, {"sequence": "MENDEL"}, {"sequence": "ISOTYPE"}]
    file_path = tmp_path / "batch.jsonl"
    await model.predict(items=items, output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        assert "mean_plddt" in rec

@pytest.mark.asyncio
async def test_batch_predict_to_disk_stop_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    await model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=True)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 2
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        if i == 0:
            assert "mean_plddt" in rec
        else:
            assert "error" in rec

@pytest.mark.asyncio
async def test_batch_predict_to_disk_continue_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    await model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        if i == 1:
            assert "error" in rec
        else:
            assert "mean_plddt" in rec

@pytest.mark.asyncio
async def test_invalid_input_items_type(model, monkeypatch):
    # Defensive: monkeypatch the _batch_call_autoschema_or_manual method to return None
    monkeypatch.setattr(model, "_batch_call_autoschema_or_manual", lambda *a, **kw: None)
    with pytest.raises(TypeError, match="Parameter 'items' must be of type list, tuple"):
        await model.predict(items={"sequence": "MDNELE"})


