import json
import logging

import pytest

from biolmai.client import BioLMApi

LOGGER = logging.getLogger(__name__)

@pytest.fixture(scope='module')
def model():
    return BioLMApi("esmfold", raise_httpx=False, unwrap_single=False)

def test_valid_sequence(model):
    result = model.predict(items=[{"sequence": "MDNELE"}])
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert isinstance(res, dict)
    assert "mean_plddt" in res
    assert "pdb" in res
    assert not "status_code" in res
    assert not "error" in res

def test_valid_sequences(model):
    result = model.predict(items=[{"sequence": "MDNELE"},
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

def test_invalid_sequence_single(model):
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    result = model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 1
    res = result[0]
    assert "error" in res
    assert "status_code" in res
    # Should exist a single [0] error for the 0th 'sequence' item
    assert "Consecutive occurrences of ':' " in res['error']['items__0__sequence'][0]
    # Should really be HTTP 422, but need to change API-side
    assert res["status_code"] in (400, 422)

def test_mixed_sequence_batch_lol_items_first_invalid(model):
    items = [[{"sequence": "MENDELSEMYEFF::FEEFMLYRRTELSYYYUPPPPPU"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "error" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

def test_mixed_sequence_batch_lol_items_first_invalid_multiple_in_batch(model):
    items = [[{"sequence": "MENDELSEMYEFF::FEEFMLYRRTELSYYYUPPPPPU"},
              {"sequence": "invalid123"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items)
    assert isinstance(result, list)
    assert len(result) == 3
    assert "error" in result[0]
    assert "error" in result[1]
    assert "mean_plddt" in result[2]
    assert "pdb" in result[2]


def test_mixed_valid_invalid_sequence_batch_continue_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items, stop_on_error=False)
    assert isinstance(result, list)
    assert len(result) == 2
    assert "error" in result[0]
    assert "mean_plddt" in result[1]
    assert "pdb" in result[1]

def test_mixed_valid_invalid_sequence_batch_stop_on_error(model):
    items = [[{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
             [{"sequence": "MDNELE"}]]
    result = model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 1
    # Should stop on first error, so only one result
    assert len(result) == 1
    assert "error" in result[0]

def test_stop_on_error_with_previous_success(model):
    items = [
        [{"sequence": "MDNELE"}],
        [{"sequence": "MDNELE"}],
        [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}],
        [{"sequence": "MDNELE"}],
    ]
    result = model.predict(items=items, stop_on_error=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert "pdb" in result[0]
    assert "pdb" in result[1]
    assert "error" in result[2]

def test_raise_httpx():
    model = BioLMApi("esmfold", raise_httpx=True)
    items = [{"sequence": "MENDELSEMYEFFFEEFMLYRRTELSYYYUPPPPPU::"}]
    with pytest.raises(Exception):
        model.predict(items=items)

def test_no_double_error_key(model):
    # Single bad item should produce one error key only
    items = [{"sequence": "BAD::BAD"}]
    result = model.predict(items=items)
    # Normalize to single result dict
    res = result[0] if isinstance(result, list) else result
    # Must have exactly one "error" key at top level
    keys = list(res.keys())
    assert keys.count("error") == 1
    # If the error value is a dict, it must not itself contain "error"
    if isinstance(res["error"], dict):
        assert "error" not in res["error"]

def test_single_predict_to_disk(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    model.predict(items=[{"sequence": "MDNELE"}], output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "mean_plddt" in data

def test_single_predict_to_disk_with_error(tmp_path, model):
    file_path = tmp_path / "out.jsonl"
    model.predict(items=[{"sequence": "MD::NELE"}], output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "error" in data

def test_batch_predict_to_disk(tmp_path, model):
    items = [{"sequence": "MDNELE"}, {"sequence": "MENDEL"}, {"sequence": "ISOTYPE"}]
    file_path = tmp_path / "batch.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 3
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        assert "mean_plddt" in rec

def test_batch_predict_to_disk_stop_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=True)
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

def test_batch_predict_to_disk_continue_on_error(tmp_path, model):
    items = [[{"sequence": "MDNELE"}], [{"sequence": "DN::A"}], [{"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
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

def test_batch_predict_to_disk_continue_on_error_multiple_per_batch(tmp_path, model):
    items = [[{"sequence": "MDNELE"},
              {"sequence": "MDNELE"}],
             [{"sequence": "DN::A"},
              {"sequence": "MDNELE"}],
             [{"sequence": "ISOTYPE"},
              {"sequence": "ISOTYPE"}]]
    file_path = tmp_path / "batch_multiple.jsonl"
    model.predict(items=items, output='disk', file_path=str(file_path), stop_on_error=False)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    LOGGER.warning(lines)
    assert len(lines) == 6
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert isinstance(rec, dict)
        if i in (2, 3):
            assert "error" in rec
        else:
            assert "mean_plddt" in rec

def test_invalid_input_items_type(model, monkeypatch):
    # Defensive: monkeypatch the _batch_call_autoschema_or_manual method to return None
    monkeypatch.setattr(model, "_batch_call_autoschema_or_manual", lambda *a, **kw: None)
    # Should raise TypeError because items is not a list/tuple
    with pytest.raises(TypeError, match="Parameter 'items' must be of type list, tuple"):
        model.predict(items={"sequence": "MDNELE"})
