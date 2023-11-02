#!/usr/bin/env python

"""Tests for `biolmai` package."""

import json
import pytest
import random
import copy
from asyncio import create_task, gather, run, sleep

import biolmai.auth
import biolmai.cls
from biolmai.asynch import async_main

from click.testing import CliRunner

import biolmai
from biolmai import cli

import logging
log = logging.getLogger(__name__)

N = 6


def test_authentication():
    """Test to make sure the environment variables for auth work, and
    that you get tokens back from the site to use for requests."""
    biolmai.auth.get_user_auth_header()


def return_shuffle(l):
    c = copy.copy(l)
    random.shuffle(c)
    return c


def test_async():
    urls = [
        "https://github.com",
        "https://stackoverflow.com",
        "https://python.org",
    ]
    concurrency = 3
    resp = run(async_main(urls, concurrency))
    print(resp)

def test_esm2_embeddings_predict_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ESM2Embeddings()
    resp = cls.transform(seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert all(['error' not in r for r in resp])
    assert all(['predictions' in r for r in resp])
    assert all([len(r['predictions']) == 1 for r in resp])
    assert all(['33' in item['mean_representations'].keys()
                for subitem in resp
                for item in subitem['predictions']])


def test_esmfold_singlechain_predict_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert all(['error' not in r for r in resp])
    assert all(['predictions' in r for r in resp])
    assert all([len(r['predictions']) == 1 for r in resp])
    assert all([r['predictions'][0].startswith('PARENT N/A') for r in resp])


def test_esmfold_multichain_predict_all_valid_singlechain_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert all(['error' not in r for r in resp])
    assert all(['predictions' in r for r in resp])
    assert all([len(r['predictions']) == 1 for r in resp])
    assert all([r['predictions'][0].startswith('PARENT N/A') for r in resp])


def test_esmfold_singlechain_predict_all_locally_invalid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    bad_seqs = [''.join(return_shuffle(base_seqs))[:30] + 'i1' for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(bad_seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all([r['status_code'] is None for r in resp])
    assert all([r['batch_id'] is None for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert all(['error' in r for r in resp])
    assert all(['predictions' not in r for r in resp])
    assert all(['Unambiguous residues' in json.dumps(r['error']) for r in resp])


def test_esmfold_singlechain_predict_all_api_too_long_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    bad_seqs = [''.join(return_shuffle(base_seqs) * 100) for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(bad_seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all([r['status_code'] == 400 for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert all(['error' not in r for r in resp])
    assert all(['predictions' not in r for r in resp])
    assert all(['no more than 512 character' in json.dumps(r) for r in resp])


def test_esmfold_singlechain_predict_good_and_locally_invalid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [''.join(return_shuffle(base_seqs))[:30] + 'i1' for _ in range(int(N / 2))]
    all_seqs = seqs + bad_seqs
    random.shuffle(all_seqs)
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(all_seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert any(['error' in r for r in resp])
    assert not all(['error' in r for r in resp])
    assert any(['predictions' in r for r in resp])
    assert not all(['predictions' in r for r in resp])
    assert all([len(r['predictions']) == 1 for r in resp if 'predictions' in r])
    assert all([r['predictions'][0].startswith('PARENT N/A') for r in resp if 'predictions' in r])


def test_esmfold_singlechain_predict_ruin_all_api_batches():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [''.join(return_shuffle(base_seqs)) * 100 for _ in range(int(N / 2))]
    ruined_api_batches = []
    for a, b in list(zip(seqs, bad_seqs)):
        ruined_api_batches.extend([a, b])
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(ruined_api_batches)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert not any(['error' in r for r in resp])
    assert not any(['predictions' in r for r in resp])
    # TODO: make some assertion about format
    assert all([len(r['predictions']) == 1 for r in resp if 'predictions' in r])
    assert all([r['predictions'][0].startswith('PARENT N/A') for r in resp if 'predictions' in r])


def test_esmfold_singlechain_predict_good_and_api_too_long_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [''.join(return_shuffle(base_seqs)) * 100 for _ in range(int(N / 2))]
    all_seqs = [seqs[0], seqs[1], bad_seqs[0], bad_seqs[1],
                seqs[2], bad_seqs[2]]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(all_seqs)
    assert isinstance(resp, list)
    assert all([isinstance(r, dict) for r in resp])
    assert all(['status_code' in r for r in resp])
    assert all(['batch_id' in r for r in resp])
    assert all(['batch_item' in r for r in resp])
    assert not any(['error' in r for r in resp])
    assert any(['predictions' in r for r in resp])
    assert not all(['predictions' in r for r in resp])
    assert all([len(r['predictions']) == 1 for r in resp if 'predictions' in r])
    assert all([r['predictions'][0].startswith('PARENT N/A') for r in resp if 'predictions' in r])


def test_esmfold_multichain_predict_good_and_locally_invalid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [''.join(return_shuffle(base_seqs))[:30] + 'i1' for _ in range(int(N / 2))]
    all_seqs = seqs + bad_seqs
    random.shuffle(all_seqs)
    cls = biolmai.cls.ESMFoldMultiChain()
    resp = cls.predict(all_seqs)
    assert isinstance(resp, list)


# TODO: test one seq
# TODO: test multiprocessing ability
# TODO: test DF or list of seqs


def test_esmfold_singlechain_predict():
    seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFA"
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(seq)
    assert 1 == 1


def test_esmfold_singlechain_predict_bad_sequence():
    bad_seq = "Nota Real sequence"
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(bad_seq)
    assert 1 == 1


def test_esmfold_singlechain_bad_action():
    seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFA"
    cls = biolmai.cls.ESMFoldSingleChain()
    with pytest.raises(Exception) as e_info:
        resp = cls.tokenize(seq)
        assert 'Only' in str(e_info) and 'supported on' in str(e_info)


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'biolmai.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
