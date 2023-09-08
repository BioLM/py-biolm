#!/usr/bin/env python

"""Tests for `biolmai` package."""

import pytest
import random
import copy

from click.testing import CliRunner

import biolmai
from biolmai import cli

import logging
log = logging.getLogger(__name__)

N = 5


def test_authentication():
    """Test to make sure the environment variables for auth work, and
    that you get tokens back from the site to use for requests."""
    biolmai.biolmai.get_user_auth_header()


def return_shuffle(l):
    c = copy.copy(l)
    random.shuffle(c)
    return c


def test_esmfold_singlechain_predict_many():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.ESMFoldSingleChain()
    resp = cls.predict(seqs)
    assert all([r.startswith('PARENT ') for r in resp])


def test_esmfold_singlechain_predict_all_bad_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    bad_seqs = [''.join(return_shuffle(base_seqs))[:30] + 'i1' for _ in range(N)]
    cls = biolmai.ESMFoldSingleChain()
    with pytest.raises(Exception) as e_info:
        resp = cls.predict(bad_seqs)
        assert 'no valid sequences' in str(e_info).lower()


def test_esmfold_singlechain_predict_good_and_bad_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [''.join(return_shuffle(base_seqs))[:30] + 'i1' for _ in range(int(N / 2))]
    all_seqs = seqs + bad_seqs
    random.shuffle(all_seqs)
    cls = biolmai.ESMFoldSingleChain()
    resp = cls.predict(all_seqs)
    # assert 'predictions' in resp

def test_esmfold_multichain_predict_good_and_bad_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = [''.join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [''.join(return_shuffle(base_seqs))[:30] + 'i1' for _ in range(int(N / 2))]
    all_seqs = seqs + bad_seqs
    random.shuffle(all_seqs)
    cls = biolmai.ESMFoldMultiChain()
    resp = cls.predict(all_seqs)
    assert isinstance(resp, list)


# TODO: test one seq
# TODO: test multiprocessing ability
# TODO: test DF or list of seqs
# TODO: test batching in POST request ability
def test_esmfold_singlechain_predict():
    seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFA"
    cls = biolmai.ESMFoldSingleChain()
    resp = cls.predict(seq)
    # assert 'predictions' in resp


def test_esmfold_singlechain_predict_bad_sequence():
    bad_seq = "Nota Real sequence"
    cls = biolmai.ESMFoldSingleChain()
    with pytest.raises(Exception) as e_info:
        resp = cls.predict(bad_seq)
        assert 'ambiguous residues' in str(e_info)


def test_esmfold_singlechain_bad_action():
    seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFA"
    cls = biolmai.ESMFoldSingleChain()
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
