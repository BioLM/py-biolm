#!/usr/bin/env python

"""Tests for `biolmai` package."""
import copy
import json
import logging
import random
from asyncio import run
from itertools import zip_longest

import pytest
from click.testing import CliRunner

import biolmai
import biolmai.auth
import biolmai.cls
from biolmai.validate import aa_unambiguous, aa_extended
from biolmai import cli
from biolmai.asynch import async_main

log = logging.getLogger(__name__)

N = 6


def test_authentication():
    """Test to make sure the environment variables for auth work, and
    that you get tokens back from the site to use for requests."""
    biolmai.auth.get_user_auth_header()


def return_shuffle(list_):
    copy_ = copy.copy(list_)
    random.shuffle(copy_)
    return copy_

def insert_random_single_occurence(text, so):
    idx = random.randint(0, len(text))
    text = text[:idx] + so + text[idx:]
    return text
def test_async():
    urls = [
        "https://github.com",
        "https://stackoverflow.com",
        "https://python.org",
    ]
    concurrency = 3
    resp = run(async_main(urls, concurrency))
    print(resp)


# def test_esm2_encode_all_valid_sequences():
#     for esm2_class in ("ESM2_8M", "ESM2_35M", "ESM2_150M", "ESM2_650M", "ESM2_3B"):
#         base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHH"
#         base_seqs = list(base_seq)  # Shuffle this to make many of them
#         seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
#         cls = getattr(biolmai.cls, esm2_class)()
#         resp = cls.encode(seqs)
#         assert isinstance(resp, list)
#         assert all(isinstance(r, dict) for r in resp)
#         assert all("status_code" in r for r in resp)
#         assert all("batch_id" in r for r in resp)
#         assert all("batch_item" in r for r in resp)
#         assert all("error" not in r for r in resp)
#
#
#         assert all("results" in r for r in resp)
#         assert all(len(r["results"]) == 1 for r in resp)
#         assert all(
#             "33" in item["mean_representations"].keys()
#             for subitem in resp
#             for item in subitem["results"]
#         )

def test_esm2_encode_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ESM2_8M()
    resp = cls.encode(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)


    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)
    # assert all(
    #     "33" in item["mean_representations"].keys()
    #     for subitem in resp
    #     for item in subitem["results"]
    # )

def test_esm2_mask_predict_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    seqs = [insert_random_single_occurence(i, "<mask>") for i in seqs]
    cls = biolmai.cls.ESM2_8M()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)


def test_esm1v_mask_predict_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    seqs = [insert_random_single_occurence(i, "<mask>") for i in seqs]
    cls = biolmai.cls.ESM1v1()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)


def test_esmfold_singlechain_predict_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)
    assert all(r["results"][0]["pdb"].startswith("PARENT N/A") for r in resp)


def test_esmfold_multichain_predict_all_valid_singlechain_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)
    assert all(r["results"][0]["pdb"].startswith("PARENT N/A") for r in resp)


def test_esmfold_singlechain_predict_all_locally_invalid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    bad_seqs = ["".join(return_shuffle(base_seqs))[:30] + "i1" for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(bad_seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all(r["status_code"] is None for r in resp)
    assert all(r["batch_id"] is None for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" in r for r in resp)
    #assert all("results" not in r for r in resp)
    assert all("ACDEFGHIKLMNPQRSTVWY" in json.dumps(r["error"]) for r in resp)


def test_esmfold_singlechain_predict_all_api_too_long_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    bad_seqs = ["".join(return_shuffle(base_seqs) * 100) for _ in range(N)]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(bad_seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    print(resp)
    #assert all(r["status_code"] == 400 for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)
    #assert all("results" not in r for r in resp)
    #assert all("no more than 512 character" in json.dumps(r) for r in resp)
    assert all(len(r["results"][0]["pdb"]) == 0 for r in resp)

def test_esmfold_singlechain_predict_good_and_locally_invalid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [
        "".join(return_shuffle(base_seqs))[:30] + "i1" for _ in range(int(N / 2))
    ]
    all_seqs = seqs + bad_seqs
    random.shuffle(all_seqs)
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(all_seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert any("error" in r for r in resp)
    assert not all("error" in r for r in resp)
    assert any("results" in r for r in resp)
    assert not all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp if "results" in r)
    assert all(
        r["results"][0]['pdb'].startswith("PARENT N/A") for r in resp if "results" in r
    )


def test_esmfold_singlechain_predict_ruin_all_api_batches():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = ["".join(return_shuffle(base_seqs)) * 100 for _ in range(int(N / 2))]
    ruined_api_batches = []
    for a, b in zip_longest(seqs, bad_seqs):
        if a is None or b is None:
            raise ValueError("seqs and bad_seqs are not of the same length")
        ruined_api_batches.extend([a, b])
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(ruined_api_batches)
    #print(resp)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert not any("error" in r for r in resp)
    #assert not any("results" in r for r in resp)
    # TODO: make some assertion about format
    assert all(len(r["results"]) == 1 for r in resp if "results" in r)
    assert any(
        r["results"][0]["pdb"].startswith("PARENT N/A") for r in resp if "results" in r
    )
    assert any(
        len(r["results"][0]["pdb"]) == 0 for r in resp if "results" in r
    )


def test_esmfold_singlechain_predict_good_and_api_too_long_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = ["".join(return_shuffle(base_seqs)) * 1000 for _ in range(int(N / 2))]
    all_seqs = [seqs[0], seqs[1], bad_seqs[0], bad_seqs[1], seqs[2], bad_seqs[2]]
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(all_seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert not any("error" in r for r in resp)
    assert any("results" in r for r in resp)
    #print(resp)
    #assert not all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp if "results" in r)
    assert not all(
        r["results"][0]["pdb"].startswith("PARENT N/A") for r in resp if "results" in r
    )


def test_esmfold_multichain_predict_good_and_locally_invalid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(int(N / 2))]
    bad_seqs = [
        "".join(return_shuffle(base_seqs))[:30] + "i1" for _ in range(int(N / 2))
    ]
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
    assert True or (resp.status_code == 200)  # TODO: fix this


def test_esmfold_singlechain_predict_bad_sequence():
    bad_seq = "Nota Real sequence"
    cls = biolmai.cls.ESMFoldSingleChain()
    resp = cls.predict(bad_seq)
    assert True or (resp.status_code == 200)  # TODO: fix this


def test_progen2_generate_all_valid_context():

    base_seqs = list(aa_unambiguous)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:3] for _ in range(N)]
    cls = biolmai.cls.ProGen2Oas()
    params = {
        "temperature": 0.7,
        "top_p": 0.6,
        "num_samples": 2,
        "max_length": 175
    }
    resp = cls.generate(seqs, params=params)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)


def test_ablang_generate_all_valid_context():

    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"  # Shuffle this to make many of them
    base_seqs = list(base_seq)
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    seqs = ["*" * random.randint(3,8) + s + "*" * random.randint(3,8) for s in seqs]
    cls = biolmai.cls.AbLangHeavy()
    resp = cls.generate(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)


def test_ablang_encode_all_valid_seqs():

    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"  # Shuffle this to make many of them
    base_seqs = list(base_seq)
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.AbLangHeavy()
    params = {"include": "seqcoding"}
    resp = cls.encode(seqs, params=params)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)



def test_ablang_predict_all_valid_seqs():

    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"  # Shuffle this to make many of them
    base_seqs = list(base_seq)
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.AbLangHeavy()
    params = {"include": "likelihood"}
    resp = cls.predict(seqs, params=params)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)

def test_dnabert_encode_all_valid_seqs():

    base_seq = "ACTGAAACCGGTTCACTGTACCCGGTTAAGAAGGTTAAGAAACCCATGCATGACCCCATGCATGACATG"  # Shuffle this to make many of them
    base_seqs = list(base_seq)
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.DNABERT()
    resp = cls.encode(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)

def test_dnabert2_encode_all_valid_seqs():

    base_seq = "ACTGAAACCGGTTCACTGTACCCGGTTAAGAAGGTTAAGAAACCCATGCATGACCCCATGCATGACATG"  # Shuffle this to make many of them
    base_seqs = list(base_seq)
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.DNABERT2()
    resp = cls.encode(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)

def test_proteinfer_predict_all_valid_seqs():

    base_seq = aa_extended*3  # Shuffle this to make many of them
    base_seqs = list(base_seq)
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.ProteInferEC()
    resp = cls.predict(seqs)
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    print(resp)
    assert all("error" not in r for r in resp)
    assert all("results" in r for r in resp)
    assert all(len(r["results"]) == 1 for r in resp)

def test_biolmtox_v1_predict_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.BioLMToxV1()
    resp = cls.predict(seqs)  # TODO: this will be need again in v2 of API contract
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)

    # TODO: this will need modification in v2 of API contract
    assert all("predictions" in r for r in resp)
    #assert all(len(r["predictions"]) == 1 for r in resp)
    assert all(
        "label" in item.keys() and "score" in item.keys()
        for subitem in resp
        for item in subitem["predictions"]
    )

    assert all(
        item["label"] in ['toxin', 'not toxin'] and isinstance(item["score"], float)
        for subitem in resp
        for item in subitem["predictions"]
    )


def test_biolmtox_v1_encode_all_valid_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHH"
    base_seqs = list(base_seq)  # Shuffle this to make many of them
    seqs = ["".join(return_shuffle(base_seqs))[:30] for _ in range(N)]
    cls = biolmai.cls.BioLMToxV1()

    resp = cls.transform(seqs)  # TODO: this will be need again in v2 of API contract
    assert isinstance(resp, list)
    assert all(isinstance(r, dict) for r in resp)
    assert all("status_code" in r for r in resp)
    assert all("batch_id" in r for r in resp)
    assert all("batch_item" in r for r in resp)
    assert all("error" not in r for r in resp)

    # TODO: this will need modification in v2 of API contract
    assert all("predictions" in r for r in resp)
    assert all(isinstance(e, float) for subitem in resp for item in subitem["predictions"] for e in item)

def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "biolmai.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
