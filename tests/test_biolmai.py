#!/usr/bin/env python

"""Tests for `biolmai` package."""

import pytest

from click.testing import CliRunner

import biolmai
from biolmai import cli

import logging
log = logging.getLogger(__name__)


def test_authentication():
    """Test to make sure the environment variables for auth work, and
    that you get tokens back from the site to use for requests."""
    resp = biolmai.get_api_token()
    # Make sure we have access and refresh keys in dictionary response
    assert 'access' in resp, log.warning(resp)
    assert 'refresh' in resp, log.warning(resp)


def test_arbitrary_api_call_inference():
    seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFA"
    payload = {
        "instances": [{
            "data": {"text": seq}
        }]
    }
    tokens = biolmai.get_api_token()
    access = tokens.get('access')
    refresh = tokens.get('refresh')

    resp = biolmai.api_call(
        model_name='esmfold-singlechain',
        action='predict',
        payload=payload,
        access=access,
        refresh=refresh
    )

    assert 'predictions' in resp, log.warning(resp)


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'biolmai.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
