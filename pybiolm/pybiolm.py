"""Main module."""
import json
import os
import requests

import json, os, requests
import urllib3
import datetime
import time

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import logging

log = logging.getLogger('biolm_util')


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=list(range(400, 599)),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def retry_minutes(sess, URL, HEADERS, dat, timeout, mins):
    """Retry for N minutes."""
    try:
        now = datetime.datetime.now()
        try_until = now + datetime.timedelta(minutes=mins)
        while datetime.datetime.now() < try_until:
            response = None
            try:
                log.info('Trying {}'.format(datetime.datetime.now()))
                response = sess.post(
                    URL,
                    headers=HEADERS,
                    data=dat,
                    timeout=timeout
                )
                response.raise_for_status()
                if 'error' in response.json():
                    raise ValueError(response.json().dumps())
                else:
                    break
            except Exception as e:
                log.warning(e)
                if response:
                    log.warning(response.text)
                time.sleep(5)  # Wait 5 seconds between tries
        if response is None:
            err = "Got Nonetype response"
            raise ValueError(err)
        elif 'Server Error' in response.text:
            err = "Got Server Error"
            raise ValueError(err)
        else:
            response.raise_for_status()
    except Exception as e:
        raise
    else:
        return response


def get_api_token():
    """Get a BioLM API token to use with future API requests.

    Copied from https://api.biolm.ai/#d7f87dfd-321f-45ae-99b6-eb203519ddeb.
    """
    url = "https://biolm.ai/api/auth/token/"

    payload = json.dumps({
        "username": os.environ.get("BIOLM_USER"),
        "password": os.environ.get("BIOLM_PASSWORD")
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = response.json()

    return response_json


def api_call(model_name, action, payload, access, refresh):
    """Hit an arbitrary BioLM model inference API."""
    # Normally would POST multiple sequences at once for greater efficiency,
    # but for simplicity sake will do one at at time right now
    url = f'https://biolm.ai/api/v1/models/{model_name}/{action}/'

    payload = json.dumps(payload)

    try:
        assert access
        assert refresh
    except AssertionError:
        raise AssertionError("BioLM access or refresh token not set")

    headers = {
        'Cookie': 'access={};refresh={}'.format(access, refresh),
        'Content-Type': 'application/json'
    }

    session = requests_retry_session()
    tout = urllib3.util.Timeout(total=180, read=180)
    response = retry_minutes(session, url, headers, payload, tout, mins=12)

    resp_json = response.json()

    return resp_json
