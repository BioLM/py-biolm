"""References to API endpoints."""
import datetime
import time

import requests
from requests.adapters import HTTPAdapter

import biolmai.auth
import biolmai
import inspect
import pandas as pd
import numpy as np
from biolmai.asynch import async_api_call_wrapper

from biolmai.biolmai import log
from biolmai.const import MULTIPROCESS_THREADS
from functools import lru_cache

from biolmai.payloads import INST_DAT_TXT, predict_resp_many_in_one_to_many_singles


@lru_cache(maxsize=64)
def validate_endpoint_action(allowed_classes, method_name, api_class_name):
    action_method_name = method_name.split('.')[-1]
    if action_method_name not in allowed_classes:
        err = 'Only {} supported on {}'
        err = err.format(
            list(allowed_classes),
            api_class_name
        )
        raise AssertionError(err)


def text_validator(text, c):
    """Validate some text against a class-based validator, returning a string
    if invalid, or None otherwise."""
    try:
        c(text)
    except Exception as e:
        return str(e)


def validate(f):
    def wrapper(*args, **kwargs):
        # Get class instance at runtime, so you can access not just
        # APIEndpoints, but any *parent* classes of that,
        # like ESMFoldSinglechain.
        class_obj_self = args[0]
        try:
            is_method = inspect.getfullargspec(f)[0][0] == 'self'
        except:
            is_method = False

        # Is the function we decorated a class method?
        if is_method:
            name = '{}.{}.{}'.format(f.__module__,
                                     class_obj_self.__class__.__name__,
                                     f.__name__)
        else:
            name = '{}.{}'.format(f.__module__, f.__name__)

        if is_method:
            # Splits name, e.g. 'biolmai.api.ESMFoldSingleChain.predict'
            action_method_name = name.split('.')[-1]
            validate_endpoint_action(
                class_obj_self.action_class_strings,
                action_method_name,
                class_obj_self.__class__.__name__
            )

        input_data = args[1]
        # Validate each row's text/input based on class attribute `seq_classes`
        for c in class_obj_self.seq_classes:
            # Validate input data against regex
            if class_obj_self.multiprocess_threads:
                validation = input_data.text.apply(text_validator, args=(c, ))
            else:
                validation = input_data.text.apply(text_validator, args=(c, ))
            if 'validation' not in input_data.columns:
                input_data['validation'] = validation
            else:
                input_data['validation'] = input_data['validation'].str.cat(
                    validation, sep='\n', na_rep='')

        # Mark your batches, excluding invalid rows
        valid_dat = input_data.loc[input_data.validation.isnull(), :].copy()
        N = class_obj_self.batch_size  # N rows will go per API request
        # JOIN back, which is by index
        if valid_dat.shape[0] != input_data.shape[0]:
            valid_dat['batch'] = np.arange(valid_dat.shape[0])//N
            input_data = input_data.merge(
                valid_dat.batch, left_index=True, right_index=True, how='left')
        else:
            input_data['batch'] = np.arange(input_data.shape[0])//N

        res = f(class_obj_self, input_data, **kwargs)
        return res
    return wrapper


def convert_input(f):
    def wrapper(*args, **kwargs):
        # Get the user-input data argument to the decorated function
        class_obj_self = args[0]
        input_data = args[1]
        # Make sure we have expected input types
        acceptable_inputs = (str, list, tuple, np.ndarray, pd.DataFrame)
        if not isinstance(input_data, acceptable_inputs):
            err = "Input must be one or many DNA or protein strings"
            raise ValueError(err)
        # Convert single-sequence input to list
        if isinstance(input_data, str):
            input_data = [input_data]
        # Make sure we don't have a matrix
        if isinstance(input_data, np.ndarray) and len(input_data.shape) > 1:
            err = "Detected Numpy matrix - input a single vector or array"
            raise AssertionError(err)
        # Make sure we don't have a >=2D DF
        if isinstance(input_data, pd.DataFrame) and len(input_data.shape) > 1:
            err = "Detected Pandas DataFrame - input a single vector or Series"
            raise AssertionError(err)
        input_data = pd.DataFrame(input_data, columns=['text'])
        return f(args[0], input_data, **kwargs)
    return wrapper


class APIEndpoint(object):
    batch_size = 3  # Overwrite in parent classes as needed

    def __init__(self, multiprocess_threads=None):
        # Check for instance-specific threads, otherwise read from env var
        if multiprocess_threads is not None:
            self.multiprocess_threads = multiprocess_threads
        else:
            self.multiprocess_threads = MULTIPROCESS_THREADS  # Could be False
        # Get correct auth-like headers
        self.auth_headers = biolmai.auth.get_user_auth_header()
        self.action_class_strings = tuple([
            c.__name__.replace('Action', '').lower() for c in self.action_classes
        ])

    def post_batches(self, dat, slug, action, payload_maker, resp_key):
        keep_batches = dat.loc[~dat.batch.isnull(), ['text', 'batch']]
        if keep_batches.shape[0] == 0:
            pass  # Do nothing - we made nice JSON errors to return in the DF
            # err = "No inputs found following local validation"
            # raise AssertionError(err)
        if keep_batches.shape[0] > 0:
            api_resps = async_api_call_wrapper(
                keep_batches,
                slug,
                action,
                payload_maker,
                resp_key
            )
            if isinstance(api_resps, pd.DataFrame):
                batch_res = api_resps.explode('api_resp')  # Should be lists of results
                len_res = batch_res.shape[0]
            else:
                batch_res = pd.DataFrame({'api_resp': api_resps})
                len_res = batch_res.shape[0]
            orig_request_rows = keep_batches.shape[0]
            if len_res != orig_request_rows:
                err = "Response rows ({}) mismatch with input rows ({})"
                err = err.format(len_res, orig_request_rows)
                raise AssertionError(err)

            # Stack the results horizontally w/ original rows of batches
            keep_batches['prev_idx'] = keep_batches.index
            keep_batches.reset_index(drop=False, inplace=True)
            batch_res.reset_index(drop=True, inplace=True)
            keep_batches['api_resp'] = batch_res
            keep_batches.set_index('prev_idx', inplace=True)
            dat = dat.join(keep_batches.reindex(['api_resp'], axis=1))
        else:
            dat['api_resp'] = None
        return dat

    def unpack_local_validations(self, dat):
        dat.loc[
            dat.api_resp.isnull(), 'api_resp'
        ] = dat.loc[~dat.validation.isnull(), 'validation'].apply(
            predict_resp_many_in_one_to_many_singles,
            args=(None, None, True, None)).explode()

        return dat

    @convert_input
    @validate
    def predict(self, dat):
        dat = self.post_batches(dat, self.slug, 'predict', INST_DAT_TXT, 'predictions')
        dat = self.unpack_local_validations(dat)
        return dat.api_resp.replace(np.nan, None).tolist()

    def infer(self, dat):
        return self.predict(dat)

    @convert_input
    @validate
    def transform(self, dat):
        dat = self.post_batches(dat, self.slug, 'transform', INST_DAT_TXT, 'predictions')
        dat = self.unpack_local_validations(dat)
        return dat.api_resp.replace(np.nan, None).tolist()

    @convert_input
    @validate
    def generate(self, dat):
        dat = self.post_batches(dat, self.slug, 'generate', INST_DAT_TXT, 'generated')
        dat = self.unpack_local_validations(dat)
        return dat.api_resp.replace(np.nan, None).tolist()


def retry_minutes(sess, URL, HEADERS, dat, timeout, mins):
    """Retry for N minutes."""
    HEADERS.update({'Content-Type': 'application/json'})
    attempts, max_attempts = 0, 5
    try:
        now = datetime.datetime.now()
        try_until = now + datetime.timedelta(minutes=mins)
        while datetime.datetime.now() < try_until and attempts < max_attempts:
            response = None
            try:
                log.info('Trying {}'.format(datetime.datetime.now()))
                response = sess.post(
                    URL,
                    headers=HEADERS,
                    data=dat,
                    timeout=timeout
                )
                if response.status_code not in (400, 404):
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
            attempts += 1
        if response is None:
            err = "Got Nonetype response"
            raise ValueError(err)
        elif 'Server Error' in response.text:
            err = "Got Server Error"
            raise ValueError(err)
    except Exception as e:
        return response
    return response


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


class PredictAction(object):

    def __str__(self):
        return 'PredictAction'


class GenerateAction(object):

    def __str__(self):
        return 'GenerateAction'


class TransformAction(object):

    def __str__(self):
        return 'TransformAction'


class ExplainAction(object):

    def __str__(self):
        return 'ExplainAction'


class SimilarityAction(object):

    def __str__(self):
        return 'SimilarityAction'


class FinetuneAction(object):

    def __str__(self):
        return 'FinetuneAction'
