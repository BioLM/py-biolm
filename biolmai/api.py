"""References to API endpoints."""
from biolmai import biolmai
import inspect
import pandas as pd
import numpy as np
from asyncio import create_task, gather, run, sleep
from biolmai.asynch import async_main, async_api_calls

from biolmai.biolmai import get_user_auth_header
from biolmai.const import MULTIPROCESS_THREADS
if MULTIPROCESS_THREADS:
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False,
                           nb_workers=int(MULTIPROCESS_THREADS), verbose=2)
from functools import lru_cache

from biolmai.payloads import INST_DAT_TXT
from biolmai.validate import ExtendedAAPlusExtra, SingleOccurrenceOf, \
    UnambiguousAA, \
    UnambiguousAAPlusExtra


def predict_resp_many_in_one_to_many_singles(resp_json, status_code,
                                             batch_id, local_err, batch_size):
    expected_root_key = 'predictions'
    to_ret = []
    if not local_err and status_code and status_code == 200:
        list_of_individual_seq_results = resp_json[expected_root_key]
    elif local_err:
        list_of_individual_seq_results = [{'error': resp_json}]
    elif status_code and status_code != 200 and isinstance(resp_json, dict):
        list_of_individual_seq_results = [resp_json] * batch_size
    else:
        raise ValueError("Unexpected response in parser")
    for idx, item in enumerate(list_of_individual_seq_results):
        d = {'status_code': status_code,
             'batch_id': batch_id,
             'batch_item': idx}
        if not status_code or status_code != 200:
            d.update(item)  # Put all resp keys at root there
        else:
            # We just append one item, mimicking a single seq in POST req/resp
            d[expected_root_key] = []
            d[expected_root_key].append(item)
        to_ret.append(d)
    return to_ret


def async_api_call_wrapper(grouped_df, slug, action, payload_maker,
                           response_key):
    """Wrap API calls to assist with sequence validation as a pre-cursor to
    each API call.
    """
    model_name = slug
    # payload = payload_maker(grouped_df)
    init_ploads = grouped_df.groupby('batch').apply(payload_maker, include_batch_size=True)
    ploads = init_ploads.to_list()
    init_ploads = init_ploads.to_frame(name='pload')
    init_ploads['batch'] = init_ploads.index
    init_ploads = init_ploads.reset_index(drop=True)
    assert len(ploads) == init_ploads.shape[0]
    for inst, b in zip(ploads, init_ploads['batch'].to_list()):
        inst['batch'] = b

    headers = get_user_auth_header()  # Need to pull each time
    urls = [
        "https://github.com",
        "https://stackoverflow.com",
        "https://python.org",
    ]
    # concurrency = 3
    api_resp = run(async_api_calls(model_name, action, headers,
                                   ploads, response_key))
    api_resp = [item for sublist in api_resp for item in sublist]
    api_resp = sorted(api_resp, key=lambda x: x['batch_id'])
    # print(api_resp)
    # api_resp = biolmai.api_call(model_name, action, headers, payload,
    #                             response_key)
    # resp_json = api_resp.json()
    # batch_id = int(grouped_df.batch.iloc[0])
    # batch_size = grouped_df.shape[0]
    # response = predict_resp_many_in_one_to_many_singles(
    #     resp_json, api_resp.status_code, batch_id, None, batch_size)
    return api_resp


def api_call_wrapper(df, args):
    """Wrap API calls to assist with sequence validation as a pre-cursor to
    each API call.
    """
    model_name, action, payload_maker, response_key = args
    payload = payload_maker(df)
    headers = get_user_auth_header()  # Need to pull each time
    api_resp = biolmai.api_call(model_name, action, headers, payload,
                                response_key)
    resp_json = api_resp.json()
    batch_id = int(df.batch.iloc[0])
    batch_size = df.shape[0]
    response = predict_resp_many_in_one_to_many_singles(
        resp_json, api_resp.status_code, batch_id, None, batch_size)
    return response


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
            name = '{}.{}.{}'.format(f.__module__, args[0].__class__.__name__,
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
                validation = input_data.text.parallel_apply(text_validator, args=(c, ))
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
        self.auth_headers = biolmai.get_user_auth_header()
        self.action_class_strings = tuple([
            c.__name__.replace('Action', '').lower() for c in self.action_classes
        ])

    @convert_input
    @validate
    def predict(self, dat):
        keep_batches = dat.loc[~dat.batch.isnull(), ['text', 'batch']]
        if keep_batches.shape[0] == 0:
            err = "No inputs found following local validation"
            # raise AssertionError(err)
        elif self.multiprocess_threads:
            api_resps = async_api_call_wrapper(
                    keep_batches,
                    self.slug,
                    'predict',
                    INST_DAT_TXT,
                    'predictions'
            )
        else:
            api_resps = keep_batches.groupby('batch').apply(
                api_call_wrapper,
                (
                    self.slug,
                    'predict',
                    INST_DAT_TXT,
                    'predictions'
                ),
            )
        if keep_batches.shape[0] > 0:
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

        dat.loc[
            dat.api_resp.isnull(), 'api_resp'
        ] = dat.loc[~dat.validation.isnull(), 'validation'].apply(
            predict_resp_many_in_one_to_many_singles,
            args=(None, None, True, None)).explode()

        return dat.api_resp.replace(np.nan, None).tolist()

    def infer(self, dat):
        return self.predict(dat)

    @validate
    def tokenize(self, dat):
        payload = {"instances": [{"data": {"text": dat}}]}
        resp = biolmai.api_call(
            model_name=self.slug,
            headers=self.auth_headers,  # From APIEndpoint base class
            action='transform',
            payload=payload
        )
        return resp


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


class ESMFoldSingleChain(APIEndpoint):
    slug = 'esmfold-singlechain'
    action_classes = (PredictAction, )
    seq_classes = (UnambiguousAA(), )
    batch_size = 2


class ESMFoldMultiChain(APIEndpoint):
    slug = 'esmfold-multichain'
    action_classes = (PredictAction, )
    seq_classes = (ExtendedAAPlusExtra(extra=[':']), )
    batch_size = 2


class ESM2Embeddings(APIEndpoint):
    """Example.

    ```python
    {
      "instances": [{
        "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQ"}
      }]
    }
    ```
    """
    slug = 'esm2_t33_650M_UR50D'
    action_classes = (TransformAction,)
    seq_classes = (UnambiguousAA(), )
    batch_size = 3


class ESM1v1(APIEndpoint):
    """Example.

    ```python
    {
      "instances": [{
        "data": {"text": "QERLEUTGR<mask>SLGYNIVAT"}
      }]
    }
    ```
    """
    slug = 'esm1v_t33_650M_UR90S_1'
    action_classes = (PredictAction, )
    seq_classes = (SingleOccurrenceOf('<mask>'),
                   ExtendedAAPlusExtra(extra=['<mask>']))
    batch_size = 5


class ESM1v2(APIEndpoint):
    slug = 'esm1v_t33_650M_UR90S_2'
    action_classes = (PredictAction, )
    seq_classes = (SingleOccurrenceOf('<mask>'),
                   ExtendedAAPlusExtra(extra=['<mask>']))
    batch_size = 5


class ESM1v3(APIEndpoint):
    slug = 'esm1v_t33_650M_UR90S_3'
    action_classes = (PredictAction, )
    seq_classes = (SingleOccurrenceOf('<mask>'),
                   ExtendedAAPlusExtra(extra=['<mask>']))
    batch_size = 5


class ESM1v4(APIEndpoint):
    slug = 'esm1v_t33_650M_UR90S_4'
    action_classes = (PredictAction, )
    seq_classes = (SingleOccurrenceOf('<mask>'),
                   ExtendedAAPlusExtra(extra=['<mask>']))
    batch_size = 5


class ESM1v5(APIEndpoint):
    slug = 'esm1v_t33_650M_UR90S_5'
    action_classes = (PredictAction, )
    seq_classes = (SingleOccurrenceOf('<mask>'),
                   ExtendedAAPlusExtra(extra=['<mask>']))
    batch_size = 5
