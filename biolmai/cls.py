"""API inference classes."""
from biolmai.api import APIEndpoint, PredictAction, TransformAction, GenerateAction
from biolmai.validate import UnambiguousAA, ExtendedAAPlusExtra, SingleOccurrenceOf


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

    .. highlight:: python
    .. code-block:: python

       {
         "instances": [{
           "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQ"}
         }]
       }
    """
    slug = 'esm2_t33_650M_UR50D'
    action_classes = (TransformAction,)
    seq_classes = (UnambiguousAA(), )
    batch_size = 1


class ESM1v1(APIEndpoint):
    """Example.

    .. highlight:: python
    .. code-block:: python

       {
          "instances": [{
            "data": {"text": "QERLEUTGR<mask>SLGYNIVAT"}
          }]
       }
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


class ESMIF1(APIEndpoint):
    slug = 'esmif1'
    action_classes = (GenerateAction, )
    seq_classes = tuple([])
    batch_size = 2


class Progen2(APIEndpoint):
    slug = 'progen2'
    action_classes = (GenerateAction, )
    seq_classes = tuple([])
    batch_size = 1
