Evo2-1B-Base API
================

Evo2 is a multimodal language model trained jointly on DNA, RNA and protein.  The 1 B-parameter "base" checkpoint supports embeddings (encode), log-probability scoring (predict) and autoregressive generation (generate).

Encode
------

``POST /api/v3/evo2-1b-base/encode/`` returns embeddings from user-selected transformer layers.

Predict
-------

``POST /api/v3/evo2-1b-base/predict/`` (alias of ``predict_log_prob``) sums log-probabilities over all tokens in the input sequence.

Generate
--------

``POST /api/v3/evo2-1b-base/generate/`` continues a DNA/RNA sequence prompt using standard temperature / top-k / top-p sampling parameters.

Example encode request (Python):

.. code:: python

    from biolmai import BioLM
    response = BioLM(
        entity="evo2-1b-base",
        action="encode",
        params={
            "embedding_layers": [-2],
            "include": ["mean"]
        },
        items=[{"sequence": "ATGGATT..."}]
    )
    print(response)

Resource usage
--------------

Runs on 4 vCPU / 16 GB RAM with one NVIDIA L4 GPU (approx. 15 min timeout per job).

Schema reference
----------------

The request / response models correspond to `algorithms.schemas_v3.static.evo2`. 