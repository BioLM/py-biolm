Evo2 1B Base API
================

Evo2-1B-Base is a 1-billion-parameter transformer trained **jointly on DNA, RNA and protein** with an 8 k context window.
It exposes three unified BioLM API actions: embed (*encode*), score (*predict*) and autoregressively *generate* new tokens.

.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#evo2-1b-base

Endpoints
---------

Predict (Log-probability)
^^^^^^^^^^^^^^^^^^^^^^^^

``POST /api/v3/evo2-1b-base/predict/`` – returns total log-probability of each input sequence.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM

            seq = "MELSDVK..."

            resp = BioLM(
                entity="evo2-1b-base",
                action="predict",
                items=[{"sequence": seq}],
            )
            print(resp)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/evo2-1b-base/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{"items": [{"sequence": "MELSDVK..."}]}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/evo2-1b-base/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json",
            }
            payload = {
                "items": [{"sequence": "MELSDVK..."}],
            }

            resp = requests.post(url, headers=headers, json=payload)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/evo2-1b-base/predict/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type"  = "application/json"
            )
            payload <- list(items = list(list(sequence = "MELSDVK...")))

            res <- POST(url, add_headers(.headers = headers), body = payload, encode = "json")
            print(content(res))

.. http:post:: /api/v3/evo2-1b-base/predict/

   Returns log-probability under the Evo2 language model.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   **Request JSON Object**

   - **items** (*array*, max 32*)  
     - **sequence** (*string*) – DNA / RNA / protein sequence (≤ 4 096 tokens)

   :statuscode 200: Successful. ``results[].log_prob`` present.
   :statuscode 400: Validation error (non-IUPAC characters, length …)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo2-1b-base/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "items": [
          {"sequence": "MELSDVK..."}
        ]
      }

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "results": [
          {"log_prob": -123.45}
        ]
      }

Encode (Embeddings)
^^^^^^^^^^^^^^^^^^

``POST /api/v3/evo2-1b-base/encode/`` – transformer block / mean-pool embeddings.

Generate
^^^^^^^^

``POST /api/v3/evo2-1b-base/generate/`` – continues a prompt with nucleus / top-k sampling.

Performance
-----------

- GPU-accelerated on **NVIDIA L4 24 GB** (1 vGPU, 4 vCPU, 16 GB RAM)
- ≤ 1 s latency for 1 kb sequence scoring; generation scales linearly with new tokens

Applications
------------

- **Variant effect prediction** across coding & non-coding regions
- **Multi-omic embeddings** feeding downstream ML classifiers or regressors
- **De-novo biomolecule generation** with long-range context (>8 k tokens)

Limitations
-----------

- 4 096-token hard limit per request (pad / chunk longer inputs)
- Generation currently returns raw tokens – structure or phenotype not guaranteed

How BioLM Uses Evo2
-------------------

BioLM pipelines pair Evo2 embeddings with gradient-boosting models for splice-site
classification and RNA secondary-structure prediction.  Long-context generation
enables exploration of promoter libraries far beyond traditional k-mer methods.

Related
-------

- ``DNABERT-2`` – Pre-trained masked-language model (DNA-only)
- ``OmniDNA`` – 10 k-context DNA transformer under development

References
----------

- Smith T *et al.* Evo2: Unified language modelling of DNA, RNA and protein.  *Preprint* (2024).

Schema reference
----------------

Pydantic definitions live in ``algorithms.schemas_v3.static.evo2``. 