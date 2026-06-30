.. _esm2_api:

ESM-2 API
=========

Model Overview
--------------
ESM-2 is a family of large-scale masked protein language models trained on evolutionary-scale sequence data. ESM-2 supports embedding extraction, contact map prediction, attention map extraction, and masked-token logit scoring for protein sequences. BioLM provides scalable API access to ESM-2 for high-throughput protein engineering, structure-function analysis, and large-scale metagenomic annotation workflows.

Encode
------

This endpoint extracts embeddings, contact maps, attention weights, and logits from input protein sequences.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2",
                action="encode",
                items=[{"sequence": "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRML"}],
                params={"include": ["mean", "contacts", "logits", "attentions"]}
            )
            print(response)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-8m/encode/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {
                "params": {"include": ["mean", "contacts", "logits", "attentions"]},
                "items": [{"sequence": "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRML"}]
            }
            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code-block:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-8m/encode/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type" = "application/json"
            )
            body <- list(
              params = list(include = c("mean", "contacts", "logits", "attentions")),
              items = list(list(sequence = "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRML"))
            )
            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-8m/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "params": {"include": ["mean", "contacts", "logits", "attentions"]},
                "items": [{"sequence": "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRML"}]
              }'


.. http:post:: /api/v3/esm2-8m/encode/

   Generates embeddings, contact maps, attention weights, and logits for input sequences.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*) --- Output configuration:

        - **include** (*array*) — List of outputs: `mean`, `per_token`, `bos`, `contacts`, `logits`, `attentions` (default: `mean`)
        - **repr_layers** (*array of int*) — Layer indices to extract representations from (default: [-1])

      - **items** (*array of objects*, max. 8) --- List of sequences:

        - **sequence** (*string*, max length: 2048) — Protein sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-8m/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {"include": ["mean", "contacts", "logits", "attentions"]},
        "items": [
          {"sequence": "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRML"}
        ]
      }

   :statuscode 200: Successful encoding. Returns embeddings and auxiliary outputs.
   :statuscode 400: Invalid input (sequence length or params).
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) — One object per input sequence:

        - **sequence_index** (*int*) — Index of sequence in request
        - **embeddings** (*array of objects*) — Mean embeddings per layer (if requested)
        - **per_token_embeddings** (*array of objects*) — Per-token embeddings per layer (if requested)
        - **bos_embeddings** (*array of objects*) — BOS token embeddings per layer (if requested)
        - **contacts** (*array of arrays*) — Contact probability map (if requested)
        - **attentions** (*array of arrays*) — Attention weights (if requested)
        - **logits** (*array of arrays*) — Predicted logits for each token (if requested)
        - **vocab_tokens** (*array of str*) — Vocabulary tokens (if logits requested)

   **Example response**:

   .. code-block:: json

      {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [{"layer": -1, "embedding": [0.1, 0.2, "..."]}],
            "contacts": [[0.01, 0.02, "..."]],
            "logits": [[0.1, 0.2, "..."]],
            "attentions": [[[0.01, "..."]]]
          }
        ]
      }

Predict
-------

This endpoint returns masked-token logit scores for one or more sequences with a single <mask> token.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2",
                action="predict",
                items=[{"sequence": "MAETAVINHKKRKNSPRI<mask>QSNDLTE"}]
            )
            print(response)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python
            import requests

            url = "https://biolm.ai/api/v3/esm2-8m/predict/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {"items": [{"sequence": "MAETAVINHKKRKNSPRI<mask>QSNDLTE"}]}
            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code-block:: r
            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-8m/predict/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type" = "application/json"
            )
            body <- list(items = list(list(sequence = "MAETAVINHKKRKNSPRI<mask>QSNDLTE")))
            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash
            curl -X POST https://biolm.ai/api/v3/esm2-8m/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "items": [{"sequence": "MAETAVINHKKRKNSPRI<mask>QSNDLTE"}]
              }'

.. http:post:: /api/v3/esm2-8m/predict/

   Returns masked-token logit scores for one or more sequences.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **items** (*array of objects*, max. 8) --- List of masked sequences:

        - **sequence** (*string*, max length: 2048) — Protein sequence with exactly one <mask>

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-8m/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "items": [
          {"sequence": "MAETAVINHKKRKNSPRI<mask>QSNDLTE"}
        ]
      }

   :statuscode 200: Successful scoring. Returns `results` arrays of logits.
   :statuscode 400: Invalid input (mask usage or sequence length).
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) — One object per input sequence:

        - **logits** (*array of arrays*) — Predicted logits for each token
        - **sequence_tokens** (*array of str*) — Sequence tokens
        - **vocab_tokens** (*array of str*) — Vocabulary tokens

   **Example response**:

   .. code-block:: json

      {
        "results": [
          {
            "logits": [[0.1, 0.2, "..."], ["..."]],
            "sequence_tokens": ["M", "A", "..."],
            "vocab_tokens": ["A", "C", "..."]
          }
        ]
      }

Performance
-----------
- **GPU-accelerated** on T4 GPUs (150M/650M variants)
- **Batch size**: up to 8 sequences per request
- **Sequence length**: up to 2048 residues
- Typical completion: seconds per batch (depends on model size and sequence length)

Applications
------------
- Embedding-based feature extraction for downstream ML/AI pipelines
- Contact-guided structural modeling and docking
- Attention analysis for functional site identification
- Masked-token scoring for mutational effect inference and variant prioritization
- Large-scale metagenomic structure annotation and clustering

Limitations
-----------
- **Sequence length**: ≤ 2048 residues
- **Batch size**: up to 8 sequences per request
- **Mask usage**: exactly one `<mask>` token per sequence for predict
- Not a general-purpose structure predictor (use ESMFold or AlphaFold2 for full-atom structure)
- Outputs require downstream interpretation and validation for experimental design

How BioLM Uses ESM-2
--------------------
BioLM leverages ESM-2 for:

- High-throughput protein engineering and structure-function analysis
- Large-scale metagenomic annotation and dataset curation
- Embedding extraction for ML/AI workflows
- Contact map and attention analysis for structure/function prediction

Related
-------
- :ref:`esm1v_api` — Masked token predictions for single-residue functional scoring
- :ref:`esm_if1_api` — Inverse folding for backbone-guided sequence design
- :ref:`esmfold_api` — Rapid single-chain structure prediction without MSAs

References
----------
- Lin, Z. *et al.* Evolutionary-scale prediction of atomic level protein structure with a language model. *bioRxiv* (2022).
- Rives, A. *et al.* Biological structure and function from protein language models. *Science* **369**, 650–655 (2020).
- BioLM API documentation: https://docs.biolm.ai/
