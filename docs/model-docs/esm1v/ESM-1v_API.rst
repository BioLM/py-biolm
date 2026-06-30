.. _esm1v_api:

ESM-1v API
==========

ESM-1v is a masked protein language model ensemble for zero-shot prediction of the effects of mutations on protein function. ESM-1v supports single-residue masked token prediction for mutation effect inference, library scoring, and protein design. BioLM provides scalable API access to ESM-1v for high-throughput mutational scanning, variant prioritization, and protein engineering workflows.

Predict
-------

This endpoint scores masked tokens in protein sequences and returns per-token probabilities for all ensemble variants.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm1v",
                action="predict",
                items=[{"sequence": "AVILTI<mask>HGPR"}]
            )
            for variant in response["results"][0]:
                print(variant)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python

            import requests
            url = "https://biolm.ai/api/v3/esm1v-all/predict/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {"items": [{"sequence": "AVILTI<mask>HGPR"}]}
            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code-block:: r

            library(httr)
            url <- "https://biolm.ai/api/v3/esm1v-all/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(items = list(list(sequence = "AVILTI<mask>HGPR")))
            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash

            curl -X POST https://biolm.ai/api/v3/esm1v-all/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "items": [
                  {"sequence": "AVILTI<mask>HGPR"}
                ]
              }'

.. http:post:: /v3/esm1v-all/predict/

   Scores masked tokens across all ensemble variants and returns per-variant per-token probability labels.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **items** (*array of objects*, max. 5) --- List of input sequences:

        - **sequence** (*string*, max length: 512) — Protein sequence containing exactly one `<mask>` token

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm1v-all/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "items": [
          {"sequence": "AVILTI<mask>HGPR"}
        ]
      }

   :statuscode 200: Successful scoring. Returns per-input label lists.
   :statuscode 400: Invalid input (mask usage, sequence length, or batch size).
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) — List of prediction objects for each request item, where each object contains:

        - **esm1v-n1** (*array of objects*) — List of predictions from model variant n1:
          - **token** (*int*) — Index of masked position
          - **token_str** (*string*) — Predicted amino acid
          - **score** (*float*) — Probability of amino acid at masked position
          - **sequence** (*string*) — Full sequence with `<mask>` replaced
        - **esm1v-n2** (*array of objects*) — ...
        - **esm1v-n3** (*array of objects*) — ...
        - **esm1v-n4** (*array of objects*) — ...
        - **esm1v-n5** (*array of objects*) — ...

   **Example response**:

   .. code-block:: json

      {
        "results": [
          {
            "esm1v-n1": [
              {"token": 6, "token_str": "A", "score": 0.12, "sequence": "AVILTIAHGPR"},
              {"token": 6, "token_str": "V", "score": 0.10, "sequence": "AVILTVHGPR"}
            ],
            "esm1v-n2": [
              {"token": 6, "token_str": "A", "score": 0.13, "sequence": "AVILTIAHGPR"},
              {"token": 6, "token_str": "V", "score": 0.09, "sequence": "AVILTVHGPR"}
            ],
            "...": "..."
          }
        ]
      }

Performance
-----------
- **Batch size**: up to 5 sequences per request
- **Sequence length**: up to 512 residues
- **CPU**: 2 CPUs & 2GB RAM for individual variants; **GPU (T4)** for ensemble endpoint
- Typical completion: seconds per batch

Applications
------------
- Mutation effect prediction via masked language modeling
- Single-residue functional scoring and variant prioritization
- Ensemble scoring by averaging across model variants
- High-throughput mutational scanning and library design

Limitations
-----------
- **Mask requirement**: Exactly one `<mask>` token per sequence
- **Sequence length**: ≤ 512 residues
- **Batch size**: max 5 sequences per call
- Not a general-purpose structure predictor (use ESMFold or AlphaFold2 for structure)
- Outputs require downstream interpretation and validation for experimental design

How BioLM Uses ESM-1v
---------------------
BioLM leverages ESM-1v for:

- Automated, high-throughput mutation effect prediction and variant prioritization
- In silico combinatorial mutagenesis and library scoring
- Iterative design cycles for protein engineering and directed evolution
- Integration into ML/AI pipelines for protein function prediction

Related
-------
- :ref:`esm2_api` — Advanced protein language model
- :ref:`esmfold_api` — Sequence-to-structure prediction
- :ref:`esm_if1_api` — Inverse folding for sequence design

References
----------
- Meier, J. *et al.* Language models enable zero-shot prediction of the effects of mutations on protein function. *bioRxiv* (2021).
- Rives, A. *et al.* Biological structure and function from protein language models. *Science* **369**, 650–655 (2020).
- BioLM API documentation: https://docs.biolm.ai/

