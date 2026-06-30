.. _progen2_api:

ProGen2 API
===========

ProGen2 is a large-scale autoregressive protein language model suite for protein sequence generation and fitness prediction. BioLM provides scalable API access to multiple ProGen2 variants (OAS, Medium, Large, BFD90) for modern protein engineering, antibody/nanobody design, and synthetic biology workflows.

Generate
--------

This endpoint generates protein sequences from a given context using ProGen2.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            # Generate protein sequences from context
            result = BioLM(
                entity="progen2-oas",
                action="generate",
                items=[{"context": "EVQ"}],
                params={
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "num_samples": 2,
                    "max_length": 128
                }
            )
            print(result)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python

            import requests

            url = "https://biolm.ai/api/v3/progen2-oas/generate/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {
                "params": {"temperature": 0.8, "top_p": 0.9, "num_samples": 2, "max_length": 128},
                "items": [{"context": "EVQ"}]
            }
            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code-block:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/progen2-oas/generate/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type" = "application/json"
            )
            body <- list(
              params = list(temperature = 0.8, top_p = 0.9, num_samples = 2, max_length = 128),
              items = list(list(context = "EVQ"))
            )
            res <- POST(url, add_headers(.headers=headers), body=body, encode="json")
            print(content(res))

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash

            curl -X POST https://biolm.ai/api/v3/progen2-oas/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "params": {
                  "temperature": 0.8,
                  "top_p": 0.9,
                  "num_samples": 2,
                  "max_length": 128
                },
                "items": [
                  {"context": "EVQ"}
                ]
              }'

.. http:post:: /api/v3/progen2-oas/generate/

   Generate protein sequences from a given context using ProGen2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*) --- Generation parameters:

        - **temperature** (*float*) — Sampling temperature (default: 0.8, range: 0.0–8.0)
        - **top_p** (*float*) — Nucleus sampling probability (default: 0.9, range: 0.0–1.0)
        - **num_samples** (*int*) — Number of sequences to generate per context (default: 1, range: 1–3)
        - **max_length** (*int*) — Maximum length of generated sequence (default: 128, range: 12–512)

      - **items** (*array of objects*, max. 1) --- List of input contexts:

        - **context** (*string*, length 1–512) — Protein sequence context (AAs, unambiguous)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/progen2-oas/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {
          "temperature": 0.8,
          "top_p": 0.9,
          "num_samples": 2,
          "max_length": 128
        },
        "items": [
          {"context": "EVQ"}
        ]
      }

   :statuscode 200: Successful. Returns generated sequences and log-likelihoods.
   :statuscode 400: Invalid input (parameter, context, or batch size).
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of arrays*) — For each input context, a list of generated sequence objects:

        - **sequence** (*string*) — Generated protein sequence
        - **ll_sum** (*float*) — Sum of log-likelihoods for the sequence
        - **ll_mean** (*float*) — Mean log-likelihood per token

   **Example response**:

   .. code-block:: json

      {
        "results": [
          [
            {
              "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDGGGYS",
              "ll_sum": -22.09,
              "ll_mean": -0.21
            },
            {
              "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDRYSSS",
              "ll_sum": -20.83,
              "ll_mean": -0.20
            }
          ]
        ]
      }

Performance
-----------
- **Batch size**: 1 context per request (per schema)
- **Max sequence length**: 512 residues (input context and output)
- **CPU/GPU**: OAS/Medium: 2 CPUs, 8GB RAM; Large/BFD90: 4 CPUs, 16GB RAM; GPU (T4) for Medium/Large/BFD90
- Typical latency: seconds per request

Applications
------------
- De novo protein sequence generation from motifs or partial sequences
- Antibody/nanobody library design (e.g., CDR diversification)
- Enzyme engineering and synthetic protein design
- Fitness landscape exploration and zero-shot fitness prediction
- Large-scale protein library curation for screening

Limitations
-----------
- **Input context**: Only unambiguous amino acids (no ambiguous or non-standard codes)
- **Max sequence length**: 512 residues (input and output)
- **Batch size**: 1 context per request
- **No structure conditioning**: Sequence-only generation (no structure input)

How BioLM Uses ProGen2
----------------------
BioLM enables scalable, programmatic access to ProGen2 for advanced protein design workflows:

- Antibody/nanobody library design: Generate diverse CDR or full-length antibody/nanobody sequences for library construction, using context motifs or frameworks as prompts.
- Enzyme and protein engineering: Design novel enzymes or functional proteins by seeding with active site motifs or partial sequences, generating variants for screening.
- Fitness landscape exploration: Sample and score large numbers of variants for zero-shot fitness prediction, prioritizing candidates for experimental validation.
- Dataset curation: Generate synthetic protein libraries for ML training, benchmarking, or augmentation of sparse datasets.

Related
-------
- :ref:`esm1v_api` — Masked language model for mutation effect prediction
- :ref:`esm2_api` — Advanced protein language model
- :ref:`ablang2_api` — Antibody-specific sequence restoration and embedding
- :ref:`protgpt2_api` — Alternative protein sequence generator

References
----------
- Madani, A., et al. (2023). ProGen2: Exploring the Boundaries of Protein Language Models. *bioRxiv*. https://doi.org/10.1101/2023.06.21.546019

