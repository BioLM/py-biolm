ESM3 Open Small API
===================

ESM3 Open Small is a GPU-accelerated protein language model for single-sequence analysis that encodes amino acid sequences into numerical embeddings and predicts single-chain 3D structures. The API supports sequences up to 2,048 residues and batched inference of up to 5 sequences per request, returning mean or per-residue embeddings, optional sequence logits with vocabulary tokens, and PDB-format structures with mean pLDDT. These representations enable downstream tasks such as fitness modeling, clustering, annotation, and structure-guided protein engineering.

Predict
-------

Predict 3D structure (PDB) and mean pLDDT for two protein sequences using the esm3-open-small model.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm3-open-small",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTFFVVALALLAVASA"
                  },
                  {
                    "sequence": "GQDPYVPQENPNTQATF"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm3-open-small/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTFFVVALALLAVASA"
                },
                {
                  "sequence": "GQDPYVPQENPNTQATF"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm3-open-small/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTFFVVALALLAVASA"
                    },
                    {
                      "sequence": "GQDPYVPQENPNTQATF"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm3-open-small/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTFFVVALALLAVASA"
                ),
                list(
                  sequence = "GQDPYVPQENPNTQATF"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm3-open-small/predict/

   Predict endpoint for ESM3 Open Small.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding outputs to return, using values from {"mean", "per_token", "logits"}


      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm3-open-small/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTFFVVALALLAVASA"
          },
          {
            "sequence": "GQDPYVPQENPNTQATF"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **pdb** (*string*) — Predicted protein structure in PDB format as a single concatenated text block

        - **mean_plddt** (*float*, range: 0.0-1.0) — Mean predicted pLDDT confidence score across all residues

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "ATOM      1  N   MET A   1       8.949   7.186  -7.857  1.00  0.98           N  \nATOM      2  CA  MET A   1       9.422   6.492  -6.664  1.00  0.98           C  \nATOM      3  C   MET A   1       8.392... (truncated for documentation)",
            "mean_plddt": 0.9874640107154846
          },
          {
            "pdb": "ATOM      1  N   GLY A   1       8.356  -0.380  -4.298  1.00  0.67           N  \nATOM      2  CA  GLY A   1       6.965  -0.255  -3.875  1.00  0.67           C  \nATOM      3  C   GLY A   1       6.555... (truncated for documentation)",
            "mean_plddt": 0.6948099136352539
          }
        ]
      }


Encode
------

Compute mean and per-token embeddings plus logits for two short protein sequences using the esm3-open-small model.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm3-open-small",
                action="encode",
                params={
                  "include": [
                    "mean",
                    "per_token",
                    "logits"
                  ]
                },
                items=[
                  {
                    "sequence": "MKTFFVVALALLAVASA"
                  },
                  {
                    "sequence": "GSSGSSGSSGSSGSSGS"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm3-open-small/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean",
                  "per_token",
                  "logits"
                ]
              },
              "items": [
                {
                  "sequence": "MKTFFVVALALLAVASA"
                },
                {
                  "sequence": "GSSGSSGSSGSSGSSGS"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm3-open-small/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean",
                      "per_token",
                      "logits"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MKTFFVVALALLAVASA"
                    },
                    {
                      "sequence": "GSSGSSGSSGSSGSSGS"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm3-open-small/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean",
                  "per_token",
                  "logits"
                )
              ),
              items = list(
                list(
                  sequence = "MKTFFVVALALLAVASA"
                ),
                list(
                  sequence = "GSSGSSGSSGSSGSSGS"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm3-open-small/encode/

   Encode endpoint for ESM3 Open Small.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:


      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, required, min length: 1, max length: 771) — Protein sequence composed of standard amino acid codes and ":" separators for multimers (up to 3 occurrences).

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm3-open-small/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean",
            "per_token",
            "logits"
          ]
        },
        "items": [
          {
            "sequence": "MKTFFVVALALLAVASA"
          },
          {
            "sequence": "GSSGSSGSSGSSGSSGS"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **pdb** (*string*) — Predicted protein structure in standard PDB format

        - **mean_plddt** (*float*, range: 0.0-1.0) — Mean predicted Local Distance Difference Test (pLDDT) score indicating prediction confidence

        - **ptm** (*float*, range: 0.0-1.0) — Predicted Template Modeling (pTM) score evaluating global structural accuracy

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              43.61764907836914,
              15.047794342041016,
              "... (truncated for documentation)"
            ],
            "per_token_embeddings": [
              [
                170.0,
                69.0,
                "... (truncated for documentation)"
              ],
              [
                51.0,
                -19.25,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "logits": [
              [
                -1.2734375,
                -3.546875,
                "... (truncated for documentation)"
              ],
              [
                0.26171875,
                -2.296875,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "vocab_tokens": [
              "A",
              "C",
              "... (truncated for documentation)"
            ]
          },
          {
            "embeddings": [
              -64.04412078857422,
              -59.0,
              "... (truncated for documentation)"
            ],
            "per_token_embeddings": [
              [
                -19.0,
                187.0,
                "... (truncated for documentation)"
              ],
              [
                41.75,
                -171.0,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "logits": [
              [
                -0.65234375,
                -2.75,
                "... (truncated for documentation)"
              ],
              [
                0.96484375,
                -1.59375,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "vocab_tokens": [
              "A",
              "C",
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- ESM3 Open Small is deployed on NVIDIA L4 GPUs with mixed‑precision inference and memory‑efficient attention, enabling rapid structure prediction for sequences up to 2,048 residues per request
- For single‑sequence structure prediction, ESM3 Open Small (via the ``predictor`` endpoint) is typically several times faster than AlphaFold2 and ESMFold at similar lengths, as it avoids MSA and template searches and uses a single forward pass with argmax decoding of structure tokens
- On benchmark datasets (CAMEO, CASP14/15), the 1.4B ESM3 family slightly underperforms AlphaFold2 in mean lDDT / TM-score, but outperforms ESM2-based pipelines and is comparable to or better than ESMFold; ESM3 also maintains stronger accuracy on single-sequence and orphan proteins where MSA-dependent models degrade
- Compared to larger internal ESM3 variants (7B, 98B), ESM3 Open Small trades some structure accuracy and generative capability for substantially lower latency and cost, making it suitable for high‑throughput embedding (``encoder`` endpoint), log‑probability scoring, and rapid structure prediction in production workflows

Applications
------------

- Single-sequence protein structure prediction for design and engineering workflows
- Generating candidate folds and backbones for de novo protein design
- Scoring and ranking designed or natural variants via log-likelihood for prioritization in wet-lab campaigns
- Embedding protein sequences for downstream ML tasks such as activity prediction, clustering, or retrieval across large sequence libraries
- Not optimal for very long proteins beyond 2048 residues or for systems requiring explicit multi-chain complex modeling

Limitations
-----------

- **Maximum Sequence Length**: The API accepts sequences up to ``2048`` amino acids. Longer proteins must be truncated or processed in overlapping windows; very long or highly disordered chains may lead to lower-quality structures.
- **Batch Size**: A maximum of ``5`` sequences per request is supported across all endpoints (``encoder``, ``predictor``, ``log_prob``). Larger workloads require batching over multiple API calls and client-side aggregation.
- **Input Alphabet**: All endpoints require unambiguous amino-acid sequences; sequences containing ambiguous or non-standard residues are rejected by ``validate_aa_unambiguous`` and must be cleaned or mapped before submission.
- **Model Scope (ESM3-open-small)**: The hosted model is the ``open-small`` ESM3 variant. It omits certain viral and toxin-related training data and function keywords, so performance on viral proteins, select agents, or highly engineered toxins is intentionally degraded and not representative of full-scale ESM3 models.
- **Prediction Accuracy and Confidence**: Structure predictions return ``pdb`` and ``mean_plddt`` only. ``mean_plddt`` is a model-internal confidence estimate and can be miscalibrated, especially for de novo or highly out-of-distribution designs; low or moderate values should be treated as preliminary and validated with orthogonal tools or experiments.
- **Use-Case Fit**: ESM3-open-small is optimized for sequence-level embeddings, log-probabilities, and single-chain structure prediction. It is not a full multimer, ligand-binding, or function-design engine, and is less suitable as the sole model for high-precision engineering of complexes, binding interfaces, or pathogen-related sequences.

How We Use It
-------------

BioLM uses ESM3 Open Small as a standardized, API-accessible engine for protein sequence embeddings, structural inference, and sequence scoring that accelerates design–build–test cycles. ESM3's multimodal training on sequence and structure allows us to encode proteins into rich latent vectors, estimate sequence log-probabilities as a proxy for evolutionary plausibility, and obtain fast single-sequence structure predictions, which together enable high-throughput virtual screening, in silico stability filtering, and prioritization of candidates for experimental validation in protein engineering programs.

- Embeddings from the ``encoder`` endpoint integrate into downstream ML models for fitness prediction, clustering, and navigation of sequence space.
- The ``predictor`` endpoint provides rapid 3D structure hypotheses and mean pLDDT for structure-aware filtering alongside other BioLM models.

Related
-------

- ``ESMFold`` – Single-sequence structure prediction model using ESM embeddings; useful for independently validating or refining structures generated or implied by ESM3 Open Small.
- ``ESM-2 150M`` – Lightweight protein language model for fast sequence embeddings; complements ESM3 Open Small when you only need sequence representations without multimodal reasoning or structure generation.
- ``AlphaFold2`` – High-accuracy MSA-based structure predictor; provides a strong baseline to benchmark ESM3 Open Small’s single-sequence structure predictions on key targets.
- ``ESM-IF1`` – Inverse folding model for sequence design from structure; can be paired with ESM3 Open Small structure predictions for round‑trip designability checks and alternative sequence proposals.

References
----------

- Lin, Z., Akin, H., Rao, R., et al. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*, 379(6637), 1123–1130.
