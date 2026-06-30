Omni-DNA 1B API
===============

Omni-DNA 1B is a 1B-parameter autoregressive genomic foundation model exposed as a GPU-accelerated encoder and sequence log-probability predictor for DNA. The API supports unambiguous DNA sequences up to 2048 nt and batched inference (up to 2 sequences per request), returning pooled (mean) and last-token embeddings suitable for downstream classifiers, regressors, or retrieval systems, as well as overall sequence log-likelihoods. Omni-DNA 1B attains an average MCC of 0.767 across 18 NT benchmark tasks.

Predict
-------

Predict log probabilities for input DNA sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="omni-dna-1b",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  },
                  {
                    "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/omni-dna-1b/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                },
                {
                  "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/omni-dna-1b/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                    },
                    {
                      "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/omni-dna-1b/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                ),
                list(
                  sequence = "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/omni-dna-1b/predict/

   Predict endpoint for Omni-DNA 1B.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to include in the response:

          - Enum values: "mean", "last"

      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — DNA sequence using unambiguous nucleotides (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/omni-dna-1b/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
          },
          {
            "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
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

        - **mean** (*array of objects*, optional) — Mean embedding representation across tokens in the input sequence

          - **embedding** (*array of floats*) — Embedding vector for the mean representation

        - **last** (*array of objects*, optional) — Embedding representation of the last token in the input sequence

          - **embedding** (*array of floats*) — Embedding vector for the last-token representation

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **log_prob** (*float*, range: negative infinity to 0.0) — Log probability score of the input sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -245.78988647460938
          },
          {
            "log_prob": -164.30740356445312
          }
        ]
      }


Encode
------

Generate embeddings for input DNA sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="omni-dna-1b",
                action="encode",
                params={
                  "include": [
                    "mean",
                    "last"
                  ]
                },
                items=[
                  {
                    "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  },
                  {
                    "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/omni-dna-1b/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean",
                  "last"
                ]
              },
              "items": [
                {
                  "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                },
                {
                  "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/omni-dna-1b/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean",
                      "last"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                    },
                    {
                      "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/omni-dna-1b/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean",
                  "last"
                )
              ),
              items = list(
                list(
                  sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                ),
                list(
                  sequence = "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/omni-dna-1b/encode/

   Encode endpoint for Omni-DNA 1B.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to include in response


          - Allowed values:

            - "mean" — Mean embedding across all sequence positions
            - "last" — Embedding from the last sequence position


      - **items** (*array of objects*, min: 1, max: 2) --- DNA sequences to encode:

        - **sequence** (*string*, required, min length: 1, max length: 2048) — DNA sequence containing only unambiguous nucleotide characters (A, T, G, C)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/omni-dna-1b/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean",
            "last"
          ]
        },
        "items": [
          {
            "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
          },
          {
            "sequence": "TTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGATTGACCTGAACCTGAACTGA"
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

        - **mean** (*array of objects*, optional) — Mean-pooled embeddings over all tokens in the encoded sequence:

          - **embedding** (*array of floats*) — Embedding vector for the sequence; length equals the model hidden size for the selected Omni-DNA variant (e.g., 2048 for "1b"); values typically range from approximately -40.0 to 40.0.

        - **last** (*array of objects*, optional) — Embeddings from the final token position of the encoded sequence:

          - **embedding** (*array of floats*) — Embedding vector for the last token; length equals the model hidden size for the selected Omni-DNA variant (e.g., 2048 for "1b"); values typically range from approximately -40.0 to 40.0.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "mean": [
              {
                "embedding": [
                  0.30445659160614014,
                  -0.13346552848815918,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "last": [
              {
                "embedding": [
                  1.8188480138778687,
                  -0.9933197498321533,
                  "... (truncated for documentation)"
                ]
              }
            ]
          },
          {
            "mean": [
              {
                "embedding": [
                  0.05789097025990486,
                  -0.18520615994930267,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "last": [
              {
                "embedding": [
                  0.7221266627311707,
                  -0.7872365713119507,
                  "... (truncated for documentation)"
                ]
              }
            ]
          }
        ]
      }


Performance
-----------

- Omni-DNA 1B is deployed on NVIDIA L4 GPUs with 16 GB memory and optimized transformer inference (RoPE positional embeddings, fused attention, mixed-precision) to provide stable latency for both embedding and log-probability endpoints.
- On Nucleotide Transformer downstream benchmarks, Omni-DNA 1B achieves an average score of 0.767 across 18 classification tasks, outperforming DNABERT-2 and NT Transformer models on 13/18 tasks, and exceeding the next-best Omni-DNA variant (116M) by 0.012 absolute.
- Within the Omni-DNA family (20M–1B parameters), the 1B model systematically gives the highest downstream accuracy; the gain over 116M is largest on histone modification tasks (e.g., +0.019 average MCC on histone markers) while maintaining comparable promoter and splice-site performance.
- Architectural choices in Omni-DNA 1B—non-parametric LayerNorm and RoPE—reduce pretraining loss and improve fine-tuning accuracy relative to smaller Omni-DNA variants that use RMSNorm or ALiBi, and make the 1B model more robust to performance degradation under tokenizer vocabulary expansion.

Applications
------------

- Identification and annotation of regulatory DNA elements in genomic sequences using sequence embeddings and log-probabilities, helping teams prioritize promoters, enhancers, and splice sites when building expression constructs or screening regulatory libraries; most useful on sequences long enough to capture flanking context, and less informative for extremely short inputs (<50 nucleotides).
- In silico prioritization of epigenetic marker assays (e.g., histone acetylation and methylation) by using Omni-DNA embeddings as features in downstream predictive models, reducing the number of chromatin experiments needed in cell-line engineering or biomarker discovery; performance can be lower for rare marks or cell types underrepresented in public training data and requires task-specific supervised training.
- Functional characterization of novel or engineered constructs by embedding DNA sequences and training lightweight heads to classify sequence type (e.g., promoter vs enhancer vs background) or approximate functional labels, accelerating design–build–test cycles in synthetic biology pipelines; predictions for highly synthetic or out-of-distribution constructs should be experimentally validated.
- Motif and pattern exploration in regulatory DNA using embeddings to cluster sequences or compare related variants, supporting design of libraries for promoter/enhancer optimization or splice-site tuning; well-suited to ranking and grouping candidates rather than providing absolute quantitative estimates of motif strength.
- Multi-task genomic modeling workflows that reuse a single set of Omni-DNA embeddings across many tasks (regulatory classification, epigenetic prediction, species/source attribution), simplifying bioinformatics pipelines and reducing retraining costs for companies running large-scale genomic or synthetic library screens; in scenarios where a single task demands maximum possible accuracy, a specialized model trained end-to-end for that task may still perform slightly better.

Limitations
-----------

- **Maximum Sequence Length**: Input DNA ``sequence`` strings are limited to ``2048`` nucleotides. Longer sequences must be truncated or split client-side; requests exceeding this limit will be rejected.

- **Batch Size**: Each ``encoder`` or ``predictor`` call supports at most ``2`` ``items`` per request. Larger datasets should be processed in multiple batched calls.

- **GPU Type**: The ``"1b"`` variant is provisioned on **GPU Type** ``L4`` with stricter ``timeout``; very long or many small requests in sequence may approach this limit. Smaller variants (``"20m"``–``"700m"``) use ``T4`` GPUs with lower memory and may be preferable for high-throughput, lower-accuracy use cases.

- **Embedding Outputs**: The ``encoder`` endpoint only returns token-level embeddings aggregated as ``mean`` or ``last`` (controlled via the ``include`` option). It does not expose the full cross-modal capabilities from the paper (e.g., DNA-to-text or DNA-to-image generation) through this API.

- **Cross-Modal and Task Limitations**: Although Omni-DNA was finetuned on multiple task types, the hosted API does not provide task-specific classification heads or DNA-to-function/image outputs. For those tasks, Omni-DNA embeddings must be combined with separate downstream models built by the user.

- **Use Case Suitability**: This model family is not designed for long-range genomic interaction prediction, variant effect prediction, or other tasks requiring context substantially beyond ``2048`` bases or specialized architectures; dedicated models are recommended for those applications.

How We Use It
-------------

Omni-DNA 1B enables unified DNA sequence representation within BioLM’s protein engineering workflows, providing consistent embeddings and sequence likelihoods that connect genomic context to downstream protein design decisions. By standardizing DNA encoding via scalable APIs, Omni-DNA 1B accelerates tasks such as linking regulatory regions to expression outcomes, triaging construct variants, and prioritizing edits that are more consistent with learned genomic patterns before moving into protein-level generative design and predictive modeling.

- Integrates with BioLM’s protein and antibody design pipelines by supplying DNA embeddings and log-probability scores that inform construct selection and variant pruning.
- Supports multi-round design–build–test cycles by enabling rapid, batched evaluation (up to 2 sequences per request, up to 2048 bases each) through consistent encoder and predictor endpoints.

Related
-------

- ``DNABERT-2`` – Bidirectional transformer for genomic sequence modeling, useful as a baseline or complementary model when comparing Omni-DNA's autoregressive embeddings and log-probabilities on the same DNA tasks.
- ``NT-Transformer`` – Genomic foundation model family trained on similar benchmarks; helpful for benchmarking Omni-DNA embeddings and log-probabilities on NT and Genomic Benchmark classification tasks.
- ``HyenaDNA`` – Long-range genomic sequence model; can be compared with Omni-DNA when evaluating how well sequence embeddings capture long-range regulatory patterns.
- ``Caduceus`` – Bidirectional long-range DNA model; provides a strong reference for assessing Omni-DNA performance on regulatory element and splice-site prediction tasks using its embeddings.

References
----------

- Li, Z., Subasri, V., Shen, Y., Li, D., Zhao, Y., Stan, G.-B., & Shan, C. (2024). *Omni-DNA: A Unified Genomic Foundation Model for Cross-Modal and Multi-Task Learning*. arXiv:2405.14333. https://arxiv.org/abs/2405.14333
