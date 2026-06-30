ESM C 600M API
==============

ESM C 600M is a 600M-parameter transformer protein language model (36 layers, width 1152, 18 attention heads) trained with masked language modeling on large UniRef, MGnify, and JGI sequence datasets. The API provides GPU-accelerated encoding and masked-token prediction for batches of up to 8 protein sequences, each up to 2048 residues. It returns mean and per-token embeddings from selectable layers, as well as logits, enabling downstream use in protein design, variant effect prediction, functional annotation, and large-scale representation learning workflows.

Predict
-------

Predict masked residues in input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esmc-600m",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACDEFGHIK<mask>L"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esmc-600m/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACDEFGHIK<mask>L"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esmc-600m/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACDEFGHIK<mask>L"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esmc-600m/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACDEFGHIK<mask>L"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esmc-600m/predict/

   Predict endpoint for ESM C 600M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers from which to return representations

        - **include** (*array of strings*, default: ["mean"]) — Output types to include; allowed values: "mean", "per_token", "logits"


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for encoding:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid codes plus "-" character


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for masked prediction:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid codes plus "<mask>" token; must contain at least one "<mask>" token


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for log probability calculation:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using unambiguous amino acid codes only

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmc-600m/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACDEFGHIK<mask>L"
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

        - **embeddings** (*array of objects*, optional) — Mean embeddings per requested layer

          - **layer** (*int*) — Layer index as specified in the request

          - **embedding** (*array of floats*, size: 960 for 300m, 1152 for 600m) — Mean embedding vector for the sequence

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer

          - **layer** (*int*) — Layer index as specified in the request

          - **embeddings** (*array of arrays of floats*, shape: [sequence_length, embedding_size], embedding_size: 960 for 300m, 1152 for 600m) — Embedding vectors for each token position

        - **logits** (*array of arrays of floats*, optional, shape: [sequence_length, vocab_size]) — Raw logits for each token position; vocab_size: 20 (20 standard amino acids)

        - **vocab_tokens** (*array of strings*, optional, size: 20) — Vocabulary tokens corresponding to logits indices


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **logits** (*array of arrays of floats*, shape: [num_masked_positions, vocab_size]) — Raw logits for masked positions; vocab_size: 20 (20 standard amino acids)

        - **sequence_tokens** (*array of strings*, size: sequence_length) — Tokenized input sequence with mask tokens replaced by predicted tokens

        - **vocab_tokens** (*array of strings*, size: 20) — Vocabulary tokens corresponding to logits indices

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "logits": [
              [
                24.125,
                23.625,
                "... (truncated for documentation)"
              ],
              [
                25.375,
                25.125,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "A",
              "C",
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


Encode
------

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esmc-600m",
                action="encode",
                params={
                  "repr_layers": [
                    -1
                  ],
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "sequence": "ACDEFGHIKL"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esmc-600m/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "repr_layers": [
                  -1
                ],
                "include": [
                  "mean"
                ]
              },
              "items": [
                {
                  "sequence": "ACDEFGHIKL"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esmc-600m/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "repr_layers": [
                      -1
                    ],
                    "include": [
                      "mean"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "ACDEFGHIKL"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esmc-600m/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                repr_layers = list(
                  -1
                ),
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  sequence = "ACDEFGHIKL"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esmc-600m/encode/

   Encode endpoint for ESM C 600M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of transformer layers for which embeddings are returned

        - **include** (*array of strings*, default: ["mean"]) — Output types to include; allowed values: "mean", "per_token", "logits"


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid alphabet plus "-" character

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmc-600m/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "repr_layers": [
            -1
          ],
          "include": [
            "mean"
          ]
        },
        "items": [
          {
            "sequence": "ACDEFGHIKL"
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

        - **embeddings** (*array of objects*, optional) — Mean embeddings per requested layer:

          - **layer** (*int*) — Layer index as returned for the requested representation (e.g. 35 for last layer in 600m model)
          - **embedding** (*array of floats*) — Mean embedding vector for the sequence; length depends on model size (1152 for 600m model)

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer:

          - **layer** (*int*) — Layer index as returned for the requested representation
          - **embeddings** (*array of arrays of floats*) — Embedding matrix with shape [sequence_length, embedding_size]; embedding_size depends on model size (1152 for 600m model)

        - **logits** (*array of arrays of floats*, optional) — Logit scores with shape [sequence_length, vocab_size]; vocab_size = 33; values are unbounded floats

        - **vocab_tokens** (*array of strings*, optional, size: 33) — Vocabulary tokens corresponding to indices in logits

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              {
                "layer": 35,
                "embedding": [
                  6.875,
                  87.5,
                  "... (truncated for documentation)"
                ]
              }
            ]
          }
        ]
      }


Performance
-----------

- ESM C 600M is hosted on NVIDIA T4 GPUs with 16 GB memory, providing GPU-accelerated inference for both encoding (embeddings/logits) and masked sequence prediction endpoints.
- Relative to BioLM's ESM-2 650M:
  
  - Delivers substantially higher predictive accuracy on structure-informed benchmarks (e.g., contact precision P@L on CASP15) while maintaining comparable latency at the same batch size (up to 8 sequences of length 2048).
  - Uses an optimized transformer architecture (Pre-LN, rotary embeddings, SwiGLU, bias-free layers) to reduce memory footprint and improve tokens-per-second throughput for long protein sequences.

- Relative to BioLM's larger ESM-2 3B and 15B models:
  
  - Matches or exceeds the predictive performance of ESM-2 3B and approaches ESM-2 15B on representation-learning tasks, with roughly 5× fewer parameters than 3B and much lower GPU memory requirements.
  - Offers a better accuracy–throughput trade-off for high-throughput embedding and masked residue scoring pipelines, enabling similar-quality downstream models at significantly lower compute cost.

- Within the ESM C family on BioLM, the 600M variant typically achieves near-6B-level contact-map and representation quality for many applications, while being substantially faster and cheaper to serve than 6B, making it preferable when large-scale batching or low latency is a priority.

Applications
------------

- Protein embedding generation for downstream predictive modeling, enabling rapid screening of large sequence libraries for properties such as thermostability, expression, or catalytic efficiency; suitable as an input to custom machine learning models in enzyme engineering and general protein optimization, but not a direct predictor of structure or function on its own.
- Unsupervised exploration and clustering of protein sequence space using embeddings from the encoder endpoint to identify novel families, scaffolds, or domains; useful for biotech teams mining metagenomic or proprietary sequence datasets for candidates with distinct sequence signatures, but not a substitute for experimental functional characterization.
- Representation learning for protein variant effect modeling, where embeddings serve as features in supervised models to prioritize mutations that improve stability, activity, or developability; valuable for protein engineering workflows that combine in silico ranking with high-throughput screening, while still requiring labeled data for the specific target and assay of interest.
- Embedding-based protein similarity search and retrieval, using encoder-generated vectors to build approximate nearest-neighbor indices and quickly locate sequences related to known functional benchmarks; effective for hit expansion and scaffold hopping in enzyme discovery pipelines, though it does not guarantee conservation of fine-grained active-site chemistry.
- Using masked-token predictions from the predictor endpoint for mutation suggestion and plausibility checks (e.g., proposing amino acid substitutions at specific positions and scoring them via log probabilities); helpful for narrowing large mutational spaces to evolutionarily consistent variants, but not sufficient alone for precise quantitative predictions such as ΔΔG or kinetic parameters.

Limitations
-----------

- **Maximum Sequence Length**: Input sequences are limited to ``2048`` amino acids; longer sequences must be truncated or processed in segments.
- **Batch Size**: The maximum allowed batch size per request is ``8`` sequences; larger datasets must be split into multiple requests.
- **GPU Type**: Inference is performed on ``T4`` GPUs; performance may vary depending on sequence length, requested ``repr_layers``, and whether ``include`` options like ``per_token`` or ``logits`` are used.
- ESM C 600M is optimized for representation learning and masked token prediction; it does not perform full-sequence generative design or 3D structure prediction through this API.
- The model is trained on natural protein sequences; sequences with extensive non-standard residues or ambiguous tokens may yield less meaningful embeddings or logits.
- For antibody-specific structural or affinity optimization (e.g., detailed CDR modeling), specialized antibody design and structure models generally provide better performance than embeddings or masked predictions from ESM C 600M.

How We Use It
-------------

BioLM uses ESM C 600M as a core encoder in protein design workflows, generating rich sequence representations that drive downstream predictive models and in silico screening. These embeddings enable rapid filtering, ranking, and clustering of candidates for enzyme engineering, antibody optimization, and variant prioritization, and are often combined with masked-residue scoring from the predictor endpoint and external structure- or property-based models to support multi-round, lab-in-the-loop optimization.

- Encodings from ESM C 600M serve as standardized inputs to custom property predictors (e.g., activity, stability, developability), improving model performance with minimal labeled data.
- Masked prediction scores from ESM C 600M help identify tolerated and beneficial mutations, guiding focused mutagenesis libraries and reducing experimental search space.

Related
-------

- ``ESM C 300M`` – Smaller ESM Cambrian variant with similar representations; useful for faster, lower-cost embedding and masked prediction before scaling workflows to ESM C 600M.
- ``ESMFold`` – Structure prediction model that converts ESM C 600M sequence embeddings into 3D structures for structure-aware analysis.
- ``ESM-IF1`` – Inverse folding model that designs sequences from target backbones; can use ESM C 600M embeddings to score or filter candidates.
- ``ESM3 Open Small`` – Multimodal generative model for sequence/structure/function; uses related training data and complements ESM C 600M embeddings in end-to-end protein design pipelines.

References
----------

- ESM Team (2024). "ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning." *EvolutionaryScale Website*, December 4, 2024. https://www.evolutionaryscale.ai/blog/esm-cambrian
