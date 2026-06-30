ESM C 300M API
==============

ESM C 300M is a GPU-accelerated protein language model that learns unsupervised representations of amino acid sequences for downstream structure- and function-related tasks. The API exposes two actions: an encoder for sequence embeddings and logits, and a predictor for masked-token completion. It supports batches of up to 8 sequences with lengths up to 2048 residues, returns per-layer mean or per-token embeddings, and provides logits with associated vocab tokens for integration into protein engineering, variant scoring, and large-scale annotation pipelines.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esmc-300m",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACD<mask>E"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esmc-300m/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACD<mask>E"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esmc-300m/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACD<mask>E"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esmc-300m/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACD<mask>E"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esmc-300m/predict/

   Predict endpoint for ESM C 300M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers to compute and return representations for

        - **include** (*array of strings*, default: ["mean"]) — Output components to include in the response; allowed values: "mean", "per_token", "logits"


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, required, min length: 1, max length: 2048) — Protein sequence using extended amino acid codes with "-" allowed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmc-300m/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACD<mask>E"
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

        - **embeddings** (*array of objects*, optional) — Layer-wise mean embeddings

          - **layer** (*int*) — Layer index for this embedding
          - **embedding** (*array of floats*) — Mean embedding vector for the layer

        - **per_token_embeddings** (*array of objects*, optional) — Layer-wise per-token embeddings

          - **layer** (*int*) — Layer index for these embeddings
          - **embeddings** (*array of arrays of floats*) — Per-token embedding vectors for the layer

        - **logits** (*array of arrays of floats*, optional) — Masked-position logits over the amino acid vocabulary

        - **vocab_tokens** (*array of strings*, optional) — Ordered amino acid vocabulary tokens corresponding to logits


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **logits** (*array of arrays of floats*) — Logit scores for each masked position over the amino acid vocabulary

        - **sequence_tokens** (*array of strings*) — Input sequence tokens after tokenization

        - **vocab_tokens** (*array of strings*) — Ordered amino acid vocabulary tokens corresponding to logits

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "logits": [
              [
                16.25,
                16,
                "... (truncated for documentation)"
              ],
              [
                24.25,
                24.25,
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
                entity="esmc-300m",
                action="encode",
                params={
                  "repr_layers": [
                    -1,
                    -2
                  ],
                  "include": [
                    "mean",
                    "per_token"
                  ]
                },
                items=[
                  {
                    "sequence": "ACDEFGHIKLMN"
                  },
                  {
                    "sequence": "MKVGA--PKLTYLV"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esmc-300m/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "repr_layers": [
                  -1,
                  -2
                ],
                "include": [
                  "mean",
                  "per_token"
                ]
              },
              "items": [
                {
                  "sequence": "ACDEFGHIKLMN"
                },
                {
                  "sequence": "MKVGA--PKLTYLV"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esmc-300m/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "repr_layers": [
                      -1,
                      -2
                    ],
                    "include": [
                      "mean",
                      "per_token"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "ACDEFGHIKLMN"
                    },
                    {
                      "sequence": "MKVGA--PKLTYLV"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esmc-300m/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                repr_layers = list(
                  -1,
                  -2
                ),
                include = list(
                  "mean",
                  "per_token"
                )
              ),
              items = list(
                list(
                  sequence = "ACDEFGHIKLMN"
                ),
                list(
                  sequence = "MKVGA--PKLTYLV"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esmc-300m/encode/

   Encode endpoint for ESM C 300M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of ints*, default: [-1]) — Indices of model layers to compute representations for
        - **include** (*array of strings*, default: ["mean"]) — Embedding or logits types to include in the response; allowed values: "mean", "per_token", "logits"

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid alphabet plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmc-300m/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "repr_layers": [
            -1,
            -2
          ],
          "include": [
            "mean",
            "per_token"
          ]
        },
        "items": [
          {
            "sequence": "ACDEFGHIKLMN"
          },
          {
            "sequence": "MKVGA--PKLTYLV"
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

        - **embeddings** (*array of objects*, optional) — One object per requested representation layer:

          - **layer** (*int*) — Layer index used to generate the embedding

          - **embedding** (*array of floats*) — Mean embedding vector for the specified layer

        - **per_token_embeddings** (*array of objects*, optional) — One object per requested representation layer:

          - **layer** (*int*) — Layer index used to generate the per-token embeddings

          - **embeddings** (*array of arrays of floats*) — Per-residue embedding vectors for the specified layer

        - **logits** (*array of arrays of floats*, optional) — Per-position logits over the amino acid vocabulary

        - **vocab_tokens** (*array of strings*, optional) — Vocabulary tokens corresponding to the logits indices

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              {
                "layer": 29,
                "embedding": [
                  -26.25,
                  -41.25,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 28,
                "embedding": [
                  -11.5625,
                  1.9609375,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 29,
                "embeddings": [
                  [
                    -64.0,
                    20.0,
                    "... (truncated for documentation)"
                  ],
                  [
                    -39.5,
                    -72.0,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 28,
                "embeddings": [
                  [
                    -28.75,
                    43.75,
                    "... (truncated for documentation)"
                  ],
                  [
                    4.0,
                    -16.375,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ]
          },
          {
            "embeddings": [
              {
                "layer": 29,
                "embedding": [
                  -90.0,
                  -69.5,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 28,
                "embedding": [
                  -7.53125,
                  -4.4375,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 29,
                "embeddings": [
                  [
                    -75.0,
                    -21.875,
                    "... (truncated for documentation)"
                  ],
                  [
                    -134.0,
                    -81.5,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 28,
                "embeddings": [
                  [
                    -11.5,
                    7.25,
                    "... (truncated for documentation)"
                  ],
                  [
                    -37.0,
                    12.25,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ]
          }
        ]
      }


Performance
-----------

- ESM C 300M is deployed on NVIDIA T4 GPUs with 2 vCPUs and 8 GB RAM, optimized for high-throughput embedding and masked-prediction workloads.
- Compared to ESM-2 650M, ESM C 300M delivers similar representation quality for structure-aware tasks (e.g., contact precision P@L on CASP15) while requiring substantially less GPU memory and offering faster inference at the same batch and sequence limits.
- For embedding-focused use (encoder endpoint), ESM C 300M provides higher accuracy and richer representations than ESM-2 150M and ESM-2 35M, while being lighter-weight than ESM-2 650M and significantly more efficient than ESM3 Open Small for sequence-only tasks.
- BioLM’s optimized deployment of ESM C 300M enables consistent latency and scalable parallel inference across large protein sets, making it suitable as a faster drop-in replacement for ESM-2 in production pipelines where ESM3’s added multimodality is not required.

Applications
------------

- Designing novel fluorescent proteins with low sequence similarity to known variants, using embeddings from the encoder endpoint to search sequence space and prioritize candidates with preserved structural features; for example, proposing new GFP-like sequences for multiplexed imaging or biosensors; not optimal for designing proteins that depend on complex post-translational modifications beyond chromophore formation.
- Engineering structurally diverse protein scaffolds around predefined functional motifs by embedding large libraries and clustering or ranking candidates by similarity to known active-site contexts, enabling rapid prototyping of ligand-binding proteins or molecular sensors; less suitable when precise control over long-timescale conformational dynamics is required.
- Generating and screening sequence variants consistent with specified secondary structure or solvent exposure patterns by combining ESM C embeddings with downstream structure predictors, allowing teams to design stable, globular proteins for therapeutic or industrial use; not optimal for de novo prediction of detailed protein-protein interfaces without additional structural models or experimental data.
- Rational protein editing workflows where single-site or local variants are scored via the predictor endpoint (masked-token logit comparisons) to maintain global fold while altering properties such as epitope exposure or sequence length, for instance compressing a protease while preserving catalytic residues; limited when edits require extensive domain rearrangements or introduce highly flexible, disordered regions.
- Rapid functional annotation and triage of large protein libraries by training lightweight classifiers on top of ESM C embeddings to predict coarse function labels (e.g., enzyme class, binding vs non-binding), helping companies prioritize candidates before assays; less accurate for proteins with truly novel or underrepresented functions lacking related sequences in the training data.

Limitations
-----------

- **Maximum Sequence Length**: Sequences may be up to ``2048`` amino acids long (``max_sequence_len``). Longer inputs must be truncated or split into overlapping segments and processed across multiple requests.
- **Batch Size**: Each request can contain at most ``8`` sequences (``batch_size``). Larger collections must be divided into multiple requests.
- **Model Variants**: This API currently serves the ``300m`` ESM C configuration (300M parameters). Larger ESM C variants (``600m``, ``6b``) described in EvolutionaryScale materials are not available through this endpoint and must be accessed via other services if needed.
- **Input Sequence Constraints**: For ``encoder`` requests, sequences may include standard amino acids plus ``-``; ambiguous or non-amino-acid characters are rejected. For ``predictor`` requests, sequences must use unambiguous amino acid letters, with at least one ``<mask>`` token; other ambiguous or unknown residues are not supported.
- **Use-Case Scope**: ESM C is an unsupervised representation model optimized for embeddings and masked-token scoring, not for full 3D structure prediction or controlled generative design. For structure prediction, de novo design, or multimodal sequence–structure–function tasks, models such as ESMFold, AlphaFold2, or ESM3 are more appropriate.
- **Output Types and Size**: In ``encoder`` requests, the ``include`` parameter accepts ``mean`` (per-sequence embeddings), ``per_token`` (per-residue embeddings), and ``logits`` (per-position token scores plus ``vocab_tokens``). ``per_token`` and especially ``logits`` outputs scale with sequence length and can substantially increase response size and latency, so they should be requested only when needed.

How We Use It
-------------

ESM C 300M accelerates protein engineering by providing scalable, standardized sequence embeddings and masked-token predictions that plug directly into design, filtering, and ranking pipelines. We use its representations as a drop-in backbone for custom supervised models (e.g., fitness, stability, developability) and as a scoring layer in active-learning loops, enabling faster triage of large variant libraries and more efficient iteration when combined with structure prediction (e.g., ESMFold) and biophysical property models.

- Drives iterative enzyme and antibody optimization by coupling ESM C embeddings with lab-in-the-loop selection
- Integrates with generative models and downstream ranking tools to focus synthesis on the most promising designs

Related
-------

- ``ESMFold`` – Predicts 3D protein structures from sequences using ESM representations, useful for validating or downstream modeling with ESM C 300M embeddings.
- ``ESM-IF1`` – Performs inverse folding to design sequences for given backbones; combine with ESM C 300M embeddings for structure-conditioned sequence analysis and design.
- ``AlphaFold2`` – Predicts protein structures from sequence; can be paired with ESM C 300M embeddings for structure-function modeling and annotation tasks.
- ``ESM3 Open Small`` – Multimodal generative model over sequence, structure, and function; use ESM C 300M for efficient large-scale embedding and ESM3 for controlled protein generation.

References
----------

- Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J., Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y., Mishra, C., Kim, C., Bartie, L. J., Nemeth, M., Hsu, P. D., Sercu, T., Candido, S., & Rives, A. (2024). Simulating 500 million years of evolution with a language model. *bioRxiv*. https://doi.org/10.1101/2024.01.09.574461
