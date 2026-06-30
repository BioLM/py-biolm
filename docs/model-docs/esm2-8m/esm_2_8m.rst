ESM-2 8M API
============

ESM-2 8M is a lightweight protein language model for sequence-level analysis, exposing masked language modeling and embedding extraction via CPU- or GPU-backed inference. Trained on UniRef with a transformer encoder, it supports up to 2048-residue sequences and batches of up to 8 items. The encoder endpoint provides mean, per-token, BOS embeddings, self-attention, contact maps, and logits, while the predictor endpoint scores masked positions. Typical uses include protein representation learning, mutation scoring, contact inference, and downstream structure or function prediction pipelines.

Predict
-------

Perform masked amino acid prediction for input protein sequences with ESM-2 8M.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2-8m",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACD<mask>FGHI"
                  },
                  {
                    "sequence": "MNOP<mask>RST"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-8m/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACD<mask>FGHI"
                },
                {
                  "sequence": "MNOP<mask>RST"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-8m/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACD<mask>FGHI"
                    },
                    {
                      "sequence": "MNOP<mask>RST"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-8m/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACD<mask>FGHI"
                ),
                list(
                  sequence = "MNOP<mask>RST"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2-8m/predict/

   Predict endpoint for ESM-2 8M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers from which to return representations; negative values index from the last layer

        - **include** (*array of strings*, default: ["mean"]) — Output groups to compute and return; allowed values: "mean", "per_token", "bos", "contacts", "logits", "attentions"


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard amino acid codes, extended amino acid codes, or special tokens such as "-" for gaps and "<mask>" for masked positions

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-8m/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACD<mask>FGHI"
          },
          {
            "sequence": "MNOP<mask>RST"
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

        - **sequence_index** (*int*) — Index of the input sequence in the original request (zero-based)

        - **embeddings** (*array of objects*, optional) — Mean sequence embeddings per requested layer:

          - **layer** (*int*) — Index of the representation layer from the ESM-2 model (range: -1 to total_layers - 1)

          - **embedding** (*array of floats*) — Mean embedding vector for the sequence; length depends on model variant:

            - "8m": 320 dimensions
            - "35m": 480 dimensions
            - "150m": 640 dimensions
            - "650m": 1280 dimensions

        - **bos_embeddings** (*array of objects*, optional) — Beginning-of-sequence (BOS) embeddings per requested layer:

          - **layer** (*int*) — Index of the representation layer from the ESM-2 model (range: -1 to total_layers - 1)

          - **embedding** (*array of floats*) — BOS embedding vector; length depends on model variant:

            - "8m": 320 dimensions
            - "35m": 480 dimensions
            - "150m": 640 dimensions
            - "650m": 1280 dimensions

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer:

          - **layer** (*int*) — Index of the representation layer from the ESM-2 model (range: -1 to total_layers - 1)

          - **embeddings** (*array of arrays of floats*) — Embedding vectors for each token in the sequence, shape: [sequence_length, embedding_size]; embedding_size depends on model variant:

            - "8m": 320 dimensions
            - "35m": 480 dimensions
            - "150m": 640 dimensions
            - "650m": 1280 dimensions

        - **contacts** (*array of arrays of floats*, optional) — Predicted inter-residue contact probabilities, shape: [sequence_length, sequence_length], range: 0.0–1.0

        - **attentions** (*array of arrays of floats*, optional) — Attention weights, shape: [num_attention_heads, sequence_length, sequence_length], range: 0.0–1.0; num_attention_heads depends on model variant:

          - "8m": 20 heads
          - "35m": 20 heads
          - "150m": 20 heads
          - "650m": 20 heads

        - **logits** (*array of arrays of floats*, optional) — Per-token logits over the vocabulary, shape: [sequence_length, vocab_size], unbounded real values

        - **vocab_tokens** (*array of strings*, optional) — Vocabulary tokens corresponding to positions in `logits`, size: vocab_size


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **logits** (*array of arrays of floats*) — Predicted logits for each masked token, shape: [num_masked_positions, vocab_size], unbounded real values

        - **sequence_tokens** (*array of strings*) — Tokenized input sequence including `<mask>` and other special tokens, size: sequence_length

        - **vocab_tokens** (*array of strings*) — Vocabulary tokens corresponding to positions in `logits`, size: vocab_size

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "logits": [
              [
                -0.27711379528045654,
                2.309023141860962,
                "... (truncated for documentation)"
              ],
              [
                0.20582802593708038,
                -0.13699881732463837,
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
              "L",
              "A",
              "... (truncated for documentation)"
            ]
          },
          {
            "logits": [
              [
                0.1745593547821045,
                -0.662738025188446,
                "... (truncated for documentation)"
              ],
              [
                0.5286823511123657,
                0.3695230185985565,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "M",
              "N",
              "... (truncated for documentation)"
            ],
            "vocab_tokens": [
              "L",
              "A",
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Encode
------

Generate embeddings for input protein sequences with ESM-2 8M.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2-8m",
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
                    "sequence": "QRSTVWYACDEFG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-8m/encode/ \
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
                  "sequence": "QRSTVWYACDEFG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-8m/encode/"
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
                      "sequence": "QRSTVWYACDEFG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-8m/encode/"
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
                  sequence = "QRSTVWYACDEFG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2-8m/encode/

   Encode endpoint for ESM-2 8M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers to return embeddings from

        - **include** (*array of strings*, default: ["mean"]) — Output types to compute; allowed values: "mean", "per_token", "bos", "contacts", "logits", "attentions"

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard amino acid codes plus "-" character for gaps

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-8m/encode/ HTTP/1.1
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
            "sequence": "QRSTVWYACDEFG"
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

        - **sequence_index** (*int*) — Index of the input sequence in the original request (0-based)

        - **embeddings** (*array of objects*, optional) — Mean sequence embeddings per requested layer:

          - **layer** (*int*) — Layer index from ESM-2 model (-1 for final layer)

          - **embedding** (*array of floats*, size: 320, 480, 640, or 1280 depending on model size) — Mean embedding vector for the layer

        - **bos_embeddings** (*array of objects*, optional) — Beginning-of-sequence embeddings per requested layer:

          - **layer** (*int*) — Layer index from ESM-2 model (-1 for final layer)

          - **embedding** (*array of floats*, size: 320, 480, 640, or 1280 depending on model size) — Embedding vector for the beginning-of-sequence token

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer:

          - **layer** (*int*) — Layer index from ESM-2 model (-1 for final layer)

          - **embeddings** (*array of arrays of floats*, shape: [sequence_length, embedding_size]) — Embedding vectors for each token in the sequence; embedding_size is 320, 480, 640, or 1280 depending on model size

        - **contacts** (*array of arrays of floats*, optional, shape: [sequence_length, sequence_length], range: 0.0–1.0) — Predicted inter-residue contact probabilities; symmetric matrix with values indicating probability of contact between residue pairs

        - **attentions** (*array of arrays of floats*, optional, shape: [sequence_length, sequence_length], range: 0.0–1.0) — Self-attention weights from the final layer; symmetric matrix indicating attention between residue pairs

        - **logits** (*array of arrays of floats*, optional, shape: [sequence_length, vocab_size]) — Predicted logits for each token position; vocab_size is 33 (20 standard amino acids, plus special tokens)

        - **vocab_tokens** (*array of strings*, optional, length: 33) — Vocabulary tokens corresponding to logits indices; included only when logits are requested

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [
              {
                "layer": 5,
                "embedding": [
                  -0.7343788743019104,
                  -0.5336151123046875,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embedding": [
                  0.16363729536533356,
                  -0.18132680654525757,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 5,
                "embeddings": [
                  [
                    -0.1730840802192688,
                    0.5365392565727234,
                    "... (truncated for documentation)"
                  ],
                  [
                    -1.2964400053024292,
                    -0.04409468173980713,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embeddings": [
                  [
                    0.3278985023498535,
                    0.3302983343601227,
                    "... (truncated for documentation)"
                  ],
                  [
                    0.037568263709545135,
                    -0.08198162913322449,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              {
                "layer": 5,
                "embedding": [
                  -0.24226808547973633,
                  -0.8612385988235474,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embedding": [
                  0.10104967653751373,
                  -0.20004889369010925,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 5,
                "embeddings": [
                  [
                    0.7492956519126892,
                    -2.3160605430603027,
                    "... (truncated for documentation)"
                  ],
                  [
                    -0.3973373472690582,
                    -0.8978397250175476,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embeddings": [
                  [
                    0.43441492319107056,
                    -0.1285461038351059,
                    "... (truncated for documentation)"
                  ],
                  [
                    0.13970370590686798,
                    -0.30551478266716003,
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

- ESM-2 8M is deployed on CPU-only instances (no GPU requirement), with lower memory and compute needs than BioLM's larger ESM-2 variants (35M, 150M, 650M), making it the lightest-weight option for large-scale or latency-sensitive workloads.
- Inference throughput is substantially higher than larger ESM-2 models on the same hardware: for typical protein sequence lengths, ESM-2 8M runs approximately 5–10x faster than the 150M variant, and even more compared to 650M, enabling rapid embedding and masked-prediction sweeps.
- Predictive quality is reduced relative to larger ESM-2 models, as expected for its smaller parameter count: validation perplexity on UniRef50 is 10.33 (8M) vs. 7.75 (150M) and 6.95 (650M), and unsupervised long-range contact precision at L is 0.17 (8M) vs. 0.44 (150M) and 0.52 (650M).
- Compared with structure-prediction APIs such as ESMFold or AlphaFold2, ESM-2 8M is orders of magnitude faster and computationally cheaper because it only computes sequence-level embeddings, attentions, contacts, and logits, without running a 3D structure module or MSA search, making it suitable when high throughput is more important than atomic-level accuracy.

Applications
------------

- Rapid screening of protein variants using single-sequence structural features from embeddings and contact predictions, enabling protein engineering teams to prioritize thousands of candidates for downstream high-accuracy folding or wet-lab assays without running MSAs; accuracy is generally lower than AlphaFold/ESMFold, especially for long or evolutionarily isolated proteins, but throughput is much higher.
- Early-stage triage of designed protein libraries by extracting per-residue embeddings and contact maps to flag grossly misfolded or unstable designs before expensive simulation or experimental work; useful in high-throughput design cycles for enzymes, binding scaffolds, and synthetic domains, though fine-grained stability or activity differences still require specialized models or experiments.
- Rapid structural annotation and clustering of large metagenomic or proprietary protein collections by using mean embeddings and approximate contact patterns as structure-informed fingerprints, allowing researchers to group sequences by fold-like similarity, identify outliers, and select diverse subsets for detailed characterization; performance is best for sequences within the training distribution and may degrade for very long or unusual proteins.
- Detection of potential structural homologs beyond simple sequence identity by comparing ESM-2 embeddings between proteins, helping teams infer likely folds or functional classes when BLAST-style searches fail; any functional hypotheses drawn from embedding or contact similarity should be validated experimentally and, where possible, cross-checked with higher-accuracy structure predictors.

Limitations
-----------

- **Maximum Sequence Length**: Input sequences are limited to ``2048`` amino acids per item (``ESM2EncodeRequestItem.sequence`` and ``ESM2PredictRequestItem.sequence``); longer proteins must be truncated or split before submission.
- **Batch Size**: API requests accept at most ``8`` sequences per call via ``items`` (``batch_size = 8``). Larger datasets must be processed in multiple requests.
- **Model Variant Accuracy**: The ``8m`` ESM-2 model is the smallest parameter variant and provides lower-quality embeddings and contact maps than larger ESM-2 models (e.g., ``150m``, ``650m``). For applications that are highly sensitive to embedding or contact-map quality, consider larger ESM-2 sizes.
- **Single-Sequence Only**: Both ``encoder`` and ``predictor`` endpoints operate on single sequences without MSAs or structural templates. For tasks where evolutionary couplings from MSAs are critical (e.g., high-accuracy structure prediction), MSA-based tools such as AlphaFold2 or RosettaFold generally perform better.
- **Masked Prediction Scope**: The ``predictor`` endpoint only supports masked language modeling over tokens in ``ESM2PredictRequestItem.sequence`` containing ``<mask>``. It does not perform full sequence design, scoring of unmasked variants, or 3D structure prediction; use downstream models or pipelines for those tasks.

How We Use It
-------------

ESM-2 8M enables rapid, scalable extraction of protein sequence embeddings and contact maps, which we use as standardized inputs to downstream models for structure-aware protein engineering. Its fast single-sequence inference allows us to score and prioritize large variant libraries, guide mutagenesis, and couple embeddings with task-specific predictors of stability, developability, or function, accelerating design–build–test cycles without dependence on MSAs or external evolutionary databases.

- Enables efficient first-pass triage of protein variants before more expensive structure prediction or lab screening
- Integrates with downstream predictors (e.g., stability, aggregation, binding) that operate on ESM-2 embeddings or contact features for end-to-end optimization

Related
-------

- ``AlphaFold2`` – High-accuracy structure prediction model for benchmarking and validating structure-related signals inferred from ESM-2 8M embeddings and contacts.
- ``ESMFold`` – Uses larger ESM-2 models to predict 3D protein structures directly from sequence; pairs naturally with ESM-2 8M when you need fast embeddings plus explicit structural models.
- ``ESM-IF1`` – Inverse folding model that designs sequences for given backbone structures; complements ESM-2 8M embeddings in protein design and sequence–structure analysis workflows.
- ``ESM-2 150M`` – Higher-capacity ESM-2 variant with improved embedding quality and contact predictions, useful when ESM-2 8M accuracy is insufficient and additional GPU resources are available.

References
----------

- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*, 379(6637), 1123–1130.
