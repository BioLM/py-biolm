ESM-2 3B API
============

ESM-2 3B is a transformer protein language model trained with masked language modeling on UniRef-derived evolutionary-scale data, providing rich sequence representations and zero-shot masked amino acid predictions. This API exposes 8M–650M parameter variants and supports up to 8 protein sequences per call, each up to 2,048 residues. Encoder and predictor endpoints return configurable mean, per-token, or BOS embeddings, attention-derived contact maps, logits, and attention weights for applications such as feature extraction, mutation scoring, and sequence-based protein engineering.

Predict
-------

Predict the masked amino acids in the input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2-3b",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MT<mask>KLAA"
                  },
                  {
                    "sequence": "GGSAA<mask>TTF"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-3b/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MT<mask>KLAA"
                },
                {
                  "sequence": "GGSAA<mask>TTF"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-3b/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MT<mask>KLAA"
                    },
                    {
                      "sequence": "GGSAA<mask>TTF"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-3b/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MT<mask>KLAA"
                ),
                list(
                  sequence = "GGSAA<mask>TTF"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2-3b/predict/

   Predict endpoint for ESM-2 3B.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers to include in the encoded representations

        - **include** (*array of strings*, default: ["mean"]) — Encoded outputs to return for each input sequence


          Allowed values:

          - mean
          - per_token
          - bos
          - contacts
          - logits
          - attentions


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid alphabet, allowing "-" characters

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-3b/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MT<mask>KLAA"
          },
          {
            "sequence": "GGSAA<mask>TTF"
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

        - **sequence_index** (*int*) — Index of the input sequence in the request ``items`` array

        - **embeddings** (*array of objects*, optional) — Mean embeddings for each requested layer

          - **layer** (*int*) — Layer index requested in ``repr_layers``

          - **embedding** (*array of floats*) — Mean embedding vector for the sequence, size: ``embedding_dim``

        - **bos_embeddings** (*array of objects*, optional) — Beginning-of-sequence embeddings for each requested layer

          - **layer** (*int*) — Layer index requested in ``repr_layers``

          - **embedding** (*array of floats*) — Embedding vector for the beginning-of-sequence token, size: ``embedding_dim``

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings for each requested layer

          - **layer** (*int*) — Layer index requested in ``repr_layers``

          - **embeddings** (*array of arrays of floats*) — Embedding vectors per token position, shape: ``[sequence_length, embedding_dim]``

        - **contacts** (*array of arrays of floats*, optional) — Pairwise contact scores, shape: ``[sequence_length, sequence_length]``

        - **attentions** (*array of arrays of floats*, optional) — Aggregated self-attention values, shape: ``[sequence_length, sequence_length]``

        - **logits** (*array of arrays of floats*, optional) — Per-position logits over vocabulary tokens, shape: ``[sequence_length, vocab_size]``

        - **vocab_tokens** (*array of strings*, optional) — Vocabulary tokens corresponding to indices in ``logits``, size: ``vocab_size``


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **logits** (*array of arrays of floats*) — Per-position logits over vocabulary tokens for masked-position prediction, shape: ``[sequence_length, vocab_size]``

        - **sequence_tokens** (*array of strings*) — Tokenized input sequence including ``"<mask>"`` tokens, size: ``sequence_length``

        - **vocab_tokens** (*array of strings*) — Vocabulary tokens corresponding to indices in ``logits``, size: ``vocab_size``

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "logits": [
              [
                -0.8711350560188293,
                -0.008733198046684265,
                "... (truncated for documentation)"
              ],
              [
                0.1018385961651802,
                -0.10829788446426392,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "M",
              "T",
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
                -0.9810609817504883,
                -0.4438960552215576,
                "... (truncated for documentation)"
              ],
              [
                -0.26015204191207886,
                -0.05383633077144623,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "G",
              "G",
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

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2-3b",
                action="encode",
                params={
                  "repr_layers": [
                    -1
                  ],
                  "include": [
                    "mean",
                    "bos",
                    "contacts"
                  ]
                },
                items=[
                  {
                    "sequence": "MTEYKLVVVG"
                  },
                  {
                    "sequence": "GGSAAGTTFVN"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-3b/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "repr_layers": [
                  -1
                ],
                "include": [
                  "mean",
                  "bos",
                  "contacts"
                ]
              },
              "items": [
                {
                  "sequence": "MTEYKLVVVG"
                },
                {
                  "sequence": "GGSAAGTTFVN"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-3b/encode/"
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
                      "mean",
                      "bos",
                      "contacts"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MTEYKLVVVG"
                    },
                    {
                      "sequence": "GGSAAGTTFVN"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-3b/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                repr_layers = list(
                  -1
                ),
                include = list(
                  "mean",
                  "bos",
                  "contacts"
                )
              ),
              items = list(
                list(
                  sequence = "MTEYKLVVVG"
                ),
                list(
                  sequence = "GGSAAGTTFVN"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2-3b/encode/

   Encode endpoint for ESM-2 3B.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers for which to return representations

        - **include** (*array of strings*, default: ["mean"]) — Output types to compute for each item; allowed values: "mean", "per_token", "bos", "contacts", "logits", "attentions"


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid codes plus "-" character

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-3b/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "repr_layers": [
            -1
          ],
          "include": [
            "mean",
            "bos",
            "contacts"
          ]
        },
        "items": [
          {
            "sequence": "MTEYKLVVVG"
          },
          {
            "sequence": "GGSAAGTTFVN"
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

        - **sequence_index** (*int*) — Index of the input sequence in the request ``items`` list

        - **embeddings** (*array of objects*, optional) — Mean embeddings per requested layer, included when ``"mean"`` is in ``params.include``:

          - **layer** (*int*) — Layer index used for this embedding, corresponding to a value in ``params.repr_layers``

          - **embedding** (*array of floats*) — Mean embedding vector over tokens for this layer, length: ``embedding_dim`` for the selected ESM-2 variant (e.g., 320 for 8M, 480 for 35M, 640 for 150M, 1280 for 650M)

        - **bos_embeddings** (*array of objects*, optional) — Beginning-of-sequence embeddings per requested layer, included when ``"bos"`` is in ``params.include``:

          - **layer** (*int*) — Layer index used for this embedding, corresponding to a value in ``params.repr_layers``

          - **embedding** (*array of floats*) — Embedding vector for the beginning-of-sequence token at this layer, length: ``embedding_dim`` (same as ``embeddings.embedding``)

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer, included when ``"per_token"`` is in ``params.include``:

          - **layer** (*int*) — Layer index used for these embeddings, corresponding to a value in ``params.repr_layers``

          - **embeddings** (*array of arrays of floats*) — Embedding vectors per token position at this layer, shape: ``[sequence_length, embedding_dim]``

        - **contacts** (*array of arrays of floats*, optional) — Predicted inter-residue contact scores, included when ``"contacts"`` is in ``params.include``, shape: ``[sequence_length, sequence_length]``, range: 0.0–1.0

        - **attentions** (*array of arrays of floats*, optional) — Aggregated self-attention scores between token positions, included when ``"attentions"`` is in ``params.include``, shape: ``[sequence_length, sequence_length]``, range: 0.0–1.0

        - **logits** (*array of arrays of floats*, optional) — Predicted per-position logits over the model vocabulary, included when ``"logits"`` is in ``params.include``, shape: ``[sequence_length, vocab_size]``

        - **vocab_tokens** (*array of strings*, optional) — Vocabulary tokens corresponding to the last dimension of ``logits``, length: ``vocab_size``

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              {
                "layer": 36,
                "embedding": [
                  0.02133369818329811,
                  -0.05615193769335747,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "bos_embeddings": [
              {
                "layer": 36,
                "embedding": [
                  0.046763572841882706,
                  -0.029284246265888214,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "contacts": [
              [
                2.672532446013065e-06,
                0.003567361505702138,
                "... (truncated for documentation)"
              ],
              [
                0.003567360108718276,
                2.9021906811976805e-05,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "embeddings": [
              {
                "layer": 36,
                "embedding": [
                  -0.0010191182373091578,
                  -0.03960900008678436,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "bos_embeddings": [
              {
                "layer": 36,
                "embedding": [
                  -0.005338006187230349,
                  -0.04863175004720688,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "contacts": [
              [
                5.831161615788005e-05,
                0.00019262765999883413,
                "... (truncated for documentation)"
              ],
              [
                0.0001926274853758514,
                0.0006442570011131465,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- ESM-2 3B is served on NVIDIA V100/A100‑class GPUs with mixed‑precision inference and kernel‑level optimizations, enabling efficient large‑batch masked‑token prediction and embedding generation.
- A single forward pass produces all encoder‑side outputs exposed by the API (embeddings, logits, attentions, contacts); requesting multiple outputs in one call has only a modest latency increase relative to requesting a single output.
- Compared to smaller ESM‑2 variants (8M, 35M, 150M, 650M), the 3B model yields stronger structural and functional signal: unsupervised long‑range contact precision and structure‑module TM‑scores are higher (e.g., CASP14 TM ≈0.55 vs. ≈0.49 for 150M and ≈0.51 for 650M), improving zero‑shot mutational effect prediction and downstream modeling quality when used as an encoder.
- The 3B model’s 2560‑dimensional hidden representations (projected in BioLM to 1280‑dimensional residue embeddings for typical workflows) offer a favorable accuracy‑per‑compute trade‑off relative to 15B‑class models, approaching their representation quality while remaining substantially cheaper to deploy at scale.

Applications
------------

- Sequence embeddings for downstream protein property models, using ESM-2 3B encoder representations (``mean``, ``bos``, or ``per_token`` via the ``encoder`` endpoint) as inputs to custom predictors for stability, solubility, expression, or other developability metrics; useful for biotech teams building ML-driven protein engineering pipelines without training large language models from scratch.
- Contact-style and distance-pattern extraction from the ``contacts`` and ``attentions`` outputs of the ``encoder`` endpoint, enabling rapid assessment of likely tertiary interaction patterns from single sequences; helpful for guiding scaffold selection or deprioritizing designs with inconsistent long-range patterns, but not a replacement for full 3D structure prediction or MD simulations.
- Zero-shot mutation scoring by masking one or more residues and calling the ``predictor`` endpoint to obtain logits over amino acids at masked positions, allowing estimation of relative likelihoods of substitutions; supports prioritization of variants in directed evolution or saturation mutagenesis, though it does not provide explicit ΔΔG, activity, or developability values without supervised calibration.
- Sequence-based filtering and ranking of large in silico protein libraries, using perplexity-like scores derived from ``predictor`` logits to down-select designs that better match natural sequence statistics; useful for triaging de novo scaffolds or metagenomic hits before experimental screening, but should be combined with additional biophysical or functional filters.
- Structural motif localization from per-residue embeddings returned by the ``encoder`` endpoint (``per_token`` option), where lightweight downstream models or clustering identify helices, loops, active-site neighborhoods, and candidate interface regions; supports design of fusion proteins, binders, or biosensors that rely on stable secondary-structure elements, but is less reliable for highly disordered or context-dependent motifs without experimental follow-up.

Limitations
-----------

- **Maximum Sequence Length**: Input protein sequences must not exceed ``2048`` amino acids (``max_sequence_len``). Longer sequences must be truncated or split client-side before calling the ``encoder`` or ``predictor`` endpoints.
- **Batch Size**: The ``items`` list in both ``encoder`` and ``predictor`` requests accepts at most ``8`` sequences per call (``batch_size``). Larger datasets must be processed over multiple requests.
- **Training Data Bias**: ESM-2 was trained on natural protein sequences from UniRef. Performance may degrade on highly synthetic, chimeric, de novo, or heavily engineered proteins; outputs in these regimes should be treated as heuristic and validated experimentally.
- **No Explicit 3D Structures**: This API exposes sequence-based outputs only. It does not run ESMFold or any other 3D structure module. Structural information is indirect, via ``contacts`` (pairwise inter-residue scores), ``attentions`` (self-attention weights), and latent embeddings returned by ``encoder``.
- **Embeddings and Auxiliary Outputs**: The ``include`` field in ``encoder`` controls which outputs are computed and returned. ``mean`` enables sequence-level ``embeddings`` (one or more ``LayerEmbedding`` vectors per sequence), ``per_token`` enables ``per_token_embeddings`` (``LayerPerTokenEmbeddings`` over residues), ``bos`` enables ``bos_embeddings`` (beginning-of-sequence vectors), and ``contacts``, ``logits``, ``attentions`` enable their corresponding fields. Options omitted from ``include`` are neither computed nor returned, which reduces runtime and response size.
- **Masked Prediction Only**: The ``predictor`` endpoint implements masked-language modeling with the ``<mask>`` token only. Each ``sequence`` must contain one or more ``<mask>`` positions; the endpoint does not support autoregressive generation, full-sequence scoring, or free-form design, and returns only position-wise ``logits`` with ``sequence_tokens`` and ``vocab_tokens`` for interpretation.

How We Use It
-------------

BioLM uses ESM-2 3B as a backbone for protein sequence representation and generative design, integrating embeddings, unsupervised contact maps, and masked-token predictions into larger model stacks for design–build–test–learn campaigns. Standardized sequence encodings from the encoder API support downstream structure prediction (for example with ESMFold or AlphaFold) and biophysical property models, while the predictor API enables masked-language-model style fixed-backbone sequence optimization and local variant exploration.

- Enables fixed-backbone optimization and localized de novo sequence generation via masked-token prediction and embedding-based scoring
- Scales to multi-round engineering campaigns by providing consistent, reusable encodings that integrate with structure-derived and physicochemical property models

Related
-------

- ``ESMFold`` – End-to-end single-sequence 3D structure prediction based on ESM-2; useful for validating or structurally interpreting sequences embedded with ``ESM-2 3B``.
- ``ESM-IF1`` – Inverse folding model that designs sequences for a given backbone structure; complements ``ESM-2 3B`` for structure-constrained sequence design.
- ``AlphaFold2`` – High-accuracy structure prediction model useful for cross-checking or further analyzing sequences first screened or embedded with ``ESM-2 3B``.
- ``ESM-2 150M`` – Smaller, faster ESM-2 variant suitable for rapid embedding generation and prototyping before scaling analyses to ``ESM-2 3B``.

References
----------

- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science, 379*(6637), 1123–1130. Earlier versions are available as the bioRxiv preprint 2022.07.20.500902.
