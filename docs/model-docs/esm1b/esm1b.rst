ESM1b API
=========

ESM-1b is a 33-layer, ~650M-parameter Transformer protein language model trained on 250M UniRef50 protein sequences (86B amino acids) with a masked language modeling objective. The API provides GPU-accelerated encoder and predictor endpoints that return per-sequence and per-residue embeddings, attention maps, logits, and masked-token predictions for batches of up to 8 proteins (length ≤1,022). These representations support tasks such as remote homology search, structure/contacts feature extraction, and mutational effect modeling.

Predict
-------

Predict masked amino acids in protein sequences using ESM-1b

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm1b",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTAYIAK<mask>QISFVKSHFS"
                  },
                  {
                    "sequence": "GAVLIPF<mask>YCMNQSTDE<mask>HK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm1b/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTAYIAK<mask>QISFVKSHFS"
                },
                {
                  "sequence": "GAVLIPF<mask>YCMNQSTDE<mask>HK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm1b/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTAYIAK<mask>QISFVKSHFS"
                    },
                    {
                      "sequence": "GAVLIPF<mask>YCMNQSTDE<mask>HK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm1b/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTAYIAK<mask>QISFVKSHFS"
                ),
                list(
                  sequence = "GAVLIPF<mask>YCMNQSTDE<mask>HK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm1b/predict/

   Predict endpoint for ESM1b.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **items** (*array of objects*, min: 1, max: 8, required) --- Masked input sequences:

        - **sequence** (*string*, min length: 1, max length: 1022, required) — Amino acid sequence validated against AAExtendedPlusExtra with "<mask>" required and allowed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm1b/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTAYIAK<mask>QISFVKSHFS"
          },
          {
            "sequence": "GAVLIPF<mask>YCMNQSTDE<mask>HK"
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

        - **logits** (*array of arrays of floats*, shape: [L, V]) — Unnormalized scores for each vocabulary token at each token position, where L is the tokenized sequence length and V is the vocabulary size

        - **sequence_tokens** (*array of strings*, length: L) — Tokenized input sequence, including special tokens such as "<mask>"

        - **vocab_tokens** (*array of strings*, length: V) — Output vocabulary tokens corresponding to the second dimension of ``logits``

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "logits": [
              [
                0.3815561830997467,
                -1.3087595701217651,
                "... (truncated for documentation)"
              ],
              [
                -0.018967360258102417,
                -0.6312656998634338,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "M",
              "K",
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
                -0.4918019771575928,
                -1.2386813163757324,
                "... (truncated for documentation)"
              ],
              [
                -0.31671813130378723,
                2.442998170852661,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "G",
              "A",
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

Encode protein sequences with ESM-1b, requesting multiple representation layers and embedding types

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm1b",
                action="encode",
                params={
                  "repr_layers": [
                    -1,
                    6,
                    12
                  ],
                  "include": [
                    "mean",
                    "per_token",
                    "bos",
                    "logits"
                  ]
                },
                items=[
                  {
                    "sequence": "MKTAYIAKQRQISFVKSHFS"
                  },
                  {
                    "sequence": "GAVLIPFWYCMNQSTDERHK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm1b/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "repr_layers": [
                  -1,
                  6,
                  12
                ],
                "include": [
                  "mean",
                  "per_token",
                  "bos",
                  "logits"
                ]
              },
              "items": [
                {
                  "sequence": "MKTAYIAKQRQISFVKSHFS"
                },
                {
                  "sequence": "GAVLIPFWYCMNQSTDERHK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm1b/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "repr_layers": [
                      -1,
                      6,
                      12
                    ],
                    "include": [
                      "mean",
                      "per_token",
                      "bos",
                      "logits"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MKTAYIAKQRQISFVKSHFS"
                    },
                    {
                      "sequence": "GAVLIPFWYCMNQSTDERHK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm1b/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                repr_layers = list(
                  -1,
                  6,
                  12
                ),
                include = list(
                  "mean",
                  "per_token",
                  "bos",
                  "logits"
                )
              ),
              items = list(
                list(
                  sequence = "MKTAYIAKQRQISFVKSHFS"
                ),
                list(
                  sequence = "GAVLIPFWYCMNQSTDERHK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm1b/encode/

   Encode endpoint for ESM1b.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers to return representations for

        - **include** (*array of strings*, default: ["mean"]) — Representation types to return; allowed values: "mean", "per_token", "bos", "logits", "attentions"


      - **items** (*array of objects*, min: 1, max: 8, required) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 1022, required) — Amino acid sequence using the extended alphabet plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm1b/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "repr_layers": [
            -1,
            6,
            12
          ],
          "include": [
            "mean",
            "per_token",
            "bos",
            "logits"
          ]
        },
        "items": [
          {
            "sequence": "MKTAYIAKQRQISFVKSHFS"
          },
          {
            "sequence": "GAVLIPFWYCMNQSTDERHK"
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

        - **sequence_index** (*int*) — Zero-based index of the input sequence in the request ``items`` array

        - **embeddings** (*array of objects*, optional) — Mean sequence-level embeddings for requested layers

          - **layer** (*int*) — Model layer index associated with the embedding

          - **embedding** (*array of floats*, length: 1280) — Layer-specific sequence embedding vector

        - **bos_embeddings** (*array of objects*, optional) — Beginning-of-sequence token embeddings for requested layers

          - **layer** (*int*) — Model layer index associated with the BOS embedding

          - **embedding** (*array of floats*, length: 1280) — Layer-specific BOS token embedding vector

        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings for requested layers

          - **layer** (*int*) — Model layer index associated with the per-token embeddings

          - **embeddings** (*array of arrays of floats*, shape: ``[L, 1280]``) — Layer-specific embeddings per token, where ``L`` is the tokenized sequence length including special tokens

        - **attentions** (*array of arrays of floats*, optional) — Self-attention weights, flattened across layers, heads, and token positions

        - **logits** (*array of arrays of floats*, optional) — Per-token output scores over the amino acid vocabulary, shape ``[L, V]`` where ``V`` is the size of ``vocab_tokens``

        - **vocab_tokens** (*array of strings*, optional) — Vocabulary entries corresponding to the last dimension of ``logits``

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
                "layer": 33,
                "embedding": [
                  0.06035921722650528,
                  0.10358679294586182,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embedding": [
                  -0.2090596705675125,
                  0.08616238087415695,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  0.8181778192520142,
                  -0.9239708781242371,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "bos_embeddings": [
              {
                "layer": 33,
                "embedding": [
                  -0.17265623807907104,
                  0.11798319220542908,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embedding": [
                  -3.965080976486206,
                  0.9067386984825134,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -3.090304136276245,
                  -1.2702486515045166,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 33,
                "embeddings": [
                  [
                    0.0877808928489685,
                    0.057009898126125336,
                    "... (truncated for documentation)"
                  ],
                  [
                    0.1496940702199936,
                    -0.0737527385354042,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embeddings": [
                  [
                    4.847900390625,
                    -0.2435275763273239,
                    "... (truncated for documentation)"
                  ],
                  [
                    1.8292689323425293,
                    -0.4840039610862732,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embeddings": [
                  [
                    4.977882385253906,
                    -0.7182919383049011,
                    "... (truncated for documentation)"
                  ],
                  [
                    2.235905170440674,
                    -0.13048285245895386,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ],
            "logits": [
              [
                0.2741698622703552,
                -1.2439606189727783,
                "... (truncated for documentation)"
              ],
              [
                -0.07653619349002838,
                -0.5435468554496765,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "vocab_tokens": [
              "L",
              "A",
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              {
                "layer": 33,
                "embedding": [
                  0.08701622486114502,
                  0.07660450041294098,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embedding": [
                  0.05603637918829918,
                  0.5261194109916687,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -1.042244791984558,
                  -0.7624322175979614,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "bos_embeddings": [
              {
                "layer": 33,
                "embedding": [
                  -0.12817233800888062,
                  0.17094434797763824,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embedding": [
                  -5.523139476776123,
                  0.6325452327728271,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -3.0614817142486572,
                  0.11153705418109894,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 33,
                "embeddings": [
                  [
                    0.1595464050769806,
                    0.10200859606266022,
                    "... (truncated for documentation)"
                  ],
                  [
                    0.03582605719566345,
                    -0.08742739260196686,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 6,
                "embeddings": [
                  [
                    0.07901507616043091,
                    2.2436790466308594,
                    "... (truncated for documentation)"
                  ],
                  [
                    2.0379934310913086,
                    -0.20363545417785645,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embeddings": [
                  [
                    -0.7994107604026794,
                    0.6554591655731201,
                    "... (truncated for documentation)"
                  ],
                  [
                    0.9054992198944092,
                    -0.4249749481678009,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ],
            "logits": [
              [
                -0.49218833446502686,
                -1.1896588802337646,
                "... (truncated for documentation)"
              ],
              [
                -0.32505860924720764,
                2.507049322128296,
                "... (truncated for documentation)"
              ],
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


Performance
-----------

- Model architecture and training:
  - 33-layer bidirectional Transformer language model (~650M parameters) trained on UR50/S (high-diversity UniRef50) with a masked language modeling objective
  - Encodes global evolutionary statistics across families rather than relying on per-target MSAs, enabling single-sequence usage at scale
  - Representations contain linearly decodable information about secondary structure, long-range contacts, and remote homology (as shown in the ESM-1b paper benchmarks)
- Comparative representation quality vs. other BioLM encoders:
  - Versus smaller ESM-2 encoders (8M/35M/150M), ESM-1b generally provides stronger embeddings for remote homology detection, secondary structure probes, and long-range contact probes, particularly when paired with simple downstream heads
  - Versus ESM-2 650M, ESM-1b has slightly weaker structure-aware representations on very large, diverse datasets but remains a practical default where historical benchmarks, tooling, and stability are preferred over marginal accuracy gains
- Use as a feature source vs. structure-prediction models:
  - ESM-1b is not a 3D structure predictor; it exposes sequence-level and token-level representations and logits suited for downstream models
  - When paired with lightweight supervised heads, ESM-1b features can match or exceed alignment-based baselines (e.g., CCMpred-style contacts, sequence-profile-based secondary structure), but will not reach end-to-end 3D accuracy of models such as AlphaFold2 or ESMFold
- Masked prediction and mutation scoring performance:
  - Within BioLM’s catalog, ESM-1b offers high-quality amino-acid distributions at masked positions, making it preferable to lightweight encoders for mutation-impact ranking and masked-design loops
  - Newer Evo/E1-series generative models can achieve similar or better ranking quality at comparable compute cost, but ESM-1b remains better characterized and more widely benchmarked for variant-effect and mutational scanning tasks

Applications
------------

- Protein fitness and mutational scanning surrogates for engineering campaigns: use ESM1b log-likelihoods or masked-token predictions via the ``predictor`` endpoint to prioritize single and combinatorial mutations in directed evolution or library design, reducing wet-lab screening when optimizing stability, activity, or manufacturability; most informative when the target function is well represented in natural sequence space and less so for highly novel functions
- Structure- and contact-aware feature generation for downstream ML models: use the ``encoder`` endpoint to extract mean, per-token, BOS, and attention-derived embeddings from selected layers as input features to custom supervised models (e.g., stability predictors, aggregation-risk classifiers, interface-scoring models) that require structural signal without running full 3D prediction on every variant, enabling large-scale in silico screening of thousands to millions of protein designs
- Remote homology and scaffolding search in large sequence libraries: embed proprietary or public protein sequence collections with ESM1b and run nearest-neighbor or clustering methods on the resulting vectors to detect remote structural relatives and alternative scaffolds that simple sequence identity thresholds miss, supporting reuse of known assays and expression systems; less suitable when precise domain boundaries or high-quality alignments are the primary requirement
- MSA-free pre-filtering for protein design and variant triage: apply ESM1b log-probabilities or embeddings to very large mutational or generative libraries where building robust MSAs is infeasible (for example, highly diverse synthetic libraries or metagenomic-like spaces), quickly discarding sequences that are strongly out-of-distribution with respect to evolutionary statistics and focusing experimental campaigns on candidates more likely to be foldable and functional
- Integration into automated protein engineering pipelines: incorporate ESM1b sequence embeddings and log-likelihoods as one stage in multi-model ranking stacks (alongside stability, solubility, and developability predictors) to orchestrate design–test–learn loops for industrial enzymes and other production proteins; ESM1b provides a general-purpose sequence representation capturing secondary and tertiary trends but does not replace task-specific models or experimental validation

Limitations
-----------

- **Maximum sequence length**: Input ``sequence`` strings are limited to ``ESM1bParams.max_sequence_len`` = ``1022`` amino acids (excluding BOS/EOS). Longer proteins must be truncated, split into overlapping windows, or summarized with domain-level sequences. This constraint applies to all request types (``ESM1bEncodeRequest``, ``ESM1bPredictRequest``, ``ESM1bLogProbRequest``).
- **Batch size and throughput**: Each request can include at most ``ESM1bParams.batch_size`` = ``8`` items in ``items``. Large sequence collections must be sharded across multiple API calls. The model is relatively large (~650M parameters), so high-volume or latency-sensitive workloads may require additional engineering (asynchronous batching, caching, pre-computed embeddings).
- **Representation-only model (no 3D structure)**: ESM-1b provides sequence-level encodings and token-level scores via ``encoder`` (``repr_layers`` and ``include`` options such as ``mean``, ``per_token``, ``bos``, ``logits``, ``attentions``), but does not perform explicit 3D structure prediction. It encodes information correlated with secondary and tertiary structure, but for detailed structure (backbone coordinates, side-chain conformations) and stability ranking of a small candidate set, structure-specific models (e.g. AlphaFold2, ESMFold, antibody-specific structure models) are usually more appropriate.
- **Masked language modeling, not full generative design**: The ``predictor`` endpoint (``ESM1bPredictRequest``) requires at least one ``<mask>`` token in each ``sequence`` (enforced by ``SingleOrMoreOccurrencesOf(token="<mask>")``) and returns per-token logits over the vocabulary. It is optimized for scoring substitutions and local infills around ``<mask>``, not for unconditional de novo generation or long autoregressive design. For large-scale sequence generation, CausalLM-style models (e.g. ProGen2, ProtGPT2) or diffusion-based backbone generators are often better suited.
- **Embeddings are generic, not task-specific**: Encodings returned via ``encoder`` (``embeddings``, ``per_token_embeddings``, ``bos_embeddings``, ``attentions``, optional ``logits`` with ``vocab_tokens``) capture broad evolutionary and structural signals learned from UR50/S, but they are not specialized for any one downstream task. For high-precision applications (e.g. quantitative activity prediction, developability screening, antibody optimization), these representations typically need to be combined with domain-specific models, MSAs, structural features, or fine-tuning.
- **Scope of biological generalization**: ESM-1b is trained on natural protein sequences using a 20–amino-acid–centric vocabulary plus a small set of extra tokens (validated by ``AAExtendedPlusExtra`` in ``sequence``). It may perform poorly on highly non-natural sequences (e.g. heavily synthetic repeats, non-standard amino acids encoded as arbitrary symbols, extremely long low-complexity regions) and does not model DNA/RNA. For nucleic acid–focused applications or very unusual protein-like polymers, dedicated DNA/RNA models or custom training are typically required.

How We Use It
-------------

ESM1b underpins many of our protein design and optimization workflows by providing rich single-sequence embeddings and structure-aware features that plug into standardized APIs for screening, ranking, and iterative optimization. Teams use these embeddings as a shared representation across enzyme engineering, antibody maturation, and developability assessment, combining them with supervised models (e.g., stability, activity, immunogenicity), structure-derived metrics from separate structure tools, and assay readouts to prioritize variants for synthesis and multi-round improvement.

- ESM1b embeddings integrate with downstream property predictors (stability, expression, binding, aggregation) to enable high-throughput in silico triage and reduce experimental burden.
- In antibody and enzyme campaigns, ESM1b-based features are combined with structural models, physicochemical summaries (charge, size, hydrophobicity), and assay results to drive multi-round design–test–learn cycles through scalable, API-driven workflows.

Related
-------

- ``ESM-2 650M`` – Next-generation ESM model; use when you want ESM-1b–style single-sequence embeddings and masked-token predictions with improved accuracy and scaling, while keeping similar API patterns and downstream use cases.
- ``MSA Transformer`` – ESM family model that conditions on multiple sequence alignments; complements ESM-1b by capturing explicit evolutionary couplings when high-quality MSAs are available.
- ``ESMFold`` – Structure prediction model built on ESM representations; combine ESM-1b sequence embeddings for tasks like mutation effect modeling or annotation with ESMFold when you also need 3D structure.
- ``ESM-1v`` – Variant-effect–oriented ESM model; use alongside ESM-1b when you need general-purpose embeddings or masked-token outputs (ESM-1b) plus zero-shot mutation effect scores (ESM-1v) on the same proteins.

References
----------

- Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J., & Fergus, R. (2021). `Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences <https://doi.org/10.1073/pnas.2016239118>`_. *Proceedings of the National Academy of Sciences of the United States of America*, 118(15), e2016239118.
