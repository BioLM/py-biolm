ESM-2 35M API
=============

ESM-2 35M is a 12-layer, 35M-parameter protein language model trained on UniRef50/UniRef90 sequences with a masked language modeling objective. The API exposes two actions: encoder, which returns sequence-level and per-residue embeddings, BOS embeddings, attention-based contact maps, logits, and attentions for up to 8 protein sequences of length ≤2048; and predictor, which performs masked-token prediction on single or multiple <mask> sites. Typical uses include representation learning, mutation scoring via logit differences, and contact-map extraction for structure-aware protein design workflows.

Predict
-------

Masked language modeling with ESM2-35M: predict the amino acid at one masked site in each sequence

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2-35m",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTFFV<mask>LVLLLSGALAAPVA"
                  },
                  {
                    "sequence": "ACDEFGHIKLMNPQRSTVWY<mask>"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-35m/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTFFV<mask>LVLLLSGALAAPVA"
                },
                {
                  "sequence": "ACDEFGHIKLMNPQRSTVWY<mask>"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-35m/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTFFV<mask>LVLLLSGALAAPVA"
                    },
                    {
                      "sequence": "ACDEFGHIKLMNPQRSTVWY<mask>"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-35m/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTFFV<mask>LVLLLSGALAAPVA"
                ),
                list(
                  sequence = "ACDEFGHIKLMNPQRSTVWY<mask>"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2-35m/predict/

   Predict endpoint for ESM-2 35M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Layer indices to include in embeddings

        - **include** (*array of strings*, default: ["mean"]) — Embedding components to return; allowed values: "mean", "per_token", "bos", "contacts", "logits", "attentions"


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for encoding:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard amino acid codes; may include "-" gap characters

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-35m/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTFFV<mask>LVLLLSGALAAPVA"
          },
          {
            "sequence": "ACDEFGHIKLMNPQRSTVWY<mask>"
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

        - **logits** (*array of arrays of floats*, shape: [num_masks, vocab_size]) — Per-mask unnormalized prediction scores over the amino acid vocabulary

        - **sequence_tokens** (*array of strings*, length: input sequence length) — Tokenized input sequence including special tokens such as ``<mask>``

        - **vocab_tokens** (*array of strings*, length: vocab_size) — Amino acid vocabulary tokens corresponding to indices in ``logits``

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "logits": [
              [
                0.20047184824943542,
                -0.4815218448638916,
                "... (truncated for documentation)"
              ],
              [
                -0.7963054180145264,
                0.3801613450050354,
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
                -0.11371655762195587,
                2.547285556793213,
                "... (truncated for documentation)"
              ],
              [
                0.4559939503669739,
                0.11516737937927246,
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
          }
        ]
      }


Encode
------

Compute mean, per-token, BOS embeddings and contact maps from ESM2-35M for two short protein sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2-35m",
                action="encode",
                params={
                  "repr_layers": [
                    -1,
                    6
                  ],
                  "include": [
                    "mean",
                    "per_token",
                    "bos",
                    "contacts"
                  ]
                },
                items=[
                  {
                    "sequence": "MKTFFVLVLLLSGALAAPVA"
                  },
                  {
                    "sequence": "GSSGSSGSSGSSGSSGSSGS"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2-35m/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "repr_layers": [
                  -1,
                  6
                ],
                "include": [
                  "mean",
                  "per_token",
                  "bos",
                  "contacts"
                ]
              },
              "items": [
                {
                  "sequence": "MKTFFVLVLLLSGALAAPVA"
                },
                {
                  "sequence": "GSSGSSGSSGSSGSSGSSGS"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2-35m/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "repr_layers": [
                      -1,
                      6
                    ],
                    "include": [
                      "mean",
                      "per_token",
                      "bos",
                      "contacts"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MKTFFVLVLLLSGALAAPVA"
                    },
                    {
                      "sequence": "GSSGSSGSSGSSGSSGSSGS"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2-35m/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                repr_layers = list(
                  -1,
                  6
                ),
                include = list(
                  "mean",
                  "per_token",
                  "bos",
                  "contacts"
                )
              ),
              items = list(
                list(
                  sequence = "MKTFFVLVLLLSGALAAPVA"
                ),
                list(
                  sequence = "GSSGSSGSSGSSGSSGSSGS"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2-35m/encode/

   Encode endpoint for ESM-2 35M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Types of embeddings or logits to return; allowed values: "mean", "per_token", "logits"

      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard unambiguous amino acid codes; ambiguous amino acids not allowed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-35m/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "repr_layers": [
            -1,
            6
          ],
          "include": [
            "mean",
            "per_token",
            "bos",
            "contacts"
          ]
        },
        "items": [
          {
            "sequence": "MKTFFVLVLLLSGALAAPVA"
          },
          {
            "sequence": "GSSGSSGSSGSSGSSGSSGS"
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

        - **pdb** (*string*) — Predicted protein structure in standard PDB file format.

        - **mean_plddt** (*float*, range: 0.0 - 1.0) — Mean predicted Local Distance Difference Test (pLDDT) confidence score for the predicted structure, indicating prediction accuracy (0.0 = low confidence, 1.0 = high confidence).

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
                "layer": 6,
                "embedding": [
                  -1.252715826034546,
                  -0.41427677869796753,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -0.057384051382541656,
                  -0.3180755078792572,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "bos_embeddings": [
              {
                "layer": 6,
                "embedding": [
                  -2.809511184692383,
                  -3.4495561122894287,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -0.17458851635456085,
                  -0.31701692938804626,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 6,
                "embeddings": [
                  [
                    -2.5074832439422607,
                    -0.6743671894073486,
                    "... (truncated for documentation)"
                  ],
                  [
                    -1.1302917003631592,
                    0.028207749128341675,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embeddings": [
                  [
                    -0.5093696713447571,
                    -0.07652397453784943,
                    "... (truncated for documentation)"
                  ],
                  [
                    0.2325054109096527,
                    -0.15158739686012268,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ],
            "contacts": [
              [
                5.763190458196732e-08,
                0.0001390503894072026,
                "... (truncated for documentation)"
              ],
              [
                0.00013905025843996555,
                0.7919971942901611,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              {
                "layer": 6,
                "embedding": [
                  -0.45312127470970154,
                  -0.6850371360778809,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -0.3599430024623871,
                  -0.07315696775913239,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "bos_embeddings": [
              {
                "layer": 6,
                "embedding": [
                  -3.612804412841797,
                  -3.0009207725524902,
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embedding": [
                  -0.24680757522583008,
                  -0.15286189317703247,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 6,
                "embeddings": [
                  [
                    -0.49513113498687744,
                    0.3145851194858551,
                    "... (truncated for documentation)"
                  ],
                  [
                    -0.9800776839256287,
                    0.3134635090827942,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              },
              {
                "layer": 12,
                "embeddings": [
                  [
                    -0.47163817286491394,
                    -0.22130785882472992,
                    "... (truncated for documentation)"
                  ],
                  [
                    -0.44780775904655457,
                    0.24118921160697937,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ],
            "contacts": [
              [
                0.0011023097904399037,
                4.918707418255508e-06,
                "... (truncated for documentation)"
              ],
              [
                4.918702870781999e-06,
                0.21327124536037445,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- ESM-2 35M is deployed for CPU-only inference with 2 vCPUs and 8 GB RAM; no GPU is required for either ``encoder`` or ``predictor`` endpoints.
- Typical latency is on the order of 1–2 seconds per single-sequence request at maximum schema limits (up to 8 sequences per request, up to 2048 residues each), and scales approximately linearly with both sequence length and batch size.
- On unsupervised structure-related benchmarks, ESM-2 35M shows lower accuracy than larger ESM-2 variants while remaining usable for coarse structural signals:
  
  - Long-range contact precision at L: 0.30 vs. 0.44 (150M), 0.52 (650M), 0.54 (3B).
  - TM-score on CASP14: 0.41 vs. 0.47 (150M), 0.51 (650M), 0.52 (3B); on CAMEO: 0.56 vs. 0.65, 0.70, 0.72 respectively.

- Within BioLM’s ESM-2 family, the 35M model offers the best throughput and lowest resource footprint, making it well suited for high-volume embedding or masked-prediction workloads where the slight loss in structural signal relative to 150M/650M is acceptable.

Applications
------------

- Rapid structural feature extraction from protein sequences for engineering workflows, using embeddings and contact maps from the encoder endpoint to prioritize candidates by foldability, packing, or putative stability before expensive simulations or assays; especially useful when screening many variants up to 2048 residues; less informative for proteins with large conformational changes or long disordered regions where single-sequence models underperform.
- High-throughput structural annotation of metagenomic or proprietary protein collections by computing embeddings and contact maps through the encoder API, enabling teams to cluster sequences by inferred structural class, detect remote structural similarity, and flag candidates with novel folds for follow-up characterization; performance may be reduced for sequences with very low similarity to training distributions.
- Single-sequence protein design assessment by embedding designed variants with the encoder endpoint and comparing their representations or contact patterns to known, functional backbones, helping design teams quickly filter unstable or misfolded designs without MSAs; accuracy is typically lower than full structure-prediction pipelines for very large or multidomain proteins.
- Embedding-based feature generation for downstream ML models in protein engineering, where mean or per-token embeddings from the encoder are used as inputs to custom predictors of stability, expression, localization, or other assay readouts; valuable when paired with in-house lab data to train task-specific models; embeddings alone are not sufficient for precise prediction of binding affinities or complex interfaces without additional training.

Limitations
-----------

- **Maximum Sequence Length**: Sequences in ``items`` must be at most ``2048`` amino acids (``ESM2Params.max_sequence_len``). Longer proteins must be truncated or split across multiple ``ESM2EncodeRequest`` / ``ESM2PredictRequest`` calls.
- **Batch Size**: The maximum ``batch_size`` is ``8`` sequences per request (length-constrained by ``max_sequence_len``). Larger datasets require batching across multiple API calls and downstream aggregation.
- **Single-Sequence Context Only**: Both ``encoder`` and ``predictor`` operate on individual sequences without multiple sequence alignments or template structures. Tasks that depend critically on deep MSAs (e.g. AlphaFold2-style high-accuracy folding of low-depth or orphan proteins) may perform better with MSA-based models.
- **No End-to-End Structure Prediction**: The ``encoder`` can return ``contacts`` (inter-residue distance/contact scores) and rich embeddings (``mean``, ``per_token``, ``bos``, ``attentions``, ``logits``), but it does not produce full 3D atomic coordinates. For atomic-resolution structures, use dedicated folding models such as ESMFold or AlphaFold2.
- **Language-Model-Based Reliability**: All outputs (embeddings, ``contacts``, masked-token ``logits``) are derived from a masked language model trained on natural protein sequences. For highly artificial, low-homology, or strongly out-of-distribution sequences, representations and contact maps may be less biologically meaningful, and masked-residue predictions less informative.
- **Model Size vs. Accuracy**: The 35M-parameter variant is optimized for speed and CPU-only deployment (no ``gpu`` required in ``ESM2_VARIANT_RESOURCE_SPECS``). It is less accurate than larger ESM-2 models (e.g. 150M, 650M, 3B, 15B) on structure- and function-related benchmarks, so for applications where accuracy is critical and latency/memory are less constrained, larger ESM-2 variants may be more appropriate.

How We Use It
-------------

ESM-2 35M enables rapid, scalable exploration of protein sequence space in early-stage protein engineering and optimization campaigns. We use its embeddings, attention maps, and contact predictions as standardized API features to drive downstream predictive models and rank large mutational libraries, then combine these with structural metrics from ESMFold and other BioLM models to prioritize designs for synthesis, screening, and multi-round optimization.

- Integrates efficiently into predictive and generative modeling workflows, providing consistent sequence encodings for tasks such as enzyme design, antibody maturation, and epitope optimization.
- Supports fast sequence-based ranking and filtering across large variant pools, reducing wet-lab screening burden and accelerating iteration cycles.

Related
-------

- ``ESMFold`` – Builds on ESM-2 representations to predict atomic-level protein structures directly from sequence, useful for validating or downstreaming embeddings and contacts from ESM-2 35M.
- ``AlphaFold2`` – High-accuracy structure predictor using MSAs; complements ESM-2 35M when you need slower but more accurate structures for sequences you analyze with embeddings or contacts.
- ``ESM-2 150M`` – Larger ESM-2 variant exposed by the same API with improved representations and contact maps for more demanding downstream tasks.
- ``ESM-IF1`` – Inverse folding model that designs sequences for a given backbone; pairs with ESM-2 35M embeddings or contacts for analyzing or optimizing designed sequences.

References
----------

- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*, 379(6637), 1123–1130.
