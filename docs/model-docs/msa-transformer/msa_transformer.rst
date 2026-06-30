MSA Transformer API
===================

MSA Transformer is an unsupervised protein language model that operates directly on multiple sequence alignments using interleaved row and column (axial) attention to capture both evolutionary covariation and sequence-level patterns with ~100M parameters. The API encodes MSAs of up to 256 aligned sequences (length ≤1024) and returns layer-resolved mean and per-position embeddings for the query sequence, tied row attention maps, and attention-derived contact probabilities, enabling downstream structure modeling, contact-based features, and MSA-aware protein engineering workflows at GPU-accelerated throughput.

Encode
------

Encode two MSAs (multiple sequence alignments) with different depths, requesting mean and per-token embeddings, tied row attentions, and contact maps from layers -1 and 12.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="msa-transformer",
                action="encode",
                params={
                  "repr_layers": [
                    -1,
                    12
                  ],
                  "include": [
                    "mean",
                    "per_token",
                    "row_attention",
                    "contacts"
                  ]
                },
                items=[
                  {
                    "msa": [
                      "MKTFFVAGLAA",
                      "M-TFFVAGLAA",
                      "MKTYFVA-LAA",
                      "MKTFFVAGL-A"
                    ]
                  },
                  {
                    "msa": [
                      "ACDEFGHIKLMNP",
                      "ACDEYGHIKLMNP",
                      "AC-EFGHIKLMNP",
                      "ACDEFGHIKLM-P"
                    ]
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/msa-transformer/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "repr_layers": [
                  -1,
                  12
                ],
                "include": [
                  "mean",
                  "per_token",
                  "row_attention",
                  "contacts"
                ]
              },
              "items": [
                {
                  "msa": [
                    "MKTFFVAGLAA",
                    "M-TFFVAGLAA",
                    "MKTYFVA-LAA",
                    "MKTFFVAGL-A"
                  ]
                },
                {
                  "msa": [
                    "ACDEFGHIKLMNP",
                    "ACDEYGHIKLMNP",
                    "AC-EFGHIKLMNP",
                    "ACDEFGHIKLM-P"
                  ]
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/msa-transformer/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "repr_layers": [
                      -1,
                      12
                    ],
                    "include": [
                      "mean",
                      "per_token",
                      "row_attention",
                      "contacts"
                    ]
                  },
                  "items": [
                    {
                      "msa": [
                        "MKTFFVAGLAA",
                        "M-TFFVAGLAA",
                        "MKTYFVA-LAA",
                        "MKTFFVAGL-A"
                      ]
                    },
                    {
                      "msa": [
                        "ACDEFGHIKLMNP",
                        "ACDEYGHIKLMNP",
                        "AC-EFGHIKLMNP",
                        "ACDEFGHIKLM-P"
                      ]
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/msa-transformer/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                repr_layers = list(
                  -1,
                  12
                ),
                include = list(
                  "mean",
                  "per_token",
                  "row_attention",
                  "contacts"
                )
              ),
              items = list(
                list(
                  msa = list(
                    "MKTFFVAGLAA",
                    "M-TFFVAGLAA",
                    "MKTYFVA-LAA",
                    "MKTFFVAGL-A"
                  )
                ),
                list(
                  msa = list(
                    "ACDEFGHIKLMNP",
                    "ACDEYGHIKLMNP",
                    "AC-EFGHIKLMNP",
                    "ACDEFGHIKLM-P"
                  )
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/msa-transformer/encode/

   Encode endpoint for MSA Transformer.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **repr_layers** (*array of integers*, default: [-1]) — Indices of model layers to return representations from

        - **include** (*array of strings*, default: ["mean"]) — Output types to include; allowed values: "mean", "per_token", "row_attention", "contacts"


      - **items** (*array of objects*, min length: 1, max length: 4) --- Input MSAs:

        - **msa** (*array of strings*, min length: 2, max length: 256, required) — List of aligned sequences, all same length, each string length: 1–1024, first sequence is the query, allowed characters: extended amino acids plus "-" and "."

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/msa-transformer/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "repr_layers": [
            -1,
            12
          ],
          "include": [
            "mean",
            "per_token",
            "row_attention",
            "contacts"
          ]
        },
        "items": [
          {
            "msa": [
              "MKTFFVAGLAA",
              "M-TFFVAGLAA",
              "MKTYFVA-LAA",
              "MKTFFVAGL-A"
            ]
          },
          {
            "msa": [
              "ACDEFGHIKLMNP",
              "ACDEYGHIKLMNP",
              "AC-EFGHIKLMNP",
              "ACDEFGHIKLM-P"
            ]
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

        - **sequence_index** (*int*) — Index of the sequence in the input ``msa`` used as the reference (0-based, typically 0 for the first sequence)

        - **embeddings** (*array of objects*, optional) — Mean-pooled embeddings of the reference sequence for selected layers

          - **layer** (*int*) — Model layer index for this embedding (negative indices allowed, e.g. ``-1`` for final layer)

          - **embedding** (*array of floats*, size: d) — Mean embedding vector over aligned positions of the reference sequence; d is the model hidden dimension (d = 768 for the 100M parameter MSA Transformer)

        - **per_token_embeddings** (*array of objects*, optional) — Per-position embeddings of the reference sequence for selected layers

          - **layer** (*int*) — Model layer index for these embeddings (negative indices allowed, e.g. ``-1`` for final layer)

          - **embeddings** (*array of arrays of floats*, shape: [L, d]) — Embedding vectors for each aligned position of the reference sequence; L is the aligned sequence length, d is the model hidden dimension (d = 768 for the 100M parameter MSA Transformer)

        - **row_attentions** (*array of arrays of arrays of floats*, optional, shape: [num_layers, L, L]) — Tied row self-attention maps over aligned positions of the reference sequence; one L×L matrix per included layer with attention weights aggregated over heads, typically in the range [0.0, 1.0]

        - **contacts** (*array of arrays of floats*, optional, shape: [L, L]) — Pairwise contact scores between aligned positions of the reference sequence; each entry is a contact probability or score in the range [0.0, 1.0] derived from attention-based contact prediction

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              {
                "layer": 12,
                "embedding": [
                  -0.6860689520835876,
                  -1.1436715126037598,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 12,
                "embeddings": [
                  [
                    -1.353301763534546,
                    -1.2815747261047363,
                    "... (truncated for documentation)"
                  ],
                  [
                    -1.1626347303390503,
                    -1.0148133039474487,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ],
            "row_attentions": [
              [
                [
                  0.07133398205041885,
                  0.10691503435373306,
                  "... (truncated for documentation)"
                ],
                [
                  0.106469064950943,
                  0.1222657859325409,
                  "... (truncated for documentation)"
                ],
                "... (truncated for documentation)"
              ],
              [
                [
                  0.06474870443344116,
                  0.09530974179506302,
                  "... (truncated for documentation)"
                ],
                [
                  0.10945310443639755,
                  0.05670018121600151,
                  "... (truncated for documentation)"
                ],
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "contacts": [
              [
                0.0004204990400467068,
                0.013430573046207428,
                "... (truncated for documentation)"
              ],
              [
                0.013430578634142876,
                0.0006078745936974883,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "embeddings": [
              {
                "layer": 12,
                "embedding": [
                  -0.7907589673995972,
                  -1.1145281791687012,
                  "... (truncated for documentation)"
                ]
              }
            ],
            "per_token_embeddings": [
              {
                "layer": 12,
                "embeddings": [
                  [
                    -1.207494854927063,
                    -0.9583343267440796,
                    "... (truncated for documentation)"
                  ],
                  [
                    -0.55595463514328,
                    -1.9209963083267212,
                    "... (truncated for documentation)"
                  ],
                  "... (truncated for documentation)"
                ]
              }
            ],
            "row_attentions": [
              [
                [
                  0.09318715333938599,
                  0.14087234437465668,
                  "... (truncated for documentation)"
                ],
                [
                  0.09702638536691666,
                  0.11914250254631042,
                  "... (truncated for documentation)"
                ],
                "... (truncated for documentation)"
              ],
              [
                [
                  0.06364841759204865,
                  0.04230216145515442,
                  "... (truncated for documentation)"
                ],
                [
                  0.08090747147798538,
                  0.043891292065382004,
                  "... (truncated for documentation)"
                ],
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "contacts": [
              [
                0.006547844968736172,
                0.002017058664932847,
                "... (truncated for documentation)"
              ],
              [
                0.002017058664932847,
                0.04606044292449951,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Computational complexity and scaling on MSAs:
  
  - Axial attention over rows and columns reduces attention cost from :math:`O(M^2 L^2)` (naive) to :math:`O(L M^2) + O(M L^2)`, where :math:`M` is MSA depth and :math:`L` is sequence length
  - In practice, BioLM constrains depth to at most 256 sequences and length to 1024 residues (matching the original model’s training regime) to keep memory and latency predictable
  - For a fixed length, runtime grows approximately linearly with the number of sequences used after subsampling/filtering

- Unsupervised structural-contact accuracy relative to other BioLM models:
  
  - On a 14,842‑protein benchmark, MSA Transformer reaches ~57–58% long‑range top‑L precision, versus ~41–42% for ESM‑1b single‑sequence transformers and ~39% for Potts/CCMpred‑style models
  - It matches or exceeds Potts performance on ~97% of proteins, despite using a filtered subset of the MSA, and provides stronger long‑range contact signal than ESM‑2 / E1 embeddings when an MSA is available
  - Contact scores exposed via the ``contacts`` output closely track these benchmarked unsupervised attention‑derived contacts

- Performance as a feature encoder in supervised pipelines:
  
  - Using only query‑sequence embeddings from MSA Transformer, downstream ResNet contact heads outperform comparable heads built on co‑evolutionary‑only or ESM‑1b/ESM‑2 embeddings
  - Adding row‑attention maps from the API (``row_attentions`` output) yields ~7–10 percentage‑point gains in long‑range top‑L precision over trRosetta‑like baselines on CASP13‑FM and CAMEO
  - In BioLM protein engineering workflows, these improved contacts and geometry‑aware embeddings translate into more precise contact/distance priors for design and filtering than pipelines driven purely by Potts or single‑sequence transformers

- Comparative robustness and practical model choice:
  
  - MSA Transformer maintains non‑trivial contact accuracy when either covariance (column shuffling) or sequence‑order patterns (position shuffling) are ablated, unlike Potts (fails when covariance is destroyed) or single‑sequence transformers (fail when sequence order is scrambled)
  - Accuracy increases primarily with MSA diversity, not just depth: diversity‑maximizing subsampling can match or exceed ESM‑1b‑level unsupervised contact precision with as few as 8–16 sequences, whereas nearly identical sequences require far more depth
  - For large‑scale structure‑aware screening, MSA Transformer is typically more data‑efficient and generalizes better across families than Potts, while being faster and cheaper per sequence than full 3D predictors such as AlphaFold2 or ESMFold for contact‑focused workloads

Applications
------------

- Unsupervised contact map estimation from MSAs for structure-aware protein engineering: use attention-derived contact probabilities (``include=["contacts"]``) to approximate long-range residue–residue contacts from an existing alignment, supporting downstream 3D modeling, stability assessment, and interface analysis when full structure prediction pipelines are too slow or unavailable
- MSA-driven feature generation for supervised property predictors: extract per-residue embeddings and pairwise features from the query sequence (``include=["per_token", "row_attention"]``) and feed them into custom regression/classification models for stability, activity, or aggregation, reducing reliance on hand-crafted coevolution features
- Triage of designed or mutagenized variant libraries using approximate structural constraints: run candidate variant MSAs through the encoder to identify substitutions likely to disrupt core contacts or long-range networks via changes in contact maps, enabling early filtering of structurally implausible protein designs before expression when only sequence-level information is available
- Assessment of MSA depth and diversity sufficiency for structure-guided pipelines: exploit the model’s robustness to shallow, diversity-filtered alignments (up to 256 sequences per MSA in the API) to quickly gauge whether current MSAs are likely adequate for contact-guided modeling, and when further sequence mining or single-sequence models should be considered; not suitable when no meaningful alignment can be built
- Portfolio-scale, structure-informed sequence embeddings for protein families: generate mean and per-token embeddings (``include=["mean"]`` or ``["mean", "per_token"]``) from MSAs of related industrial proteins to obtain representations that capture evolutionary variation and inferred structural constraints, enabling clustering, similarity search, and transfer learning across multiple engineering programs beyond what single-sequence models provide

Limitations
-----------

- **Maximum sequence length and MSA depth**: All sequences in ``msa`` must have identical length and cannot exceed ``max_sequence_len = 1024`` characters. Each ``MSATransformerEncodeRequestItem`` must contain between 2 and ``max_msa_depth = 256`` aligned sequences, with the first sequence treated as the query. Inputs violating these constraints (e.g. variable-length rows, longer sequences, too many or too few sequences) will be rejected.
- **Batch size and output volume**: Each ``MSATransformerEncodeRequest`` can contain at most ``batch_size = 4`` MSAs in ``items``. Requesting many layers in ``repr_layers`` and heavy outputs in ``include`` (e.g. ``"per_token"``, ``"row_attention"``, ``"contacts"``) increases response size and latency. For large-scale pipelines, prefer ``"mean"`` embeddings unless you specifically need position-wise or pairwise outputs.
- **MSA format and alphabet requirements**: ``msa`` must be a list of pre-aligned protein sequences of equal length, using only the allowed MSA alphabet (extended amino acids plus gap ``"-"`` and alignment insert ``"."``). The API does not perform alignment, filtering, or MSA construction; low-quality or low-diversity MSAs (e.g. many nearly identical sequences or very shallow MSAs) will reduce the quality of embeddings, row attentions, and derived ``contacts``.
- **Contact predictions are approximate features, not full structure models**: ``"contacts"`` are unsupervised contact scores inferred from tied row attention, not full 3D structures. They are best used as features or ranking signals in a broader structure or design pipeline, not as a replacement for dedicated structure predictors (such as AlphaFold2 or ESMFold) when atomic-resolution models are required.
- **Not optimal for single-sequence or generative tasks**: The model is trained with a masked language modeling objective over MSAs and is intended for inference on multiple aligned sequences. It is not a good choice for single-sequence-only use cases, de novo sequence generation, or applications needing autoregressive likelihoods. In those cases, single-sequence language models or causal generative models are more appropriate; MSA Transformer is most useful when you can provide a reasonably deep and diverse alignment.
- **Biological and domain coverage limits**: Performance depends on evolutionary signal and training coverage. Very short alignments, highly novel or poorly characterized protein families, or MSAs that mix unrelated domains can yield unreliable embeddings and ``contacts``. For such edge cases, you may need to complement MSA Transformer with other models or carefully curated MSAs to obtain robust downstream results.

How We Use It
-------------

MSA Transformer enables us to bring evolutionary constraints directly into protein engineering campaigns by turning MSAs into standardized contact features and sequence representations that downstream models can share. We use its embeddings and attention-derived contact maps as common inputs across generative sequence models, structure predictors, and developability/property predictors to prioritize variants and guide exploration of sequence space in multi-round design cycles such as enzyme optimization, antibody affinity and liability tuning, and library triage.

- Integrates MSA-derived embeddings and contact features with single-sequence language models to steer mutation proposals toward structurally and evolutionarily plausible regions.  
- Supplies consistent contact maps and representations into automated pipelines for sequence ranking, multi-objective optimization, and lab-in-the-loop design, accelerating identification of high-performing protein variants.

Related
-------

- ``ESM1b`` – Single-sequence protein language model whose attention-based contact signals provide a baseline for unsupervised structure learning; useful when no MSA is available or to compare MSA vs. single-sequence performance.
- ``ESM-2 650M`` – Next-generation transformer for single protein sequences; complements MSA Transformer by providing higher-capacity single-sequence embeddings when MSAs are shallow or unavailable.
- ``Evo 1.5 8k Base`` – Evolutionary sequence model operating on families of related sequences; complements MSA Transformer by providing family-aware embeddings that can be combined with MSA-derived contacts for downstream tasks.
- ``ESMFold`` – End-to-end 3D structure prediction model; useful as an independent structure baseline to benchmark contact maps derived from MSA Transformer attention or to obtain structures when only single sequences are used.

References
----------

- Rao, R., Meier, J., Sercu, T., Ovchinnikov, S., & Rives, A. (2021). `Transformer protein language models are unsupervised structure learners <https://doi.org/10.48550/arXiv.2010.06085>`_. *International Conference on Learning Representations (ICLR)*.
