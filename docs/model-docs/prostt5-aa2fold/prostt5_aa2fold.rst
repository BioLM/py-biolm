ProstT5 AA2Fold API
===================

ProstT5 AA2Fold is a GPU-accelerated encoder-decoder model that translates amino acid (AA) sequences into 3Di structure tokens, enabling rapid protein structure-aware annotation. Fine-tuned from ProtT5 on 17M AlphaFoldDB structures encoded with Foldseek’s 20-letter 3Di alphabet, it provides sequence-to-3Di folding at up to 512 residues per sequence and batches of up to 2 sequences. The API exposes encoder and generator endpoints for high-throughput 3Di token generation, supporting remote homology searches, structural classification, and large-scale proteome analyses.

Encode
------

Generate embeddings for input amino acid sequences (AA2fold)

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="prostt5-aa2fold",
                action="encode",
                params={
                  "direction": "AA2fold"
                },
                items=[
                  {
                    "sequence": "ACDEFGHIKGL"
                  },
                  {
                    "sequence": "MNPQRSTVEWY"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/prostt5-aa2fold/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "direction": "AA2fold"
              },
              "items": [
                {
                  "sequence": "ACDEFGHIKGL"
                },
                {
                  "sequence": "MNPQRSTVEWY"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/prostt5-aa2fold/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "direction": "AA2fold"
                  },
                  "items": [
                    {
                      "sequence": "ACDEFGHIKGL"
                    },
                    {
                      "sequence": "MNPQRSTVEWY"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/prostt5-aa2fold/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                direction = "AA2fold"
              ),
              items = list(
                list(
                  sequence = "ACDEFGHIKGL"
                ),
                list(
                  sequence = "MNPQRSTVEWY"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/prostt5-aa2fold/encode/

   Encode endpoint for ProstT5 AA2Fold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature for sequence generation

        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Nucleus sampling cumulative probability threshold

        - **top_k** (*int*, range: 1-20, default: 3) — Number of highest probability tokens to consider for sampling

        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor applied to repeated tokens

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of generated sequences per input item

      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, required, length: 1-512) — Protein sequence using standard amino acid codes (20 standard amino acids plus extended codes B, J, O, U, X, Z)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-aa2fold/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "direction": "AA2fold"
        },
        "items": [
          {
            "sequence": "ACDEFGHIKGL"
          },
          {
            "sequence": "MNPQRSTVEWY"
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

        - **mean_representation** (*array of floats*, size: 1024) — Mean embedding vector for the input sequence; element values typically range from approximately -1.5 to +1.5

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "mean_representation": [
              -0.0159149169921875,
              0.01357269287109375,
              "... (truncated for documentation)"
            ]
          },
          {
            "mean_representation": [
              -0.11297607421875,
              -0.0172576904296875,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate 3Di (fold) sequences from amino acid inputs (AA2fold)

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="prostt5-aa2fold",
                action="generate",
                params={
                  "direction": "AA2fold",
                  "temperature": 1.2,
                  "top_p": 0.95,
                  "top_k": 6,
                  "repetition_penalty": 1.2,
                  "num_samples": 1,
                  "num_beams": 3
                },
                items=[
                  {
                    "sequence": "ACDEFGHIKGL"
                  },
                  {
                    "sequence": "MNPQRSTVEWY"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/prostt5-aa2fold/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "direction": "AA2fold",
                "temperature": 1.2,
                "top_p": 0.95,
                "top_k": 6,
                "repetition_penalty": 1.2,
                "num_samples": 1,
                "num_beams": 3
              },
              "items": [
                {
                  "sequence": "ACDEFGHIKGL"
                },
                {
                  "sequence": "MNPQRSTVEWY"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/prostt5-aa2fold/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "direction": "AA2fold",
                    "temperature": 1.2,
                    "top_p": 0.95,
                    "top_k": 6,
                    "repetition_penalty": 1.2,
                    "num_samples": 1,
                    "num_beams": 3
                  },
                  "items": [
                    {
                      "sequence": "ACDEFGHIKGL"
                    },
                    {
                      "sequence": "MNPQRSTVEWY"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/prostt5-aa2fold/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                direction = "AA2fold",
                temperature = 1.2,
                top_p = 0.95,
                top_k = 6,
                repetition_penalty = 1.2,
                num_samples = 1,
                num_beams = 3
              ),
              items = list(
                list(
                  sequence = "ACDEFGHIKGL"
                ),
                list(
                  sequence = "MNPQRSTVEWY"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/prostt5-aa2fold/generate/

   Generate endpoint for ProstT5 AA2Fold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **direction** (*string*, enum: ["AA2fold", "fold2AA"], required) — Translation direction for sequence generation

        - **temperature** (*float*, range: 0.0-8.0, default: 1.2) — Sampling temperature for AA2fold or fold2AA generation, depending on direction

        - **top_p** (*float*, range: 0.0-1.0, default: 0.95 for AA2fold, 0.85 for fold2AA) — Cumulative probability threshold for token selection

        - **top_k** (*int*, range: 1-20, default: 6 for AA2fold, 3 for fold2AA) — Number of top tokens considered for next prediction

        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty for repeated tokens in the output

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate per input item

        - **num_beams** (*int*, range: 1-3, default: 3, used only for AA2fold) — Beam width for beam search in sequence generation


      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences for generation:

        - **sequence** (*string*, length: 1-512, required) — Input sequence; uppercase amino acid sequence for AA2fold or lowercase 3Di sequence (characters in "acdefghiklmnpqrstvwy") for fold2AA

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-aa2fold/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "direction": "AA2fold",
          "temperature": 1.2,
          "top_p": 0.95,
          "top_k": 6,
          "repetition_penalty": 1.2,
          "num_samples": 1,
          "num_beams": 3
        },
        "items": [
          {
            "sequence": "ACDEFGHIKGL"
          },
          {
            "sequence": "MNPQRSTVEWY"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of arrays of objects*) --- One result per input item, in the order requested:

        - **sequence** (*string*) — Generated sequence for this sample (amino acids or 3Di tokens, depending on the generation direction)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "dadpvrddpdd"
            }
          ],
          [
            {
              "sequence": "ddpppppppdd"
            }
          ]
        ]
      }


Performance
-----------

- ProstT5 AA2Fold runs on Nvidia L4 GPUs in half-precision (fp16), using encoder-only inference where possible to maximize throughput and reduce memory footprint.
- For structural remote homology detection (AA→3Di embeddings + Foldseek), ProstT5 AA2Fold delivers a three orders-of-magnitude runtime speedup compared to workflows that first predict full 3D structures with models such as AlphaFold2 or ESMFold before searching.
- In SCOPe40 remote homology benchmarks, ProstT5 AA2Fold 3Di predictions combined with Foldseek reach near-experimental structural sensitivity (ROC-AUC ≈ 0.45 at the superfamily level vs. 0.49 with experimental structures) and clearly outperform MMseqs2 sequence-only search (ROC-AUC ≈ 0.06).
- For CATH structural classification from embeddings, ProstT5 AA2Fold consistently outperforms earlier sequence-only pLMs served on BioLM (e.g., ProtT5, ESM-1b), achieving ≈73% accuracy at the CATH superfamily level versus ≈64% for ProtT5 and ≈57% for ESM-1b.

Applications
------------

- Rapid structural annotation of novel protein sequences by predicting 3Di structure tokens directly from amino acid sequences, enabling large-scale remote homology detection with Foldseek without running 3D structure prediction; valuable for biotech teams screening metagenomic or proteome-scale datasets for new protein folds or scaffolds, but not a substitute for atomic-resolution models or binding-site level analysis.
- Accelerated inverse folding and scaffold diversification by generating diverse amino acid sequences from a given 3Di fold description, allowing protein engineers to explore alternative sequences predicted to adopt similar overall structures; useful for early-stage scaffold or enzyme family diversification, though not optimized for exact sequence recovery or designs requiring residue-level functional motifs.
- High-throughput fold and superfamily classification of large protein libraries by using ProstT5 embeddings for embedding-based annotation transfer, supporting rapid triage of variants or constructs by structural class or novelty; beneficial in directed evolution or library design workflows, but does not provide explicit functional annotation or activity prediction.
- Structure-aware mutation impact screening by comparing ProstT5-predicted 3Di strings or embeddings for wild-type and variant sequences to flag mutations that likely perturb global or local fold, helping prioritize candidates in stability engineering or affinity maturation campaigns; may miss subtle effects related to dynamics, allostery, or specific chemistry at active sites.
- Embedding-based clustering and redundancy reduction of combinatorial protein libraries by using ProstT5 amino acid embeddings to group variants with similar inferred structures, reducing experimental screening burden while maintaining structural diversity; effective for down-selection of large design sets, but less suited for separating variants that differ mainly in functional properties not strongly tied to fold.

Limitations
-----------

- **Maximum Sequence Length**: Input ``sequence`` strings must not exceed ``1000`` residues for ``encode`` requests and ``512`` residues for ``generate`` requests. Longer proteins must be truncated or split client-side.
- **Batch Size**: The maximum number of ``items`` per request is ``16`` for ``encode`` and ``2`` for ``generate``. Larger collections of sequences require multiple API calls.
- **GPU Type**: Inference runs on ``L4`` GPUs. Throughput and latency are suitable for typical batched analysis, but very large screens (e.g., whole-metagenome) may need asynchronous or sharded workflows.
- ProstT5 is tuned for structure-related tasks (e.g., remote homology detection via 3Di, inverse folding) and to provide 3Di encodings. For purely functional tasks (e.g., subcellular localization, ligand-binding site prediction), embeddings from sequence-focused models such as ProtT5, ESM-2, or Ankh can perform better.
- The 3Di representation emphasizes well-folded, structured regions. Because the training data filtered for high-confidence AlphaFold2 structures, ProstT5 may underperform on intrinsically disordered proteins, very low-complexity regions, or other poorly structured targets.
- ProstT5-generated 3Di strings do not include 3D atomic coordinates. For applications needing explicit coordinates or fine-grained geometric detail (e.g., docking, high-resolution modeling), use dedicated structure predictors such as AlphaFold2 or ESMFold downstream of ProstT5 or instead of it.

How We Use It
-------------

BioLM uses ProstT5 AA2Fold to convert amino acid sequences into 3Di structure strings for fast, sequence-level access to structure-informed features. These 3Di encodings integrate into our standardized, scalable pipelines for protein engineering, where they drive rapid structural similarity screening, remote homology detection, and prioritization of variants before more expensive 3D modeling or wet-lab testing. Combined with downstream tools such as Foldseek and with other sequence- and property-based filters, ProstT5 AA2Fold enables tighter design–test–learn cycles in antibody maturation, enzyme optimization, and targeted protein design.

- Reduces the need for full 3D prediction in early screening by using ProstT5-derived 3Di structure surrogates.
- Supports large-scale, structure-aware candidate ranking for libraries of up to hundreds of thousands of sequences via standardized APIs.

Related
-------

- ``ProstT5 Fold2AA`` – Inverse of ``ProstT5 AA2Fold``, translates 3Di structure sequences back into amino acid sequences for inverse folding and round‑trip validation workflows.
- ``Foldseek`` – Consumes 3Di sequences predicted by ``ProstT5 AA2Fold`` to search for structurally related proteins at sequence-search speed.
- ``ESMFold`` – Full-atom structure predictor that can turn ProstT5‑generated sequences into 3D coordinates for validating inverse folding designs.
- ``AlphaFold2`` – Source of the high-quality 3D structure predictions used to derive 3Di training data for ``ProstT5 AA2Fold`` and to benchmark downstream structure-based tasks.

References
----------

- Heinzinger, M., Weissenow, K., Gomez Sanchez, J., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). Bilingual language model for protein sequence and structure. *NAR Genomics and Bioinformatics*, 6(4), lqae150. https://doi.org/10.1093/nargab/lqae150
