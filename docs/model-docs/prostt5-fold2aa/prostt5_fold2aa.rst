ProstT5 Fold2AA API
===================

ProstT5 Fold2AA is a GPU-accelerated bilingual protein language model that translates structural protein representations encoded as Foldseek 3Di tokens (lowercase a–y) into amino acid sequences. The service exposes encoder and generator endpoints for 3Di-to-AA workflows, supporting up to 16 sequences per batch for embedding and up to 2 sequences per batch for conditional sequence generation (up to 512 residues, ≤3 samples per input). Typical use cases include structure-guided sequence design, inverse folding, remote homolog exploration, and structure-aware library generation.

Encode
------

Generate embeddings for input fold (3Di) sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="prostt5-fold2aa",
                action="encode",
                params={},
                items=[
                  {
                    "sequence": "acdefghiklm"
                  },
                  {
                    "sequence": "cdefghiklmnpq"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/prostt5-fold2aa/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "acdefghiklm"
                },
                {
                  "sequence": "cdefghiklmnpq"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/prostt5-fold2aa/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "acdefghiklm"
                    },
                    {
                      "sequence": "cdefghiklmnpq"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/prostt5-fold2aa/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "acdefghiklm"
                ),
                list(
                  sequence = "cdefghiklmnpq"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/prostt5-fold2aa/encode/

   Encode endpoint for ProstT5 Fold2AA.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature for sequence generation

        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Nucleus sampling cumulative probability threshold

        - **top_k** (*int*, range: 1-20, default: 3) — Number of highest probability tokens to consider at each step

        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor for repeated tokens

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate per input sequence

      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Input sequence as a 3Di token string using characters "acdefghiklmnpqrstvwy"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-fold2aa/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "acdefghiklm"
          },
          {
            "sequence": "cdefghiklmnpq"
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

        - **mean_representation** (*array of floats*, size: 1024) — Mean embedding vector for the input sequence, derived from the ProstT5 encoder

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "mean_representation": [
              -0.059417724609375,
              -0.072265625,
              "... (truncated for documentation)"
            ]
          },
          {
            "mean_representation": [
              -0.049267448484897614,
              -0.053131457418203354,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate amino acid sequences from input fold (3Di) sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="prostt5-fold2aa",
                action="generate",
                params={
                  "temperature": 1.5,
                  "top_p": 0.9,
                  "top_k": 5,
                  "repetition_penalty": 1.3,
                  "num_samples": 2
                },
                items=[
                  {
                    "sequence": "acdefghiklmnpqrs"
                  },
                  {
                    "sequence": "cdghiklmnpqrstvwy"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/prostt5-fold2aa/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 1.5,
                "top_p": 0.9,
                "top_k": 5,
                "repetition_penalty": 1.3,
                "num_samples": 2
              },
              "items": [
                {
                  "sequence": "acdefghiklmnpqrs"
                },
                {
                  "sequence": "cdghiklmnpqrstvwy"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/prostt5-fold2aa/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 1.5,
                    "top_p": 0.9,
                    "top_k": 5,
                    "repetition_penalty": 1.3,
                    "num_samples": 2
                  },
                  "items": [
                    {
                      "sequence": "acdefghiklmnpqrs"
                    },
                    {
                      "sequence": "cdghiklmnpqrstvwy"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/prostt5-fold2aa/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 1.5,
                top_p = 0.9,
                top_k = 5,
                repetition_penalty = 1.3,
                num_samples = 2
              ),
              items = list(
                list(
                  sequence = "acdefghiklmnpqrs"
                ),
                list(
                  sequence = "cdghiklmnpqrstvwy"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/prostt5-fold2aa/generate/

   Generate endpoint for ProstT5 Fold2AA.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 1.2) — Sampling temperature for token generation

        - **top_p** (*float*, range: 0.0-1.0, default: 0.95) — Nucleus sampling cumulative probability threshold

        - **top_k** (*int*, range: 1-20, default: 6) — Number of highest probability tokens considered at each decoding step

        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor applied to repeated tokens during decoding

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of generated sequences returned per input sequence

        - **num_beams** (*int*, range: 1-3, default: 3) — Beam search width used during decoding


      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Amino acid sequence using standard extended amino acid codes


      - **params** (*object*, optional) --- Configuration parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature for token generation

        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Nucleus sampling cumulative probability threshold

        - **top_k** (*int*, range: 1-20, default: 3) — Number of highest probability tokens considered at each decoding step

        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor applied to repeated tokens during decoding

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of generated sequences returned per input sequence


      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — 3Di sequence using characters "acdefghiklmnpqrstvwy"


      - **items** (*array of objects*, min: 1, max: 16) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 1000, required) — Amino acid sequence using standard extended amino acid codes


      - **items** (*array of objects*, min: 1, max: 16) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 1000, required) — 3Di sequence using characters "acdefghiklmnpqrstvwy"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-fold2aa/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 1.5,
          "top_p": 0.9,
          "top_k": 5,
          "repetition_penalty": 1.3,
          "num_samples": 2
        },
        "items": [
          {
            "sequence": "acdefghiklmnpqrs"
          },
          {
            "sequence": "cdghiklmnpqrstvwy"
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

        - **sequence** (*string*) — Generated sequence; amino acids (upper-case A–Z, including X, B, Z, U, O) or 3Di tokens (lower-case ``acdefghiklmnpqrstvwy``), length: 1–512 characters

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "GTSGYAGCNGVFGK"
            },
            {
              "sequence": "SGKGYGCWNGAEGV"
            }
          ],
          [
            {
              "sequence": "DGTCASGPK"
            },
            {
              "sequence": "DGCGRAGSY"
            }
          ]
        ]
      }


Performance
-----------

- Hardware and precision: inference for both encoder (embeddings) and generator (3Di→AA) runs on NVIDIA L4 GPUs in mixed half-precision (fp16), matching the ProstT5 architecture and giving similar throughput to ProtT5 for embedding-only workloads.
- Inverse folding accuracy vs. ProteinMPNN: ProstT5 Fold2AA sequences evaluated with ESMFold achieve an average lDDT of ~0.72, RMSD ~2.9 Å, and TM-score ~0.58 to their target structures, compared with ProteinMPNN’s ~0.77 lDDT, 2.61 Å RMSD, and 0.61 TM-score to the same templates. This places ProstT5 slightly below ProteinMPNN in structural fidelity while using a sequence-to-sequence transformer instead of a graph neural network.
- Sequence diversity and design trade-off: ProstT5 Fold2AA designs show lower pairwise identity to native sequences (~22% PIDE) than ProteinMPNN designs (~30% PIDE) for similar target structures, indicating higher sequence diversity at modest cost in lDDT/TM-score. This can be useful when exploring broader sequence space around a given fold.
- Optimization relative to other BioLM models: compared with encoder-only protein LMs on BioLM (e.g., ProtT5- or ESM-2–style embedding endpoints), ProstT5 Fold2AA adds an auto-regressive decoding step and sampling controls (beam search up to width 3, top-p/top-k, temperature, repetition penalty). This makes generation slower per sequence than pure embedding extraction but substantially cheaper and faster than structure-based inverse-design workflows that require explicit 3D prediction (e.g., AlphaFold2 + ProteinMPNN).

Applications
------------

- Rapid screening of large protein and metagenomic sequence collections for remote homolog detection by using ProstT5 to translate amino acid sequences into 3Di structural tokens and then searching with Foldseek; enables structure-level sensitivity at near sequence-search speed for companies mining sequence databases to discover novel protein folds or distant structural homologs; not suitable when atomic-resolution models or detailed active-site geometries are required.
- High-throughput inverse folding for protein design by generating candidate amino acid sequences from target 3Di structural strings, allowing teams to propose diverse sequences expected to adopt a desired fold or scaffold before downstream structure prediction and wet-lab testing; useful for exploring backbone-compatible variants in early design cycles, but requires additional filters and experimental validation for functional activity or developability.
- Structural embedding extraction from 3Di token strings with the ProstT5 encoder for fast fold-level classification, clustering, and annotation transfer (e.g. CATH/SCOPe-like grouping) across large protein libraries; valuable for biotech pipelines where sequence identity is low but structural similarity informs function or platform selection; less informative for tasks dominated by local chemistry (e.g. specific catalytic mechanisms) rather than overall fold.
- Generation and ranking of structurally constrained sequence variants by iterating between 3Di representations and ProstT5-generated amino acid sequences, enabling protein engineering workflows to sample diverse sequences predicted to preserve a given structural framework for stability or solubility optimization; not optimal as a standalone tool for fine-tuning binding specificity or catalytic efficiency without complementary predictive models and assays.
- Accelerated structural redundancy reduction and clustering of proprietary protein collections by first predicting 3Di strings from sequences with ProstT5 (outside this API) and then encoding or searching those 3Di strings via the API, allowing rapid removal of structurally similar candidates before expensive structure prediction or experimental screening; less appropriate for applications that depend on precise domain boundaries or quaternary structure modeling.

Limitations
-----------

- **Maximum Sequence Length**: ProstT5 supports sequences up to ``max_sequence_len = 1000`` residues for ``encode`` requests and up to ``max_sequence_len = 512`` residues for ``generate`` requests. Longer amino acid or 3Di sequences must be truncated or split before submission.
- **Batch Size**: The maximum ``batch_size`` is ``16`` for ``encode`` and ``2`` for ``generate``. Larger datasets must be processed in multiple API calls.
- ProstT5 is optimized for well-structured, globular proteins. Because training filtered out low-confidence AlphaFold2 models and highly repetitive 3Di strings, performance may degrade on intrinsically disordered proteins, very short peptides, transmembrane segments with poorly defined folds, or highly repetitive 3Di sequences.
- ProstT5-predicted 3Di sequences and derived embeddings are highly effective for rapid fold-based remote homology detection and structure-aware tasks, but they do not replace full 3D structure prediction or comparison methods (e.g. AlphaFold2/ESMFold plus structural alignment) when fine-grained atomic detail is required.
- ProstT5 embeddings are biased toward structural information learned from 3Di; tasks dominated by functional context (e.g. detailed subcellular localization, ligand-binding specificity, GO-term prediction) may perform worse than with general-purpose protein language models such as ProtT5 used alone.
- ProstT5 inverse folding (``fold2AA`` generation) can propose diverse amino acid sequences consistent with an input 3Di fold, but sequence quality and structural fidelity are typically lower than specialized graph-based design tools (e.g. ProteinMPNN) for precise protein engineering or therapeutic design.

How We Use It
-------------

ProstT5 Fold2AA lets us start from a target 3Di structure string and generate diverse amino acid sequences predicted to adopt that structural pattern, which is valuable for inverse folding, library design around known folds, and exploring remote sequence solutions for a given scaffold. In practice, we pair ProstT5 Fold2AA with ESMFold for rapid structural plausibility checks and Foldseek or other ProstT5-based encoders for remote homology analysis, embedding-based filtering, and prioritization in protein engineering campaigns.

- Enables scalable inverse-folding libraries (3Di→AA) that can be triaged with downstream structural prediction and biophysical filters.
- Integrates into broader BioLM workflows that use sequence embeddings, remote homology search, and downstream stability/activity models to focus experimental effort on the most promising designs.

Related
-------

- ``ProstT5 AA2Fold`` – Complementary to ProstT5 Fold2AA; translates amino acid sequences into 3Di structural strings, enabling sequence-to-structure modeling and round‑trip translation workflows.
- ``Foldseek`` – Consumes 3Di sequences for fast structural similarity search; can take ProstT5 Fold2AA‑generated 3Di as input to perform rapid remote homology detection without full 3D structure prediction.
- ``ESMFold`` – Predicts atomic‑resolution 3D structures from amino acid sequences; often used to evaluate ProstT5 Fold2AA‑generated sequences by comparing their predicted structures to native folds.
- ``AlphaFold2`` – Source of the high‑quality structural models (via AlphaFoldDB) used to derive 3Di tokens for ProstT5 training, providing the structural ground truth behind its fold‑to‑sequence translation.

References
----------

- Heinzinger, M., Weissenow, K., Gomez Sanchez, J., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). Bilingual language model for protein sequence and structure. *NAR Genomics and Bioinformatics*, 6(4), lqae150. https://doi.org/10.1093/nargab/lqae150
