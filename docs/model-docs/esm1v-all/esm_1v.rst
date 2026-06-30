ESM-1v API
==========

ESM-1v is a 650M-parameter protein language model for zero-shot prediction of mutational effects on protein function, scoring substitutions directly from single amino acid sequences without task-specific training or MSAs. The API accepts batches of 1–5 sequences (length ≤ 512, optionally containing a single ``<mask>`` token) and returns per-position log-odds scores over the 20 canonical amino acids from one selected ESM-1v model (n1–n5) or a GPU-backed ensemble (all). Typical uses include variant prioritization, protein engineering, and enzyme or antibody optimization.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm1v-all",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACDG<mask>HIKLMN"
                  },
                  {
                    "sequence": "XPQRS<mask>FGT"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm1v-all/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {},
              "items": [
                {
                  "sequence": "ACDG<mask>HIKLMN"
                },
                {
                  "sequence": "XPQRS<mask>FGT"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm1v-all/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {},
                  "items": [
                    {
                      "sequence": "ACDG<mask>HIKLMN"
                    },
                    {
                      "sequence": "XPQRS<mask>FGT"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm1v-all/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(),
              items = list(
                list(
                  sequence = "ACDG<mask>HIKLMN"
                ),
                list(
                  sequence = "XPQRS<mask>FGT"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm1v-all/predict/

   Predict endpoint for ESM-1v.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **model_number** (*string*, enum: ["n1", "n2", "n3", "n4", "n5", "all"], default: "all") — ESM-1v model variant used to score each input sequence

      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence containing exactly one "<mask>" token, using extended amino acid characters plus "<mask>"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm1v-all/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {},
        "items": [
          {
            "sequence": "ACDG<mask>HIKLMN"
          },
          {
            "sequence": "XPQRS<mask>FGT"
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

        - **esm1v-n1** (*array of objects*) — Predictions from ESM-1v model variant n1

          - **token** (*int*) — Integer token index for the amino acid at the masked position

          - **token_str** (*string*) — Single-letter amino acid code or special token corresponding to `token`

          - **score** (*float*, range: 0.0-1.0) — Predicted probability for `token` at the masked position

          - **sequence** (*string*) — Input sequence with `<mask>` replaced by `token_str`

        - **esm1v-n2** (*array of objects*) — Predictions from ESM-1v model variant n2

          - **token** (*int*) — Integer token index for the amino acid at the masked position

          - **token_str** (*string*) — Single-letter amino acid code or special token corresponding to `token`

          - **score** (*float*, range: 0.0-1.0) — Predicted probability for `token` at the masked position

          - **sequence** (*string*) — Input sequence with `<mask>` replaced by `token_str`

        - **esm1v-n3** (*array of objects*) — Predictions from ESM-1v model variant n3

          - **token** (*int*) — Integer token index for the amino acid at the masked position

          - **token_str** (*string*) — Single-letter amino acid code or special token corresponding to `token`

          - **score** (*float*, range: 0.0-1.0) — Predicted probability for `token` at the masked position

          - **sequence** (*string*) — Input sequence with `<mask>` replaced by `token_str`

        - **esm1v-n4** (*array of objects*) — Predictions from ESM-1v model variant n4

          - **token** (*int*) — Integer token index for the amino acid at the masked position

          - **token_str** (*string*) — Single-letter amino acid code or special token corresponding to `token`

          - **score** (*float*, range: 0.0-1.0) — Predicted probability for `token` at the masked position

          - **sequence** (*string*) — Input sequence with `<mask>` replaced by `token_str`

        - **esm1v-n5** (*array of objects*) — Predictions from ESM-1v model variant n5

          - **token** (*int*) — Integer token index for the amino acid at the masked position

          - **token_str** (*string*) — Single-letter amino acid code or special token corresponding to `token`

          - **score** (*float*, range: 0.0-1.0) — Predicted probability for `token` at the masked position

          - **sequence** (*string*) — Input sequence with `<mask>` replaced by `token_str`

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "esm1v-n1": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.09930282086133957,
                "sequence": "A C D G L H I K L M N"
              },
              {
                "token": 12,
                "token_str": "I",
                "score": 0.07655816525220871,
                "sequence": "A C D G I H I K L M N"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n2": [
              {
                "token": 8,
                "token_str": "S",
                "score": 0.0869770348072052,
                "sequence": "A C D G S H I K L M N"
              },
              {
                "token": 4,
                "token_str": "L",
                "score": 0.08455677330493927,
                "sequence": "A C D G L H I K L M N"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n4": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.09174387156963348,
                "sequence": "A C D G L H I K L M N"
              },
              {
                "token": 8,
                "token_str": "S",
                "score": 0.07290017604827881,
                "sequence": "A C D G S H I K L M N"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n5": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.09874340891838074,
                "sequence": "A C D G L H I K L M N"
              },
              {
                "token": 8,
                "token_str": "S",
                "score": 0.07558094710111618,
                "sequence": "A C D G S H I K L M N"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n3": [
              {
                "token": 6,
                "token_str": "G",
                "score": 0.10201912373304367,
                "sequence": "A C D G G H I K L M N"
              },
              {
                "token": 4,
                "token_str": "L",
                "score": 0.08753731101751328,
                "sequence": "A C D G L H I K L M N"
              },
              "... (truncated for documentation)"
            ]
          },
          {
            "esm1v-n1": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.10145029425621033,
                "sequence": "X P Q R S L F G T"
              },
              {
                "token": 8,
                "token_str": "S",
                "score": 0.09379883855581284,
                "sequence": "X P Q R S S F G T"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n2": [
              {
                "token": 8,
                "token_str": "S",
                "score": 0.0989096611738205,
                "sequence": "X P Q R S S F G T"
              },
              {
                "token": 4,
                "token_str": "L",
                "score": 0.09306105226278305,
                "sequence": "X P Q R S L F G T"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n4": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.3064996600151062,
                "sequence": "X P Q R S L F G T"
              },
              {
                "token": 12,
                "token_str": "I",
                "score": 0.11773010343313217,
                "sequence": "X P Q R S I F G T"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n5": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.6074760556221008,
                "sequence": "X P Q R S L F G T"
              },
              {
                "token": 18,
                "token_str": "F",
                "score": 0.05280425772070885,
                "sequence": "X P Q R S F F G T"
              },
              "... (truncated for documentation)"
            ],
            "esm1v-n3": [
              {
                "token": 4,
                "token_str": "L",
                "score": 0.10762723535299301,
                "sequence": "X P Q R S L F G T"
              },
              {
                "token": 6,
                "token_str": "G",
                "score": 0.10102758556604385,
                "sequence": "X P Q R S G F G T"
              },
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- GPU-accelerated inference for the 5-model ESM-1v ensemble (``esm1v-all``) runs on NVIDIA T4 GPUs, while individual ensemble members (``esm1v-n1``–``n5``) are served on lightweight CPU instances optimized for high-throughput scoring.
- Zero-shot mutation effect prediction achieves average |Spearman ρ| ≈ 0.51 across 41 deep mutational scanning datasets, matching state-of-the-art unsupervised MSA-based methods (DeepSequence and EVMutation, each ≈ 0.51) without per-protein training or MSA generation.
- Compared to earlier single-sequence models available on BioLM (ESM-1b |ρ| ≈ 0.46, ProtBERT |ρ| ≈ 0.43 on the same benchmarks), ESM-1v provides higher zero-shot accuracy; the 5-model ensemble attains the strongest correlations and exceeds DeepSequence on 17/41 datasets.
- Inference is substantially more efficient than MSA-dependent architectures (e.g., MSA Transformer, DeepSequence, EVMutation): ESM-1v uses masked forward passes on single, unaligned sequences, enabling faster, lower-cost scoring for large mutational libraries while approaching or matching their predictive performance.

Applications
------------

- Zero-shot scoring of single or multiple amino acid substitutions to prioritize variants in protein engineering campaigns, enabling rapid in silico filtering before deep mutational scanning or focused library construction; most reliable when variants remain close to natural homologs in length and composition, and less reliable for highly unnatural sequences or poorly constrained regions.
- Computational triage of industrial enzyme libraries to enrich for likely functional or stabilizing mutations before expression and assay, reducing experimental load for teams optimizing catalytic efficiency, specificity, or process robustness; performance may degrade for proteins with very limited representation in natural sequence databases or atypical cofactors.
- Identifying functionally constrained or putative active-site residues by ranking per-position mutation effects, guiding where to focus saturation mutagenesis or combinatorial design in workflows for metabolic pathway enzymes, transporters, or receptors; less informative for positions with weak evolutionary constraints, long-range epistasis, or strong context-dependent allosteric effects.
- Variant effect scoring to support assessment of protein-coding changes in research, diagnostics development, or safety workflows, providing an additional sequence-based line of evidence for classifying missense variants as more likely benign vs. more likely damaging; should be combined with structural, population, and clinical evidence and not used as a sole decision criterion.
- Pre-screening development or manufacturing variants (e.g., manufacturability edits or sequence optimizations) to avoid substitutions predicted to strongly reduce protein fitness, helping biopharma teams de-risk sequence changes that might impact yield or stability; not optimal for predicting effects driven primarily by post-translational modifications, formulation conditions, or non-sequence process parameters.

Limitations
-----------

- **Maximum Sequence Length**: Input protein sequences in the ``sequence`` field must be ``<= 512`` amino acids (including the single ``<mask>`` token). Longer proteins must be truncated or split into overlapping windows before calling the ``predictor`` endpoint.
- **Batch Size**: Each ``predictor`` request accepts at most ``5`` items in the ``items`` list. Larger mutational scans must be split across multiple requests and merged client-side.
- **GPU Type**: GPU acceleration (``gpu=ModalGPU.T4``) is available only when ``model`` is set to the ensemble ``all``. Individual models ``n1``–``n5`` use ``gpu=None`` and run on CPU only, which may be slower for large libraries.
- **Scoring Task Only**: The API returns per-amino-acid log-probability scores around the single ``<mask>`` site as ``ESM1vPredictResponseLabel`` entries (``token``, ``token_str``, ``score``, ``sequence``). It does not expose structure prediction, sequence generation, or embeddings.
- **Single Masked Site**: Each ``sequence`` may contain at most one ``<mask>`` token (enforced by ``SingleOccurrenceOf``). Scoring mutations at multiple positions requires separate request items or calls, one masked position per ``sequence``.
- **Biological Domain and Training Data**: ESM-1v is optimized for zero-shot variant effect prediction on natural-like protein sequences. Scores on highly synthetic, non-protein, or very low-homology sequences can be less reliable and should be interpreted together with experimental or structural data.

How We Use It
-------------

BioLM uses ESM-1v as a zero-shot variant-effect scorer in protein engineering workflows to prioritize large mutational libraries before lab work. Standardized, API-based access to per-position mutation scores enables rapid ranking of single and simple combinatorial variants, and these scores are integrated with embedding-based clustering, supervised fitness models, and structure- and developability-aware analyses to guide iterative design cycles for antibodies, enzymes, and other proteins.

- Enables rapid triage of candidate protein sequences prior to synthesis and experimental screening
- Integrates with embedding-based and 3D-structure workflows to align sequence-level ranking with downstream functional and biophysical objectives

Related
-------

- ``ESM-2 650M`` – Larger general-purpose protein language model; useful as a baseline to compare zero-shot mutation scoring and embeddings against ESM-1v’s variant-focused ensemble.
- ``ESMFold`` – Single-sequence structure prediction model based on ESM-2; can be used to place ESM-1v mutation scores in 3D structural context.
- ``AlphaFold2`` – High-accuracy structure predictor that complements ESM-1v by relating predicted variant effects to structural and active-site features.
- ``ESM-IF1`` – Inverse folding model that scores sequences on fixed backbones, providing a structure-conditioned counterpart to ESM-1v’s sequence-only variant effect predictions.

References
----------

- Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021). *Language models enable zero-shot prediction of the effects of mutations on protein function*. bioRxiv, 2021.07.09.450648. https://doi.org/10.1101/2021.07.09.450648
