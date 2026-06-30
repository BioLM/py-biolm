DSM 650M Base API
=================

DSM 650M Base is a 650M-parameter diffusion-based protein language model built on an ESM2-650M encoder with a masked diffusion objective for unified representation learning and generative design. The service provides GPU-accelerated endpoints for embeddings (mean, per-residue, CLS) of sequences up to 2,048 residues, log-probability and perplexity scoring, and diffusion-based sequence generation via masking or unconditional sampling. Typical applications include enzyme and binder design, fitness-guided mutational scans, and large-scale functional annotation workflows.

Predict
-------

Score protein sequences with log-probability and perplexity using DSM 650M Base

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-650m-base",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTFFVAGILLLASVAA"
                  },
                  {
                    "sequence": "GPNLGVWGQNTKST"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-650m-base/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTFFVAGILLLASVAA"
                },
                {
                  "sequence": "GPNLGVWGQNTKST"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-650m-base/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTFFVAGILLLASVAA"
                    },
                    {
                      "sequence": "GPNLGVWGQNTKST"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-650m-base/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTFFVAGILLLASVAA"
                ),
                list(
                  sequence = "GPNLGVWGQNTKST"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-650m-base/predict/

   Predict endpoint for DSM 650M Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **num_sequences** (*int*, range: 1-32, default: 1) — Number of sequences to generate per input item

        - **temperature** (*float*, range: 0.1-2.0, default: 1.0) — Sampling temperature

        - **top_k** (*int*, minimum: 1, optional, default: null) — Top-k sampling cutoff (null = disabled)

        - **top_p** (*float*, range: 0.0-1.0, optional, default: null) — Nucleus sampling threshold (null = disabled)

        - **max_length** (*int*, range: 10-2048, optional, default: null) — Maximum sequence length (null = derived from input)

        - **step_divisor** (*int*, range: 1-1000, default: 100) — Step divisor for diffusion

        - **remasking** (*string*, default: "random") — Remasking strategy:

          - **"low_confidence"** — Remask lowest-confidence tokens

          - **"random"** — Remask random tokens

          - **"low_logit"** — Remask tokens with lowest logits

          - **"dual"** — Combined remasking strategy


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences:

        - **sequence** (*string*, max length: 2048, optional, default: "") — Amino acid sequence that may include "<mask>" and "<eos>" tokens; remaining characters must be unambiguous amino acids if present

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-650m-base/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTFFVAGILLLASVAA"
          },
          {
            "sequence": "GPNLGVWGQNTKST"
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

        - Each inner array has length = ``num_sequences`` from the request

        - **sequence** (*string*) — Generated amino acid sequence, length: 1–2048, alphabet: unambiguous amino acids, special tokens removed

        - **log_prob** (*float*) — Total log probability of ``sequence``, summed over positions, natural logarithm

        - **perplexity** (*float*) — ``exp(-log_prob / L)`` where ``L`` is the length of ``sequence``

        - **sequence2** (*string*, optional) — Second generated amino acid sequence for PPI variants, length: 1–2048, omitted for base variants


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **sequence_index** (*int*) — Zero-based index of the input sequence in the request

        - **embeddings** (*array of floats*, optional) — Mean-pooled embedding vector, size: model-dependent hidden_dim (e.g., 640–2560), omitted if not requested

        - **per_residue_embeddings** (*array of arrays of floats*, optional) — Per-residue embeddings with shape ``[L, hidden_dim]``, where ``L`` is input sequence length (1–2048) and hidden_dim is model-dependent, omitted if not requested

        - **cls_embeddings** (*array of floats*, optional) — [CLS] token embedding vector, size: hidden_dim, omitted if not requested


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **log_prob** (*float*) — Total log probability of the input sequence, natural logarithm

        - **perplexity** (*float*) — ``exp(-log_prob / L)`` where ``L`` is ``sequence_length``

        - **sequence_length** (*int*) — Length of the scored sequence in residues, range: 1–2048

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -67.97557371854782,
            "perplexity": 43.65949402805405,
            "sequence_length": 17
          },
          {
            "log_prob": -78.83341121673584,
            "perplexity": 191.62891121930247,
            "sequence_length": 14
          }
        ]
      }


Encode
------

Encode protein sequences into mean, per-residue, and CLS embeddings using DSM 650M Base

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-650m-base",
                action="encode",
                params={
                  "include": [
                    "mean",
                    "per_residue",
                    "cls"
                  ]
                },
                items=[
                  {
                    "sequence": "MKTFFVAGILLLASVAA"
                  },
                  {
                    "sequence": "GSSGSSGSSGSSGSSGS"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-650m-base/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean",
                  "per_residue",
                  "cls"
                ]
              },
              "items": [
                {
                  "sequence": "MKTFFVAGILLLASVAA"
                },
                {
                  "sequence": "GSSGSSGSSGSSGSSGS"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-650m-base/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean",
                      "per_residue",
                      "cls"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MKTFFVAGILLLASVAA"
                    },
                    {
                      "sequence": "GSSGSSGSSGSSGSSGS"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-650m-base/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean",
                  "per_residue",
                  "cls"
                )
              ),
              items = list(
                list(
                  sequence = "MKTFFVAGILLLASVAA"
                ),
                list(
                  sequence = "GSSGSSGSSGSSGSSGS"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-650m-base/encode/

   Encode endpoint for DSM 650M Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Encoder configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding representations to return; allowed values: "mean", "per_residue", "cls"


      - **items** (*array of objects*, min: 1, max: 16, required) --- Input protein sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using AAExtendedPlusExtra alphabet with "-" allowed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-650m-base/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean",
            "per_residue",
            "cls"
          ]
        },
        "items": [
          {
            "sequence": "MKTFFVAGILLLASVAA"
          },
          {
            "sequence": "GSSGSSGSSGSSGSSGS"
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

        - **sequence_index** (*int*) — Zero-based index of the input sequence in the request

        - **embeddings** (*array of floats*, optional) — Mean-pooled embedding vector for the sequence (size: *d*, where *d* is the DSM hidden dimension for the selected model size)

        - **per_residue_embeddings** (*array of arrays of floats*, optional) — Per-residue embedding vectors (shape: [L, *d*], where L is the sequence length and *d* is the DSM hidden dimension for the selected model size)

        - **cls_embeddings** (*array of floats*, optional) — [CLS] token embedding vector (size: *d*, where *d* is the DSM hidden dimension for the selected model size)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [
              0.04345281049609184,
              0.013784312643110752,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                0.012877407483756542,
                0.04565651714801788,
                "... (truncated for documentation)"
              ],
              [
                -0.03472462296485901,
                0.09256646037101746,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "cls_embeddings": [
              0.04319912567734718,
              -0.002943706465885043,
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              0.1153685599565506,
              0.020157085731625557,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                0.06304744631052017,
                -0.19785350561141968,
                "... (truncated for documentation)"
              ],
              [
                -0.0029798513278365135,
                0.1588726043701172,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "cls_embeddings": [
              0.0731930211186409,
              0.02582411654293537,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Diffuse and sample new protein variants from a masked seed sequence using DSM 650M Base

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-650m-base",
                action="generate",
                params={
                  "num_sequences": 4,
                  "temperature": 0.9,
                  "top_k": 20,
                  "top_p": 0.9,
                  "max_length": 80,
                  "step_divisor": 150,
                  "remasking": "low_confidence"
                },
                items=[
                  {
                    "sequence": "MKTFFVA<mask>ILLLASVAA<eos>"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-650m-base/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "num_sequences": 4,
                "temperature": 0.9,
                "top_k": 20,
                "top_p": 0.9,
                "max_length": 80,
                "step_divisor": 150,
                "remasking": "low_confidence"
              },
              "items": [
                {
                  "sequence": "MKTFFVA<mask>ILLLASVAA<eos>"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-650m-base/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "num_sequences": 4,
                    "temperature": 0.9,
                    "top_k": 20,
                    "top_p": 0.9,
                    "max_length": 80,
                    "step_divisor": 150,
                    "remasking": "low_confidence"
                  },
                  "items": [
                    {
                      "sequence": "MKTFFVA<mask>ILLLASVAA<eos>"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-650m-base/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                num_sequences = 4,
                temperature = 0.9,
                top_k = 20,
                top_p = 0.9,
                max_length = 80,
                step_divisor = 150,
                remasking = "low_confidence"
              ),
              items = list(
                list(
                  sequence = "MKTFFVA<mask>ILLLASVAA<eos>"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-650m-base/generate/

   Generate endpoint for DSM 650M Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Generation parameters:

        - **num_sequences** (*int*, range: 1-32, default: 1) — Number of sequences to generate per input item

        - **temperature** (*float*, range: 0.1-2.0, default: 1.0) — Sampling temperature

        - **top_k** (*int*, optional, min: 1, default: null) — Top-k sampling cutoff (null = disabled)

        - **top_p** (*float*, optional, range: 0.0-1.0, default: null) — Nucleus sampling threshold (null = disabled)

        - **max_length** (*int*, optional, range: 10-2048, default: null) — Maximum generated sequence length (null = inferred from input)

        - **step_divisor** (*int*, range: 1-1000, default: 100) — Step divisor controlling the number of diffusion steps

        - **remasking** (*string*, enum: ["low_confidence", "random", "low_logit", "dual"], default: "random") — Remasking strategy used during diffusion generation


      - **items** (*array of objects*, min: 1, max: 1) --- Input specification:

        - **sequence** (*string*, max length: 2048, optional, default: "") — Input protein sequence that may include "<mask>" and "<eos>" tokens; remaining characters must be unambiguous amino acids if present (empty string allowed for unconditional generation)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-650m-base/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "num_sequences": 4,
          "temperature": 0.9,
          "top_k": 20,
          "top_p": 0.9,
          "max_length": 80,
          "step_divisor": 150,
          "remasking": "low_confidence"
        },
        "items": [
          {
            "sequence": "MKTFFVA<mask>ILLLASVAA<eos>"
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

        - **[i]** (*array of objects*) — Generated sequences for input item *i*, length = ``num_sequences``

          - **sequence** (*string*) — Generated amino acid sequence, length: 1–2048 residues, alphabet: unambiguous amino acids
          - **log_prob** (*float*) — Total log probability of ``sequence`` under the model, units: natural log
          - **perplexity** (*float*, > 0.0) — ``exp(-log_prob / sequence_length)`` where ``sequence_length`` is the number of residues in ``sequence``
          - **sequence2** (*string*, optional) — Second generated amino acid sequence for PPI variants, length: 1–2048 residues, alphabet: unambiguous amino acids

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKTFFVASILLLASVAA",
              "log_prob": -68.39641150832176,
              "perplexity": 44.69227372441637
            },
            {
              "sequence": "MKTFFVAGILLLASVAA",
              "log_prob": -67.97557371854782,
              "perplexity": 43.65949402805405
            },
            "... (truncated for documentation)"
          ]
        ]
      }


Performance
-----------

- Representation quality (DSM encode) relative to other BioLM models:
  - DSM 650M Base yields higher downstream linear‑probe performance than its ESM‑2 650M backbone across secondary structure, EC/GO/InterPro, stability, and PPI benchmarks when using frozen embeddings
  - On the same probe suite, DSM 650M Base matches or exceeds similarly sized encoder‑only models such as E1 600M and ESM C 300M in weighted F1, while approaching the average classification performance of substantially larger sequence‑to‑sequence models like ProstT5 AA2Fold
  - Versus generative encoders such as ProGen2 Medium and ProGen2 BFD90, DSM 650M Base embeddings are more linearly predictive of functional (GO/EC) and structural labels, making it preferable when encoding or annotation quality is the primary objective

- Generative performance (DSM generate) versus other BioLM protein generators:
  - Compared with ESM‑2 650M used in mask‑fill mode, DSM 650M Base maintains higher reconstruction F1 and Alignment Score (ASc) at high corruption (≥50–90% masked), while MLM reconstructions degrade rapidly; at 90% masking, DSM 650M Base reaches ASc ≈0.27, over four standard deviations above random natural sequence pairs
  - Relative to discrete diffusion models such as DSM 150M Base and DPLM‑like approaches, DSM 650M Base matches or slightly exceeds alignment (ASc) at high mask rates but improves token‑wise F1 by up to ~30–35% under heavy corruption
  - For unconditional generation, DSM 650M Base reproduces amino‑acid 1–3‑mer and predicted secondary‑structure distributions with Jensen–Shannon divergence <0.01 to natural validation sequences, while still sampling from a distinct distribution that is useful for de novo design rather than memorization

- Scoring and calibration (DSM score) versus other encoders:
  - When reconstructing masked sequences (via generate) or scoring unmasked sequences (via score), DSM 650M Base shows a slower increase in cross‑entropy with mask rate than ESM‑2 650M and ESM‑2 3B on OMG‑derived validation and test sets, so perplexity remains informative even when >70% of residues are perturbed
  - Diffusion‑style training improves calibration under strong distribution shift (e.g., heavy random masking or template corruption) compared with pure MLMs, making DSM 650M Base more reliable for ranking candidates by log probability or perplexity in aggressive redesign workloads

- Comparison within the DSM family and practical implications:
  - Compared to DSM 150M Base, DSM 650M Base consistently improves reconstruction ASc and F1 across 5–90% masking and better matches natural amino‑acid 3‑mer and 9‑class secondary‑structure distributions, closing much of the gap to very large encoders while remaining smaller and cheaper than models like ESM‑2 3B or ProstT5 AA2Fold
  - DSM 650M Base provides more PPI‑relevant signal than non‑diffusion encoders of similar size on HPPI‑like linear probes, but for strongly target‑conditioned binder design tasks, DSM 650M PPI or Synteract2 remain preferable and can be combined with DSM 650M Base as a fast front‑end generator whose outputs are filtered and ranked by structure and PPI predictors

Applications
------------

- De novo protein sequence generation for industrial enzyme leads, using DSM 650M’s unconditional diffusion sampling (empty or fully masked input via the ``generator`` endpoint) to propose biomimetic but novel proteins whose amino acid and predicted secondary-structure statistics track natural proteins, reducing the need for brute-force random mutagenesis when starting new engineering campaigns for detergents, food processing, or chemical manufacturing
- High-corruption sequence reconstruction to explore local fitness landscapes, by masking large regions (e.g., 30–90%) of a known enzyme sequence with ``<mask>`` tokens and calling the ``generator`` endpoint to inpaint the missing residues in a single diffusion run, enabling rapid in silico generation of diverse yet structurally coherent variant panels for stability, activity, or specificity optimization when experimental screening capacity is limited
- Protein representation for downstream predictive models, by encoding sequences with the ``encoder`` endpoint (mean, per-residue, or CLS embeddings) and feeding frozen embeddings into task-specific heads (e.g., thermostability, solubility, expression, or process tolerance predictors), improving model performance over standard MLM encoders while keeping supervised training data and compute moderate for proprietary property predictors in commercial enzyme programs
- Template-guided redesign of known enzyme scaffolds, using the ``generator`` endpoint to partially mask an existing catalytic or binding region in a validated protein and regenerate key segments, allowing exploration of sequence space around established motifs while tending to preserve global fold-like patterns; useful for affinity or specificity tuning, but not a replacement for 3D docking, physics-based affinity prediction, or full developability assessment
- Likelihood-based filtering and ranking of candidate enzyme variants, by scoring libraries with the ``predictor`` endpoint to obtain log-probability and perplexity under DSM 650M, then prioritizing sequences that are closer to the learned distribution of natural proteins, which helps triage large in silico or DNA-synthesized libraries before costly wet-lab testing, while still requiring orthogonal structure/function prediction and experimental validation for final selection

Limitations
-----------

- **Maximum sequence length and batching**: DSM 650M Base accepts protein sequences up to ``2048`` residues on all endpoints. For ``DSMGenerateRequestItem.sequence``, ``DSMEncodeRequestItem.sequence``, and ``DSMScoreRequestItem.sequence``, the ``max_length`` is ``2048`` (with ``min_length=1`` where enforced). The ``generator`` endpoint (``DSMGenerateRequest``) currently supports a single input item per request (``items`` has ``min_length=1``, ``max_length=1``) with up to ``num_sequences=32`` outputs per item. The ``encoder`` and ``predictor`` endpoints (``DSMEncodeRequest`` and ``DSMScoreRequest``) accept up to ``16`` sequences per request (``items`` has ``max_length=16``). Larger workloads must be split client-side across multiple API calls.

- **Token and alphabet restrictions**: All endpoints are protein-only. Non-amino-acid symbols (including nucleotides) are rejected. For generation, ``DSMGenerateRequestItem.sequence`` may be empty (unconditional generation) or contain standard amino acids plus ``"<mask>"`` and ``"<eos>"``; after stripping these special tokens, any remaining characters must be valid unambiguous amino acids or the request will fail validation. For encoding, ``DSMEncodeRequestItem.sequence`` uses ``AAExtendedPlusExtra(extra=["-"])``, allowing the extended amino acid alphabet plus the gap character ``"-"``. For scoring, ``DSMScoreRequestItem.sequence`` must contain strictly unambiguous amino acids (no gaps, no ``"<mask>"``, no ``"<eos>"``). Mixed alphabets, masked tokens in encoding/scoring, or other symbols are not supported.

- **Diffusion generation behavior and quality–speed trade-offs**: The ``generator`` endpoint uses a masked diffusion process, not autoregressive sampling. Sequence quality and runtime are sensitive to ``DSMGenerateRequestParams.step_divisor`` and ``DSMGenerateRequestParams.remasking``. Lower ``step_divisor`` values (``>=1``, default ``100``) increase diffusion steps and usually improve reconstruction/realism but slow inference; higher values speed up sampling at the cost of more noise. The default ``remasking="random"`` is robust but not optimal for every domain; alternative values (``"low_confidence"``, ``"low_logit"``, ``"dual"``) may change convergence and diversity but are not guaranteed to improve function. Aggressive sampling settings near the bounds (e.g., ``temperature`` close to ``0.1`` or ``2.0``, extreme ``top_k``/``top_p``) can yield low-likelihood or unrealistic proteins even if the call is valid.

- **Embeddings and scores are sequence-only and context-agnostic**: DSM 650M Base provides sequence-level embeddings and log-probability–based scores for **single proteins** only. The ``encoder`` endpoint returns representations such as ``embeddings`` (mean-pooled), ``per_residue_embeddings`` (shape ``[seq_len, hidden_dim]``), and ``cls_embeddings``, but these are generic features, not direct predictions of stability, activity, binding, or expression. The ``predictor`` endpoint exposes only total ``log_prob`` and ``perplexity`` per sequence. Structure, ligands, textual prompts, or partner sequences are **not** inputs to these Base endpoints (PPI conditioning and ``sequence2`` outputs are only available in DSM PPI variants via different routing). Per-residue outputs are large and can be memory-intensive for long sequences and large batches.

- **Scientific and algorithmic limitations**: DSM 650M is trained on unlabeled protein sequence corpora and evaluated mostly with in-silico metrics (reconstruction, secondary structure surrogates, annotation proxies, and binding predictors). The API does **not** ensure that generated proteins will express, fold, or function as intended in any organism or assay without experimental validation. Unconditional or weakly conditioned generation tends to produce “biomimetic but generic’’ sequences; without strong templates, targeted conditioning, or external filters, DSM is not suited to highly localized design tasks (e.g., fine active-site sculpting, antibody CDR grafting, or tight specificity changes). Its diffusion objective focuses on denoising high mask rates and global context and is not a substitute for structure-native 3D design tools when atomic geometry or specific interfaces are the primary constraint.

- **When DSM 650M Base is not the optimal choice**: DSM 650M Base is best used as a **sequence-level generator/encoder** that feeds into broader design and ranking pipelines. It is generally not the right tool for: (1) final ranking among a small set of high-value candidates where structure- or physics-based models (e.g., structure predictors, 3D diffusion/energy models) provide better discrimination; (2) codon- or nucleotide-level design, where protein-only models cannot handle DNA/RNA constraints; (3) strict binder co-design workflows that require joint modeling of two partners—these should use DSM PPI variants or dedicated PPI/affinity models, not the Base ``encoder``/``predictor``; (4) antibody/nanobody structure prediction, where antibody-specialized 3D models typically outperform sequence-only pLMs; or (5) applications that require sequences longer than ``2048`` residues or extremely large single-call batches, which must instead be truncated, tiled, or split across multiple requests.

How We Use It
-------------

DSM 650M Base enables end-to-end protein engineering workflows where a single model supports both sequence understanding and generative design. Its diffusion-style denoising objective lets teams use the same representations for ranking, clustering, and property prediction while also exploring sequence space via masked or unconditional generation at high mask rates. In practice, DSM 650M embeddings feed into similarity search, developability and function predictors, and structure tools (e.g., AlphaFold-class models), while its generator proposes variants around natural or template sequences that can be iteratively refined with new assay data using standardized, scalable APIs.

- Typical applications include enzyme improvement, antibody and binder optimization (via template- and mask-based design), and rapid exploration of mutational neighborhoods around lead sequences.
- DSM 650M’s unified representation–generation capability simplifies pipeline architecture: the same model can support encode/score/generate stages, reducing integration overhead and enabling reproducible multi-round design campaigns.

Related
-------

- ``DSM 150M Base`` – Smaller DSM variant with the same masked-diffusion objective; useful for rapid prototyping or benchmarking before scaling designs or reconstruction tasks to ``DSM 650M Base``.
- ``DSM 650M PPI`` – DSM variant fine-tuned for conditional binder design; uses target–binder formatting compatible with sequences and templates produced by ``DSM 650M Base``.
- ``ESM-2 650M`` – Underlying MLM checkpoint used to initialize DSM; serves as a baseline for comparing representation quality and reconstruction performance of ``DSM 650M Base`` encodings and scores.
- ``ESM C 600M`` – High-performance protein encoder; its embeddings can be paired with sequences generated or scored by ``DSM 650M Base`` for downstream structure/function prediction or screening.

References
----------

- Hallee, L., Rafailidis, N., Bichara, D. B., & Gleghorn, J. P. (2025). *Diffusion Sequence Models for Enhanced Protein Representation and Generation*. bioRxiv.
