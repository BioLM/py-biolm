DSM 150M Base API
=================

DSM 150M Base is a 150M-parameter diffusion-based protein language model extending the ESM2-150M encoder with a masked diffusion objective for both representation learning and sequence generation. The API provides GPU-accelerated embeddings (mean, per-residue, CLS), high-corruption denoising and scoring for unambiguous sequences (log-probability, perplexity), and unconditional or template-based generation via mask-filling for sequences up to length 2,048. Typical uses include sequence embedding for downstream models, mutational scanning, template-free or template-guided design, and large-scale screening workflows.

Predict
-------

Score (log-probability and perplexity) for unambiguous protein sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-150m-base",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTFFVAIASVLALA"
                  },
                  {
                    "sequence": "GHHHHHSSGVDLGTENLYFQ"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-150m-base/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTFFVAIASVLALA"
                },
                {
                  "sequence": "GHHHHHSSGVDLGTENLYFQ"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-150m-base/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTFFVAIASVLALA"
                    },
                    {
                      "sequence": "GHHHHHSSGVDLGTENLYFQ"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-150m-base/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTFFVAIASVLALA"
                ),
                list(
                  sequence = "GHHHHHSSGVDLGTENLYFQ"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-150m-base/predict/

   Predict endpoint for DSM 150M Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **num_sequences** (*int*, range: 1-32, default: 1) — Number of sequences to generate per input item

        - **temperature** (*float*, range: 0.1-2.0, default: 1.0) — Sampling temperature

        - **top_k** (*int*, optional, minimum: 1, default: null) — Top-k sampling cutoff (null = disabled)

        - **top_p** (*float*, optional, range: 0.0-1.0, default: null) — Nucleus sampling probability (null = disabled)

        - **max_length** (*int*, optional, range: 10-2048, default: null) — Maximum sequence length in residues (null = derived from input)

        - **step_divisor** (*int*, range: 1-1000, default: 100) — Step divisor for diffusion process

        - **remasking** (*string*, enum: "low_confidence" \| "random" \| "low_logit" \| "dual", default: "random") — Remasking strategy identifier for diffusion


      - **items** (*array of objects*, min: 1, max: 1) --- Input specifications:

        - **sequence** (*string*, max length: 2048, default: "") — Amino acid sequence with optional "<mask>" and "<eos>" tokens (empty string allowed)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-150m-base/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTFFVAIASVLALA"
          },
          {
            "sequence": "GHHHHHSSGVDLGTENLYFQ"
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

        - Each inner array element corresponds to one sampled sequence for the input item, in sampling order:

          - **sequence** (*string*) — Generated amino acid sequence, length 1-2048, uppercase unambiguous residues

          - **log_prob** (*float*) — Total log probability of the generated sequence, summed over positions (natural log, unnormalized by length)

          - **perplexity** (*float*) — exp(-log_prob / sequence_length), dimensionless

          - **sequence2** (*string*, optional) — Second generated amino acid sequence for PPI variants, same constraints as ``sequence`` or absent for non-PPI models


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **sequence_index** (*int*) — Zero-based index of the input sequence in the original request

        - **embeddings** (*array of floats*, optional) — Mean-pooled sequence embedding, size = model hidden dimension

        - **per_residue_embeddings** (*array of arrays of floats*, optional) — Per-residue embeddings with shape [L, H], where L = input sequence length (1-2048), H = model hidden dimension

        - **cls_embeddings** (*array of floats*, optional) — [CLS] token embedding, size = model hidden dimension


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **log_prob** (*float*) — Total log probability of the input sequence under the model, summed over positions (natural log)

        - **perplexity** (*float*) — exp(-log_prob / sequence_length), dimensionless

        - **sequence_length** (*int*) — Length of the scored amino acid sequence in residues (1-2048)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -62.99843406677246,
            "perplexity": 51.285195776445775,
            "sequence_length": 15
          },
          {
            "log_prob": -62.360413044691086,
            "perplexity": 19.483023142990238,
            "sequence_length": 20
          }
        ]
      }


Encode
------

Compute mean, per-residue, and CLS embeddings for two protein sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-150m-base",
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
                    "sequence": "MKTFFVAIASVLALA"
                  },
                  {
                    "sequence": "GHHHHHSSGVDLGTENLYFQ"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-150m-base/encode/ \
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
                  "sequence": "MKTFFVAIASVLALA"
                },
                {
                  "sequence": "GHHHHHSSGVDLGTENLYFQ"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-150m-base/encode/"
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
                      "sequence": "MKTFFVAIASVLALA"
                    },
                    {
                      "sequence": "GHHHHHSSGVDLGTENLYFQ"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-150m-base/encode/"
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
                  sequence = "MKTFFVAIASVLALA"
                ),
                list(
                  sequence = "GHHHHHSSGVDLGTENLYFQ"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-150m-base/encode/

   Encode endpoint for DSM 150M Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Encoder configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to include in the response

          - Allowed values: "mean", "per_residue", "cls"

      - **items** (*array of objects*, min: 1, max: 16, required) --- Input sequences to encode:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid alphabet plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-150m-base/encode/ HTTP/1.1
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
            "sequence": "MKTFFVAIASVLALA"
          },
          {
            "sequence": "GHHHHHSSGVDLGTENLYFQ"
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

        - **embeddings** (*array of floats*, optional) — Mean-pooled sequence embedding, length = model hidden size, unitless

        - **per_residue_embeddings** (*array of arrays of floats*, optional) — Per-residue embeddings with shape ``[sequence_length, hidden_size]``, unitless

        - **cls_embeddings** (*array of floats*, optional) — [CLS] token embedding, length = model hidden size, unitless

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [
              -0.10783202201128006,
              -0.04181931912899017,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                -0.3604585528373718,
                -0.37475961446762085,
                "... (truncated for documentation)"
              ],
              [
                -0.2334814816713333,
                -0.19240154325962067,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "cls_embeddings": [
              -0.07805260270833969,
              0.025669991970062256,
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              -0.20002518594264984,
              -0.1761600524187088,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                -0.20035408437252045,
                -0.6979106664657593,
                "... (truncated for documentation)"
              ],
              [
                -0.16627338528633118,
                -0.3506596088409424,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "cls_embeddings": [
              -0.06080200523138046,
              -0.0903371274471283,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate three candidate completions for a partially masked protein sequence using low-confidence remasking

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-150m-base",
                action="generate",
                params={
                  "num_sequences": 3,
                  "temperature": 0.9,
                  "top_k": 50,
                  "top_p": 0.9,
                  "max_length": 80,
                  "step_divisor": 150,
                  "remasking": "low_confidence"
                },
                items=[
                  {
                    "sequence": "MKTFFVAIASV<mask>ALA<eos>"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-150m-base/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "num_sequences": 3,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.9,
                "max_length": 80,
                "step_divisor": 150,
                "remasking": "low_confidence"
              },
              "items": [
                {
                  "sequence": "MKTFFVAIASV<mask>ALA<eos>"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-150m-base/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "num_sequences": 3,
                    "temperature": 0.9,
                    "top_k": 50,
                    "top_p": 0.9,
                    "max_length": 80,
                    "step_divisor": 150,
                    "remasking": "low_confidence"
                  },
                  "items": [
                    {
                      "sequence": "MKTFFVAIASV<mask>ALA<eos>"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-150m-base/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                num_sequences = 3,
                temperature = 0.9,
                top_k = 50,
                top_p = 0.9,
                max_length = 80,
                step_divisor = 150,
                remasking = "low_confidence"
              ),
              items = list(
                list(
                  sequence = "MKTFFVAIASV<mask>ALA<eos>"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-150m-base/generate/

   Generate endpoint for DSM 150M Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **num_sequences** (*int*, range: 1-32, default: 1) — Number of sequences to generate per input item

        - **temperature** (*float*, range: 0.1-2.0, default: 1.0) — Sampling temperature for sequence generation

        - **top_k** (*int*, minimum: 1, optional, default: null) — Top-k sampling cutoff (null disables top-k)

        - **top_p** (*float*, range: 0.0-1.0, optional, default: null) — Nucleus sampling probability (null disables top-p)

        - **max_length** (*int*, range: 10-2048, optional, default: null) — Maximum length of each generated sequence (null derives length from input or model defaults)

        - **step_divisor** (*int*, range: 1-1000, default: 100) — Diffusion step divisor controlling the number of denoising steps

        - **remasking** (*string*, enum: ["low_confidence", "random", "low_logit", "dual"], default: "random") — Remasking strategy applied between diffusion steps


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences:

        - **sequence** (*string*, max length: 2048, optional, default: "") — Input sequence as unambiguous amino acids with optional "<mask>" and "<eos>" tokens; empty string allowed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-150m-base/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "num_sequences": 3,
          "temperature": 0.9,
          "top_k": 50,
          "top_p": 0.9,
          "max_length": 80,
          "step_divisor": 150,
          "remasking": "low_confidence"
        },
        "items": [
          {
            "sequence": "MKTFFVAIASV<mask>ALA<eos>"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of arrays*) --- One result per input item, in the order requested:

        - (*inner array of objects*) — One entry per generated sequence for the corresponding input:

          - **sequence** (*string*) — Generated amino acid sequence after filling masks and removing special tokens, length: 0–2048 characters

          - **log_prob** (*float*) — Total log probability of ``sequence``, sum over all positions, units: natural logarithm

          - **perplexity** (*float*) — Per-sequence perplexity, defined as ``exp(-log_prob / length)``, where ``length`` is the number of amino acid residues in ``sequence``, range: > 0

          - **sequence2** (*string*, optional) — Generated second amino acid sequence for PPI variants, length: 0–2048 characters, omitted or ``null`` for base variants

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKTFFVAIASVAALA",
              "log_prob": -63.505016803741455,
              "perplexity": 52.93493671298972
            },
            {
              "sequence": "MKTFFVAIASVVALA",
              "log_prob": -62.80944871902466,
              "perplexity": 50.68299981193348
            },
            {
              "sequence": "MKTFFVAIASVIALA",
              "log_prob": -64.51391386985779,
              "perplexity": 56.380289296678804
            }
          ]
        ]
      }


Performance
-----------

- Relative reconstruction performance within the ESM2/DSM family:
  - DSM 150M Base is retrained from ESM-2 150M with a diffusion-style denoising objective, substantially improving recovery of heavily corrupted sequences in a single forward pass
  - At low corruption (5–15% masked residues), ESM-2 150M has slightly better token-level accuracy and cross-entropy, consistent with its MLM pretraining at light mask rates
  - At high corruption (≥50% masked residues), DSM 150M Base maintains lower reconstruction loss and higher F1 over masked positions than ESM-2 150M, with alignment score (ASc) well above random-protein baselines even at 90% masking; DSM 650M Base follows the same trend with higher absolute F1/ASc at higher compute cost
- Generative behavior versus other BioLM sequence models:
  - DSM 150M Base performs non-autoregressive denoising with bidirectional context, yielding amino-acid and secondary-structure k-mer distributions that closely track natural proteins while remaining statistically distinct, unlike left-to-right generators such as ProGen2 variants
  - On unconditional generation, DSM-family models achieve very low Jensen–Shannon divergence (<0.01 for amino-acid 3-mers) to natural sequence corpora; DSM 150M reproduces this behavior at smaller scale, while DSM 650M better matches higher-order and functional statistics
  - DSM 150M Base produces realistic, structurally plausible sequences but, relative to DSM 650M Base, tends to favor more generic folds and common annotation patterns, reflecting its lower capacity; for template-like completion (partially masked inputs), it typically outperforms ESM-2 150M at high mask fractions while being cheaper to sample than large generators such as Evo 2 1B or ProGen2 Large
- Embedding / representation quality compared to other encoders:
  - DSM 150M Base reuses the ESM-2 150M backbone with a modified head (additional projection, logit capping, tied embeddings), which yields modest gains on some downstream linear probes relative to the original ESM-2 150M checkpoint
  - Across annotation and structure-based tasks (e.g., EC, GO BP/MF/CC, domain and secondary-structure prediction), DSM 150M embeddings clearly outperform random-vector and random-transformer controls, indicating that diffusion pretraining preserves and slightly refines ESM-2’s semantic organization
  - Compared with larger encoders from the same lineage (ESM-2 650M, ESM C 600M, DSM 650M), DSM 150M Base has lower absolute probe accuracy but a better accuracy-per-FLOP trade-off and more consistent task performance than autoregressive encoders such as ProtCLM1B, making it suitable for high-throughput embedding workloads
- DSM 150M Base versus DSM 650M / DSM 650M PPI and structure-prediction models:
  - DSM 150M Base is a sequence-only generator, scorer, and encoder without explicit conditioning on partner chains; it is less accurate than DSM 650M PPI for reconstructing one chain given its interactor and for binder design (e.g., BenchBB), but offers substantially higher throughput and lower cost per sequence
  - Relative to DSM 650M Base, DSM 150M trades some reconstruction accuracy, functional fidelity, and rare-motif coverage for cheaper deployment and easier horizontal scaling in dense scoring or embedding scenarios
  - Compared with structure-prediction models (ESMFold, AlphaFold2, AlphaFold3-class services), DSM 150M Base can generate and score orders of magnitude more variants per GPU-hour; when used upstream in multi-stage pipelines (e.g., DSM 150M for candidate generation plus Synteract2 and then structural models), DSM typically contributes a small fraction of total wall-clock time while determining most of the design search space and filtering load

Applications
------------

- High-throughput generative protein design for large-scale sequence exploration, using DSM 150M to sample biomimetic but novel proteins that follow natural-like amino acid, secondary structure, and functional-annotation distributions; this enables companies to populate design spaces for downstream screening when they need many diverse candidates but do not yet have a strong structural hypothesis
- Sequence completion and repair for noisy or partially known protein constructs, leveraging DSM’s ability to reconstruct proteins with very high mask rates in a single forward pass via the ``generator`` endpoint, to fill gaps, repair low-quality regions, or propose local variants around a template sequence; useful when working with legacy constructs, proprietary libraries, or metagenomic hits where segments are missing or intentionally masked
- Embedding-based protein property modeling and ranking, by calling the ``encoder`` endpoint to obtain DSM 150M latent representations (mean, per-residue, or CLS) as features for custom in-house predictors of stability, expression, manufacturability, or client-specific assays, allowing biotech teams to build more informative surrogate models than with one-hot or simple k-mer encodings; downstream model quality will still depend on the amount and quality of labeled data
- Template-guided diversification of known binders or interaction partners, where teams start from an existing protein–protein interaction template (e.g., receptor-binding proteins or scaffolded binding domains) and use DSM-style masked diffusion over the sequence via the ``generator`` endpoint to explore local sequence neighborhoods while maintaining global coherence, generating variants for subsequent structure prediction and affinity screening; DSM 150M does not itself compute 3D structure or binding affinity, so it should be paired with external predictors and wet-lab validation
- Early-stage design-direction triage via unconditional or lightly conditioned sequence generation combined with automated functional annotation pipelines, using DSM outputs together with external predictors (e.g., secondary structure or function models) to quickly assess whether a proposed design strategy tends to yield plausible, on-mechanism proteins; this helps R&D teams decide which target classes or scaffolds merit more intensive structural modeling or experimental campaigns, with the understanding that DSM 150M is not a replacement for high-resolution structure determination or experimental testing

Limitations
-----------

- **Maximum sequence length and batch limits**: All DSM 150M Base endpoints enforce a hard per-sequence limit of ``2048`` residues (including any ``<mask>`` / ``<eos>`` tokens) via the ``max_sequence_len`` constraint. Generation requests must contain exactly one input item (``DSMGenerateRequest.items`` length ``= 1``) and can return up to ``num_sequences <= 32`` candidates for that single input. Encoding (``DSMEncodeRequest``) and scoring (``DSMScoreRequest``) endpoints accept up to ``16`` sequences per request (``items`` length ``<= 16``); all encode/score sequences must be non-empty.

- **Input token constraints and masking semantics**: For generation, ``DSMGenerateRequestItem.sequence`` may be empty (unconditional design) or contain standard unambiguous amino acids plus optional ``<mask>`` and ``<eos>`` tokens; any other characters are rejected. For encoding, ``DSMEncodeRequestItem.sequence`` must be composed of allowed amino acids plus ``'-'`` for gap-style positions; for scoring, ``DSMScoreRequestItem.sequence`` must be unambiguous amino acids only (no masks, gaps, or special tokens). The model does not parse structural annotations, chain delimiters, ligands, or other formatting; these must be stripped or preprocessed before calling the API.

- **Diffusion generation speed–quality trade-offs**: DSM generation is iterative diffusion, not single-pass autoregression. The ``step_divisor`` parameter (``1 <= step_divisor <= 1000``, default ``100``) controls how many positions are updated per diffusion step: lower values generally improve realism and reconstruction quality but increase latency, while higher values increase throughput at the cost of sequence quality and diversity. Extreme sampling settings (very low ``temperature`` near ``0.1`` or aggressive ``top_k`` / ``top_p``) can collapse diversity and reduce the benefits of diffusion. Large design campaigns should budget higher wall-clock time per generated sequence than for encoding or scoring and may need to tune ``step_divisor`` and sampling parameters for their throughput/quality trade-off.

- **Model scope and non-optimal use cases**: DSM 150M is a sequence-only model derived from an ESM2-style encoder and does not consume 3D structures, ligands, small molecules, or assay conditions, nor does it output structures or contact maps. It is primarily suited for protein sequence design (unconditional or masked), generic binder design when used with DSM-PPI variants, and sequence embeddings via the encoder endpoint. It is not ideal as the only tool for (a) final structural ranking or interface modeling (structure predictors such as AF2/ESMFold or antibody-focused models are better suited), (b) codon optimization or mRNA design, (c) antibody CDR-focused engineering where antibody-specific models outperform generic pLMs, or (d) fine-grained local fitness or mutational effect prediction when specialized supervised models or experimental data are available.

- **Probabilistic outputs and scoring limitations**: The ``predictor`` endpoint (``DSMScore``) returns sequence-level ``log_prob`` and ``perplexity`` only; it does not provide residue-level mutation scores, explicit fitness estimates, or structural metrics. Higher log-probability / lower perplexity indicates model preference under its training distribution, not guaranteed foldability, stability, function, or binding. In production pipelines, DSM scores should be combined with downstream structure prediction, stability and aggregation filters, and task-specific predictors (for example, affinity or developability models).

- **Generalization and safety considerations**: DSM 150M was trained on large protein corpora using masked diffusion and shows strong *in silico* reconstruction and distributional metrics, but it has not been comprehensively validated across all protein families, organisms, or unusual chemistries. Generated sequences can be novel or lie outside well-characterized functional space, especially under high corruption or high ``temperature`` settings. The API does not enforce biosafety, immunogenicity, expression, or manufacturability constraints; users are responsible for appropriate expert review, experimental validation, and compliance with institutional and regulatory biosafety requirements.

How We Use It
-------------

DSM 150M Base is used as a fast, low-cost model for exploratory protein design and annotation, where teams need to generate and triage large sequence libraries before investing in synthesis. Its diffusion training enables reliable sequence generation and reconstruction even at high corruption levels, which we apply to in silico mutagenesis around templates, focused resampling of poorly behaving regions, and broad exploration around lab hits. Through standardized APIs, DSM 150M Base embeddings and scores feed directly into downstream ML models and partner tools for structure prediction, interaction screening, and developability assessment, supporting iterative design–build–test cycles and enabling teams to reserve larger DSM variants or specialized predictors for high-value follow-up on shortlisted candidates.

- Enables rapid generation and refinement of design-ready sequence libraries around templates, motifs, or lab-derived hits, with standardized embeddings suitable for internal models of fitness, developability, or manufacturability.
- Integrates with structure, PPI, and annotation models over scalable APIs so teams can automate ranking, diversity control, and safety filtering, and align design output with constraints from expression systems, assay formats, and downstream manufacturing.

Related
-------

- ``DSM 650M Base`` – Larger DSM model with the same masked diffusion objective as ``DSM 150M Base``, providing higher-capacity embeddings and unconditional sequence generation for more challenging design tasks.
- ``DSM 650M PPI`` – Conditional DSM variant fine-tuned for protein–protein interaction design; pairs with ``DSM 150M Base`` when moving from general unconditional generation to target-aware binder design workflows.
- ``ESM-2 150M`` – The underlying masked language model that DSM further trains with a diffusion objective; useful as a baseline for standard MLM pretraining, fine-tuning, or non-diffusion embeddings at a similar parameter scale.
- ``Evo 2 1B Base`` – A larger, general-purpose sequence modeling and design model that complements ``DSM 150M Base`` when you want broader evolutionary priors or to compare diffusion-based designs with autoregressive/evolution-based designs.

References
----------

- Hallee, L., Rafailidis, N., Bichara, D. B., & Gleghorn, J. P. (2025). `Diffusion Sequence Models for Enhanced Protein Representation and Generation <https://github.com/Gleghorn-Lab/DSM>`_. *Preprint / Software repository*.
