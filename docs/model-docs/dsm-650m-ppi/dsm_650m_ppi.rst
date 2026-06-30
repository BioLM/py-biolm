DSM 650M PPI API
================

DSM 650M PPI is a 650M-parameter diffusion sequence model for protein–protein interaction–aware design, extended from ESM2 and fine-tuned on ∼646k high-confidence STRING PPI pairs. It generates candidate binders conditioned on a target protein or fills masked regions in interacting partners using a masked-diffusion denoising process, maintaining useful accuracy under corruption (∼38% recovery at 15% masking on PPI tests). The API exposes GPU-accelerated, batch generation and embedding for PPI-conditioned design, template diversification, and in silico binder screening.

Predict
-------

Score two interacting protein partners for overall log-probability and perplexity

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-650m-ppi",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTFFVAGLLASAAAGALA"
                  },
                  {
                    "sequence": "GQDPYVRYLENGKLTK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-650m-ppi/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTFFVAGLLASAAAGALA"
                },
                {
                  "sequence": "GQDPYVRYLENGKLTK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-650m-ppi/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTFFVAGLLASAAAGALA"
                    },
                    {
                      "sequence": "GQDPYVRYLENGKLTK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-650m-ppi/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTFFVAGLLASAAAGALA"
                ),
                list(
                  sequence = "GQDPYVRYLENGKLTK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-650m-ppi/predict/

   Predict endpoint for DSM 650M PPI.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Generation parameters:

        - **num_sequences** (*int*, range: 1-32, default: 1) — Number of sequences to generate per input item

        - **temperature** (*float*, range: 0.1-2.0, default: 1.0) — Sampling temperature for sequence generation

        - **top_k** (*int*, minimum: 1, optional, default: null) — Top-k sampling cutoff (null = disabled)

        - **top_p** (*float*, range: 0.0-1.0, optional, default: null) — Nucleus sampling threshold (null = disabled)

        - **max_length** (*int*, range: 10-2048, optional, default: null) — Maximum sequence length for generation (null = derived from input)

        - **step_divisor** (*int*, range: 1-1000, default: 100) — Step divisor for the diffusion process

        - **remasking** (*string*, enum: ["low_confidence", "random", "low_logit", "dual"], default: "random") — Remasking strategy for the diffusion process


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences:

        - **sequence** (*string*, max length: 2048, optional, default: "") — Input sequence that may contain standard unambiguous amino acids plus "<mask>" and "<eos>" tokens, or be empty for unconditional generation

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-650m-ppi/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTFFVAGLLASAAAGALA"
          },
          {
            "sequence": "GQDPYVRYLENGKLTK"
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

        - **[i]** (*array of objects*) — One element per generated sequence for the i-th input:

          - **sequence** (*string*) — Generated amino acid sequence, length: 1–2048

          - **log_prob** (*float*) — Total log probability of the generated sequence, aggregated over all positions, unit: natural logarithm

          - **perplexity** (*float*) — Per-sequence perplexity, computed as exp(-log_prob / sequence_length), range: > 0

          - **sequence2** (*string*, optional) — Generated amino acid sequence for the second chain in PPI variants, length: 1–2048 when present

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -73.92243732511997,
            "perplexity": 40.29074807349975,
            "sequence_length": 19
          },
          {
            "log_prob": -74.88023114204407,
            "perplexity": 81.83618285048628,
            "sequence_length": 16
          }
        ]
      }


Encode
------

Compute mean, per-residue, and CLS embeddings for two protein chains

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-650m-ppi",
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
                    "sequence": "MKTFFVAGLLASAAAGALA"
                  },
                  {
                    "sequence": "GSSGSSSSENLYFQG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-650m-ppi/encode/ \
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
                  "sequence": "MKTFFVAGLLASAAAGALA"
                },
                {
                  "sequence": "GSSGSSSSENLYFQG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-650m-ppi/encode/"
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
                      "sequence": "MKTFFVAGLLASAAAGALA"
                    },
                    {
                      "sequence": "GSSGSSSSENLYFQG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-650m-ppi/encode/"
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
                  sequence = "MKTFFVAGLLASAAAGALA"
                ),
                list(
                  sequence = "GSSGSSSSENLYFQG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-650m-ppi/encode/

   Encode endpoint for DSM 650M PPI.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Encoder configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding representations to return; allowed values: "mean", "per_residue", "cls"


      - **items** (*array of objects*, min: 1, max: 16, required) --- Input sequences to encode:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid alphabet plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-650m-ppi/encode/ HTTP/1.1
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
            "sequence": "MKTFFVAGLLASAAAGALA"
          },
          {
            "sequence": "GSSGSSSSENLYFQG"
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

        - **sequence_index** (*int*) — Zero-based index of the input sequence in the request batch

        - **embeddings** (*array of floats*, optional) — Mean-pooled embedding; shape: [hidden_dim], where hidden_dim depends on model size and variant

        - **per_residue_embeddings** (*array of arrays of floats*, optional) — Per-residue embeddings; shape: [L, hidden_dim], where L is the input sequence length (1–2048) and hidden_dim depends on model size and variant

        - **cls_embeddings** (*array of floats*, optional) — CLS token embedding; shape: [hidden_dim], where hidden_dim depends on model size and variant

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [
              0.07354628294706345,
              0.03296946734189987,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                0.10539563000202179,
                -0.004589089658111334,
                "... (truncated for documentation)"
              ],
              [
                0.0031044939532876015,
                0.13089734315872192,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "cls_embeddings": [
              0.10415444523096085,
              -0.032084524631500244,
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              0.09462907165288925,
              0.1801355928182602,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                0.06166168302297592,
                -0.04662762209773064,
                "... (truncated for documentation)"
              ],
              [
                0.046410296112298965,
                0.25953978300094604,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "cls_embeddings": [
              0.048445917665958405,
              0.04147320240736008,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate three sequence variants for a masked region in a protein, with diffusion-based sampling

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dsm-650m-ppi",
                action="generate",
                params={
                  "num_sequences": 3,
                  "temperature": 0.9,
                  "top_k": 50,
                  "top_p": 0.9,
                  "max_length": 80,
                  "step_divisor": 120,
                  "remasking": "dual"
                },
                items=[
                  {
                    "sequence": "MKTFFVAGLLA<mask><mask>GALA<eos>"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dsm-650m-ppi/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "num_sequences": 3,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.9,
                "max_length": 80,
                "step_divisor": 120,
                "remasking": "dual"
              },
              "items": [
                {
                  "sequence": "MKTFFVAGLLA<mask><mask>GALA<eos>"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dsm-650m-ppi/generate/"
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
                    "step_divisor": 120,
                    "remasking": "dual"
                  },
                  "items": [
                    {
                      "sequence": "MKTFFVAGLLA<mask><mask>GALA<eos>"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dsm-650m-ppi/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                num_sequences = 3,
                temperature = 0.9,
                top_k = 50,
                top_p = 0.9,
                max_length = 80,
                step_divisor = 120,
                remasking = "dual"
              ),
              items = list(
                list(
                  sequence = "MKTFFVAGLLA<mask><mask>GALA<eos>"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dsm-650m-ppi/generate/

   Generate endpoint for DSM 650M PPI.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **num_sequences** (*int*, range: 1-32, default: 1) — Number of sequences to generate per input item

        - **temperature** (*float*, range: 0.1-2.0, default: 1.0) — Sampling temperature for sequence generation

        - **top_k** (*int*, optional, min: 1, default: null) — Top-k sampling limit (null = disabled)

        - **top_p** (*float*, optional, range: 0.0-1.0, default: null) — Nucleus sampling probability (null = disabled)

        - **max_length** (*int*, optional, range: 10-2048, default: null) — Maximum sequence length for generation (null = inferred from input)

        - **step_divisor** (*int*, range: 1-1000, default: 100) — Step divisor for the diffusion process

        - **remasking** (*string*, default: "random") — Remasking strategy for diffusion; one of:

          - "low_confidence"

          - "random"

          - "low_logit"

          - "dual"


      - **items** (*array of objects*, min: 1, max: 1) --- Input specification:

        - **sequence** (*string*, max length: 2048, default: "") — Input protein sequence, may be empty or contain unambiguous amino acids plus optional "<mask>" and "<eos>" tokens

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dsm-650m-ppi/generate/ HTTP/1.1
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
          "step_divisor": 120,
          "remasking": "dual"
        },
        "items": [
          {
            "sequence": "MKTFFVAGLLA<mask><mask>GALA<eos>"
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

        - Each inner array contains one object per generated sequence for that input, ordered as generated:

          - **sequence** (*string*) — Generated amino acid sequence after diffusion-based mask filling, using the DSM amino acid vocabulary, length: 1–2048

          - **log_prob** (*float*) — Total log probability of the generated sequence under the model, summed over all positions, natural logarithm units (unbounded real value)

          - **perplexity** (*float*) — Per-token perplexity of the generated sequence, computed as exp(-log_prob / sequence_length), range: > 0.0

          - **sequence2** (*string*, optional) — Second generated amino acid sequence for PPI variants, using the DSM amino acid vocabulary, length: 1–2048

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKTFFVAGLLAATGALA",
              "log_prob": -87.85799538344145,
              "perplexity": 131.76232659318225
            },
            {
              "sequence": "MKTFFVAGLLAASGALA",
              "log_prob": -83.99137063324451,
              "perplexity": 106.29170595298885
            },
            {
              "sequence": "MKTFFVAGLLAFTGALA",
              "log_prob": -84.94105643033981,
              "perplexity": 112.05026659151326
            }
          ]
        ]
      }


Performance
-----------

- Architectural scale and relative capacity:
  - DSM 650M PPI is a DSM 650M–scale encoder fine‑tuned for PPI, with higher conditional design capacity for binders than DSM 150M Base and DSM 650M Base
  - On PPI sequence reconstruction at 15% masking, DSM 650M PPI reduces cross‑entropy from 2.225 (DSM 150M Base) and 2.046 (DSM 650M Base) to 1.989 and improves token accuracy from 31.4% (DSM 150M Base) to 38.3%
  - Compared to ESM‑2 650M (2.407) and ESM‑2 3B (2.234), DSM 650M PPI achieves lower cross‑entropy (1.989) on partner reconstruction, indicating higher conditional modeling capacity at smaller or similar parameter scale
- Binder‑specific predictive performance:
  - On a held‑out, high‑confidence STRING‑derived PPI test set, DSM 650M PPI attains the lowest cross‑entropy (1.989) and highest masked‑token F1 (0.382) among tested encoder‑family models, improving over DSM 650M Base (F1 0.365) and ESM‑2 3B (F1 0.334)
  - In template‑based BenchBB Binder Benchmark runs scored by Synteract2, DSM 650M PPI generates libraries in which a large fraction of candidates exceed the best known binder by predicted pKd, with success rates up to ~88% for BHRF1 and consistent ppKd improvements across EGFR, BBF‑14, SpCas9, IL‑7Rα, MBP, and PD‑L1
  - Conditional DSM 650M PPI tends to produce designs that more closely track template/target interfaces (higher structural consistency) with strong but slightly more conservative ppKd gains, complementing the broader exploratory behavior of unconditional DSM 650M Base
- Representation quality vs. other BioLM embedding models:
  - DSM 650M (the base encoder for DSM 650M PPI) matches or exceeds ESM‑2 650M on diverse linear‑probe benchmarks (EC, GO BP/MF/CC, domain families, local structural labels), and is only clearly surpassed on average by substantially larger models such as ProtT5
  - Relative to autoregressive protein LMs (e.g., ProGen‑like models), DSM embeddings yield higher linear‑probe F1 on most classification tasks, reflecting stronger bidirectional context modeling that benefits PPI‑focused downstream predictors (e.g., Synteract2)
  - For sequence‑level similarity and clustering, DSM 650M embeddings show tighter correlation with functional/structural labels than DSM 150M Base and comparable or better separation than ESM‑2 650M, enabling a single representation stack for both design and discriminative screening
- Generative robustness and scaling behavior:
  - Under high corruption (70–90% masking), DSM 650M and DSM 650M PPI maintain realistic reconstructions with ASc ≈ 0.27, more than four standard deviations above random natural protein pairs and higher than masked‑LM ESM‑2 baselines at similar mask rates
  - Diffusion‑based DSM models show 2.4–37.8% higher F1 than discrete diffusion baselines (DPLM) across mask rates, and maintain lower cross‑entropy than MLM‑only ESM‑2 as corruption increases, enabling aggressive in silico exploration from heavily masked templates in a single denoising trajectory
  - Compared to autoregressive generators exposed via BioLM, DSM 650M PPI avoids left‑to‑right sampling bottlenecks and leverages bidirectional context, improving reconstruction quality and interface coherence at comparable or higher throughput for large virtual libraries (10⁵–10⁶ sequences per target)

Applications
------------

- De novo design of protein binders against therapeutic and industrial targets (DSM 650M PPI) for rapid hit discovery, where the model is fine-tuned on >600k high-confidence STRING v12 protein–protein interactions to generate a binding partner sequence (SeqB) conditioned on a target protein sequence (SeqA), enabling sequence-based proposal of receptor antagonists, decoys, or target-specific capture reagents without requiring structural input
- Virtual affinity maturation and diversification of known binders by partially masking an existing binder sequence and regenerating large variant panels while conditioning on the same target sequence, enabling exploration of local sequence neighborhoods around moderate-affinity scaffolds to improve predicted binding or escape liabilities while broadly preserving target-aware binding features; DSM remains a generative proposal engine and should be combined with structure prediction, liability filters, and experimental assays
- Large-scale virtual binder library generation for downstream in silico screening by conditionally or unconditionally generating millions of protein sequences with realistic amino acid and secondary-structure statistics, then passing these through external structure, PPI, and developability predictors; this reduces dependence on natural binders and helps platform biotechs or CDMOs assemble reusable, target-focused libraries, but is not suited to strict enforcement of specific folds or repeat architectures
- Embedding-based triage and prioritization of candidate binders using DSM’s encoder endpoint to compute sequence embeddings for clustering, redundancy reduction, and similarity search across internal or DSM-generated panels, enabling selection of structurally and functionally diverse subsets and proximity-based comparison to internal “gold standard” binders, with the caveat that embeddings provide a prioritization signal rather than a guarantee of binding or manufacturability
- AI-assisted design of protein interaction partners for non-therapeutic applications such as affinity purification tags, biosensor components, or capture reagents in manufacturing and quality control, by conditioning DSM 650M PPI on arbitrary process-relevant targets (e.g., host-cell proteins or recombinant products) to generate candidate interactors; use in dual-use or virulence-related contexts is not supported and should be avoided in line with BioLM’s governance and dataset restrictions

Limitations
-----------

- **Maximum sequence length and batching**: All DSM 650M PPI endpoints enforce a hard per-sequence limit of ``2048`` residues. For generation, ``DSMGenerateRequest.items`` accepts exactly ``1`` item, and each ``DSMGenerateRequestItem.sequence`` must be ``<= 2048`` characters (including ``"<mask>"`` and ``"<eos>"`` tokens). For encoding and scoring, ``DSMEncodeRequest.items`` and ``DSMScoreRequest.items`` accept up to ``16`` sequences per request, each with length ``1–2048``. Requests that exceed these limits will fail validation and must be split client-side.

- **Generation behavior and diffusion controls**: DSM uses a masked-diffusion style reverse process rather than left-to-right autoregression. ``DSMGenerateRequestParams.step_divisor`` (``1–1000``, default ``100``) trades off speed vs. quality: lower values generally improve sample quality but add more diffusion steps and latency. ``DSMGenerateRequestParams.num_sequences`` (``1–32``, default ``1``) controls how many samples are returned per input item. Remasking is limited to the discrete strategies in ``DSMRemaskingStrategy`` (``"low_confidence"``, ``"random"``, ``"low_logit"``, ``"dual"``; default ``"random"``). Very aggressive settings (e.g., very low ``step_divisor`` combined with high ``num_sequences``) can be slow and are not suitable for high-throughput screening.

- **Masking semantics and input tokens**: For generation, ``DSMGenerateRequestItem.sequence`` may be empty (unconditional design) or contain amino acids plus optional ``"<mask>"`` and ``"<eos>"`` tokens. DSM is trained to denoise heavily corrupted inputs but is not designed to implement arbitrary templating schemes (for example, complex pattern constraints or fixed motifs at many non-contiguous positions). It will attempt to fill all ``"<mask>"`` positions, but it does not enforce hard design constraints beyond the provided tokens; for strict constraint satisfaction or codon-level design, other models or downstream filtering are usually required.

- **Embedding/encoding output scope**: ``DSMEncodeRequestParams.include`` controls which embeddings are returned in ``DSMEncodeResponseResult.embeddings``, ``per_residue_embeddings``, and ``cls_embeddings``: ``"mean"`` (mean-pooled), ``"per_residue"``, and/or ``"cls"``. DSM provides sequence-level embeddings only and does not output 3D structural coordinates, binding poses, or explicit functional labels. For tasks that require explicit structure (for example, interface geometry or loop conformations), a structure prediction model is typically needed, with DSM encodings used for clustering, similarity search, or coarse ranking.

- **PPI-focused generation vs. general binding design**: The PPI variant (``DSMVariants.PPI``) is fine-tuned on high-confidence StringDB pairs to generate a second binder sequence (returned as ``DSMGenerateResponseResult.sequence2``) conditioned on an input target in ``DSMGenerateResponseResult.sequence``. This conditioning is statistical and trained on generic protein-protein interactions, not on a specific therapeutic or industrial target class. It does not guarantee high affinity, specificity, stability, or developability and may be less suitable for specialized modalities (for example, nanobodies, membrane multipass receptors, Fc engineering) than dedicated models or structure/physics-based workflows. Downstream screening with binding predictors, structural models, and experimental validation remains essential.

- **Model scope and non-optimal use cases**: DSM 650M PPI is optimized for amino acid sequences and is not intended for nucleotide design, codon optimization, or explicit evolutionary modeling (for example, phylogenetic reconstruction). It is also not ideal as the only tool for late-stage ranking when structure-based methods (for example, AlphaFold-class predictors or specialized binder-design frameworks) can provide more precise interface information. In large design campaigns, DSM is best used for high-throughput sequence generation (``generator`` endpoint), embedding-based analysis (``encoder`` endpoint), and coarse scoring (``predictor``/``DSMScoreRequest``), followed by more expensive structural or physics-based methods on a narrowed candidate set.

How We Use It
-------------

DSM 650M PPI enables teams to move from binder concepts to ranked, synthesis-aware designs by standardizing protein–protein interaction generation behind a scalable API that fits into existing discovery pipelines. By conditioning on a target protein and jointly optimizing for natural-like sequence statistics and model likelihood, DSM 650M PPI integrates with structure prediction (e.g., AlphaFold-style workflows), sequence- and function-annotation models, and downstream biophysical filters (charge, size, stability, developability) to support closed-loop design. Organizations use the model to expand around known binders, explore *de novo* sequence space for new modalities (enzymes, antibodies, scaffolds), and connect in vitro screening data back into ML pipelines for multi-round optimization, all via a consistent interface for sequence generation, scoring, and selection.

- Combines DSM PPI generation with predictive PPI scoring (e.g., Synteract2), secondary structure and function predictors, and in-house assay data to prioritize candidates with stronger in silico and experimental performance.  
- Supports large design campaigns (hundreds of thousands of candidates) where automated post-processing applies lab-informed constraints and business criteria (IP landscape, manufacturability, safety) before synthesis.

Related
-------

- ``DSM 650M Base`` – Shares the same diffusion sequence backbone as ``DSM 650M PPI`` but is trained unconditionally; useful for generic sequence generation, scoring, or optimization of binders before PPI-aware refinement.
- ``Synteract2`` – Protein–protein interaction predictor from the DSM*ppi* work; naturally complements ``DSM 650M PPI`` by ranking or filtering designed binders based on predicted interaction probability and binding affinity.
- ``ProteinMPNN`` – Structure-conditioned sequence design model; can refine or diversify ``DSM 650M PPI`` candidates around a fixed 3D interface for joint sequence–structure optimization of binders.
- ``ESM-2 650M`` – The masked language model backbone extended by DSM; embeddings or MLM scores from ``ESM-2 650M`` can be combined with ``DSM 650M PPI`` outputs for additional sequence quality checks, mutational analysis, or downstream transfer learning.

References
----------

- Hallee, L., Rafailidis, N., Bichara, D. B., & Gleghorn, J. P. (2025). Diffusion Sequence Models for Enhanced Protein Representation and Generation. Preprint and software/data release. Available at: https://github.com/Gleghorn-Lab/DSM
