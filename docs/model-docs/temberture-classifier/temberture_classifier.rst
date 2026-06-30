TemBERTure Classifier API
=========================

TemBERTure Classifier (TemBERTureCLS) predicts protein thermostability class (thermophilic >60°C vs non-thermophilic) from primary amino acid sequence using a protBERT-BFD backbone with Pfeiffer adapter fine-tuning. The API supports GPU-accelerated batch inference on up to 8 sequences of length ≤512 residues, returning a thermophilicity score in [0,1] and a discrete class label. Trained on the curated TemBERTureDB, the model reports ~0.89 accuracy, F1 0.90, MCC 0.78, enabling triage for enzyme engineering, library pruning, and metagenome annotation.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="temberture-classifier",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKGSILGFVFGDE"
                  },
                  {
                    "sequence": "ASTTSIHR-GGKP"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/temberture-classifier/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKGSILGFVFGDE"
                },
                {
                  "sequence": "ASTTSIHR-GGKP"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/temberture-classifier/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKGSILGFVFGDE"
                    },
                    {
                      "sequence": "ASTTSIHR-GGKP"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/temberture-classifier/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKGSILGFVFGDE"
                ),
                list(
                  sequence = "ASTTSIHR-GGKP"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/temberture-classifier/predict/

   Predict endpoint for TemBERTure Classifier.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **items** (*array of objects*, min length: 1, max length: 8) --- Input records:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using extended amino acid codes plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/temberture-classifier/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKGSILGFVFGDE"
          },
          {
            "sequence": "ASTTSIHR-GGKP"
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

        - **prediction** (*float*) — Model output value; probability score in [0.0, 1.0] for classifier or predicted melting temperature in °C for regression

        - **classification** (*string*, optional) — Predicted protein thermal class label (e.g. "Thermophilic", "Non-thermophilic")

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "prediction": 0.44348000620737454,
            "classification": "Non-thermophilic"
          },
          {
            "prediction": 0.1699508151440416,
            "classification": "Non-thermophilic"
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
                entity="temberture-classifier",
                action="encode",
                params={
                  "include": [
                    "mean",
                    "per_residue"
                  ]
                },
                items=[
                  {
                    "sequence": "MKVALGAIFVDK"
                  },
                  {
                    "sequence": "GGAKKLY-PQMV"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/temberture-classifier/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean",
                  "per_residue"
                ]
              },
              "items": [
                {
                  "sequence": "MKVALGAIFVDK"
                },
                {
                  "sequence": "GGAKKLY-PQMV"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/temberture-classifier/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean",
                      "per_residue"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MKVALGAIFVDK"
                    },
                    {
                      "sequence": "GGAKKLY-PQMV"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/temberture-classifier/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean",
                  "per_residue"
                )
              ),
              items = list(
                list(
                  sequence = "MKVALGAIFVDK"
                ),
                list(
                  sequence = "GGAKKLY-PQMV"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/temberture-classifier/encode/

   Encode endpoint for TemBERTure Classifier.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to include in the response (possible values: "mean", "per_residue", "cls")

      - **items** (*array of objects*, min length: 1, max length: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using extended amino acid codes plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/temberture-classifier/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean",
            "per_residue"
          ]
        },
        "items": [
          {
            "sequence": "MKVALGAIFVDK"
          },
          {
            "sequence": "GGAKKLY-PQMV"
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

        - **embeddings** (*array of float*, size: 1024, optional) — Mean protein embedding vector for the sequence

        - **per_residue_embeddings** (*array of arrays of float*, shape: [L, 1024], optional) — Per-residue embedding vectors for the sequence (L ≤ 512)

        - **cls_embeddings** (*array of float*, size: 1024, optional) — CLS token embedding vector for the sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [
              0.0038513324689120054,
              0.03376583755016327,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                0.048194464296102524,
                0.019800299778580666,
                "... (truncated for documentation)"
              ],
              [
                0.003919513430446386,
                0.0680345892906189,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "embeddings": [
              0.047257035970687866,
              0.03965606167912483,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                -0.0397711843252182,
                0.033104993402957916,
                "... (truncated for documentation)"
              ],
              [
                -0.02825094945728779,
                0.02881569415330887,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Hardware and runtime characteristics
  - Deployed on NVIDIA H100, A100, and L4 GPUs for high-throughput mixed-precision (FP16/BF16) inference with fused attention kernels
  - For 512-aa sequences with dynamic batching, typical throughput is: H100 ~1,600–2,100 sequences/min; A100 ~900–1,300 sequences/min; L4 ~400–650 sequences/min
  - At 512 aa, a single process with adapters active uses ~1–3 GB GPU memory; adapter overhead at inference is negligible relative to the base protBERT-BFD model
- Model architecture and computational cost
  - Based on protBERT-BFD (~420M parameters, 30 transformer layers, 16 heads, 1024 hidden size) with standard O(L²) attention in sequence length L
  - Adapter-based fine-tuning (Pfeiffer adapters) reduces train-time parameters to ~5M without materially changing inference latency compared to a fully fine-tuned 420M-parameter model
- Predictive performance (TemBERTure Classifier)
  - In-domain (TemBERTureDB clustered split): accuracy 0.89, F1 0.90, MCC 0.78 with balanced class F1 (non-thermophilic 0.88, thermophilic 0.90) and low seed-to-seed variance
  - Cross-dataset after 50% identity filtering: accuracy ~0.86 on iThermo and ~0.83 on a TemStaPro-derived test subset; thermophile recall degrades mainly below 20% sequence identity, while non-thermophile performance is more stable
- Comparative performance within BioLM’s model family
  - Versus TemBERTure regression used as a classifier (70°C threshold): the classifier achieves higher accuracy (0.89 vs ~0.82), substantially better thermophile recall, and lower variance; the regression model shows bimodal predictions around class boundaries when interpreted as Tm
  - Versus an internal ESM-2 650M + classifier head baseline on matched splits: TemBERTure Classifier offers comparable or better thermophile recall and overall class balance while running ~1.4–1.8× faster per 512-aa sequence on A100-class GPUs due to its smaller backbone and optimized kernels

Applications
------------

- High-throughput triage of enzyme variant libraries for hot-process biocatalysis: use TemBERTureCLS via the ``predictor`` endpoint to rank sequences by thermophilic class score before expression, reducing wet-lab screening by focusing on variants more likely to remain active at ≥60°C; example uses include cellulases for biomass saccharification, amylases for starch liquefaction, and lipases/esterases for solvent-rich reactions; limitations: binary class (thermophilic vs non-thermophilic) rather than exact Tm, sequences must be ≤512 aa (longer inputs are rejected by the API), treat the score as an enrichment prior rather than a final release criterion
- Genome and metagenome mining for thermostable homolog discovery: apply the classifier score across UniProt/NCBI/metagenome assemblies to prioritize candidates predicted thermophilic, accelerating hit-finding for high-temperature reactors and solvent-tolerant processes; example uses include selecting DNA/RNA polymerase or transaminase homologs for 65–80°C workflows; limitations: training data is enriched for bacterial/archaeal proteomes and organism growth temperatures, predictions for eukaryotic secreted proteins or extreme membrane proteins may be less reliable, always confirm experimentally
- Design-loop guidance in enzyme engineering pipelines: integrate the TemBERTureCLS class score as a lightweight objective to bias ML-guided design, recombination, or directed evolution toward variants more likely to be thermophilic while filtering out destabilizing proposals early; example uses include narrowing combinatorial libraries for oxidoreductases or hydrolases prior to structural modeling/MD and wet-lab rounds; limitations: not calibrated for fine-grained ΔTm at single-mutation resolution, and the regression model (TemBERTureTm) is not currently exposed in this API, so use class scores together with structure/biophysics filters and experimental counterscreens
- Process fit and host/process selection: rapidly assess whether a biocatalyst family is compatible with thermophilic process conditions, or select homologs predicted thermophilic for expression in high-temperature hosts or reactors; example uses include choosing heat-tolerant dehydrogenases for continuous flow at 70°C or proteases for high-temperature detergent formulations; limitations: the model does not account for buffer composition, cofactors/metals, pH, or formulation excipients, use as one input alongside stability assays
- Pre-synthesis QC for construct design and fusion architectures: screen designed constructs (tags, linkers, domain swaps) with TemBERTureCLS to flag sequences likely to be non-thermophilic when the application requires heat robustness, reducing wasted DNA synthesis and expression runs; example uses include selecting truncation boundaries for thermostable catalytic domains or choosing linkers for thermostable fusions intended for hot reactors; limitations: sequences must be ≤512 residues, and chimeric/fusion behavior depends on context beyond primary sequence so treat results as triage signals rather than definitive stability assessments

Limitations
-----------

- **Maximum Sequence Length** is 512 amino acids and **Batch Size** is 8 per request. Any ``items`` entry with a ``sequence`` longer than 512 or any request with more than 8 sequences is rejected. Long proteins are not auto-truncated, tiled, or split; you must segment multi-domain constructs yourself and combine results downstream. Only raw one-line sequences are accepted (no FASTA headers, newlines, or other whitespace).
- **Input alphabet and formatting:** Each ``items`` element must include a ``sequence`` composed of the standard amino-acid alphabet; extended tokens and ``-`` are accepted via the ``AAExtendedPlusExtra`` validator. Sequences containing many ambiguous or non-standard symbols may still pass validation but can substantially degrade prediction quality.
- **Output semantics (predictor):** Each classifier call returns a scalar ``prediction`` and, when available, a ``classification`` label (``thermophilic`` or ``non-thermophilic``). The ``prediction`` is a model score, not a probability-calibrated value; you should set application-specific thresholds and, if needed, perform your own calibration or ranking.
- **Embeddings vs. classification:** The ``encoder`` endpoint does not perform classification or Tm prediction. It returns sequence-level and/or position-level embeddings depending on ``params.include`` (``mean``, ``per_residue``, ``cls``). Use these vectors with your own downstream models if you need custom scoring, calibration, clustering, or retrieval beyond the built-in classifier.
- **Scientific scope and dataset bias:** TemBERTureCLS predicts a coarse thermophilicity class primarily derived from organism growth temperature (>60°C vs. <30°C) and curated Meltome/BacDive labels. It does not provide absolute melting temperature (Tm), mutation effects (ΔΔG/ΔTm), or environment-specific stability (pH, buffer, ligands, membranes). Training data are enriched for bacterial and archaeal proteins; performance can degrade for eukaryotic, viral, antibody, orphan, or heavily engineered sequences, especially when sequence identity to training data is <20%.
- **When this API is not ideal:** For tasks requiring accurate Tm regression, fine-grained mutational scanning, or structure-aware stability assessment, TemBERTure via this API is not sufficient (the published TemBERTureTm regression model is not exposed as a separate ``model_type``). Use complementary stability regressors, ΔΔG tools, or structure models, optionally driven by embeddings from the ``encoder`` endpoint, and reserve the classifier for coarse thermophilic vs. non-thermophilic triage.

How We Use It
-------------

TemBERTure Classifier enables rapid, sequence-only assessment of thermophilic class and is used as a decision layer across protein design and optimization workflows. Its calibrated probability score helps triage variant libraries from generative models, gate temperature-aware regression ensembles, and guide assay design (for example, screening temperatures and host selection). Combined with structure-derived metrics (such as AlphaFold2 models, interface packing, Rosetta ΔΔG) and physicochemical features (charge, pI, hydrophobicity), the classifier accelerates downselection and focuses wet-lab effort on variants most likely to meet process temperature targets. Attention-derived residue saliency highlights mutational hot spots and stability motifs for targeted diversification, improving iteration speed in active-learning campaigns. Standardized, scalable APIs support high-throughput batch scoring and consistent feature logging into multi-objective ranking models used for enzyme design, antibody maturation, and broader developability risk reduction.

- Upstream filter in generative loops to enforce thermostability constraints and raise hit quality before synthesis.
- Routing signal for class-specific models (e.g. TemBERTureTm ensembles, solubility/aggregation predictors) and DOE planning at relevant temperature regimes.
- Feature in multi-objective optimization alongside activity and expression, reducing experimental cycles to reach required Topt/Tm bands.

Related
-------

- ``TemBERTure Regression`` – Predicts melting temperature (Tm) from sequence; use after classification to estimate Tm for proteins flagged as thermophilic or to rank variants within a class.
- ``TEMPRO 3B`` – Alternative thermostability predictor based on protein language models; run alongside TemBERTure Classifier to cross-check thermophilicity calls and flag borderline sequences.
- ``ESM-2 650M`` – Provides high-quality sequence embeddings; useful for building niche, organism-specific binary thermostability classifiers or probing features when TemBERTure confidence is low.
- ``ESMFold`` – Predicts 3D structure; map TemBERTure attention hotspots or per-residue embeddings onto structures to interpret stabilizing regions and prioritize mutations.

References
----------

- Rodella, C., Lazaridi, S., & Lemmin, T. (2024). `TemBERTure: advancing protein thermostability prediction with deep learning and attention mechanisms <https://doi.org/10.1093/bioadv/vbae103>`_. *Bioinformatics Advances*, 4(1), vbae103.
