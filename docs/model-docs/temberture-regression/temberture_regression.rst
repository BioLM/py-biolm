TemBERTure Regression API
=========================

TemBERTure Regression (TemBERTureTm) predicts protein melting temperature (°C) from amino acid sequence using a protBERT-BFD backbone with adapter-based fine-tuning, exposed via classifier/regression prediction and embedding encoder endpoints. The API accepts raw sequences up to 512 residues in batched requests (up to 8 per call) and returns per-sequence Tm predictions (regression; optional thermal class) or sequence embeddings. Trained on Meltome Atlas data, a class-conditioned ensemble achieved Pearson r≈0.78 and MAE≈6.3°C. Typical uses include variant ranking, proteome-scale Tm profiling, and stability screening.

Predict
-------

Predicts temperature stability for protein sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="temberture-regression",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MNKLLGRQAWL"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/temberture-regression/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MNKLLGRQAWL"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/temberture-regression/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MNKLLGRQAWL"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/temberture-regression/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MNKLLGRQAWL"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/temberture-regression/predict/

   Predict endpoint for TemBERTure Regression.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **items** (*array of objects*, min length: 1, max length: 8) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using extended amino acid codes plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/temberture-regression/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MNKLLGRQAWL"
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

        - **prediction** (*float*) — Predicted melting temperature (Tm) in degrees Celsius
        - **classification** (*string*, optional) — Predicted thermal class label ("thermophilic" or "non-thermophilic")

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "prediction": 51.04593276977539
          }
        ]
      }


Encode
------

Encodes protein sequences into embeddings

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="temberture-regression",
                action="encode",
                params={
                  "include": [
                    "mean",
                    "per_residue"
                  ]
                },
                items=[
                  {
                    "sequence": "MQDRVKRPMNAF"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/temberture-regression/encode/ \
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
                  "sequence": "MQDRVKRPMNAF"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/temberture-regression/encode/"
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
                      "sequence": "MQDRVKRPMNAF"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/temberture-regression/encode/"
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
                  sequence = "MQDRVKRPMNAF"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/temberture-regression/encode/

   Encode endpoint for TemBERTure Regression.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output embedding types to include; allowed values: "mean", "per_residue", "cls"

      - **items** (*array of objects*, min length: 1, max length: 8, required) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using extended amino acid codes plus "-"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/temberture-regression/encode/ HTTP/1.1
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
            "sequence": "MQDRVKRPMNAF"
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

        - **sequence_index** (*integer*) — Zero-based index of the input sequence in the request

        - **embeddings** (*array of floats*, size: 1024, optional) — Mean sequence embedding vector

        - **per_residue_embeddings** (*array of arrays of floats*, shape: [≤512, 1024], optional) — Embedding vectors for individual residues

        - **cls_embeddings** (*array of floats*, size: 1024, optional) — CLS token embedding vector

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "embeddings": [
              0.08685418218374252,
              0.11291765421628952,
              "... (truncated for documentation)"
            ],
            "per_residue_embeddings": [
              [
                0.1003321036696434,
                0.11256718635559082,
                "... (truncated for documentation)"
              ],
              [
                0.10935599356889725,
                0.12453897297382355,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Model architecture and deployment: TemBERTure Regression uses a 30-layer protBERT-BFD backbone with an adapter-based regression head (≈420M frozen backbone parameters; ≈5M trainable adapter parameters). Inference runs deterministically with dropout disabled and mixed-precision FP16 kernels on NVIDIA GPUs (A100 80GB for high-throughput tiers; L4 24GB for cost-optimized tiers).
- Latency and scaling behavior: On 512-residue sequences, typical warm inference on an A100 80GB is ≈0.25–0.45 s per sequence and ≈1.8–3.0 s for 8-way server-side batches; L4 24GB is ≈0.5–0.9 s per sequence and ≈3.5–5.5 s for 8-way batches. Wall-clock time scales approximately O(L²) with sequence length due to transformer attention, so halving sequence length (e.g., 512→256) usually yields ≈40–60% faster responses. Horizontal autoscaling with dynamic batching maintains low p95 latency under bursty workloads by aggregating small requests to keep GPUs saturated.
- Accuracy and numerical stability: On Meltome-like held-out data, BioLM’s class-conditional ensemble (routing via TemBERTure Classifier, then averaging predictions from 5 non-thermophilic and 2 thermophilic regressors) achieves MAE ≈ 6.3°C and R² ≈ 0.78. If the regression output is thresholded at 70°C and used as a classifier, accuracy is ≈82%. Ensemble inference removes most training-seed variance seen in single regressors, with repeated calls on the same sequence typically varying by <0.3°C.
- Comparison to related BioLM models: Compared to TemBERTure Classifier, the regression head is heavier; at equal batch sizes the classifier is usually 5–10% faster and should be preferred when only thermophilic/non-thermophilic labeling is needed (classifier F1 ≈ 0.90, MCC ≈ 0.78, vs ≈82% accuracy from thresholded regression). Relative to TEMPRO 650M and 3B property regressors on A100 at 512 residues, TemBERTure Regression is ≈1.3–1.8× faster than TEMPRO 650M and ≈2–5× faster than TEMPRO 3B, with substantially lower GPU memory usage, making it better suited for routine large-scale Tm scoring where TEMPRO’s additional expressiveness is not critical.

Applications
------------

- High-throughput triage of enzyme variant libraries for thermostability
  - Use predicted melting temperature (Tm) from sequence to rank and down-select large mutational or recombination libraries before wet-lab screening, focusing assays on candidates more likely to remain folded near the intended process temperature
  - Valuable for industrial biocatalysts (e.g., hydrolases for detergents, lignocellulose-degrading enzymes for biomass processing, polymer-degrading enzymes) where operating at 55–75°C improves mass transfer and reduces contamination
  - Not optimal for fine-grained, per-variant optimization within a narrow window (e.g., +2–3°C shifts); best used for coarse prioritization and elimination of clearly underperforming sequences, since regression predictions are biased toward broad thermal classes and do not account for buffer, pH, cofactors, or stabilizing excipients
- Prospective mining of public or proprietary sequence repositories for thermostable homologs
  - Scan UniProt/metagenomes to identify and rank homologs more likely to retain structure at elevated temperatures, accelerating hit identification for high-temperature steps (e.g., biomass saccharification at 65°C, high-solids bioprocessing)
  - Reduces costly expression and assay of poor candidates by narrowing to families and clades enriched for higher predicted Tm distributions and thermophilic classifications
  - TemBERTureTm performs best as a ranking/filtering tool rather than an absolute Tm oracle and may degrade on sequences far outside its training distribution; validate top hits experimentally
- Stability gating in generative protein design workflows
  - Integrate Tm prediction as a fast filter to remove low-stability designs prior to structure modeling, docking, or multi-parameter optimization, enabling design loops that target stability and function simultaneously
  - Useful when exploring sequence space far from known scaffolds, where quick thermostability screening prevents wasting compute and bench time on fragile constructs
  - Not intended to serve as a single optimization objective for precise Tm targeting; pair with orthogonal predictors (e.g., structure-based stability, aggregation propensity) and confirm with experimental thermal shift assays
- Process setpoint compatibility checks for bioconversion and manufacturing
  - Evaluate whether candidate enzymes are likely to tolerate planned reactor setpoints (e.g., 60–70°C continuous operation) based on predicted Tm and thermal class, informing go/no-go, buffer selection, and whether stabilization engineering is required
  - Helps CDMOs and process development teams de-risk tech transfer by flagging sequences that are unlikely to withstand anticipated thermal exposure during upstream processing or formulation stress screens
  - Predictions are sequence-intrinsic and do not capture chaperone assistance, formulation additives, immobilization effects, or process transients; maintain safety margins around desired operating temperatures
- Proteome-scale assessment to select source organisms or donors for high-temperature applications
  - Estimate Tm and thermal class distributions across proteomes to identify organisms or gene donors whose proteins are broadly predisposed to higher thermal stability, guiding homolog selection for pathway engineering in thermotolerant chassis
  - Effective for narrowing species panels before cloning and expression, especially when growth temperature metadata are incomplete or inconsistent
  - Best for distribution-level insights rather than exact Tm per protein; sequences longer than 512 amino acids may need domain-centric evaluation due to model truncation limits in the API

Limitations
-----------

- **Batch Size** and **Maximum Sequence Length**: ``TemBERTurePredictRequest.items`` accepts up to 8 sequences per call (``batch_size`` = 8). Each ``sequence`` must be at most 512 amino acids (``max_sequence_len`` = 512). Longer inputs are hard-truncated at 512 residues by the tokenizer, which can remove N-/C-terminal regions or domains that influence predicted Tm. The supported alphabet is the extended amino acid set with ``-`` allowed; any other characters are rejected at validation.
- **Regression I/O semantics**: For the regression API, each request item contains only a ``sequence`` field; there are no tunable parameters for the ``predictor`` endpoint. Each result in ``TemBERTurePredictResponse.results`` returns a single ``prediction`` float (interpreted as melting temperature in °C). The ``classification`` field is usually ``None`` for the ``"regression"`` model type; if you need thermo-class labels, use the ``"classifier"`` model. The regression endpoint does not expose uncertainty estimates, confidence intervals, or per-residue scores.
- **Truncation and long/multi-domain proteins**: Because all sequences are truncated to 512 residues, very long or multi-domain proteins, fusion constructs, or proteins with critical terminal tags may be incompletely represented, which can reduce accuracy. When possible, pre-trim inputs to the biologically relevant construct (e.g. a single domain or mature chain) before calling the API.
- **Coarse-grained temperature predictions**: The TemBERTureTm regression model was trained on Meltome-derived data and tends to produce a bimodal distribution of predictions (mostly <~60°C or >~80°C). Within each broad class, correlations between predicted and true Tm are weak, so the regression scores are not well calibrated for fine-grained ranking of variants with similar melting temperatures, especially when small ΔTm shifts matter or values lie in the 60–80°C range.
- **Domain shift and novel designs**: Model performance can degrade on sequences from organisms, proteomes, or design spaces underrepresented in the training data (e.g. completely new species or highly engineered proteins). In such cases, the model may output overconfident class-like Tm values. Experimental validation is recommended before making safety-critical or high-impact design decisions based on predictions alone.
- **Sequence-only context**: Both ``encoder`` and ``predictor`` endpoints use only the primary amino acid ``sequence``. Predictions ignore experimental and environmental context (pH, buffer, cofactors, ligands, metal ions, chaperones, PTMs, oligomeric state, assay format). The regression model is therefore not suitable for context-dependent stability predictions, precise per-mutation ΔTm estimation without retraining, or cases where stability is dominated by quaternary structure or complex assembly effects.

How We Use It
-------------

TemBERTure Regression (TemBERTureTm) provides sequence-based Tm estimates that we use as a thermostability signal within multi-objective protein engineering campaigns. We integrate its regression outputs with TemBERTure classification, protein language model embeddings, 3D structure–derived metrics, and physicochemical property estimators to rank and down-select large enzyme and antibody libraries, while treating predictions as approximate Tm and focusing on class-aware trends and population-level distributions rather than single-value accuracy. Standardized, scalable APIs expose these estimates so teams can orchestrate high-throughput triage, calibrate model behavior to their own DSC or thermal-shift readouts with limited wet-lab data, and run active learning loops that iteratively propose variants (e.g. via masked language models) to meet combined stability, activity, and developability targets.

- Accelerates lead selection by enriching libraries for variants projected to clear application-specific thermostability thresholds before synthesis, improving hit rates and shortening optimization cycles.
- Supports lab-in-the-loop optimization by combining Tm estimates with measured data and other developability signals to rebalance objectives across rounds and reduce unnecessary synthesis and assay effort.

Related
-------

- ``TemBERTure Classifier`` – Predicts thermophilic/non-thermophilic class from sequence; use it to route sequences to class-specific TemBERTure Regression ensembles for improved Tm estimates within each class.
- ``TEMPRO 3B`` – Independent thermostability/Tm regressor; ensemble or cross-check with TemBERTure Regression to increase robustness, especially outside the Meltome Atlas temperature distribution.
- ``ESM-1v`` – Variant effect model for single/multi-point mutations; combine with TemBERTure Regression to prioritize sequence variants predicted to improve thermostability.
- ``ESMFold`` – Structure prediction from sequence; use with TemBERTure Regression to interpret predicted Tm in terms of structural features (packing, salt bridges, disulfides) before experimental validation.

References
----------

- Rodella, C., Lazaridi, S., & Lemmin, T. (2024). *TemBERTure: advancing protein thermostability prediction with deep learning and attention mechanisms* (https://doi.org/10.1093/bioadv/vbae103). Bioinformatics Advances.
