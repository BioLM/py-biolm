ESM2StabP API
=============

ESM2StabP predicts protein melting temperature (Tm, °C) from amino acid sequence using esm2_t33_650M_UR50 layer-33 embeddings and a random forest regressor augmented with thermophilic classification, optimal growth temperature, and experimental-condition (cell/lysate) features when provided. The API returns per-sequence Tm estimates and a derived thermophilic flag (Tm > 60 °C), with reported performance around R²≈0.94 and PCC≈0.92 on curated TPP-derived datasets, enabling stability-aware screening and protein/enzyme engineering workflows.

Predict
-------

Predict protein melting temperature (Tm) and thermophilic classification from amino acid sequences, optionally using growth temperature and experimental condition (cell or lysate) as additional features.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm2stabp",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "M"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm2stabp/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "M"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm2stabp/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "M"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm2stabp/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "M"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm2stabp/predict/

   Predict endpoint for ESM2StabP.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:


      - **items** (*array of objects*, min: 1, max: 8, required) --- Input protein records:

        - **sequence** (*string*, min length: 1, max length: 1022, required) — Protein sequence using AAExtendedPlusExtra character set (includes standard amino acids and "-")


        - **growth_temp** (*int*, range: -20 to 150, optional, default: null) — Optimal growth temperature in Celsius


        - **experimental_condition** (*string*, optional, default: null) — Experimental condition value; allowed values: "cell", "lysate"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2stabp/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "M"
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

        - **melting_temperature** (*float*) — Predicted melting temperature (Tm) in Celsius

        - **is_thermophilic** (*boolean*, optional) — Derived classification; True if predicted melting_temperature > 60°C, otherwise False

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "melting_temperature": 61.979739870772825,
            "is_thermophilic": true
          }
        ]
      }


Performance
-----------

- Predictive accuracy: on the reference test set, ESM2StabP achieves R² ≈ 0.94 and PCC ≈ 0.92 with MAE ≈ 3.4 °C and RMSE ≈ 4.1 °C for melting temperature (Tm) regression, outperforming DeepSTABp (R² ≈ 0.81, PCC ≈ 0.88) and ProTstab2 (R² ≈ 0.51, PCC ≈ 0.68) trained on the same data
- Error characteristics: predictions are approximately unbiased across the evaluated Tm range, with a mild tendency to underestimate at very high Tm; most predicted values lie within a few °C of experimental labels, comparable to uncertainty in many high-throughput TPP assays
- Relative performance vs. other BioLM thermostability / temperature models:
  
  - Compared to TemBERTure Regression (BERT-based, classification-then-regression), ESM2StabP yields lower regression error for single-sequence Tm prediction, particularly when growth temperature and experimental condition are supplied, by combining ESM-2 (t33, 650M) embeddings with these features in a single random forest
  - Compared to TEMPRO 650M / TEMPRO 3B, ESM2StabP provides similar or better accuracy per parameter on classical Tm prediction tasks while using a frozen 650M ESM-2 backbone plus a tree ensemble, avoiding larger transformer passes or generative decoding
  - Compared to general-purpose sequence-property regressors like Pro4S Regression, ESM2StabP is specialized for thermostability and shows substantially higher correlation and lower error on Tm, at the cost of not modeling unrelated properties
- Hardware and scalability: ESM2StabP runs ESM-2 650M on datacenter GPUs (e.g., NVIDIA A10/A100/L4-class) with CPU-side random forest inference; GPU embedding computation is parallelized and tree evaluation is batched, so throughput and latency are close to ESM-2 650M embedding endpoints and significantly faster than structure-based workflows (e.g., AlphaFold2, ESMFold) for stability ranking across large variant sets

Applications
------------

- In-silico screening of large protein variant libraries for improved thermostability in industrial enzymes, by ranking candidate sequences based on predicted melting temperature (*T*m) before committing to expression and purification; this reduces wet-lab burden and cost when optimizing enzymes for high-temperature bioprocesses such as detergent additives, biomass deconstruction, and chemical manufacturing
- Stabilization of biocatalysts used in continuous manufacturing, by using ESM2StabP predictions to prioritize sequence variants that are more likely to remain folded at target reactor temperatures (e.g., 50–80 °C), helping process engineers select constructs that are less likely to denature and foul reactors or require frequent enzyme replenishment; predictions are sequence-based and should be validated in the relevant process buffer
- Thermostability-aware protein engineering campaigns, where directed evolution or generative design workflows produce thousands of variants and ESM2StabP is used as a regression filter to discard sequences predicted to have unacceptably low *T*m, enabling teams to focus experimental testing on designs that better balance activity with stability; this is particularly valuable when only sequence data are available and no structural model or assay miniaturization is feasible
- Formulation and storage risk assessment for protein-based products (e.g., industrial proteins, research reagents), by using predicted *T*m to flag constructs likely to be marginally stable at planned storage or shipping temperatures, guiding decisions such as introducing stabilizing mutations, adjusting cold-chain requirements, or changing buffer conditions; this is most informative for relative ranking across related sequences rather than determining exact shelf-life
- Early-stage feasibility assessments when repurposing known proteins to higher-temperature processes, by computationally comparing candidate homolog sequences and using ESM2StabP to identify those predicted to tolerate the new temperature regime, helping business and R&D teams rapidly decide whether to invest in further engineering versus sourcing alternative proteins; predictions reflect intrinsic sequence-level thermostability and should be confirmed experimentally, especially in non-standard buffer or extreme pH conditions

Limitations
-----------

- **Maximum sequence length**: Each ``sequence`` must be at least 1 amino acid and no more than **1022** residues (``max_length=1022``). Longer proteins must be truncated or split, which can change the predicted ``melting_temperature`` because ESM2StabP does not account for sequence context beyond the first 1022 residues.
- **Batch size and throughput**: Each request can include between 1 and **8** items in ``items`` (``min_length=1``, ``max_length=8``). Very large libraries should be sharded across multiple requests; the model is not designed for single-shot scoring of tens of thousands of sequences in one call.
- **Input feature constraints**: The ``growth_temp`` field is optional and must be between ``-20`` and ``150`` °C when provided. The ``experimental_condition`` field is optional and must be either ``"cell"`` or ``"lysate"``. When these are omitted or approximate, performance can degrade relative to the reported benchmarks, since the underlying model leverages these features when available.
- **Prediction scope and calibration**: The model outputs a single ``melting_temperature`` (Tm in °C) and a derived Boolean ``is_thermophilic`` (``True`` if predicted Tm > 60 °C). These values are continuous predictions from a regression model trained on thermo-proteome profiling data and are not guarantees of experimental Tm; ESM2StabP tends to slightly underestimate Tm and may be less reliable for proteins or organisms that are poorly represented in the training data.
- **Not a local mutational effect or design tool**: ESM2StabP is trained at the whole-protein level and does not model per-residue contributions or ΔTm for point mutations. It is not ideal for fine-grained mutational scanning, sequence ranking within very tight Tm windows, or de novo design evaluation without additional models; use it as a coarse stability filter, not as the sole decision-maker in protein engineering campaigns.
- **Domain and model-architecture limitations**: Predictions depend on ESM2 embeddings and a random forest regressor trained on a combined but still biased dataset (over-representation of non-thermophilic proteins and limited organism diversity). Performance may degrade on atypical sequences (e.g., highly engineered chimeras, fusion proteins, repeats) or very sparse families, and the model does not consider 3D structure, ligands, complexes, or environmental factors such as pH, cofactors, or formulation conditions.

How We Use It
-------------

ESM2StabP enables thermostability-aware protein design by providing calibrated melting temperature predictions that integrate directly into in silico engineering workflows. Teams use ESM2StabP outputs alongside structure-based scores, developability filters, and sequence-embedding models to prioritize variants before synthesis, de-risk stability liabilities, and steer multi-round optimization toward sequence space with favorable stability profiles. Standardized, scalable APIs allow data science and ML engineering teams to embed thermostability scoring into automated pipelines—for example, as a post-filter on generative design outputs, a feature in multi-objective ranking models, or a checkpoint in lab-in-the-loop campaigns—so that wet-lab resources are focused on variants with a stronger probability of meeting stability specifications across enzyme, antibody, and other protein programs.

- Used together with BioLM generative models, ESM2StabP accelerates design-make-test cycles by coupling de novo or local sequence exploration with stability-aware triage and ranking.
- Integrated with other predictive endpoints (e.g., activity, aggregation, liability motifs), ESM2StabP supports portfolio-level decisions such as selecting lead scaffolds, defining stability design targets, and quantifying trade-offs between potency and manufacturability.

Related
-------

- ``TemBERTure Regression`` – Alternative sequence-based *T\ :sub:`m`\* regressor using BERT embeddings; useful for benchmarking or combining thermostability predictions with ``ESM2StabP``.
- ``TemBERTure Classifier`` – Predicts thermophilic vs. non-thermophilic sequences, mirroring the thermophilic feature used in ``ESM2StabP`` when growth temperature labels are unavailable.
- ``ThermoMPNN`` – Structure-aware protein design model that proposes stabilizing mutations; pair with ``ESM2StabP`` to iteratively design and rescore variants for higher predicted *T\ :sub:`m`\*.
- ``ThermoMPNN-D`` – Design-optimized variant of ``ThermoMPNN`` for stability-focused sequence generation, complementary to ``ESM2StabP`` in closed-loop thermostability optimization.

References
----------

- Ramos, M., Jernigan, R. L., & Kilinc, M. (2025). ESMStabP: A Regression Model for Protein Thermostability Prediction. *[Journal name to be inserted]*. https://doi.org/10.XXXX/esmstabp
