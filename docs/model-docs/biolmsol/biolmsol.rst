BioLMSol API
============

BioLMTox is a GPU-accelerated, single-sequence protein toxin classifier fine-tuned from the 650M-parameter ESM-2 model on a unified benchmark dataset spanning bacterial and eukaryotic proteins and peptides (5–15,639 AA), with ambiguously labeled sequences removed. It takes raw amino acid sequences and returns toxin vs non-toxin probabilities, achieving 0.964 accuracy and 0.984 recall on a held-out validation set. Sub-second per-sequence inference supports screening in protein engineering, peptide therapeutics, toxicology, and biosecurity workflows.

Predict
-------

Predict solubility scores at physiological pH with per-residue solubility profiles for each protein sequence

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="biolmsol",
                action="predict",
                params={
                  "ph": 7.0,
                  "include_profile": true
                },
                items=[
                  {
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEER"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/biolmsol/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "ph": 7.0,
                "include_profile": true
              },
              "items": [
                {
                  "sequence": "MKTAYIAKQRQISFVKSHFSRQLEER"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/biolmsol/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "ph": 7.0,
                    "include_profile": true
                  },
                  "items": [
                    {
                      "sequence": "MKTAYIAKQRQISFVKSHFSRQLEER"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/biolmsol/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                ph = 7.0,
                include_profile = TRUE
              ),
              items = list(
                list(
                  sequence = "MKTAYIAKQRQISFVKSHFSRQLEER"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/biolmsol/predict/

   Predict endpoint for BioLMSol.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **ph** (*float*, range: 0.0-14.0, default: 7.0) — pH value for charge calculations

        - **include_profile** (*boolean*, default: false) — Whether to include per-residue solubility profile data in the response


      - **items** (*array of objects*, min: 1, max: 10, required) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Amino acid sequence using extended validated residue codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/biolmsol/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "ph": 7.0,
          "include_profile": true
        },
        "items": [
          {
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEER"
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

        - **solubility_score** (*float*) — Overall solubility score for the input sequence (higher = more soluble)

        - **profile** (*array of floats*, optional, length: sequence length) — Per-residue intrinsic solubility scores (present only if ``params.include_profile`` is true)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "solubility_score": 0.85717,
            "profile": [
              1.822185,
              1.108933,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Model family and architecture: BioLMSol is BioLM's hosted implementation inspired by the CamSol intrinsic solubility model. It is a lightweight, sequence-only regression model (no MSA, structure, or evolutionary profiles) optimized for fast global and optional per-residue solubility scoring.
- Predictive behavior and calibration: Solubility scores are returned as IEEE‑754 32‑bit floats, roughly centered around 0 for neutral/in‑distribution intrinsic solubility, with higher positive values indicating progressively higher intrinsic solubility. Optional per-residue profiles share the same numeric range and calibration.
- Relative predictive performance:
  
  - Compared with general protein language models with generic stability heads (e.g., ESM2-based stability predictors), BioLMSol is more specialized for intrinsic solubility and aggregation‑risk style endpoints and typically provides clearer rank‑ordering of variants for solubility‑focused screens.
  - Compared with structure‑aware solubility optimization workflows such as Soluble ProteinMPNN (which can redesign sequences given a structure), BioLMSol has narrower scope but is substantially faster and better suited for one‑shot scoring of large candidate sets prior to any structure prediction.
- Throughput and scaling characteristics:
  
  - On modern NVIDIA data‑center GPUs (A10/A100 class), BioLMSol can score sequence panels that would require many ESMFold or AlphaFold2 structure predictions in a small fraction of the wall‑clock time, making it practical for 10²–10⁴ sequence campaigns.
  - The stateless sequence‑only architecture parallelizes efficiently across GPU workers; high‑volume users generally observe near‑linear scaling in effective throughput with increased request concurrency, in contrast to more memory‑bound structural or PPI models.

Applications
------------

- High-throughput solubility triage in protein and peptide engineering campaigns
  BioLMSol can rapidly score variant libraries (10⁴–10⁶ sequences, batched via the API) of therapeutic peptides, enzymes, antibodies, or other biologics to flag designs with poor predicted solubility before expression and purification. This reduces wet-lab load by deprioritizing constructs likely to aggregate or express poorly around a chosen pH. It is best used as an early computational filter combined with separate models or assays for potency, stability, and other developability attributes.

- Expression construct design and buffer selection for recombinant protein production
  CROs, CDMOs, and platform teams can integrate BioLMSol into construct registration and build pipelines to evaluate solubility at specific pH values for new protein-coding sequences. By adjusting the pH parameter and, when requested, inspecting per-residue solubility profiles, teams can identify problematic regions, choose more compatible expression buffers, or plan truncations and fusions. Predictions should complement, not replace, empirical small-scale expression tests.

- Preclinical developability assessment for biologic portfolios
  Companies developing protein therapeutics (e.g., enzymes, cytokines, scaffold proteins, antibodies) can apply BioLMSol across candidate panels to highlight constructs with low intrinsic solubility scores that may require formulation work or sequence optimization. This supports portfolio-level risk assessment and helps prioritize which constructs advance to resource-intensive in vivo and formulation studies. The model does not account for all aggregation mechanisms (e.g., excipient effects, interfaces), so it should be combined with orthogonal biophysical assays.

- Solubility-aware generative and optimization loops for protein design
  In workflows where generative models or directed evolution strategies propose new protein or peptide sequences, BioLMSol can be used as an in-the-loop constraint or post-filter to remove low-solubility candidates while preserving sequence diversity. Optional per-residue profiles enable targeted mutation of aggregation-prone segments rather than global redesign. It does not model full pharmacokinetics or immunogenicity, but is effective when paired with other in silico developability predictors.

- Platform-level monitoring and sequence governance for protein design operations
  Organizations running many concurrent protein programs can use BioLMSol to regularly screen design registries and historical constructs for solubility liabilities at relevant pH values. This enables systematic tracking of solubility risk, supports internal quality standards, and can feed dashboards that flag constructs likely to fail during expression or formulation. It is most useful as an advisory signal integrated into review workflows rather than an automatic hard gate.

Limitations
-----------

- **Maximum Sequence Length**: Each ``sequence`` in a ``CamSolPredictRequestItem`` must be 1–``2048`` amino acids long (validated by ``validate_aa_extended``). Longer inputs must be truncated or split upstream, and non‑standard amino acids outside the extended set will be rejected. This can limit use on very large multi‑domain proteins where context beyond ``2048`` residues influences solubility.
- **Batch Size**: A ``CamSolPredictRequest`` can include at most ``10`` items in ``items``. Larger datasets must be split across multiple requests, and per‑request latency scales roughly with the number of sequences. For interactive design loops with very large libraries, BioLMSol is typically better as a mid‑ or late‑stage filter rather than the first pass.
- **Output Scope**: Each result contains an overall ``solubility_score`` and, if ``include_profile=True`` in ``CamSolPredictRequestParams``, a per‑residue ``profile``. These are intrinsic, sequence‑only solubility estimates. They do not model expression host, fusion tags, formulation conditions, post‑translational modifications, aggregation state, or manufacturability and should not be interpreted as direct predictions of yield or developability risk.
- **pH and Condition Dependence**: The only tunable condition is ``ph`` (``0.0``–``14.0``) in ``CamSolPredictRequestParams``. It modulates charge calculations but does not capture buffer composition, ionic strength, temperature, co‑solvents, excipients, or concentration effects. For formulation or developability studies across complex condition spaces, BioLMSol should be paired with experimental screening or more detailed biophysical models.
- **Model Design Assumptions**: BioLMSol operates on single, linear amino‑acid sequences and does not explicitly model 3D structure, oligomerization, membrane insertion, or interaction with partner proteins. It is therefore not sufficient on its own for membrane proteins, large assemblies, or formats where quaternary or higher‑order structure dominates solubility behavior (e.g. full IgG versus isolated domains).
- **Use in Broader Pipelines**: The API is optimized for high‑throughput solubility scoring, not for toxicity, functional annotation, or structure prediction. For tasks such as general toxicity screening, foldability assessment, or large‑scale generative design triage, other BioLM models (e.g. BioLMTox, structure predictors, or embedding‑based ranking models) are often more appropriate as primary filters, with BioLMSol used as a complementary solubility signal.

How We Use It
-------------

BioLMSol integrates into protein design workflows as a standardized solubility screen that can be applied wherever protein sequences are generated, filtered, or ranked. Teams use it to prioritize generative model outputs, therapeutic candidates, and engineered variants at relevant pH values, reducing low-solubility liabilities before expression and formulation. In practice, BioLMSol scores are combined with structure-based tools, developability metrics (charge, aggregation risk, size), and task-specific models (binding, stability, expression) in API-driven pipelines so that sequence libraries can be generated, scored across models, and downselected systematically. Because the same solubility predictor is available as a consistent API across discovery, optimization, and multi-round maturation, organizations can standardize thresholds, compare designs across programs, and link per-sequence and per-residue solubility outputs to assay and LIMS data when closing the design–make–test loop.

- Integrated with generative design, stability prediction, and sequence-embedding tools, BioLMSol enables faster go/no-go decisions for large sequence libraries and iterative optimization campaigns.
- Standardized solubility scores and optional per-residue profiles support portfolio-wide risk management, from early peptide exploration through late-stage variant refinement in antibody, enzyme, and other protein engineering programs.

Related
-------

- ``BioLMTox-2`` – General protein toxin classifier fine-tuned on an improved multi-domain toxin dataset; use alongside BioLMSol to exclude toxic designs before solubility-focused optimization.
- ``SoluProt`` – Sequence-based protein solubility predictor; pair with BioLMSol to pre-screen candidates for baseline solubility before structure-aware refinement.
- ``Soluble ProteinMPNN`` – Sequence-design model biased toward soluble proteins; generate soluble variants and then rank or refine them with BioLMSol solubility predictions.
- ``ESM-2 650M`` – Protein language model family underlying several BioLM models; use to generate or embed protein sequences that can then be evaluated with BioLMSol for solubility and BioLMTox-2 for toxicity.

References
----------

- Challacombe, C. A., & Haas, N. S. (2024). Towards a Dataset for State of the Art Protein Toxin Classification. *bioRxiv*. https://doi.org/10.1101/2024.04.14.589430
