SoluProt API
============

SoluProt predicts the probability that a protein will be expressed in a soluble form in *E. coli* using a gradient boosting classifier trained on a curated, length-balanced subset of the TargetTrack database and evaluated on an independent NESG benchmark (AUC 0.62, 58.5% accuracy on a balanced 3 100-sequence test set). The API accepts unambiguous amino acid sequences (20–5 000 residues, up to 100 per request) and returns a calibrated solubility probability and binary call, enabling sequence ranking and filtering in high-throughput enzyme discovery and protein engineering workflows.

Predict
-------

Predict probability of soluble expression in E. coli for multiple protein sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="soluprot",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MHHHHHHSSGVDLGTENLYFQSMASMTGGQQMGRGSEFDDDDKAMADIGSEF"
                  },
                  {
                    "sequence": "MKKTAIAIAVALAGFATAQAADKDKKAVVNAAEKLAAEAGADKATVTKDLGAAE"
                  },
                  {
                    "sequence": "MNTQYENLQEFKQALDLAGLDGVVTVELHDTDGKTLVANAVKALGDVVVSTGNT"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/soluprot/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MHHHHHHSSGVDLGTENLYFQSMASMTGGQQMGRGSEFDDDDKAMADIGSEF"
                },
                {
                  "sequence": "MKKTAIAIAVALAGFATAQAADKDKKAVVNAAEKLAAEAGADKATVTKDLGAAE"
                },
                {
                  "sequence": "MNTQYENLQEFKQALDLAGLDGVVTVELHDTDGKTLVANAVKALGDVVVSTGNT"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/soluprot/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MHHHHHHSSGVDLGTENLYFQSMASMTGGQQMGRGSEFDDDDKAMADIGSEF"
                    },
                    {
                      "sequence": "MKKTAIAIAVALAGFATAQAADKDKKAVVNAAEKLAAEAGADKATVTKDLGAAE"
                    },
                    {
                      "sequence": "MNTQYENLQEFKQALDLAGLDGVVTVELHDTDGKTLVANAVKALGDVVVSTGNT"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/soluprot/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MHHHHHHSSGVDLGTENLYFQSMASMTGGQQMGRGSEFDDDDKAMADIGSEF"
                ),
                list(
                  sequence = "MKKTAIAIAVALAGFATAQAADKDKKAVVNAAEKLAAEAGADKATVTKDLGAAE"
                ),
                list(
                  sequence = "MNTQYENLQEFKQALDLAGLDGVVTVELHDTDGKTLVANAVKALGDVVVSTGNT"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/soluprot/predict/

   Predict endpoint for SoluProt.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*integer*, range: 1-100, default: 100) — Maximum number of sequences allowed per request

        - **max_sequence_len** (*integer*, range: 20-5000, default: 5000) — Maximum allowed sequence length in amino acids

        - **min_sequence_len** (*integer*, range: 20-5000, default: 20) — Minimum allowed sequence length in amino acids


      - **items** (*array of objects*, min: 1, max: 100) --- Input sequences:

        - **sequence** (*string*, min length: 20, max length: 5000, required) — Protein sequence using unambiguous standard amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/soluprot/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MHHHHHHSSGVDLGTENLYFQSMASMTGGQQMGRGSEFDDDDKAMADIGSEF"
          },
          {
            "sequence": "MKKTAIAIAVALAGFATAQAADKDKKAVVNAAEKLAAEAGADKATVTKDLGAAE"
          },
          {
            "sequence": "MNTQYENLQEFKQALDLAGLDGVVTVELHDTDGKTLVANAVKALGDVVVSTGNT"
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

        - **soluble** (*float*, range: 0.0-1.0) — Predicted probability of soluble expression in Escherichia coli

        - **is_soluble** (*boolean*) — Binary soluble expression classification at threshold ≥ 0.5

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "soluble": 0.6103,
            "is_soluble": true
          },
          {
            "soluble": 0.4192,
            "is_soluble": false
          },
          {
            "soluble": 0.6313,
            "is_soluble": true
          }
        ]
      }


Performance
-----------

- Predictive task and outputs:
  
  - Estimates the probability that a protein will be both expressed and recovered in soluble form in *E. coli*, based solely on primary sequence
  - Model class: gradient boosting machine over 96 engineered sequence features (composition, physicochemical descriptors, transmembrane propensity, homology to soluble *E. coli* proteins)

- Benchmark predictive accuracy (independent NESG-based SoluProt test set; 3,100 sequences; balanced classes; ≤25% global identity to training set):
  
  - Area under ROC curve (AUC): 0.62
  - Accuracy: 58.5% at probability threshold 0.5; Matthews correlation coefficient (MCC): 0.17
  - Confusion matrix: 939 TP, 873 TN, 677 FP, 611 FN

- Comparative accuracy versus other sequence-based solubility predictors on the same test set:
  
  - SoluProt achieved the highest AUC (0.62) and accuracy (58.5%) among 12 tools evaluated (including PROSO II, SWI, CamSol, DeepSol, SKADE)
  - Deep learning methods DeepSol and SKADE underperformed SoluProt (AUC ≈ 0.55–0.51; accuracies 52.9% and 49.2%), despite substantial training-set overlap with the SoluProt test data, indicating stronger generalization from SoluProt’s curated training set

- Practical enrichment and role in BioLM workflows:
  
  - Ranking sequences by SoluProt score and keeping only the top 10% yields ~49.7% more truly soluble proteins than random selection on a balanced set (232 vs. 155), making it effective for hit prioritization even with <60% absolute accuracy
  - Relative to BioLM transformer-based solubility models (for example BioLMSol, TemBERTure variants), SoluProt is narrower in scope but typically cheaper and faster per sequence and provides better enrichment for *E. coli* soluble-expression campaigns; stability-oriented models (for example TemBERTure Regression, ESM2StabP) are complementary because they estimate thermodynamic stability rather than expression outcome

Applications
------------

- Pre-screening enzyme variant libraries for soluble overexpression in *E. coli* before DNA synthesis or assembly, using SoluProt probabilities to focus cloning and screening on variants with higher predicted soluble expression and to deprioritize sequences likely to form inclusion bodies or fail to express
- Selecting soluble orthologs or homologs from metagenomic or public sequence databases as starting points for industrial enzyme development (e.g., hydrolases, oxidoreductases, transferases), by ranking thousands of candidates with SoluProt scores to reduce failed *E. coli* expression campaigns in discovery or scale-up
- Prioritizing soluble protein constructs (e.g., domain boundaries, truncations, tag-only fusions) for structural biology and assay reagent production, by comparing SoluProt outputs across construct designs to reduce the number of *E. coli* expression trials required to obtain milligram-scale soluble protein for biophysical or functional assays
- Filtering computationally designed or ML-generated enzyme sequences for predicted soluble expression in *E. coli*, using SoluProt as a downstream gate in in silico design pipelines to discard low-solubility designs before ordering DNA, with the understanding that the gradient-boosted model is trained specifically on *E. coli* intracellular expression and does not capture alternative hosts or fermentation conditions
- Portfolio-level triage of protein engineering projects that assume *E. coli* intracellular expression (e.g., early feasibility for new industrial biocatalysts), by aggregating SoluProt predictions across proposed targets to estimate relative expression risk and prioritize programs, noting that the model is not intended for membrane proteins, secreted proteins, or non-*E. coli* hosts

Limitations
-----------

- **Input size and format**: Each request must provide one or more protein sequences in ``items``, where each element is a ``SoluProtSequenceItem`` with a single ``sequence`` field containing only unambiguous amino acid codes. Individual sequences must be between ``20`` and ``5000`` residues (inclusive); shorter or longer inputs are rejected.
- **Batching constraints**: The ``items`` list in ``SoluProtPredictRequest`` must contain at least ``1`` and at most ``100`` sequences per call. Larger datasets must be split across multiple requests and batched client-side.
- **Output interpretation**: For each input sequence, the API returns a ``SoluProtPredictResult`` with ``soluble`` (a float between 0 and 1) and ``is_soluble`` (a boolean using a fixed ``0.5`` decision threshold). The ``soluble`` score is calibrated for ranking and prioritization rather than as an absolute probability of successful expression; use it comparatively within a dataset, not as a hard pass/fail guarantee.
- **Organism and expression-system specificity**: SoluProt is trained to predict soluble overexpression in *Escherichia coli* only. Predictions may be misleading for other hosts (for example, yeast, insect, mammalian, or cell‑free systems) or for conditions that differ substantially from typical *E. coli* pipelines (for example, unusual fusion tags, extreme temperatures, specialized chaperone systems).
- **Structural and protein-type coverage**: The model operates on primary sequence and was trained after filtering out transmembrane proteins, very short fragments, and sequences with undefined residues. It is not optimized for membrane proteins, secreted or complex multidomain constructs, heavily disordered fusions, or antibody-like scaffolds; for these, structure-aware or domain-specific models are often more appropriate, or SoluProt should be used only as an early coarse filter.
- **Predictive performance and role in pipelines**: On an independent balanced test set, reported accuracy is about ``58.5%`` with substantial false positives and false negatives. SoluProt is best used as an enrichment step in large-scale pipelines (for example, ranking many candidates and selecting a top fraction for experimental testing), not as the sole decision-maker for individual high‑value constructs; for critical designs, pair it with other biophysical, structural, and task-specific models.

How We Use It
-------------

BioLM uses SoluProt as a standardized solubility filter in protein design and optimization workflows to prioritize E. coli–expressible candidates before investing in cloning, expression, and assay campaigns. SoluProt scores are combined with BioLM sequence-embedding models, structural predictors, and developability/liability metrics in multi-parameter ranking schemes for enzyme discovery, antibody engineering, and iterative lead optimization. Via scalable APIs, SoluProt predictions plug into automated ML pipelines that generate, score, and narrow thousands of variants per cycle, so experimental teams focus on constructs with higher probability of soluble expression while data and ML teams refine sequence design models using feedback from lab results.

- In enzyme mining and engineering, SoluProt integrates with generative sequence models and functional predictors to exclude low-solubility sequences early, raising expression-screen hit rates and reducing re-synthesis cycles.
- In antibody and biologics optimization, SoluProt complements structure-based aggregation and stability metrics with an orthogonal, sequence-level solubility signal that feeds into composite developability scores for portfolio and project decisions.

Related
-------

- ``BioLMSol`` – Sequence-based solubility predictor that can be used alongside SoluProt to cross-check E.coli solubility estimates or to prioritize constructs by consensus solubility scores.
- ``Soluble ProteinMPNN`` – Sequence design model that optimizes both foldability and solubility; use SoluProt to select soluble natural candidates, then apply Soluble ProteinMPNN to design more soluble variants.
- ``ThermoMPNN`` – Predicts and optimizes protein thermostability; combine with SoluProt to choose variants that are both soluble in E.coli and have favorable stability profiles for expression and manufacturing.
- ``ESM-2 650M`` – General protein language model that can generate or score diverse sequence variants; use it to explore sequence space, then filter candidates with SoluProt to retain variants likely to express solubly in E.coli.

References
----------

- Hon, J., Marusiak, M., Martinek, T., Kunka, A., Zendulka, J., Bednar, D., & Damborsky, J. (2021). SoluProt: prediction of soluble protein expression in *Escherichia coli*. *Bioinformatics*, 37(10), 1559–1566. https://doi.org/10.1093/bioinformatics/btaa1102
