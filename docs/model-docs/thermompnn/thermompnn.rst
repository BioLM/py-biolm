ThermoMPNN API
==============

ThermoMPNN is a structure-based ΔΔG predictor for single amino acid substitutions that uses ProteinMPNN graph-neural embeddings and a lightweight attention/MLP head to estimate stability changes from a wild-type 3D backbone, without modeling mutant structures. The API accepts PDB or AlphaFold-style coordinates and optional mutation lists, or performs full site-saturation scans, returning per-mutation ΔΔG (kcal/mol). Typical Pearson correlations are ≈0.75 on Megascale and ≈0.65 on FireProt, with GPU-accelerated inference for proteins up to 1,024 residues.

Predict
-------

Predict ΔΔG values for specified point mutations on a short three-residue peptide backbone in chain A.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="thermompnn",
                action="predict",
                params={
                  "display_name": "ThermoMPNN predict params",
                  "model_slug": "thermompnn-predict-params",
                  "log_identifier": "ThermoMPNNPredictParams",
                  "batch_size": 1,
                  "max_sequence_len": 1024,
                  "chain": "A"
                },
                items=[
                  {
                    "display_name": "ThermoMPNN predict item",
                    "model_slug": "thermompnn-predict-item",
                    "log_identifier": "ThermoMPNNPredictItem",
                    "batch_size": 1,
                    "max_sequence_len": 1024,
                    "pdb": "ATOM      1  N   ALA A   1      11.104  13.207   6.204  1.00 20.00           N\nATOM      2  CA  ALA A   1      12.560  13.280   6.430  1.00 20.00           C\nATOM      3  C   ALA A   1      13.129  11.872   6.744  1.00 20.00           C\nATOM      4  O   ALA A   1      12.497  10.858   6.456  1.00 20.00           O\nATOM      5  N   LEU A   2      14.332  11.823   7.331  1.00 20.00           N\nATOM      6  CA  LEU A   2      15.010  10.541   7.699  1.00 20.00           C\nATOM      7  C   LEU A   2      14.252   9.825   8.829  1.00 20.00           C\nATOM      8  O   LEU A   2      13.994  10.350   9.932  1.00 20.00           O\nATOM      9  N   GLY A   3      13.890   8.603   8.493  1.00 20.00           N\nATOM     10  CA  GLY A   3      13.135   7.785   9.449  1.00 20.00           C\nATOM     11  C   GLY A   3      11.640   8.071   9.382  1.00 20.00           C\nATOM     12  O   GLY A   3      10.936   7.536  10.251  1.00 20.00           O\nTER\nEND\n",
                    "mutations": [
                      "A1V",
                      "L2A",
                      "G3D"
                    ]
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/thermompnn/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "display_name": "ThermoMPNN predict request",
              "model_slug": "thermompnn",
              "log_identifier": "ThermoMPNNPredictRequest",
              "batch_size": 1,
              "max_sequence_len": 1024,
              "params": {
                "display_name": "ThermoMPNN predict params",
                "model_slug": "thermompnn-predict-params",
                "log_identifier": "ThermoMPNNPredictParams",
                "batch_size": 1,
                "max_sequence_len": 1024,
                "chain": "A"
              },
              "items": [
                {
                  "display_name": "ThermoMPNN predict item",
                  "model_slug": "thermompnn-predict-item",
                  "log_identifier": "ThermoMPNNPredictItem",
                  "batch_size": 1,
                  "max_sequence_len": 1024,
                  "pdb": "ATOM      1  N   ALA A   1      11.104  13.207   6.204  1.00 20.00           N\nATOM      2  CA  ALA A   1      12.560  13.280   6.430  1.00 20.00           C\nATOM      3  C   ALA A   1      13.129  11.872   6.744  1.00 20.00           C\nATOM      4  O   ALA A   1      12.497  10.858   6.456  1.00 20.00           O\nATOM      5  N   LEU A   2      14.332  11.823   7.331  1.00 20.00           N\nATOM      6  CA  LEU A   2      15.010  10.541   7.699  1.00 20.00           C\nATOM      7  C   LEU A   2      14.252   9.825   8.829  1.00 20.00           C\nATOM      8  O   LEU A   2      13.994  10.350   9.932  1.00 20.00           O\nATOM      9  N   GLY A   3      13.890   8.603   8.493  1.00 20.00           N\nATOM     10  CA  GLY A   3      13.135   7.785   9.449  1.00 20.00           C\nATOM     11  C   GLY A   3      11.640   8.071   9.382  1.00 20.00           C\nATOM     12  O   GLY A   3      10.936   7.536  10.251  1.00 20.00           O\nTER\nEND\n",
                  "mutations": [
                    "A1V",
                    "L2A",
                    "G3D"
                  ]
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/thermompnn/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "display_name": "ThermoMPNN predict request",
                  "model_slug": "thermompnn",
                  "log_identifier": "ThermoMPNNPredictRequest",
                  "batch_size": 1,
                  "max_sequence_len": 1024,
                  "params": {
                    "display_name": "ThermoMPNN predict params",
                    "model_slug": "thermompnn-predict-params",
                    "log_identifier": "ThermoMPNNPredictParams",
                    "batch_size": 1,
                    "max_sequence_len": 1024,
                    "chain": "A"
                  },
                  "items": [
                    {
                      "display_name": "ThermoMPNN predict item",
                      "model_slug": "thermompnn-predict-item",
                      "log_identifier": "ThermoMPNNPredictItem",
                      "batch_size": 1,
                      "max_sequence_len": 1024,
                      "pdb": "ATOM      1  N   ALA A   1      11.104  13.207   6.204  1.00 20.00           N\nATOM      2  CA  ALA A   1      12.560  13.280   6.430  1.00 20.00           C\nATOM      3  C   ALA A   1      13.129  11.872   6.744  1.00 20.00           C\nATOM      4  O   ALA A   1      12.497  10.858   6.456  1.00 20.00           O\nATOM      5  N   LEU A   2      14.332  11.823   7.331  1.00 20.00           N\nATOM      6  CA  LEU A   2      15.010  10.541   7.699  1.00 20.00           C\nATOM      7  C   LEU A   2      14.252   9.825   8.829  1.00 20.00           C\nATOM      8  O   LEU A   2      13.994  10.350   9.932  1.00 20.00           O\nATOM      9  N   GLY A   3      13.890   8.603   8.493  1.00 20.00           N\nATOM     10  CA  GLY A   3      13.135   7.785   9.449  1.00 20.00           C\nATOM     11  C   GLY A   3      11.640   8.071   9.382  1.00 20.00           C\nATOM     12  O   GLY A   3      10.936   7.536  10.251  1.00 20.00           O\nTER\nEND\n",
                      "mutations": [
                        "A1V",
                        "L2A",
                        "G3D"
                      ]
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/thermompnn/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              display_name = "ThermoMPNN predict request",
              model_slug = "thermompnn",
              log_identifier = "ThermoMPNNPredictRequest",
              batch_size = 1,
              max_sequence_len = 1024,
              params = list(
                display_name = "ThermoMPNN predict params",
                model_slug = "thermompnn-predict-params",
                log_identifier = "ThermoMPNNPredictParams",
                batch_size = 1,
                max_sequence_len = 1024,
                chain = "A"
              ),
              items = list(
                list(
                  display_name = "ThermoMPNN predict item",
                  model_slug = "thermompnn-predict-item",
                  log_identifier = "ThermoMPNNPredictItem",
                  batch_size = 1,
                  max_sequence_len = 1024,
                  pdb = "ATOM      1  N   ALA A   1      11.104  13.207   6.204  1.00 20.00           N
            ATOM      2  CA  ALA A   1      12.560  13.280   6.430  1.00 20.00           C
            ATOM      3  C   ALA A   1      13.129  11.872   6.744  1.00 20.00           C
            ATOM      4  O   ALA A   1      12.497  10.858   6.456  1.00 20.00           O
            ATOM      5  N   LEU A   2      14.332  11.823   7.331  1.00 20.00           N
            ATOM      6  CA  LEU A   2      15.010  10.541   7.699  1.00 20.00           C
            ATOM      7  C   LEU A   2      14.252   9.825   8.829  1.00 20.00           C
            ATOM      8  O   LEU A   2      13.994  10.350   9.932  1.00 20.00           O
            ATOM      9  N   GLY A   3      13.890   8.603   8.493  1.00 20.00           N
            ATOM     10  CA  GLY A   3      13.135   7.785   9.449  1.00 20.00           C
            ATOM     11  C   GLY A   3      11.640   8.071   9.382  1.00 20.00           C
            ATOM     12  O   GLY A   3      10.936   7.536  10.251  1.00 20.00           O
            TER
            END
            ",
                  mutations = list(
                    "A1V",
                    "L2A",
                    "G3D"
                  )
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/thermompnn/predict/

   Predict endpoint for ThermoMPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **chain** (*string*, optional, default: null) — Chain identifier to use for prediction; if null, the first chain in the ``pdb`` structure is used


      - **items** (*array of objects*, required, min length: 1, max length: 1) --- Input structures and mutation specifications:

        - **pdb** (*string*, required, min length: 1, max length: max_pdb_str_len) — Protein structure in PDB format as a single string

        - **mutations** (*array of strings*, optional, default: null) — Mutation codes in format ``WT{position}MUT`` (e.g., ``A100V``); wildtype and mutant amino acids must be in ``ACDEFGHIKLMNPQRSTVWYX`` and the position segment must be numeric

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/thermompnn/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "display_name": "ThermoMPNN predict request",
        "model_slug": "thermompnn",
        "log_identifier": "ThermoMPNNPredictRequest",
        "batch_size": 1,
        "max_sequence_len": 1024,
        "params": {
          "display_name": "ThermoMPNN predict params",
          "model_slug": "thermompnn-predict-params",
          "log_identifier": "ThermoMPNNPredictParams",
          "batch_size": 1,
          "max_sequence_len": 1024,
          "chain": "A"
        },
        "items": [
          {
            "display_name": "ThermoMPNN predict item",
            "model_slug": "thermompnn-predict-item",
            "log_identifier": "ThermoMPNNPredictItem",
            "batch_size": 1,
            "max_sequence_len": 1024,
            "pdb": "ATOM      1  N   ALA A   1      11.104  13.207   6.204  1.00 20.00           N\nATOM      2  CA  ALA A   1      12.560  13.280   6.430  1.00 20.00           C\nATOM      3  C   ALA A   1      13.129  11.872   6.744  1.00 20.00           C\nATOM      4  O   ALA A   1      12.497  10.858   6.456  1.00 20.00           O\nATOM      5  N   LEU A   2      14.332  11.823   7.331  1.00 20.00           N\nATOM      6  CA  LEU A   2      15.010  10.541   7.699  1.00 20.00           C\nATOM      7  C   LEU A   2      14.252   9.825   8.829  1.00 20.00           C\nATOM      8  O   LEU A   2      13.994  10.350   9.932  1.00 20.00           O\nATOM      9  N   GLY A   3      13.890   8.603   8.493  1.00 20.00           N\nATOM     10  CA  GLY A   3      13.135   7.785   9.449  1.00 20.00           C\nATOM     11  C   GLY A   3      11.640   8.071   9.382  1.00 20.00           C\nATOM     12  O   GLY A   3      10.936   7.536  10.251  1.00 20.00           O\nTER\nEND\n",
            "mutations": [
              "A1V",
              "L2A",
              "G3D"
            ]
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

        - **mutation** (*string*) — Single-point mutation in format "WT{position}MUT" (e.g., "A100V")


        - **position** (*int*) — Residue position in the selected chain (1-indexed)


        - **wildtype** (*string*) — Wildtype amino acid single-letter code (one of "ACDEFGHIKLMNPQRSTVWYX")


        - **mutation_aa** (*string*) — Mutant amino acid single-letter code (one of "ACDEFGHIKLMNPQRSTVWYX")


        - **ddg** (*float*, units: kcal/mol) — Predicted change in folding free energy (ΔΔG° = ΔG°\_mutant − ΔG°\_wildtype), typical range ≈ −9.0 to +12.0

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          null
        ]
      }


Performance
-----------

- Predictive accuracy and calibration:
  - Trained on the Megascale dataset, ThermoMPNN achieves state-of-the-art single-mutation ΔΔG prediction: on Megascale (test, ≈28k mutations) PCC ≈ 0.75 with RMSE ≈ 0.71 kcal/mol; on FireProt HF (≈2.6k) PCC ≈ 0.65 with RMSE ≈ 1.55 kcal/mol, substantially improving over using ProteinMPNN log-probabilities alone (PCC ≈ 0.43/0.41, RMSE ≈ 1.30/2.14).
  - Compared with structure-based tools (Rosetta, FoldX, RaSP, ThermoNet) evaluated on the same Megascale and FireProt HF splits, ThermoMPNN has higher correlation and lower error (e.g., Megascale PCC ≈ 0.75 vs. 0.53 Rosetta, 0.40 FoldX, 0.71 RaSP, 0.33 ThermoNet).
  - When trained on the same Megascale split as sequence-only stability models (e.g., PROSTATA), ThermoMPNN reaches higher PCC and lower RMSE on both Megascale and FireProt HF (Megascale PCC ≈ 0.75 vs. ≈ 0.64; FireProt HF ≈ 0.65 vs. ≈ 0.59), with the largest gains for mutations toward hydrophobic residues and away from hydrophilic ones.

- Behavior across mutation types, datasets, and inverse consistency:
  - Stabilizing variants defined as ΔΔG° < −0.5 kcal/mol show positive predictive value (PPV) ≈ 56% on FireProt HF and ≈ 46% on Megascale, exceeding several commonly used tools (e.g., Rosetta, FoldX) under comparable conditions and slightly surpassing Stability Oracle’s reported PPV on similar ProTherm-derived benchmarks.
  - Despite not being explicitly constrained to be antisymmetric, ThermoMPNN’s architecture (separate ΔG predictions for wildtype and mutant followed by subtraction) yields PCC ≥ 0.60 on both direct and inverse directions across multiple inverse-mutation benchmarks (e.g., Ssym, p53, myoglobin), whereas a single-target ΔΔG regressor loses much of this inverse consistency.
  - Dynamic range is set by Megascale training data (roughly −3 to +5 kcal/mol); ThermoMPNN maintains strong correlation within this window (covering ≈ 97% of FireProt HF) but tends to compress extremely stabilizing or destabilizing outliers compared with methods explicitly tuned to broader ranges.

- Dependence on protein size and structure quality:
  - On Megascale and FireProt HF, most proteins < 100 residues exhibit per-protein RMSE < 1.0 kcal/mol, and 13/17 proteins > 100 residues remain below 1.5 kcal/mol, indicating better absolute calibration on small-to-medium domains and preserved rank-ordering for larger single-chain proteins.
  - Trained on AlphaFold-modeled backbones, ThermoMPNN transfers well to experimental structures: for >100 X-ray structures, PCC is ≈ 0.74 using crystals vs. ≈ 0.76 using AlphaFold models; for NMR structures, PCC is ≈ 0.67 vs. ≈ 0.76 on AlphaFold, reflecting typical NMR ensemble variability rather than strong sensitivity to minor backbone differences.
  - For structures with AlphaFold pLDDT > 0.75 or crystal resolution < 3 Å, no clear dependence of performance on structure “quality” is observed, so users can prioritize biologically relevant conformations over minor refinement.

- Relative performance within the BioLM ecosystem and throughput characteristics:
  - Versus ProteinMPNN-family design models (ProteinMPNN, HyperMPNN, Soluble ProteinMPNN), which are generative sequence designers conditioned on structure, ThermoMPNN is a discriminative, structure-aware stability scorer that produces numerically calibrated single-point ΔΔG estimates; a common workflow is to use MPNN variants to propose variants and ThermoMPNN to rank single-site substitutions for experimental testing.
  - Compared to structure predictors (AlphaFold2, ESMFold, Chai-1, NanobodyBuilder2, antibody/TCR builders) that output 3D structure but not ΔΔG, ThermoMPNN operates directly on a single wildtype backbone and evaluates each mutation independently, avoiding mutant structure modeling; in BioLM deployments this yields SSM-scale scans (≈1,900–2,000 single mutants for a ~100-residue domain) in seconds on a modern GPU, rather than minutes–hours for repeated structure prediction or refinement.
  - Relative to large sequence language models used as stability proxies (e.g., ESM-2 650M, Evo 2 1B Base log-likelihood or MLM-based scores), ThermoMPNN offers similar or better correlation on Megascale-style datasets with explicit kcal/mol outputs, eliminating the need for ad hoc score-to-ΔΔG mappings and simplifying downstream ranking pipelines.

Applications
------------

- Rapid in silico triaging of protein point mutations for thermodynamic stability using a single wild-type structure (experimental or high-confidence AlphaFold model), enabling biotech R&D teams to down-select large single-site mutagenesis libraries (∼10³–10⁵ variants) for industrial enzymes or therapeutic proteins based on predicted ΔΔG°; reduces wet-lab screening burden while preserving high-probability stabilizing candidates, but is less reliable for very large stability shifts (> ~5–8 kcal/mol) or proteins without a trustworthy structure

- Stability-aware lead optimization for therapeutic proteins such as Fc-fusion constructs, cytokines, and growth factors by scanning clinically relevant regions (e.g., receptor-binding interfaces, protease-sensitive loops, formulation-exposed surfaces) for point mutations predicted to be neutral or stabilizing in ΔΔG°, so formulation, PK, or manufacturability changes can be evaluated without unintentionally destabilizing the scaffold; leverages structure-aware graph neural network embeddings rather than MSAs, but is not a replacement for full developability, aggregation, or immunogenicity assessments

- Design and ranking of more thermostable variants for industrial and synthetic-biology enzymes (e.g., hydrolases, oxidoreductases, polymer-modifying enzymes) by predicting stabilizing or non-disruptive single amino acid substitutions from the wild-type structure and combining top candidates into multi-mutant designs under additive or near-additive ΔΔG° assumptions; particularly useful in early engineering campaigns to focus combinatorial testing, while recognizing that higher-order epistasis between multiple mutations is not explicitly modeled by single-mutation ΔΔG° predictions

- Optimization of de novo or AI-designed proteins for robustness before scale-up and tech transfer by passing AlphaFold or design-model backbones into ThermoMPNN to identify marginally stable regions (e.g., clusters of strongly destabilizing mutations or positions lacking stabilizing options) and to suggest targeted stabilizing point mutations that can improve expression yield, formulation robustness, or tolerance to process stresses; model performance is strong on small natural and de novo mini-proteins but may be reduced on large, highly flexible, multi-domain constructs or low-confidence structures

- Integration as a GPU-accelerated stability scoring step in automated protein engineering pipelines, where ThermoMPNN performs site-saturation mutagenesis in seconds per structure and returns ΔΔG° scores for all single-site substitutions; these scores can be used to filter, prioritize, or re-rank candidates generated by generative sequence models, structure design tools, or codon-combinatorial design, enabling closed-loop design–build–test–learn workflows that penalize destabilizing mutations while still requiring downstream biophysical and functional validation

Limitations
-----------

- **Maximum sequence length**: ThermoMPNN only scores residues from chains whose length is ≤ ``max_sequence_len`` (currently 1024 residues). If your selected chain in ``pdb`` is longer, requests will fail; split large proteins and score individual domains or chains instead.
- **Single-structure, single-chain per request**: Each ``ThermoMPNNPredictRequest`` must contain exactly one item in ``items`` (``min_length=1, max_length=1``), and only a single chain is scored per item. Use ``params.chain`` to select the chain; if omitted, only the first chain in ``pdb`` is used. ThermoMPNN predictions are conditioned only on the provided monomeric backbone and do not explicitly model inter-chain interfaces, quaternary structure effects, or ligand/partner-induced stability changes beyond what is present in the coordinates.
- **Mutation specification and scope**: ``mutations`` must be a list of strings in the exact ``'WT{position}MUT'`` format (e.g. ``'A100V'``) using the supported alphabet ``"ACDEFGHIKLMNPQRSTVWYX"``; invalid codes or non-numeric positions cause the request to be rejected. If ``mutations`` is omitted, the API performs a site-saturation mutagenesis (SSM) scan over the chain, but only for single substitutions—insertions, deletions, and multiple simultaneous mutations are not supported by this endpoint.
- **Interpretation of ``ddg``**: The response ``ddg`` field is the model-predicted change in folding free energy (ΔΔG, kcal/mol) for each ``mutation`` relative to the ``wildtype`` residue at that 1-indexed ``position``. Values are best calibrated within the dynamic range of the Megascale training data (approximately −3 to +5 kcal/mol); very large stabilizing or destabilizing effects (outside ~±5 kcal/mol) may be under- or over-estimated and should be interpreted cautiously or validated experimentally.
- **Structure and training-data dependence**: ThermoMPNN assumes a reasonably accurate monomeric backbone in the input ``pdb``. It was trained primarily on high-confidence AlphaFold models and small experimental domains (< ~100 residues) with single point mutations. Performance may degrade for low-confidence or highly flexible regions, very large proteins, heavily engineered scaffolds far from the natural/de novo folds seen in training, or mutations expected to cause large conformational changes not present in the wild-type structure.
- **When ThermoMPNN is not optimal**: This endpoint is specialized for ranking single point mutations by stability on a fixed backbone. It is not ideal for: (1) multi-mutant or global sequence design (consider generative sequence models with downstream stability ranking); (2) predicting effects on binding, activity, specificity, or expression when stability is not the main driver; (3) antibody CDR or nanobody stability where antibody-specific structure models may be preferable; or (4) early-stage screening of very diverse backbones where faster sequence-only or coarse-grained models are better suited, with ThermoMPNN used later for focused refinement.

How We Use It
-------------

ThermoMPNN enables teams to prioritize stabilizing point mutations from structure data and use those ΔΔG profiles to drive multi-objective protein engineering. Organizations use it to triage large mutational spaces around key sites, allocate synthesis and assay budgets to variants predicted to improve folding stability, and then combine these ranked sets with BioLM sequence embeddings, structure-based property predictors, and generative design models to co-optimize stability, activity, solubility, and developability across iterative design–make–test cycles.

- Integrates with generative sequence design, antibody optimization, and enzyme engineering workflows to filter and rank model-proposed variants before synthesis.
- Supports automated, high-throughput stability assessment across protein portfolios via scalable, standardized APIs, enabling portfolio-level risk reduction and more predictable timelines for therapeutic, industrial, and diagnostic programs.

Related
-------

- ``ProteinMPNN`` – Parent sequence-design GNN whose structural embeddings ThermoMPNN uses via transfer learning; pair it with ThermoMPNN to propose stabilizing mutations at sites prioritized by ThermoMPNN.
- ``ThermoMPNN-D`` – ThermoMPNN-based model for double mutants (and related variants); use alongside single-mutant ThermoMPNN when analyzing or validating epistatic mutation combinations.
- ``ESM2StabP`` – Sequence-only stability predictor using ESM-2 embeddings; complements ThermoMPNN for ΔΔG°-like scoring when no reliable structure is available or as a structure-independent cross-check.
- ``AlphaFold2`` – Structure prediction model for proteins without experimental structures; generate high-confidence monomer backbones as input to ThermoMPNN to enable structure-based ΔΔG° prediction.

References
----------

- Dieckhaus, H., Brocidiacono, M., Randolph, N. Z., & Kuhlman, B. (2024). Transfer learning to leverage larger datasets for improved prediction of protein stability changes. *Proceedings of the National Academy of Sciences*, 121(6), e2314853121. https://doi.org/10.1073/pnas.2314853121
