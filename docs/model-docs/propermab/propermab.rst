PROPERMAB API
=============

PROPERMAB is an *in silico* antibody developability framework exposed as an API for feature extraction from paired VH/VL sequences. Given heavy and light chain Fv inputs (100–200 aa, batch size 1), it predicts Fv structures via ABodyBuilder2 and computes 34 sequence- and structure-based biophysical descriptors, including charge distribution, hydrophobic and aromatic surface areas and patches, spatial statistics (ANN, Ripley’s K), and charge asymmetry metrics. These features support repertoire-scale filtering, HIC/viscosity model development, and lead optimization workflows.

Predict
-------

Extract ProperMAB developability features for a single IgG1 kappa antibody Fv using 3 structure prediction runs.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="propermab",
                action="predict",
                params={
                  "num_runs": 3,
                  "is_fv": true,
                  "isotype": "igg1",
                  "lc_type": "kappa",
                  "seed": 123
                },
                items=[
                  {
                    "heavy_seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWMGGIIPIFGANYAQKFQGRVTMTADTSTDTAYMELSSLRSEDTAVYYCARWGFGDYFDYWGQGTLVTVSS",
                    "light_seq": "DIQMTQSPSSLSASVGDRVTITCRASQSIVHSNGNTYLEWYQQKPGKAPKLLIYDASTLQSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNTYPYTFGQGTKVEIKRT"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/propermab/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "heavy_seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWMGGIIPIFGANYAQKFQGRVTMTADTSTDTAYMELSSLRSEDTAVYYCARWGFGDYFDYWGQGTLVTVSS",
                  "light_seq": "DIQMTQSPSSLSASVGDRVTITCRASQSIVHSNGNTYLEWYQQKPGKAPKLLIYDASTLQSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNTYPYTFGQGTKVEIKRT"
                }
              ],
              "params": {
                "num_runs": 3,
                "is_fv": true,
                "isotype": "igg1",
                "lc_type": "kappa",
                "seed": 123
              }
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/propermab/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "heavy_seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWMGGIIPIFGANYAQKFQGRVTMTADTSTDTAYMELSSLRSEDTAVYYCARWGFGDYFDYWGQGTLVTVSS",
                      "light_seq": "DIQMTQSPSSLSASVGDRVTITCRASQSIVHSNGNTYLEWYQQKPGKAPKLLIYDASTLQSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNTYPYTFGQGTKVEIKRT"
                    }
                  ],
                  "params": {
                    "num_runs": 3,
                    "is_fv": true,
                    "isotype": "igg1",
                    "lc_type": "kappa",
                    "seed": 123
                  }
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/propermab/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  heavy_seq = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWMGGIIPIFGANYAQKFQGRVTMTADTSTDTAYMELSSLRSEDTAVYYCARWGFGDYFDYWGQGTLVTVSS",
                  light_seq = "DIQMTQSPSSLSASVGDRVTITCRASQSIVHSNGNTYLEWYQQKPGKAPKLLIYDASTLQSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNTYPYTFGQGTKVEIKRT"
                )
              ),
              params = list(
                num_runs = 3,
                is_fv = TRUE,
                isotype = "igg1",
                lc_type = "kappa",
                seed = 123
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/propermab/predict/

   Predict endpoint for PROPERMAB.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **num_runs** (*int*, range: 1-5, default: 1) — Number of structure prediction runs to average for feature calculation

        - **is_fv** (*bool*, default: true) — Indicates whether input sequences are Fv-only domains

        - **isotype** (*string*, allowed: ["igg1", "igg2", "igg4"], default: "igg1") — Heavy chain isotype identifier used for charge-related features

        - **lc_type** (*string*, allowed: ["kappa", "lambda"], default: "kappa") — Light chain type identifier used for charge-related features

        - **seed** (*int*, minimum: 0, default: 42) — Random seed for structure prediction and feature extraction


      - **items** (*array of objects*, min: 1, max: 1) --- Input antibody chains:

        - **heavy_seq** (*string*, min length: 100, max length: 200, required) — Heavy chain amino acid sequence

        - **light_seq** (*string*, min length: 100, max length: 200, required) — Light chain amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/propermab/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "heavy_seq": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWMGGIIPIFGANYAQKFQGRVTMTADTSTDTAYMELSSLRSEDTAVYYCARWGFGDYFDYWGQGTLVTVSS",
            "light_seq": "DIQMTQSPSSLSASVGDRVTITCRASQSIVHSNGNTYLEWYQQKPGKAPKLLIYDASTLQSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNTYPYTFGQGTKVEIKRT"
          }
        ],
        "params": {
          "num_runs": 3,
          "is_fv": true,
          "isotype": "igg1",
          "lc_type": "kappa",
          "seed": 123
        }
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **sequence_features** (*object*) — 7 sequence-based features computed from input sequences

          - **theoretical_pi** (*float*) --- Isoelectric point of the antibody, unitless pH value

          - **n_charged_res** (*int*) --- Total count of charged residues (D,E,K,R) in the antibody sequence

          - **n_charged_res_fv** (*int*) --- Count of charged residues (D,E,K,R) in the Fv domain

          - **fv_charge** (*float*) --- Net charge of the Fv domain at pH 7.4, in elementary charge units

          - **fv_csp** (*float*) --- Product of VH and VL net charges, in squared elementary charge units

          - **fc_charge** (*float*) --- Net charge of the Fc domain at pH 7.4, in elementary charge units

          - **fab_fc_csp** (*float*) --- Product of Fab and Fc net charges, in squared elementary charge units

        - **structure_features** (*object*) — 27 structure-based features computed from predicted 3D Fv structures

          - **net_charge** (*float*) --- Net Fv charge from atomic partial charges, in elementary charge units

          - **exposed_net_charge** (*float*) --- Net charge of solvent-exposed atoms, in elementary charge units

          - **net_charge_cdr** (*float*) --- Net charge of CDR atoms, in elementary charge units

          - **exposed_net_charge_cdr** (*float*) --- Net charge of solvent-exposed CDR atoms, in elementary charge units

          - **scm** (*float*) --- Spatial Charge Map score, squared summed negative exposed charge, unitless

          - **dipole_moment** (*float*) --- Magnitude of Fv electric dipole moment, in Debye

          - **hyd_asa** (*float*) --- Total apolar solvent accessible surface area, in Å²

          - **hph_asa** (*float*) --- Total polar solvent accessible surface area, in Å²

          - **hyd_moment** (*float*) --- Hydrophobic moment magnitude, in Å·(hydrophobicity scale units)

          - **heiden_score** (*float*) --- Sum of positive hydrophobic surface potentials weighted by area, in Å·(lipophilicity units)

          - **hyd_patch_area** (*float*) --- Total area of hydrophobic surface patches above threshold, in Å²

          - **hyd_patch_area_cdr** (*float*) --- Total area of hydrophobic surface patches near CDRs, in Å²

          - **pos_patch_area** (*float*) --- Total area of positively charged surface patches above threshold, in Å²

          - **pos_patch_area_cdr** (*float*) --- Total area of positively charged surface patches near CDRs, in Å²

          - **neg_patch_area** (*float*) --- Total area of negatively charged surface patches above threshold, in Å²

          - **neg_patch_area_cdr** (*float*) --- Total area of negatively charged surface patches near CDRs, in Å²

          - **aromatic_asa** (*float*) --- Solvent accessible surface area of aromatic residues (F,W,Y), in Å²

          - **aromatic_cdr** (*int*) --- Count of aromatic residues (F,W,Y) in CDR regions

          - **exposed_aromatic** (*int*) --- Count of solvent-exposed aromatic residues (F,W,Y)

          - **pos_ann_index** (*float*) --- Average Nearest Neighbor index for positively charged solvent-exposed residues, unitless ratio

          - **neg_ann_index** (*float*) --- Average Nearest Neighbor index for negatively charged solvent-exposed residues, unitless ratio

          - **aromatic_ann_index** (*float*) --- Average Nearest Neighbor index for aromatic solvent-exposed residues, unitless ratio

          - **pos_ripley_k** (*float*) --- Ripley's K ratio for positively charged solvent-exposed residues at fixed distance cutoff, unitless

          - **neg_ripley_k** (*float*) --- Ripley's K ratio for negatively charged solvent-exposed residues at fixed distance cutoff, unitless

          - **aromatic_ripley_k** (*float*) --- Ripley's K ratio for aromatic solvent-exposed residues at fixed distance cutoff, unitless

          - **Fv_chml** (*float*) --- Difference between VH and VL net charges (VH − VL), in elementary charge units

          - **exposed_Fv_chml** (*float*) --- Difference between exposed VH and VL net charges, in elementary charge units

          - **cdr_h3_length** (*int*) --- Length of CDR-H3 loop from IMGT-numbered predicted structure, in residues

        - **metadata** (*object*) --- Metadata about feature extraction for this item

          - **num_runs** (*int*, range: 1–5) --- Number of independent structure prediction runs averaged

          - **isotype** (*string*) --- Heavy chain isotype used for Fc-related charge calculations, one of: "igg1", "igg2", "igg4"

          - **lc_type** (*string*) --- Light chain type used for charge calculations, one of: "kappa", "lambda"

          - **structure_prediction_method** (*string*) --- Structure prediction method identifier, fixed value: "ABodyBuilder2"

          - **feature_calculation_version** (*string*) --- PROPERMAB feature implementation version, fixed value: "propermab-0.1.0"

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_features": {
              "theoretical_pi": 7.019512926254469,
              "n_charged_res": 135,
              "n_charged_res_fv": 36,
              "fv_charge": 0.0,
              "fv_csp": 0.0,
              "fc_charge": -6.0,
              "fab_fc_csp": -18.0
            },
            "structure_features": {
              "net_charge": -0.9999999999999929,
              "exposed_net_charge": -65.24166666666582,
              "net_charge_cdr": -4.000000000000019,
              "exposed_net_charge_cdr": -35.16333333333328,
              "scm": 1071.216666666667,
              "dipole_moment": 174.9291999357993,
              "hyd_asa": 5689.426066084329,
              "hph_asa": 4534.560901240281,
              "hyd_moment": 273.14177571020673,
              "heiden_score": 126.44844806533722,
              "hyd_patch_area": 725.3431298326237,
              "hyd_patch_area_cdr": 290.38202387619174,
              "pos_patch_area": 103.76484225343812,
              "pos_patch_area_cdr": 62.5937337388002,
              "neg_patch_area": 528.6110006976515,
              "neg_patch_area_cdr": 336.1951271125561,
              "aromatic_asa": 967.1367405293745,
              "aromatic_cdr": 28,
              "exposed_aromatic": 17,
              "pos_ann_index": 1.2632999728345642,
              "neg_ann_index": 1.1507340887193036,
              "aromatic_ann_index": 0.9950437900188961,
              "pos_ripley_k": 0.48676706280013543,
              "neg_ripley_k": 0.7136081710438549,
              "aromatic_ripley_k": 1.1111985969663891,
              "Fv_chml": 1.0000000000000018,
              "exposed_Fv_chml": 1.3383333333332956,
              "cdr_h3_length": 11
            },
            "metadata": {
              "num_runs": 3,
              "isotype": "igg1",
              "lc_type": "kappa",
              "structure_prediction_method": "ABodyBuilder2",
              "feature_calculation_version": "propermab-0.1.0"
            }
          }
        ]
      }


Performance
-----------

- Computational pipeline and hardware
  - Sequence-only features are computed on CPU with negligible overhead compared to structure prediction
  - Structure-based features use ABodyBuilder2 on GPU for Fv modeling, followed by CPU-based electrostatics, hydrophobic patching, and spatial statistics; overall latency is dominated by Fv structure prediction
  - End-to-end runtime scales approximately linearly with the number of VH/VL pairs and with ``num_runs`` (1–5 structure predictions per antibody)

- Latency and relative speed vs. other BioLM structural models
  - Wall-clock time for a full sequence+structure feature pass (1 run) is on the order of tens of seconds per antibody, comparable to standalone ABodyBuilder2 Fv prediction
  - At equivalent Fv sizes, this is substantially faster and cheaper than AlphaFold2- or Chai-1-based full-atom antibody pipelines, while being slower than ESMFold-based “quick structure” calls
  - Compared to NanoBodyBuilder-style single-chain models, latency per input is higher because PROPERMAB must co-model VH/VL and compute pairwise spatial statistics across the interface

- Predictive performance for developability endpoints
  - HIC retention time: ElasticNet models trained on PROPERMAB features achieve Pearson ``r ≈ 0.71`` and Spearman ``ρ ≈ 0.75`` on a 135-mAb benchmark, outperforming any single descriptor (e.g., ``hyd_patch_area_cdr``, ``aromatic_asa``) and simple charge/hydropathy summaries from sequence-only BioLM models
  - High-concentration viscosity (IgG4, 150 mg/mL): RandomForest models using PROPERMAB features reach Spearman ``ρ ≈ 0.48`` and Pearson ``r ≈ 0.35`` on a 58-IgG4 dataset, improving on earlier single-score heuristics (e.g., SCM-like metrics) that show weak correlation on this larger panel
  - Within the BioLM model family, PROPERMAB provides a middle ground between fast but less accurate sequence-only screens and more expressive 3D-CNN/MD-derived surrogates that are materially slower per antibody

- Sequence-surrogate features and scaling behavior
  - PROPERMAB feature values computed from structures can be used as targets to train lightweight ElasticNet regressors from IMGT-aligned, one-hot VH/VL sequences
  - On a 12,000-mAb OAS-derived benchmark, these regressors reproduce structure-derived feature values with median Pearson ``r ≈ 0.87``; HIC and viscosity models retrained on sequence-predicted features show only modest performance loss while avoiding structure prediction at inference time
  - Reference implementation benchmarks show that once sequences are numbered and encoded, >140,000 VH/VL pairs can be processed in minutes on a single CPU node, enabling repertoire-scale analyses where full ABodyBuilder2+feature runs would be prohibitively slow

- Reproducibility and numerical stability
  - The ``seed`` parameter in ``ProperMABExtractFeaturesParams`` yields deterministic ABodyBuilder2 structures and downstream features for a fixed software/hardware environment, supporting reproducible ranking and iterative optimization
  - ``num_runs`` (1–5) controls internal ensemble averaging; increasing it reduces variance in structure-derived metrics (e.g., patch areas, Ripley’s K, ANN indices) approximately as expected for repeated structure predictions, at the cost of a proportional increase in compute per antibody

Applications
------------

- Lead candidate triage for therapeutic mAbs based on developability risk, using PROPERMAB-predicted HIC retention times and viscosity-related features to down-select large discovery panels (10²–10³ antibodies) into a smaller set of formulation-ready leads, reducing late-stage failures due to aggregation, poor purification yields, or non-injectable high-concentration behavior
- In silico liability mapping and sequence optimization for mAb manufacturability, by quantifying charge asymmetry, hydrophobic patch area (including CDR-localized hydrophobic patches), aromatic surface area, and spatial clustering of charged or aromatic residues to guide targeted mutations that de-risk high viscosity and hydrophobicity while preserving binding; not optimal as a standalone tool for predicting functional potency or epitope specificity
- Rapid developability screening of large antibody repertoires (10⁴–10⁶ sequences) by first extracting sequence-only charge descriptors and then, where needed, structure-derived PROPERMAB features (e.g., hydrophobic patch area, aromatic solvent-accessible area, spatial charge metrics) for a prioritized subset, enabling high-throughput ranking and filtering prior to resource-intensive structural modeling or experimental biophysics
- Formulation and CMC risk assessment for clinical mAb programs by integrating PROPERMAB’s feature set (HIC-related hydrophobic metrics, viscosity-associated charge asymmetry descriptors such as Fv\_chml and dipole\_moment, SCM) with internal assay data to prioritize molecules compatible with platform purification and high-concentration subcutaneous dosing, flagging antibodies likely to require non-standard excipients or off-platform processes
- Benchmarking and integration of internal antibody developability models and workflows, using PROPERMAB’s transparent, Python-native descriptor layer (charge distributions, hydrophobic and electrostatic surface patches, Ripley’s K statistics, 3D voxelization utilities) as standardized inputs to in-house ML models; performance remains constrained by the size and quality of available biophysical measurements and may be isotype- and assay-condition dependent

Limitations
-----------

- **Batch size and throughput**: Each request's ``items`` list has ``max_length=1`` and ``batch_size`` is fixed at ``1``. Only a single antibody pair is processed per call because 3D structure prediction and feature calculation are compute‑intensive; PROPERMAB is not intended for ultra high‑throughput screening of millions of sequences in a single stage.
- **Sequence length and format**: ``heavy_seq`` and ``light_seq`` must each be amino acid sequences of length ``100–200`` (inclusive) and should correspond to single VH/VL variable domains. Very short/long sequences, multi‑domain constructs, or non‑antibody proteins are rejected or may yield non‑interpretable ``sequence_features`` and ``structure_features``.
- **Scope of structural modeling**: All ``structure_features`` use Fv‑only structures predicted with ABodyBuilder2 (``structure_prediction_method="ABodyBuilder2"``). Constant regions (CH1–CH3, CL) and Fc are not structurally modeled; Fc‑related effects are approximated only via sequence‑level charge features in ``sequence_features`` such as ``fc_charge`` and ``fab_fc_csp``.
- **Developability property coverage and calibration**: The 34 output features (7 ``sequence_features``, 27 ``structure_features``) are generic biophysical descriptors. They were shown in the paper to support models for HIC retention time and high‑concentration viscosity on limited datasets (e.g., 135 mAbs for HIC, 58 IgG4 for viscosity) and are not validated as direct predictors of clinical developability, nor calibrated for new assay formats, isotypes, bispecifics, or non‑IgG scaffolds without user‑supplied data and model training.
- **Isotype and chain‑type assumptions**: ``isotype`` (``igg1``, ``igg2``, ``igg4``) and ``lc_type`` (``kappa``, ``lambda``) are used when computing Fc and domain‑level charge features in ``sequence_features``. These assume standard human IgG architectures; non‑canonical constant regions, Fc fusions, or engineered hinges/CH domains can lead to misleading charge features even when Fv structure prediction is successful.
- **When PROPERMAB is not optimal**: PROPERMAB is a feature extractor, not a generative model. It does not propose new sequences, does not output 3D coordinates suitable for docking or detailed epitope mapping, and—because ``batch_size=1`` and Fv structure prediction dominates runtime—is best used downstream on narrowed candidate sets, often combined with faster sequence‑only or language‑model‑based filters for repertoire‑scale design.

How We Use It
-------------

PROPERMAB enables early-stage, *in silico* developability screening of antibody libraries within BioLM’s generative design and optimization cycles, so candidates are ranked not only by affinity or epitope coverage but also by manufacturability-relevant traits linked to HIC retention time, high-concentration viscosity, aggregation risk, and formulation robustness. Through scalable, standardized APIs, PROPERMAB feature vectors are consumed alongside sequence embeddings, structure-derived metrics, and other biophysical predictors to drive multi-parameter ranking, steer sequence diversification away from problematic surface patches or charge patterns, and focus synthesis and lab testing on variants with stronger probability of surviving CMC and late-stage development.

- Integrates with BioLM antibody generators and structure/PLM-based property models to co-optimize affinity, specificity, and developability within a single ranking and selection framework.
- Exposes PROPERMAB outputs via consistent APIs, enabling automated use in enterprise ML pipelines, project triage dashboards, and lab-in-the-loop optimization campaigns at library and repertoire scale.

Related
-------

- ``ABodyBuilder3 pLDDT`` – High-accuracy antibody Fv structure prediction that can be used upstream of PROPERMAB when experimental structures are unavailable, improving structure-based feature quality for developability models.
- ``ImmuneFold Antibody`` – Alternative antibody structure predictor that can generate ensembles of Fv models for PROPERMAB’s structure featurizer, enabling robustness analyses over multiple conformations.
- ``DeepViscosity`` – A viscosity-focused deep learning model that complements PROPERMAB by providing an independent viscosity predictor; PROPERMAB features can be used to interpret or benchmark DeepViscosity outputs on the same antibody panel.
- ``BioLMSol`` – Sequence-based solubility predictor that can be run alongside PROPERMAB to provide an orthogonal developability readout (apparent solubility) using only sequence, useful for early triage before full structure-feature computation.

References
----------

- Li, B., Luo, S., Wang, W., Xu, J., Liu, D., Shameem, M., Mattila, J., Franklin, M. C., Hawkins, P. G., & Atwal, G. S. (2025). PROPERMAB: an integrative framework for *in silico* prediction of antibody developability using machine learning. *mAbs*, 17, 2474521. https://doi.org/10.1080/19420862.2025.2474521
