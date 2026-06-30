DeepViscosity API
=================

DeepViscosity is an ensemble deep learning classifier for high-concentration monoclonal antibody viscosity that predicts low (≤20 cP) vs high (>20 cP) viscosity at 150 mg/mL directly from paired Fv VH/VL sequences. The service computes 30 DeepSP-derived surface descriptors (charge and hydrophobicity across Fv/CDR regions) and evaluates a 102-model ANN ensemble, reporting class labels with mean and standard deviation of predicted probabilities and achieving ~88–90% accuracy on independent test sets. It supports batch prediction for up to 10 antibodies per request for sequence-level screening and risk assessment.

Predict
-------

Classify viscosity for two antibody Fv pairs and return ensemble probability plus 30 DeepSP spatial property features for each item.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="deepviscosity",
                action="predict",
                params={
                  "include_deepsp_features": true
                },
                items=[
                  {
                    "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGRGYFDYWGQGTLVTVSS",
                    "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQGISNNLHWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"
                  },
                  {
                    "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMNWVKQRPGQGLEWIGNINPYNGGTNYNEKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARERGGDYAMDYWGQGTLVTVSS",
                    "light_chain": "EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQHYDYPLTFGGGTKLEIK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/deepviscosity/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include_deepsp_features": true
              },
              "items": [
                {
                  "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGRGYFDYWGQGTLVTVSS",
                  "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQGISNNLHWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"
                },
                {
                  "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMNWVKQRPGQGLEWIGNINPYNGGTNYNEKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARERGGDYAMDYWGQGTLVTVSS",
                  "light_chain": "EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQHYDYPLTFGGGTKLEIK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/deepviscosity/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include_deepsp_features": true
                  },
                  "items": [
                    {
                      "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGRGYFDYWGQGTLVTVSS",
                      "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQGISNNLHWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"
                    },
                    {
                      "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMNWVKQRPGQGLEWIGNINPYNGGTNYNEKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARERGGDYAMDYWGQGTLVTVSS",
                      "light_chain": "EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQHYDYPLTFGGGTKLEIK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/deepviscosity/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include_deepsp_features = TRUE
              ),
              items = list(
                list(
                  heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGRGYFDYWGQGTLVTVSS",
                  light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQGISNNLHWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"
                ),
                list(
                  heavy_chain = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMNWVKQRPGQGLEWIGNINPYNGGTNYNEKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARERGGDYAMDYWGQGTLVTVSS",
                  light_chain = "EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQHYDYPLTFGGGTKLEIK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/deepviscosity/predict/

   Predict endpoint for DeepViscosity.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Prediction parameters:

        - **include_deepsp_features** (*boolean*, default: false) — Include 30 DeepSP spatial property features in the prediction response


      - **items** (*array of objects*, min: 1, max: 10) --- Antibody inputs:

        - **heavy_chain** (*string*, min length: 50, max length: 200, required) — Heavy chain variable region (VH) Fv amino acid sequence

        - **light_chain** (*string*, min length: 50, max length: 200, required) — Light chain variable region (VL) Fv amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/deepviscosity/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include_deepsp_features": true
        },
        "items": [
          {
            "heavy_chain": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGRGYFDYWGQGTLVTVSS",
            "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQGISNNLHWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"
          },
          {
            "heavy_chain": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMNWVKQRPGQGLEWIGNINPYNGGTNYNEKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARERGGDYAMDYWGQGTLVTVSS",
            "light_chain": "EIVLTQSPATLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQHYDYPLTFGGGTKLEIK"
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

        - **viscosity_class** (*string*) — Predicted viscosity class label, either ``"low"`` (≤20 cP at 150 mg/mL) or ``"high"`` (>20 cP at 150 mg/mL)


        - **probability_mean** (*float*, range: 0.0–1.0) — Mean predicted probability of the high-viscosity class across 102 ensemble models


        - **probability_std** (*float*, ≥0.0) — Standard deviation of predicted high-viscosity probabilities across 102 ensemble models


        - **is_high_viscosity** (*bool*) — ``true`` if ``probability_mean`` ≥ 0.5, otherwise ``false``


        - **deepsp_features** (*object*, optional) — 30 DeepSP spatial property features as scalar values when ``include_deepsp_features`` is ``true`` in the request:

          - **SAP_pos_CDRH1** (*float*) — SAP_pos descriptor for CDRH1
          - **SAP_pos_CDRH2** (*float*) — SAP_pos descriptor for CDRH2
          - **SAP_pos_CDRH3** (*float*) — SAP_pos descriptor for CDRH3
          - **SAP_pos_CDRL1** (*float*) — SAP_pos descriptor for CDRL1
          - **SAP_pos_CDRL2** (*float*) — SAP_pos descriptor for CDRL2
          - **SAP_pos_CDRL3** (*float*) — SAP_pos descriptor for CDRL3
          - **SAP_pos_CDR** (*float*) — SAP_pos descriptor aggregated over all CDRs
          - **SAP_pos_Hv** (*float*) — SAP_pos descriptor for heavy-chain variable region
          - **SAP_pos_Lv** (*float*) — SAP_pos descriptor for light-chain variable region
          - **SAP_pos_Fv** (*float*) — SAP_pos descriptor for Fv region
          - **SCM_neg_CDRH1** (*float*) — SCM_neg descriptor for CDRH1
          - **SCM_neg_CDRH2** (*float*) — SCM_neg descriptor for CDRH2
          - **SCM_neg_CDRH3** (*float*) — SCM_neg descriptor for CDRH3
          - **SCM_neg_CDRL1** (*float*) — SCM_neg descriptor for CDRL1
          - **SCM_neg_CDRL2** (*float*) — SCM_neg descriptor for CDRL2
          - **SCM_neg_CDRL3** (*float*) — SCM_neg descriptor for CDRL3
          - **SCM_neg_CDR** (*float*) — SCM_neg descriptor aggregated over all CDRs
          - **SCM_neg_Hv** (*float*) — SCM_neg descriptor for heavy-chain variable region
          - **SCM_neg_Lv** (*float*) — SCM_neg descriptor for light-chain variable region
          - **SCM_neg_Fv** (*float*) — SCM_neg descriptor for Fv region
          - **SCM_pos_CDRH1** (*float*) — SCM_pos descriptor for CDRH1
          - **SCM_pos_CDRH2** (*float*) — SCM_pos descriptor for CDRH2
          - **SCM_pos_CDRH3** (*float*) — SCM_pos descriptor for CDRH3
          - **SCM_pos_CDRL1** (*float*) — SCM_pos descriptor for CDRL1
          - **SCM_pos_CDRL2** (*float*) — SCM_pos descriptor for CDRL2
          - **SCM_pos_CDRL3** (*float*) — SCM_pos descriptor for CDRL3
          - **SCM_pos_CDR** (*float*) — SCM_pos descriptor aggregated over all CDRs
          - **SCM_pos_Hv** (*float*) — SCM_pos descriptor for heavy-chain variable region
          - **SCM_pos_Lv** (*float*) — SCM_pos descriptor for light-chain variable region
          - **SCM_pos_Fv** (*float*) — SCM_pos descriptor for Fv region

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "viscosity_class": "low",
            "probability_mean": 0.265236,
            "probability_std": 0.104694,
            "is_high_viscosity": false,
            "deepsp_features": {
              "SAP_pos_CDRH1": 2.46186,
              "SAP_pos_CDRH2": -0.3928,
              "SAP_pos_CDRH3": 3.460679,
              "SAP_pos_CDRL1": 0.486934,
              "SAP_pos_CDRL2": 10.339156,
              "SAP_pos_CDRL3": 5.498152,
              "SAP_pos_CDR": 23.083727,
              "SAP_pos_Hv": 46.939518,
              "SAP_pos_Lv": 41.697124,
              "SAP_pos_Fv": 88.750946,
              "SCM_neg_CDRH1": 23.927179,
              "SCM_neg_CDRH2": 6.452896,
              "SCM_neg_CDRH3": 12.201435,
              "SCM_neg_CDRL1": 37.813705,
              "SCM_neg_CDRL2": 13.493529,
              "SCM_neg_CDRL3": 85.697136,
              "SCM_neg_CDR": 179.130112,
              "SCM_neg_Hv": 273.16571,
              "SCM_neg_Lv": 392.668854,
              "SCM_neg_Fv": 672.63269,
              "SCM_pos_CDRH1": 45.187698,
              "SCM_pos_CDRH2": 13.92766,
              "SCM_pos_CDRH3": 97.983856,
              "SCM_pos_CDRL1": 47.607475,
              "SCM_pos_CDRL2": 47.988979,
              "SCM_pos_CDRL3": 30.018084,
              "SCM_pos_CDR": 283.50943,
              "SCM_pos_Hv": 1105.546875,
              "SCM_pos_Lv": 1030.116455,
              "SCM_pos_Fv": 2102.974121
            }
          },
          {
            "viscosity_class": "low",
            "probability_mean": 0.238198,
            "probability_std": 0.095384,
            "is_high_viscosity": false,
            "deepsp_features": {
              "SAP_pos_CDRH1": 2.773432,
              "SAP_pos_CDRH2": 3.02092,
              "SAP_pos_CDRH3": 4.208038,
              "SAP_pos_CDRL1": 1.90328,
              "SAP_pos_CDRL2": 0.759985,
              "SAP_pos_CDRL3": 4.995734,
              "SAP_pos_CDR": 17.638361,
              "SAP_pos_Hv": 41.066288,
              "SAP_pos_Lv": 35.349831,
              "SAP_pos_Fv": 76.615189,
              "SCM_neg_CDRH1": 15.759984,
              "SCM_neg_CDRH2": 11.856413,
              "SCM_neg_CDRH3": 84.668167,
              "SCM_neg_CDRL1": 89.459991,
              "SCM_neg_CDRL2": 9.934968,
              "SCM_neg_CDRL3": 119.548988,
              "SCM_neg_CDR": 341.085419,
              "SCM_neg_Hv": 357.602875,
              "SCM_neg_Lv": 445.471497,
              "SCM_neg_Fv": 813.111694,
              "SCM_pos_CDRH1": 73.222046,
              "SCM_pos_CDRH2": 46.2756,
              "SCM_pos_CDRH3": 39.836891,
              "SCM_pos_CDRL1": 15.667709,
              "SCM_pos_CDRL2": 63.980923,
              "SCM_pos_CDRL3": 6.89572,
              "SCM_pos_CDR": 242.263306,
              "SCM_pos_Hv": 1139.001465,
              "SCM_pos_Lv": 1085.151001,
              "SCM_pos_Fv": 2209.423096
            }
          }
        ]
      }


Performance
-----------

- Predictive accuracy and calibration
  - Internal AstraZeneca DV\_mAb\_229 dataset (229 mAbs, LOGO over 102 sequence clusters): mean training accuracy 80.7 %, mean leave-one-group-out validation accuracy 88.2 %
  - Independent external tests at 150 mg/mL: Lai\_mAb\_16 accuracy 87.5 % (AUPRC 0.90), Apgar\_mAb\_38 accuracy 89.5 %
  - Ensemble-averaged ``probability_mean`` and its ``probability_std`` provide a calibrated class probability and an empirical uncertainty proxy derived from 102 independent ANN members

- Ensemble architecture and deployment characteristics
  - Each ensemble member is a fully connected ANN over 30 DeepSP features with four hidden layers (128/64/32/16, ``tanh``) and a sigmoid output; all 102 members share the same 30-D input per antibody
  - On GPU, all ensemble members are evaluated as a single batched forward pass per layer, so latency scales approximately linearly with the number of antibodies in the request, not with the number of ensemble members
  - Compared with MD- or structure-based pipelines for SCM/SAP, the DeepSP+DeepViscosity stack replaces tens of nanoseconds of per-antibody MD with milliseconds-scale inference and uses a small fraction of the memory/runtime of structure-prediction models such as AlphaFold2 or ABodyBuilder3

- Comparative predictive performance vs other viscosity or developability models
  - Against DeepSCM (sequence-to-SCM surrogate): DeepViscosity improves classification accuracy from 75.0 % to 87.5 % on Lai\_mAb\_16 and from 86.8 % to 89.5 % on Apgar\_mAb\_38 by combining electrostatics (SCM) and hydrophobicity (SAP) via DeepSP features
  - Against Sharma charge/hydrophobicity rules and TAP developability flags: DeepViscosity is 18–31 percentage points more accurate than TAP and up to ~19 percentage points more accurate than Sharma on Lai\_mAb\_16, while matching Sharma on the more homogeneous Apgar\_mAb\_38 set
  - Compared with kD-based classification on DV\_mAb\_229 (79 % accuracy using kD ≤ −10 mL/g as “high viscosity”), DeepViscosity achieves higher LOGO validation accuracy (88.2 %) using only sequence-derived DeepSP descriptors, capturing viscosity-relevant cases where kD mislabels opalescent but low-viscosity antibodies

- Robustness, sequence diversity, and uncertainty
  - Training sequences are clustered by concatenated VH+VL Fv Levenshtein distance into 102 groups; leave-one-group-out training forces generalization across sequence families rather than memorization of near-clonal variants
  - The ensemble design reduces variance relative to a single ANN, improving external accuracy and stabilizing predictions across homologous antibodies
  - The returned ``probability_std`` highlights cases near the decision boundary or outside the training distribution, providing a practical signal for prioritizing experimental rheology follow-up compared with single-model classifiers used in some other BioLM antibody-property tools

Applications
------------

- In silico screening of monoclonal antibody (mAb) viscosity at clinical concentrations (formulated around 150 mg/mL in histidine-based buffers), enabling formulation and developability teams to deprioritize Fv sequences likely to exceed ~20 cP and focus limited biophysical and rheology assays on candidates with a higher probability of acceptable syringeability and manufacturability when no purified material or structures exist
- Sequence-based design-space exploration for subcutaneous mAb products, where protein engineers generate large panels of CDR and framework variants and use DeepViscosity classifications to map low- vs high-viscosity regions of sequence space, guiding targeted mutations in regions associated with charge and hydrophobic patches (captured via DeepSP-derived features) toward variants that are more compatible with high-concentration SC injection while maintaining potency and specificity
- Portfolio-level viscosity risk assessment for antibody pipelines, allowing organizations to submit all preclinical or pre-lead Fv sequences through the API to flag programs with a higher probability of problematic viscosity at ~150 mg/mL histidine conditions before committing to scale-up and fill–finish investments, informing program prioritization, resource allocation, and the need for alternative leads or formats
- Integration into computational developability and CMC workflows for therapeutic antibodies, where DeepViscosity viscosity classes and ensemble probabilities are combined with other in silico metrics (e.g., aggregation or solubility scores from separate tools) to implement automated gating rules that block sequence submissions predicted to be high viscosity, reducing downstream re-engineering cycles for candidates intended for high-concentration liquid dosage forms
- Rapid triage of legacy, life-cycle management, or biosimilar antibody variants when evaluating higher-dose or subcutaneous presentations by using DeepViscosity to estimate viscosity risk from Fv sequence alone instead of running new molecular dynamics or diffusion interaction (kD) experiments, with the important limitation that current predictions are trained on IgG-like mAbs in histidine-based buffers at 150 mg/mL and may not generalize to non-IgG isotypes, highly engineered multispecific formats, or formulations with strong buffer/excipient effects

Limitations
-----------

- **Sequence and batch size constraints**: Each ``heavy_chain`` and ``light_chain`` Fv sequence must be between ``50`` and ``200`` amino acids (``min_sequence_len`` / ``max_sequence_len``). Each ``DeepViscosityPredictRequest`` must contain at least ``1`` and at most ``10`` items (``batch_size``). Requests with sequences or batch sizes outside these bounds are rejected.
- **Binary, class-level output only**: The API returns a viscosity class label (``viscosity_class`` = ``"low"`` for ``<=20 cP`` or ``"high"`` for ``>20 cP`` at 150 mg/mL), along with ensemble statistics (``probability_mean``, ``probability_std``, ``is_high_viscosity``). It does not provide numeric viscosity values or concentration–viscosity curves; the probabilities are confidence measures for this binary classification, not quantitative viscosity predictions.
- **Scope of applicability (mAbs, Fv-only)**: DeepViscosity is trained on monoclonal antibody variable regions (Fv) provided as separate ``heavy_chain`` (VH) and ``light_chain`` (VL) sequences. It does not capture isotype or Fc-region effects (e.g., IgA, IgM, IgG subclasses), non-Ig scaffolds, highly engineered/bispecific architectures, or antibody–drug conjugates. For these modalities, treat API outputs as lower confidence and corroborate with orthogonal developability assessments.
- **Buffer and formulation assumptions**: Training data were measured at 150 mg/mL in 20 mM histidine-HCl, pH 6.0. Predictions may be less reliable for substantially different buffers, concentrations, pH/ionic strength, or excipient systems (e.g., high sucrose), and should not be assumed to generalize to all subcutaneous or IV formulations without experimental verification.
- **Sequence distribution and data bias**: The training set (229 mAbs) is diverse but imbalanced (more low- than high-viscosity examples) and underrepresents closely related panels (many variants from a single parental antibody). Accuracy may degrade for highly homologous panels or sequences far from this distribution; in such cases, use ``probability_std`` to flag uncertain calls and prioritize experimental or orthogonal in silico follow-up.
- **Model role in development pipelines**: DeepViscosity is intended for early- to mid-stage screening and ranking of candidate mAbs, not as a release criterion or replacement for rheology. It is not suitable as the sole decision-maker for late-stage formulation, device compatibility, or regulatory submissions; integrate API outputs with other in silico models (e.g., aggregation, solubility, clearance) and experimental assays for robust developability decisions.

How We Use It
-------------

DeepViscosity predictions are used as an early developability gate in antibody engineering programs, enabling teams to classify viscosity risk directly from VH/VL Fv sequences before committing to cloning, expression, and formulation. Viscosity classes from DeepViscosity are integrated with DeepSP-derived spatial descriptors, sequence embeddings, structure-aware biophysical models, and BioLM aggregation, charge, solubility, and immunogenicity predictors to support multi-objective antibody design, in silico maturation, and portfolio-level triage via scalable, standardized APIs. This helps research and process development groups prioritize low-viscosity variants for experimental testing, focus rheology assays on the most informative candidates, and run iterative cycles where new viscosity data are used to refine program-specific models.

- Integrated into generative and optimization workflows to bias sequence proposals toward low-viscosity space while co-optimizing affinity, stability, and manufacturability.  
- Combined with other developability scores to support risk assessment, asset selection, and CMC planning across large antibody portfolios.

Related
-------

- ``DeepSP`` – Generates the 30 sequence-derived DeepSP surface descriptors (SCM/SAP features) that DeepViscosity uses as inputs; call it directly when you need broader charge/hydrophobicity stability profiling beyond viscosity classification.
- ``DeepSCM`` – Earlier CNN surrogate for spatial charge maps and viscosity risk from sequence; useful as a complementary, charge-focused viscosity screen alongside DeepViscosity’s charge+hydrophobicity-based classifier.
- ``BioLMSol`` – Predicts apparent solubility and aggregation risk from sequence; combine with DeepViscosity to jointly assess high-concentration viscosity and overall developability liabilities.
- ``Pro4S Regression`` – Sequence-based regression for continuous protein stability/solubility scores; pair with DeepViscosity’s binary viscosity classes to quantify broader biophysical risk while flagging high-viscosity candidates.

References
----------

- Kalejaye, L. A., Chu, J.-M., Wu, I.-E., Amofah, B., Lee, A., Hutchinson, M., Chakiath, C., Dippel, A., Kaplan, G., Damschroder, M., Stanev, V., Pouryahya, M., Boroumand, M., Caldwell, J., Hinton, A., Kreitz, M., Shah, M., Gallegos, A., Mody, N., & Lai, P.-K. (2025). Accelerating high-concentration monoclonal antibody development with large-scale viscosity data and ensemble deep learning. *mAbs*, 17(1), 2483944. https://doi.org/10.1080/19420862.2025.2483944
