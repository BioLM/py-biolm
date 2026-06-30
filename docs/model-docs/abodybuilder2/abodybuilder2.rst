ABodyBuilder2 API
=================

ABodyBuilder2 is an antibody-specific deep learning model for rapid, high-accuracy prediction of Fv structures, including all six CDR loops, from paired heavy (H) and light (L) chain sequences. It achieves a mean CDR-H3 backbone RMSD of 2.81 Å on a benchmark of 34 antibodies and produces full all-atom models without requiring MSAs, templates, or external sequence databases. The API returns PDB-format structures suitable for antibody screening, affinity maturation, and structure-guided engineering workflows.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="abodybuilder2",
                action="predict",
                params={},
                items=[
                  {
                    "H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                    "L": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLIYHTSRLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPYTFGQGTKVEIK"
                  },
                  {
                    "H": "QVQLQQSGPELEKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLASEDSAVYYCARSNYYGSSYFDYWGQGTTLTVSS",
                    "L": "QIVLSQSPAILSASPGEKVTMTCRASSSVNYMDWYQQKPGSSPKPWIYAPSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWSFNPPTFGAGTKLEIK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/abodybuilder2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                  "L": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLIYHTSRLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPYTFGQGTKVEIK"
                },
                {
                  "H": "QVQLQQSGPELEKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLASEDSAVYYCARSNYYGSSYFDYWGQGTTLTVSS",
                  "L": "QIVLSQSPAILSASPGEKVTMTCRASSSVNYMDWYQQKPGSSPKPWIYAPSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWSFNPPTFGAGTKLEIK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/abodybuilder2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                      "L": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLIYHTSRLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPYTFGQGTKVEIK"
                    },
                    {
                      "H": "QVQLQQSGPELEKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLASEDSAVYYCARSNYYGSSYFDYWGQGTTLTVSS",
                      "L": "QIVLSQSPAILSASPGEKVTMTCRASSSVNYMDWYQQKPGSSPKPWIYAPSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWSFNPPTFGAGTKLEIK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/abodybuilder2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  H = "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                  L = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLIYHTSRLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPYTFGQGTKVEIK"
                ),
                list(
                  H = "QVQLQQSGPELEKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLASEDSAVYYCARSNYYGSSYFDYWGQGTTLTVSS",
                  L = "QIVLSQSPAILSASPGEKVTMTCRASSSVNYMDWYQQKPGSSPKPWIYAPSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWSFNPPTFGAGTKLEIK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/abodybuilder2/predict/

   Predict endpoint for ABodyBuilder2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **plddt** (*bool*, default: False) — Whether to include pLDDT scores in the response
        - **seed** (*int*, optional, default: 42) — Random seed for prediction consistency

      - **items** (*array of objects*, max: 1) --- Input sequences:

        - **H** (*string*, min length: 1, max length: 2048, required) — Heavy chain amino acid sequence
        - **L** (*string*, min length: 1, max length: 2048, required) — Light chain amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/abodybuilder2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
            "L": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLIYHTSRLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQGNTLPYTFGQGTKVEIK"
          },
          {
            "H": "QVQLQQSGPELEKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLASEDSAVYYCARSNYYGSSYFDYWGQGTTLTVSS",
            "L": "QIVLSQSPAILSASPGEKVTMTCRASSSVNYMDWYQQKPGSSPKPWIYAPSNLASGVPVRFSGSGSGTSYSLTISRVEAEDAATYYCQQWSFNPPTFGAGTKLEIK"
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

        - **pdb** (*string*) — Predicted antibody structure in standard PDB format.

        - **plddt** (*array of arrays of floats*, optional) — Predicted Local Distance Difference Test (pLDDT) scores per residue:

          - Outer array length: 2 (chains: heavy chain "H", light chain "L")
          - Inner array length: equal to the number of residues in the corresponding chain
          - Values range: 0.0–100.0 (higher values indicate higher local structure confidence)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "REMARK  ANTIBODY STRUCTURE MODELLED USING ABODYBUILDER2                         \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-18                          \nATOM      1  N   GLU H   1     -17.169... (truncated for documentation)"
          },
          {
            "pdb": "REMARK  ANTIBODY STRUCTURE MODELLED USING ABODYBUILDER2                         \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-18                          \nATOM      1  N   GLN H   1     -17.778... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- Runs on CPU-optimized instances (2 vCPUs, 8 GB RAM) without requiring a GPU, enabling cost-efficient large-scale inference for antibody structures
- Predictive accuracy (backbone RMSD, Å) on the 34-antibody benchmark:
  - CDR-H3: 2.81 (vs 2.90 for AlphaFold-Multimer; ~10% lower error than ABlooper, IgFold, EquiFold)
  - Framework regions: 0.54–0.57 (within typical X-ray experimental error)
- Chemical surface and stereochemistry:
  - All-atom outputs with accurate side-chain torsions (χ1/χ2) and exposed/buried classification comparable to AlphaFold-Multimer
  - Zero peptide-bond, cis-bond, D-amino acid, or heavy-atom clash violations in benchmarked models after refinement, unlike EquiFold and some other antibody models
- Compared to other BioLM structure predictors:
  - More accurate and substantially faster than AlphaFold2 / AlphaFold-Multimer for antibodies, especially on CDR-H3
  - More accurate for antibodies than generalist models such as ESMFold, while being far more suitable for high-throughput antibody-focused workloads

Applications
------------

- Accurate prediction of antibody variable region (VH/VL) structures for therapeutic design, providing all-atom Fv models that can be used for downstream docking, epitope mapping, and structure-based affinity maturation workflows in antibody discovery pipelines.
- Rapid structural modeling of large antibody panels (up to 8 VH/VL pairs per API call, each chain ≤2048 residues), enabling high-throughput triage of candidates from display libraries or immune repertoire sequencing by linking sequence variants to 3D CDR conformations and paratope geometry.
- Nanobody and TCR structure prediction via the broader ImmuneBuilder API, allowing teams to use the same interface for single-domain antibodies and TCRs when appropriate, and to compare structural features across immune modalities in multi-format biologics programs.
- Use of model-provided residue-level error estimates to filter or down‑weight uncertain regions (especially CDR-H3), helping focus experimental characterization on designs where predicted loop conformations and VH–VL orientations are likely to be reliable.
- Limitations include reduced accuracy for highly unusual or very long CDR-H3 loops and lack of explicit antigen context; predicted structures should be combined with experimental data and additional modeling for final drug design and developability assessment.

Limitations
-----------

- **Maximum Sequence Length**: Each antibody heavy (``H``) and light (``L``) chain sequence must not exceed ``2048`` amino acids.
- **Batch Size**: The ``items`` array in ``ImmuneBuilderABodyBuilder2PredictRequest`` supports a maximum of ``8`` antibody sequence pairs per request.
- **Antibody-specific only**: ABodyBuilder2 is trained for paired antibody variable domains (VH/VL). It is not suitable for general protein structure prediction or for other immune receptors such as nanobodies or T-cell receptors; use ``nanobodybuilder2`` or ``tcrbuilder2`` instead.
- **CDR-H3 edge cases**: Although ABodyBuilder2 achieves state-of-the-art average accuracy for CDR-H3, predictions for unusually long or rare CDR-H3 loops (e.g. beyond ~22 residues) may be less reliable than for typical lengths.
- **No evolutionary data**: The model does not use multiple sequence alignments or evolutionary information, so in settings where evolutionary couplings are critical, AlphaFold-Multimer or similar MSA-based methods may give more accurate results.
- **Output type**: The API returns only an all-atom structure in ``pdb`` format per item. It does not expose per-residue error estimates, embeddings, attention maps, or other internal model features for downstream analysis.

How We Use It
-------------

ABodyBuilder2 enables rapid, sequence-to-structure modeling of antibody variable regions (VH/VL) through a standardized API, making accurate Fv and CDR, especially CDR-H3, structures available early in antibody discovery and optimization. These structures integrate with BioLM sequence embeddings, developability predictors, and docking/affinity models to prioritize designs, focus library construction, and inform multi-round in vitro campaigns while scaling to large next-generation sequencing repertoires.

- Supports high-throughput structural screening of candidate antibodies to accelerate lead selection and maturation.
- Provides consistent 3D inputs for downstream property prediction (e.g., stability, liability, epitope/paratope analyses) and ranking.

Related
-------

- ``NanoBodyBuilder2`` – Nanobody-specific ImmuneBuilder model using the same architecture as ABodyBuilder2, useful for single-domain antibody and VHH variants.
- ``TCRBuilder2`` – ImmuneBuilder model for paired TCR α/β chains, complementary when comparing antibody and TCR recognition.
- ``ImmuneFold Antibody`` – Alternative antibody structure prediction method for cross-checking ABodyBuilder2 models and exploring method-dependent variation.
- ``ABodyBuilder3 pLDDT`` – Next-generation antibody model providing per-residue confidence scores, useful for benchmarking and confidence estimation alongside ABodyBuilder2.

References
----------

- Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins. *Communications Biology*, 6, 575. https://doi.org/10.1038/s42003-023-04927-7
