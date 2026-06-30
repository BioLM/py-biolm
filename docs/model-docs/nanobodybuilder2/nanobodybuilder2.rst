NanoBodyBuilder2 API
====================

NanoBodyBuilder2 is a nanobody-specific deep learning model from the ImmuneBuilder suite that predicts all-atom 3D structures for single-chain VHH sequences. Given a heavy-chain-only amino acid sequence (H) up to 2048 residues, the API returns a refined PDB structure, using nanobody-trained networks and an OpenMM-based post-processing pipeline. Typical applications include nanobody structure determination for docking, epitope mapping, library characterization, and structure-guided engineering workflows.

Predict
-------

Predict the 3D structure (PDB) for nanobody heavy chain sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="nanobodybuilder2",
                action="predict",
                params={},
                items=[
                  {
                    "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                  },
                  {
                    "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGINAGTGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/nanobodybuilder2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                },
                {
                  "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGINAGTGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                }
              ],
              "params": {}
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/nanobodybuilder2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                    },
                    {
                      "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGINAGTGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                    }
                  ],
                  "params": {}
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/nanobodybuilder2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  H = "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                ),
                list(
                  H = "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGINAGTGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                )
              ),
              params = list()
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/nanobodybuilder2/predict/

   Predict endpoint for NanoBodyBuilder2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - *(no parameters are currently defined for this endpoint; pass an empty object or omit this field)*

      - **items** (*array of objects*, min items: 1, max items: 8) --- Input nanobody sequences:

        - **H** (*string*, min length: 1, max length: 2048, required) — Nanobody heavy-chain amino acid sequence using unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/nanobodybuilder2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
          },
          {
            "H": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGINAGTGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
          }
        ],
        "params": {}
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **pdb** (*string*) — Predicted 3D structure in PDB format, including ATOM/HETATM records and REMARK lines

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "REMARK  NANOBODY STRUCTURE MODELLED USING NANOBODYBUILDER2                      \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-18                          \nATOM      1  N   GLN H   1     -10.553... (truncated for documentation)"
          },
          {
            "pdb": "REMARK  NANOBODY STRUCTURE MODELLED USING NANOBODYBUILDER2                      \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-18                          \nATOM      1  N   GLN H   1     -10.462... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- NanoBodyBuilder2 is specialised for nanobody 3D structure prediction and, on published benchmarks, achieves lower backbone RMSD than general-purpose models and homology-based tools (mean CDR3 RMSD 2.89 Å vs 3.44 Å for AlphaFold2; framework RMSD 0.79 Å vs 1.09 Å for ABodyBuilder and 1.19 Å for MOE).
- Compared with AlphaFold2-based structure-prediction services on BioLM, NanoBodyBuilder2 is substantially faster and cheaper for nanobody targets because it is alignment-free, does not query large external databases, and runs with a small CPU-only footprint (no GPU required).
- Within the ImmuneBuilder family (ABodyBuilder2, TCRBuilder2/TCRBuilder2+), NanoBodyBuilder2 offers nanobody accuracy on par with the antibody and TCR variants while sharing their low resource requirements (2 vCPUs, 8 GB RAM per worker), making it suitable for high-throughput nanobody structure prediction compared to GPU-dependent models such as AlphaFold2.

Applications
------------

- Structure-guided nanobody humanization by modelling how candidate framework and CDR substitutions perturb the 3D Ig fold and paratope geometry; enables therapeutic teams to down‑select humanizing mutations that are structurally compatible before expression; relies on static structural proxies and still requires experimental immunogenicity and developability assessment.
- Affinity maturation support for nanobodies by rapidly predicting 3D structures of mutational variants so in silico libraries can be ranked and filtered for plausible binding-site architectures; helps focus wet‑lab screening on variants that preserve CDR1/CDR2 topologies and side chain packing; less informative for designs that aim to radically reshape highly variable CDR3 loops without binding data.
- Stability-oriented engineering of nanobody candidates using predicted structures to flag mutations that improve core packing, reduce loop strain, or eliminate buried polar and charged residues; allows formulation and developability teams to triage variants for thermostability and aggregation assays; predictions indicate structural plausibility only and do not substitute for empirical stability measurements.
- Sequence liability mitigation by using nanobody models to locate surface-exposed hydrophobic patches, deamidation-prone Asn, or oxidation-sensitive Met in structurally important regions and redesigning them while maintaining the Ig domain; supports early reduction of downstream manufacturing and formulation risks; less effective for liabilities dominated by formulation conditions or dynamic/post-translational effects not captured in a single predicted structure.
- Structure-aware nanobody library design for discovery and optimization campaigns by generating models for large sequence sets and discarding variants that disrupt the canonical nanobody framework or collapse CDRs; enables focused libraries enriched for structurally viable binders, improving hit quality and reducing non-expressing or misfolded clones; predictions are most reliable for framework and CDR1/CDR2 variation, with CDR3 diversity still best refined via iterative experimental feedback.

Limitations
-----------

- **Maximum Sequence Length**: Each nanobody heavy chain ``H`` in ``ImmuneBuilderNanoBodyBuilder2PredictRequest.items`` must be between ``1`` and ``2048`` amino acids. Sequences outside this range, or containing ambiguous (non-standard) amino acids, are rejected by the ``aa_unambiguous_validator``.
- **Batch Size**: The ``items`` array in ``ImmuneBuilderNanoBodyBuilder2PredictRequest`` accepts up to ``8`` nanobody sequences per call (``batch_size = 8``). Larger datasets must be split across multiple API requests.
- **Input Format**: NanoBodyBuilder2 expects a single VHH-like chain provided via the required ``H`` field only. Requests including any additional keys (for example ``L``, ``A``, ``B`` or other extra fields) are not allowed (``extra = "forbid"``) and will raise a validation error.
- **Scope of Model**: The model is specialised for nanobody (single-domain VHH) structures. It is not suitable for full-length antibodies, multispecific formats, TCRs, or arbitrary proteins, where antibody- or general-purpose structure models (for example ``abodybuilder2``, AlphaFold2, ESMFold) are more appropriate.
- **CDR3 and Loop Accuracy**: As with other immune-structure models, backbone predictions for framework regions and CDR1/CDR2 are typically more reliable than for the CDR3 loop, which shows larger structural variability (CDR3 RMSDs around 2.9 Å on benchmark data). Use CDR3 conformations cautiously in applications that require precise loop geometry, such as detailed docking or affinity estimation.
- **Training Distribution**: NanoBodyBuilder2 was trained primarily on natural camelid/llama nanobody structures. Performance may degrade for highly engineered, humanised, unusually long/short, or heavily mutated nanobodies that deviate strongly from these sequence and length distributions.

How We Use It
-------------

NanoBodyBuilder2 is used in BioLM workflows to rapidly generate 3D structures from nanobody heavy-chain sequences so teams can run structure-guided design without solving experimental structures for every variant. Predicted models are consumed by downstream thermostability and developability scoring, docking, and sequence-embedding–based screening pipelines to prioritize candidates for synthesis, with particular attention to CDR3 geometry and surface-exposed liabilities.

- Enables iterative nanobody optimization and humanization by providing fast structural feedback on sequence changes.
- Integrates via standardized, scalable APIs with binding, stability, and liability prediction tools to shorten design–build–test cycles.

Related
-------

- ``nanoBERT`` – Transformer model trained for nanobody sequence infilling; complements ``NanoBodyBuilder2`` by proposing sequence variants whose 3D structures can then be modelled.
- ``ABodyBuilder2`` – Antibody-specific structure prediction; useful for comparing nanobody designs to paired heavy–light antibodies targeting the same or related epitopes.
- ``TCRBuilder2`` / ``TCRBuilder2+`` – T-cell receptor structure prediction; shares the ImmuneBuilder architecture and supports comparative analysis of nanobody versus TCR binding sites.
- ``AbLang-2`` – Antibody language model for sequence completion and optimisation; can generate or refine nanobody-like sequences prior to structural assessment with ``NanoBodyBuilder2``.

References
----------

- Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). `ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins <https://doi.org/10.1038/s42003-023-04927-7>`_. *Communications Biology*.
