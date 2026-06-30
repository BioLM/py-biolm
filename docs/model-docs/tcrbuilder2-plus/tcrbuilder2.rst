TCRBuilder2+ API
================

TCRBuilder2+ is a deep learning model for rapid prediction of paired T-cell receptor (TCR) variable domain structures from alpha (A) and beta (B) chain amino acid sequences. It predicts backbone conformations with CDR loop RMSDs on benchmark sets typically around 1–3 Å and produces refined, stereochemically consistent 3D models in PDB format. The API supports batched inference (up to 8 TCRs, sequence length ≤2048) for applications in therapeutic TCR discovery, immune repertoire structural profiling, and structure-guided TCR–pMHC interaction analysis.

Predict
-------

Predict the structure of T-cell receptors using paired alpha (A) and beta (B) chain sequences.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="tcrbuilder2-plus",
                action="predict",
                params={},
                items=[
                  {
                    "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                    "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                  },
                  {
                    "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                    "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/tcrbuilder2-plus/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                  "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                },
                {
                  "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                  "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                }
              ],
              "params": {}
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/tcrbuilder2-plus/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                      "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                    },
                    {
                      "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                      "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
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

            url <- "https://biolm.ai/api/v3/tcrbuilder2-plus/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  A = "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                  B = "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                ),
                list(
                  A = "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                  B = "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                )
              ),
              params = list()
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/tcrbuilder2-plus/predict/

   Predict endpoint for TCRBuilder2+.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output types to include:

          - Allowed values: "mean", "per_token", "bos", "contacts", "logits", "attentions"

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **H** (*string*, optional, min length: 1, max length: 2048) — Heavy chain amino acid sequence (required for ABodyBuilder2 and NanoBodyBuilder2)

        - **L** (*string*, optional, min length: 1, max length: 2048) — Light chain amino acid sequence (required for ABodyBuilder2)

        - **A** (*string*, optional, min length: 1, max length: 2048) — Alpha chain amino acid sequence (required for TCRBuilder2)

        - **B** (*string*, optional, min length: 1, max length: 2048) — Beta chain amino acid sequence (required for TCRBuilder2)

      - **ImmuneBuilderNanoBodyBuilder2PredictRequest** (*object*) --- NanoBodyBuilder2-specific request structure:

        - **items** (*array of objects*, min: 1, max: 8) — Input sequences:

          - **H** (*string*, required, min length: 1, max length: 2048) — Heavy chain amino acid sequence

      - **ImmuneBuilderABodyBuilder2PredictRequest** (*object*) --- ABodyBuilder2-specific request structure:

        - **items** (*array of objects*, min: 1, max: 8) — Input sequences:

          - **H** (*string*, required, min length: 1, max length: 2048) — Heavy chain amino acid sequence

          - **L** (*string*, required, min length: 1, max length: 2048) — Light chain amino acid sequence

      - **ImmuneBuilderTCRBuilder2PredictRequest** (*object*) --- TCRBuilder2-specific request structure:

        - **items** (*array of objects*, min: 1, max: 8) — Input sequences:

          - **A** (*string*, required, min length: 1, max length: 2048) — Alpha chain amino acid sequence

          - **B** (*string*, required, min length: 1, max length: 2048) — Beta chain amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/tcrbuilder2-plus/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
            "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
          },
          {
            "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
            "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
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

        - **pdb** (*string*) — Predicted immune protein structure in standard PDB format; includes atomic coordinates for all atoms (heavy atoms and hydrogens), residue numbering according to IMGT scheme, and chain identifiers; structure is refined to remove steric clashes, incorrect peptide bond lengths, cis-peptide bonds, and D-amino acids; coordinates are in Angstroms (Å)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "REMARK  TCR STRUCTURE MODELLED USING TCRBUILDER2+                               \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-19                          \nATOM      1  N   ALA B   2      -9.648... (truncated for documentation)"
          },
          {
            "pdb": "REMARK  TCR STRUCTURE MODELLED USING TCRBUILDER2+                               \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-19                          \nATOM      1  N   ALA B   2      -9.648... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- TCRBuilder2+ improves backbone accuracy over the original TCRBuilder2, particularly on the most challenging CDR-α3 and CDR-β3 loops, with mean RMSD reduced from 2.89 Å to 1.85 Å for CDR-α3 (~36% improvement) and from 3.12 Å to 1.93 Å for CDR-β3 (~38% improvement) on the ImmuneBuilder TCR benchmark set.
- Compared to AlphaFold-Multimer, TCRBuilder2+ achieves essentially identical loop-level accuracy for TCRs (CDR-α3 RMSD 1.85 Å vs. 1.84 Å; CDR-β3 1.93 Å vs. 1.94 Å) while avoiding MSAs and large sequence databases, making it substantially more suitable for high-throughput TCR repertoire modelling.
- Within BioLM’s structure prediction APIs, TCRBuilder2+ provides TCR-specific accuracy that is on par with AlphaFold-Multimer for TCRs but with computational cost and latency similar to other ImmuneBuilder models (ABodyBuilder2, NanoBodyBuilder2) and far lower than general-purpose predictors such as AlphaFold2/Multimer.
- The model produces all-atom TCR structures in standard PDB format with stereochemical quality comparable to experimentally determined structures in the benchmark (no peptide bond, cis-bond, D-amino acid, or heavy-atom clash violations reported for ImmuneBuilder models in the original study), and with improved loop accuracy relative to earlier specialized methods such as RepertoireBuilder and the original TCRBuilder.

Applications
------------

- Structure-guided selection of therapeutic TCR candidates from large sequence panels, using predicted α/β variable-domain conformations to flag receptors with plausible, well-packed CDRs; useful for TCR-based immunotherapy programs, while still limited for highly flexible or unusually long CDR3 loops.
- Modelling TCR–peptide–MHC binding geometry by combining TCRBuilder2+ TCR variable-domain structures with downstream docking or physics-based tools, helping guide rational engineering of specificity and affinity; suitable for canonical α/β TCRs but less reliable for non-standard domain architectures or engineered fusion formats.
- High-throughput structural triage of NGS-derived TCR repertoires, enabling rapid removal of sequences predicted to yield grossly misfolded or unstable variable domains before synthesis and expression; valuable in target discovery and biomarker efforts, with the caveat that in silico stability proxies do not replace biophysical assays.
- In silico filtering and ranking of designed or affinity-matured TCR libraries based on predicted backbone geometry and CDR packing, reducing downstream screening burden by deprioritizing models with distorted frameworks or extreme loop conformations; not intended to capture full induced-fit effects upon peptide–MHC engagement.
- Generation of multiple plausible TCR structures per sequence to estimate per-residue uncertainty, allowing teams to focus follow-up modelling and experiments on regions with consistent conformations across the ensemble; less informative for large-scale conformational rearrangements or TCR clustering phenomena beyond the variable domains.

Limitations
-----------

- **Maximum Sequence Length**: Each ``A`` (alpha) and ``B`` (beta) chain must be between ``1`` and ``2048`` amino acids. Longer sequences must be truncated or split before submission.
- **Batch Size**: Up to ``8`` TCR pairs per ``items`` list in a single request. Larger repertoires must be processed in multiple requests.
- **Input Type Restrictions**: This endpoint is specific to paired TCR chains using fields ``A`` and ``B``. It cannot accept antibody heavy/light chains (``H``, ``L``) or nanobody sequences; for those use ``abodybuilder2`` or ``nanobodybuilder2`` via their respective endpoints.
- **Model Scope**: TCRBuilder2+ is trained on conventional TCRs. Performance may degrade for very atypical or engineered receptors (e.g. extreme CDR3 lengths/compositions, non‑Ig-like domains).
- **Conformational Diversity**: The API returns a single refined ``pdb`` structure per TCR pair. It does not expose the underlying structure ensemble or per-residue error estimates, so it is not suitable for detailed flexibility analysis.
- **No Embeddings or Contacts**: The ``predictor`` endpoint only returns coordinates in ``pdb`` format; it does not expose ``ImmuneBuilderEncodeIncludeOptions`` outputs such as ``mean``, ``per_token``, ``contacts``, or ``attentions`` for downstream embedding or contact-based analyses.

How We Use It
-------------

TCRBuilder2+ enables rapid generation of TCR α/β structural models from sequence, giving protein engineers access to consistent CDR and framework geometries for large sequence panels. We use these structures to provide 3D context for TCR-based discovery, linking sequence-level design, structural clustering, and downstream developability and liability assessment via standardized, scalable APIs.

- Supports high-throughput structural annotation of TCR repertoires for epitope hypothesis generation and hit triage.
- Integrates with BioLM property prediction, embedding-based similarity search, and ranking workflows to prioritize TCRs with favorable structural and biophysical profiles.

Related
-------

- ``TCRBuilder2`` – Single-model TCR structure prediction using the same architecture and training data as ``TCRBuilder2+``, useful when you do not need ensemble-based uncertainty estimates.
- ``ImmuneFold TCR`` – Alternative TCR structure predictor based on a different architecture, useful for cross-validation and comparing predictions from ``TCRBuilder2+``.
- ``NanoBodyBuilder2`` – Predicts nanobody (single-chain Ig) structures with a similar ImmuneBuilder workflow, enabling consistent modelling of other Ig-domain therapeutics alongside TCRs.
- ``ABodyBuilder2`` – Antibody Fv structure prediction using the same ImmuneBuilder framework, useful when modelling antibodies and TCRs in a unified structural pipeline.

References
----------

- Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). ImmuneBuilder: Deep-learning models for predicting the structures of immune proteins. *Communications Biology*, 6, 575. https://doi.org/10.1038/s42003-023-04927-7
