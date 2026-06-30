ABodyBuilder3 pLDDT API
=======================

ABodyBuilder3 pLDDT is a CPU-accelerated antibody structure prediction service that infers all-atom 3D Fv structures from paired heavy (H) and light (L) chain amino acid sequences. It is based on the ImmuneBuilder/ABodyBuilder2 architecture and returns a refined PDB model plus optional per-residue pLDDT confidence scores for each chain. The API supports single-antibody inference (batch size 1, sequences up to 2048 residues each) for high-throughput analysis, virtual screening, and computational antibody design workflows.

Predict
-------

Predict 3D structure and pLDDT scores for an antibody’s heavy (H) and light (L) chain.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="abodybuilder3-plddt",
                action="predict",
                params={
                  "plddt": true,
                  "seed": 123
                },
                items=[
                  {
                    "H": "EVQLVESGGGLVKPGGSLR",
                    "L": "DIQMTQSPASLSASVGDR"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/abodybuilder3-plddt/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "plddt": true,
                "seed": 123
              },
              "items": [
                {
                  "H": "EVQLVESGGGLVKPGGSLR",
                  "L": "DIQMTQSPASLSASVGDR"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/abodybuilder3-plddt/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "plddt": true,
                    "seed": 123
                  },
                  "items": [
                    {
                      "H": "EVQLVESGGGLVKPGGSLR",
                      "L": "DIQMTQSPASLSASVGDR"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/abodybuilder3-plddt/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                plddt = TRUE,
                seed = 123
              ),
              items = list(
                list(
                  H = "EVQLVESGGGLVKPGGSLR",
                  L = "DIQMTQSPASLSASVGDR"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/abodybuilder3-plddt/predict/

   Predict endpoint for ABodyBuilder3 pLDDT.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **plddt** (*bool*, default: False) — Flag indicating whether to include per-residue pLDDT scores in the response

        - **seed** (*int*, optional, default: 42) — Random seed used to initialize model inference

      - **items** (*array of objects*, min: 1, max: 1) --- Input antibody sequences:

        - **H** (*string*, required, min length: 1, max length: 2048) — Heavy chain amino acid sequence using extended amino acid alphabet

        - **L** (*string*, required, min length: 1, max length: 2048) — Light chain amino acid sequence using extended amino acid alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/abodybuilder3-plddt/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "plddt": true,
          "seed": 123
        },
        "items": [
          {
            "H": "EVQLVESGGGLVKPGGSLR",
            "L": "DIQMTQSPASLSASVGDR"
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

        - **pdb** (*string*) — Predicted antibody structure in standard PDB file format.

        - **plddt** (*array of arrays of floats*, optional) — Predicted per-residue confidence scores for the antibody chains.

          - Outer array dimension: 2 (one array per antibody chain: heavy [H] and light [L]).
          - Inner array dimensions: equal to the length of each input sequence (H and L chains).
          - Float values represent raw model outputs and may be negative.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "REMARK   1 CREATED WITH OPENMM 7.7, 2025-06-17\nATOM      1  N   GLU H   0     -12.538   1.205   0.765  1.00  0.00           N  \nATOM      2  H   GLU H   0     -13.550   0.585   0.591  1.00  0.00      ... (truncated for documentation)",
            "plddt": [
              [
                -2.409224510192871,
                -3.094425916671753,
                "... (truncated for documentation)"
              ],
              [
                -3.281006336212158,
                -4.06817626953125,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Runs on CPU-only hardware (2 vCPUs, 8 GB RAM) for the pLDDT variant; no GPU is allocated, in contrast to ABodyBuilder3 Language, which uses a T4 GPU.
- Provides residue-level pLDDT confidence estimates for structures generated by ABodyBuilder3 Language, enabling rapid quality assessment without re-running full 3D inference.
- For typical antibody Fv lengths, inference completes in seconds per structure on CPU, enabling higher throughput than AlphaFold2-style pipelines that require GPUs and MSA/database searches.
- Confidence estimates are designed to be on a similar scale and interpretability as AlphaFold2 pLDDT, but with substantially lower computational cost, making this endpoint suitable for large antibody panels and repertoire-scale workflows.

Applications
------------

- Rapid antibody candidate screening and prioritization by predicting per-residue structural confidence (pLDDT) for heavy/light chain models, enabling biotech teams to filter large antibody libraries toward candidates with higher predicted structural reliability; not suitable as a standalone replacement for experimental developability or potency assays.
- Antibody affinity maturation workflows, where pLDDT profiles highlight well-resolved versus uncertain regions (especially CDR loops), helping protein engineers focus mutagenesis on structurally supported backbones and deprioritize designs with low-confidence core packing; particularly relevant when iterating many in silico variants before lab construction.
- Computational epitope mapping and paratope characterization by using pLDDT to assess which CDR residues are modelled with high confidence, allowing researchers to concentrate docking, MD, or interface analysis on structurally reliable loop segments; beneficial for vaccine and antibody-based diagnostic programs studying antigen–antibody interfaces.
- Antibody humanization and reformatting (e.g., isotype or framework changes), using pLDDT to compare structural integrity between parental and engineered Fv sequences, accelerating selection of variants that preserve stable frameworks and CDR geometries; less informative for predicting immunogenicity, effector function, or in vivo efficacy without further modelling and experiments.
- Structural quality control in antibody database and registry curation, applying pLDDT thresholds to automatically flag low-confidence regions or full Fv models, helping organizations maintain higher-quality structure repositories for downstream docking, QSAR, or generative design; should complement, not replace, expert structural review where decisions are high impact.

Limitations
-----------

- **Maximum Sequence Length**: The API accepts antibody heavy (``H``) and light (``L``) chain sequences up to ``2048`` amino acids each. Longer sequences cannot be processed and must be truncated or split prior to submission.
- **Batch Size**: The API enforces a fixed ``batch_size`` of ``1`` item per ``predictor`` request. Larger datasets must be processed via multiple sequential requests.
- **Model Type (``plddt``)**: Enabling the ``plddt`` flag in ``params`` returns per-residue confidence scores as a list of lists of floats (one list per chain). These scores correlate with prediction error but do not guarantee structural correctness and are primarily useful for filtering out low-confidence regions or models.
- **CDR-H3 Structural Accuracy**: ABodyBuilder3 is based on ImmuneBuilder/ABodyBuilder2, which attains high accuracy on frameworks and most CDRs but still shows larger errors for CDR-H3 (average RMSD ~2.81 Å in benchmarks). CDR-H3 conformations should be interpreted with additional caution, even when ``plddt`` scores are high.
- **Molecule Type**: The model is optimized for paired antibody variable domains (VH/VL provided as ``H`` and ``L``). It is not intended for general proteins, nanobodies, or T-cell receptors; for these, specialised models (e.g. NanoBodyBuilder2, TCRBuilder2, or general structure predictors like AlphaFold2/ESMFold) are more appropriate.
- **Throughput and Pipeline Role**: ABodyBuilder3 is designed for accurate antibody structure prediction rather than ultra-high-throughput screening. For initial ranking or filtering of very large libraries (e.g. millions of sequences), faster approximate structure predictors (such as ESMFold) are typically more suitable, with ABodyBuilder3 used in later, narrower stages.

How We Use It
-------------

ABodyBuilder3 pLDDT enables rapid, residue-level assessment of antibody structure confidence within design and optimization campaigns, allowing teams to rank large in silico libraries and focus wet-lab effort on sequences with reliable 3D models, especially in CDR loops and VH–VL interfaces. By exposing standardized pLDDT scores alongside predicted structures through scalable APIs, it integrates with sequence-based generative models, developability filters, and epitope/paratope analyses to drive faster, data-driven antibody maturation cycles.

- Supports automated triage of candidates with low-confidence structural regions before synthesis.
- Combines with biophysical property estimators and sequence embeddings to guide multi-round optimization and fine-tuning.

Related
-------

- ``NanoBodyBuilder2`` – Predicts nanobody structures with similar inputs and usage, complementing ABodyBuilder3 pLDDT for single-domain antibody variants.
- ``TCRBuilder2`` – Predicts T-cell receptor structures, enabling comparative studies alongside antibody structures from ABodyBuilder3 pLDDT.
- ``ImmuneFold Antibody`` – Provides alternative antibody structure predictions, useful for benchmarking or cross-validating ABodyBuilder3 pLDDT outputs.
- ``ABodyBuilder3 Language`` – Generates antibody structures with a sequence-only language-model approach, providing complementary structures to those scored with pLDDT.

References
----------

- Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). *ImmuneBuilder: Deep-learning models for predicting the structures of immune proteins*. Communications Biology. https://doi.org/10.1038/s42003-023-04927-7
