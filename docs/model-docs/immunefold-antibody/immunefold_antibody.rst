ImmuneFold Antibody API
=======================

ImmuneFold Antibody is a GPU-accelerated antibody and nanobody structure prediction model, obtained by LoRA-based fine-tuning of the ESMFold folding trunk on immune protein structures. The API infers unbound antibody Fv structures from heavy/light or single VHH sequences, with optional antigen PDB context for bound-state refinement. It achieves 2.65 Å RMSD on CDR H3 benchmarks, returns pLDDT and pTM confidence metrics, and typically completes inference in about 3 seconds per structure.

Predict
-------

Predict the 3D structures of antibody sequences, optionally including a PDB snippet for context.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="immunefold-antibody",
                action="predict",
                params={
                  "contact_idx": 5
                },
                items=[
                  {
                    "H": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGGGGWGQGTLVTVSS",
                    "L": "DIQMTQSPSSLSASVGDRVTITCRASSSVSYMNWYQQKPGKAPKLLIYSASNRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/immunefold-antibody/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "contact_idx": 5
              },
              "items": [
                {
                  "H": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGGGGWGQGTLVTVSS",
                  "L": "DIQMTQSPSSLSASVGDRVTITCRASSSVSYMNWYQQKPGKAPKLLIYSASNRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/immunefold-antibody/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "contact_idx": 5
                  },
                  "items": [
                    {
                      "H": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGGGGWGQGTLVTVSS",
                      "L": "DIQMTQSPSSLSASVGDRVTITCRASSSVSYMNWYQQKPGKAPKLLIYSASNRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/immunefold-antibody/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                contact_idx = 5
              ),
              items = list(
                list(
                  H = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGGGGWGQGTLVTVSS",
                  L = "DIQMTQSPSSLSASVGDRVTITCRASSSVSYMNWYQQKPGKAPKLLIYSASNRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/immunefold-antibody/predict/

   Predict endpoint for ImmuneFold Antibody.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **contact_idx** (*int*, optional) — Residue index used for contact analysis

      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences and structures:

        - **H** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of antibody heavy chain

        - **L** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of antibody light chain

        - **B** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of T-cell receptor β chain

        - **A** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of T-cell receptor α chain

        - **P** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of peptide antigen

        - **M** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of MHC molecule

        - **pdb** (*string*, optional, min length: 1, max length: 1000000) — Antigen structure in PDB format

          - Cannot be provided simultaneously with TCR inputs (`B`, `A`, `P`, `M`)

          - Requires at least one antibody chain sequence (`H` or `L`) when provided

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/immunefold-antibody/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "contact_idx": 5
        },
        "items": [
          {
            "H": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGGGGGWGQGTLVTVSS",
            "L": "DIQMTQSPSSLSASVGDRVTITCRASSSVSYMNWYQQKPGKAPKLLIYSASNRYTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLELK"
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

        - **ptm** (*float*, range: 0.0–1.0) — Predicted TM-score (pTM) confidence metric for the returned structure

        - **full_plddt** (*float*, range: 0.0–100.0) — Mean predicted Local Distance Difference Test (pLDDT) score over all residues and chains

        - **plddt** (*array of arrays of floats*, shape: [num_chains, num_residues], range: 0.0–100.0) — Per-residue pLDDT scores for each chain in the predicted structure

        - **pdb** (*string*) — Predicted 3D atomic coordinates for all chains in PDB format

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "ptm": 0.5531312823295593,
            "full_plddt": 96.76122283935547,
            "plddt": [
              [
                91.73861694335938,
                95.18270874023438,
                "... (truncated for documentation)"
              ]
            ],
            "pdb": "ATOM      1  N   GLU H   1      18.752  -1.859   4.247  1.00 91.74           N  \nATOM      2  CA  GLU H   1      17.449  -1.229   4.057  1.00 91.74           C  \nATOM      3  C   GLU H   1      16.604... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- Runs inference on NVIDIA T4 GPUs (3 vCPUs, 6 GB RAM per instance) with parameter-efficient LoRA fine-tuning merged into the base ESMFold weights, so inference cost is essentially identical to ESMFold despite improved accuracy.
- Typical single-structure inference (antibody or nanobody) completes in ~3 seconds on BioLM’s infrastructure, whereas MSA-based models such as AlphaFold2, AlphaFold-Multimer, and TCRmodel2 generally require 5–20 minutes per prediction due to MSA and template search overhead.
- Achieves consistently higher structural accuracy than general protein structure predictors (ESMFold, AlphaFold2, AlphaFold-Multimer), especially in antibody CDR H3 and nanobody CDR3 and framework regions; for example, antibody CDR H3 backbone RMSD is reduced to 2.65 Å vs. ~3.07 Å for ESMFold, AlphaFold2, and AlphaFold-Multimer, and framework RMSD to 0.72 Å vs. 0.78–0.92 Å.
- Outperforms or matches antibody-specific models (IgFold, BALMFold, ImmuneBuilder, tFold-Ab) across all antibody CDRs and frameworks, including CDR H3 RMSD (2.65 Å vs. 2.99–3.42 Å) and heavy–light chain orientation (OCD 2.66, comparable to the best tFold-Ab and better than AlphaFold-Multimer and ESMFold); ImmuneFold pLDDT confidence scores correlate with ground-truth accuracy (Pearson r ≈ 0.57 for antibodies, 0.55 for nanobodies), providing reliable per-residue and global quality estimates.

Applications
------------

- Rapid antibody and nanobody Fv structure prediction for therapeutic optimization, enabling teams to compare CDR loop conformations (especially CDR H3) across many variants and prioritize designs for stability, developability, or epitope exposure when crystallography or cryo-EM are not practical; predictions are less accurate for antibodies with very long or highly flexible CDR loops and still require experimental validation.
- Structure-guided antibody humanization and affinity maturation, allowing researchers to model how specific framework or CDR mutations alter the 3D geometry of the binding site and heavy–light chain orientation, and to filter variants that introduce clashes or large backbone distortions; not intended to directly predict immunogenicity or biophysical liabilities without follow-up assays.
- Antigen-context-aware modeling of antibody–antigen complexes, where providing a bound antigen structure (via PDB) helps refine CDR conformations and interface geometry for candidates against a known target, supporting in silico ranking before biophysical characterization; less suitable when the antigen epitope is unknown, the antigen is very large, or binding induces major conformational changes.
- High-throughput virtual screening of antibody or nanobody libraries by predicting 3D structures at scale and using model confidence scores (pLDDT, pTM) plus downstream energy or docking tools to enrich for candidates with well-formed paratopes and plausible interfaces, reducing wet-lab screening burden; not optimal when precise kinetic or thermodynamic parameters are required, which still depend on experimental measurements.

Limitations
-----------

- **Maximum Sequence Length**: Each immune chain field (``H``, ``L``, ``A``, ``B``, ``P``, ``M``) accepts up to ``256`` amino acids. Single-domain antibodies (VHH) should use ``H`` only (no ``L``). Antigen context is provided via ``pdb``; the ``pdb`` string must pass ``validate_pdb`` and respect ``max_pdb_str_len``.
- **Batch Size**: The ``items`` array allows at most ``32`` structures per request (``ImmuneFoldParams.batch_size``). Larger workloads must be split across multiple API calls.
- ImmuneFold is optimized for antibodies (including nanobodies) and T-cell receptors (TCRs) and their antigen contexts. It is not intended for general protein structure prediction, non-Ig domains, or large multi-protein assemblies outside immune protein–antigen complexes.
- Antigen usage is constrained: ``pdb`` may only be combined with antibody inputs (``H`` with optional ``L``). It cannot be used with TCR chains (``A``, ``B``, ``P``, ``M``), and cannot be provided without antibody sequence inputs.
- Because ImmuneFold adapts a single-chain language model (ESM2/ESMFold), it may not fully capture long-range inter-chain evolutionary couplings in very large or atypical antibody–antigen complexes, especially when epitopes are unknown or antigens are large.
- ImmuneFold returns a single, static 3D structure per item (``pdb`` plus confidence scores ``ptm``, ``full_plddt``, ``plddt``). It does not perform conformational sampling, dynamics, or explicit ΔΔG or binding energy prediction; for those tasks, downstream tools (for example, molecular dynamics or docking/energy calculations) are required.

How We Use It
-------------

ImmuneFold Antibody enables rapid, structure-based antibody and nanobody engineering by providing accurate Fv and CDR (especially CDR H3) predictions via standardized APIs. It integrates into our design–test–optimize loops to rank variants, de-risk developability, and prioritize candidates for synthesis, and can be combined with downstream physics-based or empirical scoring to inform affinity maturation and stability campaigns.

- Supports both single-domain (VHH) and paired heavy/light chain antibody modeling within sequence length limits, enabling broad design scenarios.
- Efficient inference and batch handling (up to 32 items) enable fast iteration in large virtual screening and optimization workflows.

Related
-------

- ``ESMFold`` – Base model that ImmuneFold fine-tunes; useful for general (non-immune) protein structure prediction when immune-specific accuracy is not required.
- ``AlphaFold2`` – General high-accuracy structure predictor that provides a baseline for comparing ImmuneFold’s immune-focused predictions.
- ``TCRBuilder2`` – TCR-specific structure prediction model that can be used alongside ImmuneFold to cross-check TCR structures.
- ``IgFold`` – Antibody-focused structure predictor offering an alternative modeling approach for antibodies that can be compared with ImmuneFold results.

References
----------

- Zhu, T., Ren, M., He, Z., Tao, S., Li, M., Bu, D., & Zhang, H. (2024). Accurate structure prediction of immune proteins using parameter-efficient transfer learning. *Journal of Chemical Theory and Computation*.
