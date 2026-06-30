AlphaFold2 API
==============

AlphaFold2 API service is BioLM's GPU-accelerated implementation of the AlphaFold2 protein structure prediction pipeline.
It integrates fast MSA generation (using MMseqs2 GPU servers or JackHMMER on NVME disks) and offers many user-input options,
such as relaxed or unrelaxed outputs.
With support for multiple databases -- UniRef90, MGnify, Small BFD -- and flexible alignment configurations,
the AlphaFold2 API helps bioinformaticians, data scientists, and life-science teams quickly generate high-confidence structural
models for proteins up to 512 residues.

AlphaFold2 is securely deployed for production workloads on BioLM and access is via REST API.
It serves well in tasks such as virtual screening and large-scale protein modeling, out-competing other implementations
on speed, scalability, and cost. By reducing inference times (e.g., ~6–10 minutes per protein for single-sequence
folding), allowing asynchronous predictions, and enabling pass-through retrieval of previously computed results,
ML researchers can focus on downstream analysis and iterative design. This service is also beneficial for technical
leaders seeking rapid R&D workflow acceleration.
Pricing details and usage plans are available on the `pricing page <https://biolm.ai/saas/>`_.

.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint performs full structure prediction, including alignment generation and final PDB outputs.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            # AlphaFold2 structure prediction
            response = BioLM(
                entity="alphafold2",
                action="predict",
                params={
                    "databases": ["mgnify", "small_bfd", "uniref90"],
                    "algorithm": "mmseqs2",
                    "predictions_per_model": 1,
                    "relax": "none",
                    "msa_iterations": 1
                },
                items=[{"sequence": "MDPGQELRVEVTK..."}]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/alphafold2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "params": {
                  "databases": ["mgnify", "small_bfd", "uniref90"],
                  "algorithm": "mmseqs2",
                  "predictions_per_model": 1,
                  "relax": "none",
                  "msa_iterations": 1
                },
                "items": [
                  {"sequence": "MDPGQELRVEVTK..."}
                ]
              }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/alphafold2/predict/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {
                "params": {
                    "databases": ["mgnify", "small_bfd", "uniref90"],
                    "algorithm": "mmseqs2",
                    "predictions_per_model": 1,
                    "relax": "none",
                    "msa_iterations": 1
                },
                "items": [
                    {"sequence": "MDPGQELRVEVTK..."}
                ]
            }

            response = requests.post(url, json=payload, headers=headers)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/alphafold2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
                params = list(
                    databases = list("mgnify", "small_bfd", "uniref90"),
                    algorithm = "mmseqs2",
                    predictions_per_model = 1,
                    relax = "none",
                    msa_iterations = 1
                ),
                items = list(
                    list(sequence = "MDPGQELRVEVTK...")
                )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/alphafold2/predict/

   Full structure prediction with alignment, optional relaxation, and final PDB outputs.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

     Request JSON Object
   
   .. container:: field-definition

      - **params** (*object*) --- Prediction configuration
        
        - **databases** (*list*) — Alignment databases
        - **algorithm** (*string*) — ``mmseqs2`` or ``jackhmmer``
        - **predictions_per_model** (*int*) — Number of structures returned per model
        - **relax** (*string*) — ``none``, ``all``, or ``best`` (default ``none``)
        - **return_templates** (*boolean*) — Include template information (disabled in predict by default)
        - **msa_iterations** (*int*) — Number of MSA iterations
        - **max_msa_sequences** (*int*) — (Optional) Max sequences in final MSA
    
      - **items** (*array of objects*, max. 1) --- List of input sequences
        
        - **sequence** (*string*, max length: 512) — Protein sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/alphafold2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {
          "databases": ["mgnify","small_bfd","uniref90"],
          "algorithm": "mmseqs2",
          "predictions_per_model": 1,
          "relax": "none",
          "msa_iterations": 1
        },
        "items": [
          {
            "sequence": "MDPGQELRVEVTK..."
          }
        ]
      }

   :statuscode 200: Successful structure prediction. Returns an array of PDB strings.
   :statuscode 400: Invalid input (sequence length or type).
   :statuscode 500: Internal error during folding pipeline.



   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "results": [
          {
            "pdbs": [
              "HEADER    ALPHAFOLD PREDICTION\nATOM      1  N   MET A   1      44.514  23.658  -2.761  1.00  0.00           N\nATOM      2  CA  MET A   1      43.147  24.281  -3.213  1.00  0.00           C\n..."
            ]
          }
        ]
      }

   Returned PDB structures are standard (ATOM/HETATM) lines. The top lines are placeholders for the standard PDB header.

Performance
-----------

- **GPU-accelerated** on A100 with up to 80GB VRAM
- Typical single-sequence turnaround: ~6–10 minutes (longer for large or repeated MSA iterations)
- ``predictions_per_model`` can increase total run time if more structures are generated

Applications
------------
- **High-accuracy single-chain protein structure prediction**: AlphaFold2 excels at predicting atomic-level 3D structures for single protein chains, even in the absence of close homologs, with accuracy competitive with experimental methods.
- **Proteome-scale modeling**: Enables large-scale prediction of protein structures across entire proteomes, supporting structural genomics and annotation projects.
- **Structure-based bioinformatics**: Facilitates downstream analyses such as molecular replacement, cryo-EM map interpretation, and structure-based function prediction.
- **Confidence and quality metrics**: Provides per-residue pLDDT and PAE (Predicted Aligned Error) JSONs, enabling users to assess model confidence, calculate custom metrics, and filter or cluster models for further study.
- **Not optimal for antibodies or multi-chain complexes**: While AlphaFold2 can model a wide range of proteins, it is less reliable for antibody variable regions and multi-chain complexes, where specialized models or additional context are preferred.
- **Rapid hypothesis generation**: Accelerates R&D by enabling fast, iterative structure prediction for candidate proteins in drug discovery, protein engineering, and synthetic biology.

Limitations
-----------

- **Sequence length limit**: 512 residues per request
- **Single-sequence per API call**: For parallel runs, submit multiple requests
- **Partial relax**: If ``relax="best"``, only the top model is relaxed. If ``none``, no relaxation is done
- **Large MSA expansions**: Setting ``max_msa_sequences`` high can delay completion

How BioLM Uses AlphaFold2
-------------------------

BioLM integrates AlphaFold2 APIs to enable researchers to quickly iterate on candidate proteins, refine structural predictions, and feed results into advanced LLM workflows. This synergy accelerates biologics discovery, structure-based drug design, and related computational biology tasks. Combining advanced structure prediction with LLM-driven analysis or text mining fosters data-rich insights, enabling faster go/no-go decisions in R&D.

Related
-------

- ``Chai-1`` – For advanced diffusion-driven folding, including RNA, DNA, and ligands
- ``ESMFold`` – Focused on single-chain proteins with no MSA required

References
----------

- Jumper, J. *et al.* Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2
