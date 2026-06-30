.. _esmfold_api:

ESMFold API
===========

ESMFold is a GPU-accelerated, single- and multi-chain protein structure prediction model that infers atomic-level 3D structures directly from protein sequences, leveraging evolutionary-scale language model representations. ESMFold supports high-throughput, end-to-end structure prediction for single proteins and complexes, with confidence metrics (mean pLDDT, pTM) and PDB-format outputs. BioLM provides scalable API access to ESMFold for protein engineering, metagenomic annotation, and structural biology workflows.

Predict
-------

This endpoint predicts 3D protein structure(s) from input sequence(s), supporting both single-chain and multi-chain (up to 4 chains) inputs.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esmfold",
                action="predict",
                items=[{"sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esmfold/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "items": [
                  {"sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
                ]
              }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esmfold/predict/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {
                "items": [
                    {"sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
                ]
            }
            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esmfold/predict/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type" = "application/json"
            )
            body <- list(
              items = list(list(sequence = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"))
            )
            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esmfold/predict/

   Predicts atomic-level 3D structure(s) for one or more protein sequences (single- or multi-chain).

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **items** (*array of objects*, max. 2) --- List of input sequences (single- or multi-chain):

        - **sequence** (*string*, max length: 768) — Protein sequence (use ":" to separate up to 4 chains)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmfold/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "items": [
          {"sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
        ]
      }

   :statuscode 200: Successful prediction. Returns structure(s) and confidence metrics.
   :statuscode 400: Invalid input (sequence length, format, or chain count).
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) — One object per input sequence:

        - **pdb** (*string*) — Predicted structure in PDB format
        - **mean_plddt** (*float*) — Mean predicted Local Distance Difference Test (pLDDT) score (0–100)
        - **ptm** (*float*) — Predicted TM-score (0–1)

   **Example response**:

   .. code-block:: json

      {
        "results": [
          {
            "pdb": "PARENT N/A\nATOM      1  N   MET A   1      -3.717 -20.294 -18.979  1.00 87.61           N  \nATOM      2  CA  MET A   1 ...",
            "mean_plddt": 94.27,
            "ptm": 0.92
          }
        ]
      }

Performance
-----------
- **GPU-accelerated** on A10G GPUs
- **Batch size**: up to 2 sequences per request
- **Sequence length**: up to 768 residues (total, including all chains)
- **Multimer support**: up to 4 chains per input (separated by ":")
- Typical completion: seconds to minutes per structure (depends on sequence length and chain count)

Applications
------------
- High-throughput single- and multi-chain protein structure prediction
- Metagenomic structure annotation and large-scale dataset curation
- Protein engineering, variant prioritization, and design workflows
- Structural biology, orphan protein, and complex modeling

Limitations
-----------
- **Sequence length**: ≤ 768 residues (total, all chains)
- **Batch size**: up to 2 sequences per request
- **Multimer support**: up to 4 chains per input (separated by ":")
- No template or MSA support (single-sequence only)
- Not intended for direct experimental design without further validation; outputs should be interpreted in the context of downstream workflows

How BioLM Uses ESMFold
----------------------
BioLM leverages ESMFold for:

- Automated, high-throughput structure prediction for protein engineering and metagenomic annotation
- Multi-chain and complex modeling for protein-protein interaction and interface analysis
- Orphan protein structure prediction and large-scale structural biology workflows
- Integration into iterative design, variant prioritization, and ML/AI pipelines

Related
-------
- :ref:`esm2_api` — Protein language modeling and feature extraction
- :ref:`esm1v_api` — Masked token predictions for single-residue functional scoring
- :ref:`esm_if1_api` — Inverse folding for backbone-guided sequence design

References
----------
- Lin, Z. *et al.* Evolutionary-scale prediction of atomic level protein structure with a language model. *Science* **379**, 1123 (2023).

