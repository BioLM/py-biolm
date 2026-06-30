Af2 Nim API
===========

AlphaFold-2 NIM is a 3-GPU Alphafold2 variant that predicts protein
structures and provides multiple-sequence alignments (MSAs).

.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai

Encode
------

Returns MSAs and template features for each input sequence.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            resp = BioLM(
                entity="af2-nim",
                action="encode",
                params={"return_templates": true},
                items=[{"sequence": "MEANLY..."}],
            )
            print(resp)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python

            import requests, os, json
            url = "https://biolm.ai/api/v3/af2-nim/encode/"
            headers = {
                "Authorization": f"Token {os.environ.get('BIOLM_TOKEN')}",
                "Content-Type": "application/json",
            }
            payload = {
              "params": {"return_templates": True},
              "items": [{"sequence": "MEANLY..."}]
            }
            res = requests.post(url, headers=headers, json=payload)
            print(res.json())

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash

            curl -X POST https://biolm.ai/api/v3/af2-nim/encode/ \
              -H "Authorization: Token $BIOLM_TOKEN" \
              -H "Content-Type: application/json" \
              -d '{"params": {"return_templates": true}, "items": [{"sequence": "MEANLY..."}]}'

    .. tab-item:: R
        :sync: r

        .. code-block:: r

            library(httr)
            url  <- "https://biolm.ai/api/v3/af2-nim/encode/"
            res  <- POST(
              url,
              add_headers(
                Authorization = sprintf("Token %s", Sys.getenv("BIOLM_TOKEN")),
                `Content-Type` = "application/json"
              ),
              body  = list(
                params = list(return_templates = TRUE),
                items  = list(list(sequence = "MEANLY..."))
              ),
              encode = "json"
            )
            print(content(res))

.. http:post:: /api/v3/af2-nim/encode/

   Returns MSAs and template features for each input sequence.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params.databases** (list[str]) — Which search DBs to use
      - **params.return_templates** (bool) — Include template hits
      - **params.msa_iterations** (int) — Jackhmmer/mmseq2 iterations
      - **params.max_msa_sequences** (int) — Cap on extracted MSA size
      - **params.algorithm** (str) — "jackhmmer" or "mmseqs2"
      - **items[].sequence** (str) — AA sequence ≤ 512 residues

   **Example request**::

      {
        "params": {"return_templates": true},
        "items": [{"sequence": "MEANLY..."}]
      }

   :statuscode 200: Successful.
   :statuscode 400: Validation error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results[].alignments.small_bfd** (list[str]) — Small-BFD MSA
      - **results[].alignments.mgnify** (list[str]) — MGnify MSA
      - **results[].alignments.uniref90** (list[str]) — UniRef90 MSA
      - **results[].templates** (list[obj]) — Top structural templates

   **Example response**::

      {
        "results": [
          {
            "alignments": {
              "small_bfd": ["..."],
              "mgnify": ["..."],
              "uniref90": ["..."]
            },
            "templates": [
              {"index": 0, "name": "1XYZ_A", "aligned_cols": 200, ...}
            ]
          }
        ]
      }

Predict
-------

Returns predicted structure files (PDB) for each input sequence.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            resp = BioLM(
                entity="af2-nim",
                action="predict",
                params={"predictions_per_model": 1},
                items=[{"sequence": "MEANLY..."}],
            )
            print(resp)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python

            import requests, os, json
            url = "https://biolm.ai/api/v3/af2-nim/predict/"
            headers = {
                "Authorization": f"Token {os.environ.get('BIOLM_TOKEN')}",
                "Content-Type": "application/json",
            }
            payload = {
              "params": {"predictions_per_model": 1},
              "items": [{"sequence": "MEANLY..."}]
            }
            res = requests.post(url, headers=headers, json=payload)
            print(res.json())

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash

            curl -X POST https://biolm.ai/api/v3/af2-nim/predict/ \
              -H "Authorization: Token $BIOLM_TOKEN" \
              -H "Content-Type: application/json" \
              -d '{"params": {"predictions_per_model": 1}, "items": [{"sequence": "MEANLY..."}]}'

    .. tab-item:: R
        :sync: r

        .. code-block:: r

            library(httr)
            url  <- "https://biolm.ai/api/v3/af2-nim/predict/"
            res  <- POST(
              url,
              add_headers(
                Authorization = sprintf("Token %s", Sys.getenv("BIOLM_TOKEN")),
                `Content-Type` = "application/json"
              ),
              body  = list(
                params = list(predictions_per_model = 1),
                items  = list(list(sequence = "MEANLY..."))
              ),
              encode = "json"
            )
            print(content(res))

.. http:post:: /api/v3/af2-nim/predict/

   Returns predicted structure files (PDB) for each input sequence.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params.databases** (list[str]) — Which search DBs to use
      - **params.predictions_per_model** (int) — # of samples per AF2 model
      - **params.relax** (str) — "all", "best" or "none"
      - **params.msa_iterations** (int) — MSA search iterations
      - **items[].sequence** (str) — AA sequence ≤ 512 residues

   **Example request**::

      {
        "params": {"predictions_per_model": 1},
        "items": [{"sequence": "MEANLY..."}]
      }

   :statuscode 200: Successful.
   :statuscode 400: Validation error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results[].pdbs** (list[str]) — Signed HTTPS URLs to PDB files

   **Example response**::

      {
        "results": [
          {
            "pdbs": [
              "https://storage.googleapis.com/biolm-af2/xyz123_unrelaxed.pdb"
            ]
          }
        ]
      }

Performance
-----------

- 3 × A100-80 GB, 24 vCPU, 96 GB RAM  
- ~6 min per 300 AA sequence (full DB search & structure)

Applications
------------

- High-accuracy monomer structure prediction
- Template-free modelling for novel proteins
- Downstream feature extraction for ML

Limitations
-----------

- Max sequence length 512 AA
- Throughput ≤ 1 job per request (batch size 1)
- Requires heavy GPU/CPU resources

References
----------

- Jumper J *et al.* Nature (2021) "High-Accuracy Protein Structure Prediction with AlphaFold 2". 