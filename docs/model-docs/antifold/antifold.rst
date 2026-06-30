AntiFold API
============

AntiFold is an antibody-specific inverse folding model fine-tuned from ESM-IF1 for structure-constrained design of variable domains, including nanobodies. It accepts IMGT-numbered antibody or antibody–antigen PDBs and returns per-residue mutation tolerance (perplexity), amino acid probabilities/logits, and optional embeddings, plus sequence-level scores. The API supports GPU-accelerated encode, predict, and generate endpoints, controlled sampling temperature, up to 50 000 sequences per target, and region-specific design across IMGT framework and CDR segments.

Predict
-------

Predict per-residue or global properties for the given antibody structure.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="antifold",
                action="predict",
                params={
                  "heavy_chain": "A",
                  "light_chain": "B",
                  "nanobody_chain": null,
                  "antigen_chain": null
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/antifold/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "heavy_chain": "A",
                "light_chain": "B",
                "nanobody_chain": null,
                "antigen_chain": null
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/antifold/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "heavy_chain": "A",
                    "light_chain": "B",
                    "nanobody_chain": null,
                    "antigen_chain": null
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/antifold/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                heavy_chain = "A",
                light_chain = "B",
                nanobody_chain = None,
                antigen_chain = None
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  
            ATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  
            END                                                                
            "
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/antifold/predict/

   Predict endpoint for AntiFold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **heavy_chain** (*string*, optional) — Chain ID for antibody heavy chain in the input PDB

        - **light_chain** (*string*, optional) — Chain ID for antibody light chain in the input PDB

        - **nanobody_chain** (*string*, optional) — Chain ID for nanobody chain in the input PDB

        - **antigen_chain** (*string*, optional) — Chain ID for antigen chain in the input PDB

        - **include** (*array of strings*, optional) — Additional outputs to include, allowed values:

          - "logprobs" — Softmax log probabilities for residues in the designed regions
          - "logits" — Raw logits before softmax for residues in the designed regions

        - **num_seq_per_target** (*integer*, range: 1-50000, default: 1) — Number of sequences to sample per input PDB

        - **sampling_temp** (*float*, range: 0.0-4.0, default: 0.2) — Temperature used when sampling residues for sequence generation

        - **regions** (*array of strings or integers*, optional, default: ["CDR1", "CDR2", "CDR3"]) — Regions or residue positions to sample mutations from, allowed string values:

          - "all"
          - "allH"
          - "allL"
          - "FWH"
          - "FWL"
          - "CDRH"
          - "CDRL"
          - "FW1"
          - "FWH1"
          - "FWL1"
          - "CDR1"
          - "CDRH1"
          - "CDRL1"
          - "FW2"
          - "FWH2"
          - "FWL2"
          - "CDR2"
          - "CDRH2"
          - "CDRL2"
          - "FW3"
          - "FWH3"
          - "FWL3"
          - "CDR3"
          - "CDRH3"
          - "CDRL3"
          - "FW4"
          - "FWH4"
          - "FWL4"
          - *integer values* — 1-based residue indices within the specified chain

        - **limit_expected_variation** (*boolean*, optional, default: false) — Whether to restrict mutations to expected variation

        - **exclude_heavy** (*boolean*, optional, default: false) — Whether to exclude the heavy chain from sequence sampling

        - **exclude_light** (*boolean*, optional, default: false) — Whether to exclude the light chain from sequence sampling


      - **items** (*array of objects*, min: 1, max: 1, required) --- Input antibody structures:

        - **pdb** (*string*, required, min length: 1, max length: 100000) — Antibody or antibody–antigen complex structure in PDB format containing only ATOM/HETATM records

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/antifold/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "heavy_chain": "A",
          "light_chain": "B",
          "nanobody_chain": null,
          "antigen_chain": null
        },
        "items": [
          {
            "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
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

        - **global_score** (*float*) — Mean log-likelihood score over all designed positions in the sequence

        - **heavy** (*string*) — Heavy-chain amino acid sequence used or generated for the input structure

        - **light** (*string*, optional) — Light-chain amino acid sequence used or generated for the input structure

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "global_score": 1.9169119596481323,
            "heavy": "M",
            "light": "M"
          }
        ]
      }


Encode
------

Generate embeddings for the specified heavy and light chains.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="antifold",
                action="encode",
                params={
                  "heavy_chain": "A",
                  "light_chain": "B",
                  "nanobody_chain": null,
                  "antigen_chain": null,
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/antifold/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "heavy_chain": "A",
                "light_chain": "B",
                "nanobody_chain": null,
                "antigen_chain": null,
                "include": [
                  "mean"
                ]
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/antifold/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "heavy_chain": "A",
                    "light_chain": "B",
                    "nanobody_chain": null,
                    "antigen_chain": null,
                    "include": [
                      "mean"
                    ]
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/antifold/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                heavy_chain = "A",
                light_chain = "B",
                nanobody_chain = None,
                antigen_chain = None,
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  
            ATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  
            END                                                                
            "
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/antifold/encode/

   Encode endpoint for AntiFold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **heavy_chain** (*string*, optional) — Chain identifier for antibody heavy chain (single character)

        - **light_chain** (*string*, optional) — Chain identifier for antibody light chain (single character)

        - **nanobody_chain** (*string*, optional) — Chain identifier for nanobody (single character)

        - **antigen_chain** (*string*, optional) — Chain identifier for antigen (single character)

        - **include** (*array of strings*, optional, default: ["mean"]) — Embedding types to include in the response, possible values: "mean", "residue", "logits"


      - **items** (*array of objects*, required, min length: 1, max length: 32) --- Input antibody structures:

        - **pdb** (*string*, required, min length: 1, max length: 100000) — Antibody structure in PDB format (ATOM/HETATM records only)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/antifold/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "heavy_chain": "A",
          "light_chain": "B",
          "nanobody_chain": null,
          "antigen_chain": null,
          "include": [
            "mean"
          ]
        },
        "items": [
          {
            "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
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

        - **embeddings** (*array of floats*, size: 512) — Mean inverse-folding embedding vector for the input structure

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 512], optional) — Per-residue inverse-folding embedding vectors

        - **logits** (*array of arrays of floats*, shape: [sequence_length, 20], optional) — Raw logits for each amino acid type at each residue position

        - **pdb_posins** (*array of integers*, length: sequence_length, optional) — Residue indices with insertion codes from the input PDB structure

        - **pdb_chain** (*array of strings*, length: sequence_length, optional) — Chain identifiers for each residue position from the input PDB structure

        - **pdb_res** (*array of strings*, length: sequence_length, optional) — Original amino acid residues from the input PDB structure

        - **top_res** (*array of strings*, length: sequence_length, optional) — Highest-logit amino acid residue at each position

        - **perplexity** (*array of floats*, length: sequence_length, optional) — Per-residue perplexity values derived from amino acid probabilities (range: ≥ 1.0)

        - **vocab** (*array of strings*, length: 20, optional) — Amino acid labels corresponding to the second dimension of logits and residue_embeddings arrays

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              0.02018200047314167,
              -0.09786378592252731,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate new sequences focused on selected CDR regions with control over sampling temperature.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="antifold",
                action="generate",
                params={
                  "heavy_chain": "A",
                  "light_chain": "B",
                  "nanobody_chain": null,
                  "antigen_chain": null,
                  "include": null,
                  "num_seq_per_target": 1,
                  "sampling_temp": 0.2,
                  "regions": [
                    "all"
                  ],
                  "limit_expected_variation": false,
                  "exclude_heavy": false,
                  "exclude_light": false
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/antifold/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "heavy_chain": "A",
                "light_chain": "B",
                "nanobody_chain": null,
                "antigen_chain": null,
                "include": null,
                "num_seq_per_target": 1,
                "sampling_temp": 0.2,
                "regions": [
                  "all"
                ],
                "limit_expected_variation": false,
                "exclude_heavy": false,
                "exclude_light": false
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/antifold/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "heavy_chain": "A",
                    "light_chain": "B",
                    "nanobody_chain": null,
                    "antigen_chain": null,
                    "include": null,
                    "num_seq_per_target": 1,
                    "sampling_temp": 0.2,
                    "regions": [
                      "all"
                    ],
                    "limit_expected_variation": false,
                    "exclude_heavy": false,
                    "exclude_light": false
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/antifold/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                heavy_chain = "A",
                light_chain = "B",
                nanobody_chain = None,
                antigen_chain = None,
                include = None,
                num_seq_per_target = 1,
                sampling_temp = 0.2,
                regions = list(
                  "all"
                ),
                limit_expected_variation = FALSE,
                exclude_heavy = FALSE,
                exclude_light = FALSE
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  
            ATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  
            END                                                                
            "
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/antifold/generate/

   Generate endpoint for AntiFold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **heavy_chain** (*string*, optional) — Chain ID for antibody heavy chain; must be present in at least one ``items[i].pdb``

        - **light_chain** (*string*, optional) — Chain ID for antibody light chain; must be present in at least one ``items[i].pdb``

        - **nanobody_chain** (*string*, optional) — Chain ID for nanobody; mutually exclusive with ``heavy_chain`` and ``light_chain``; must be present in at least one ``items[i].pdb``

        - **antigen_chain** (*string*, optional) — Chain ID for antigen; must be present in at least one ``items[i].pdb``

        - **include** (*array of strings*, optional) — Additional per-residue outputs to include; possible values:

          - logprobs
          - logits

        - **num_seq_per_target** (*int*, range: 1-50000, default: 1) — Number of sequences to generate for each input structure

        - **sampling_temp** (*float*, range: 0.0-4.0, default: 0.2) — Sampling temperature for sequence generation

        - **regions** (*array*, default: ["CDR1", "CDR2", "CDR3"]) — Regions or residue positions to use for sequence generation; each element is either a string from:

          - all
          - allH
          - allL
          - FWH
          - FWL
          - CDRH
          - CDRL
          - FW1
          - FWH1
          - FWL1
          - CDR1
          - CDRH1
          - CDRL1
          - FW2
          - FWH2
          - FWL2
          - CDR2
          - CDRH2
          - CDRL2
          - FW3
          - FWH3
          - FWL3
          - CDR3
          - CDRH3
          - CDRL3
          - FW4
          - FWH4
          - FWL4

          or an integer residue position (1-based index) valid for the specified chain(s)

        - **limit_expected_variation** (*boolean*, default: False, optional) — Whether to restrict design to residues with limited expected variation

        - **exclude_heavy** (*boolean*, default: False, optional) — Whether to exclude the heavy chain from sequence generation when both heavy and light chains are provided

        - **exclude_light** (*boolean*, default: False, optional) — Whether to exclude the light chain from sequence generation when both heavy and light chains are provided


      - **items** (*array of objects*, min: 1, max: 1, required) --- Input structures:

        - **pdb** (*string*, min length: 1, max length: 100000, required) — Protein structure in PDB format text; must contain the chains and residue positions referenced in ``params``

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/antifold/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "heavy_chain": "A",
          "light_chain": "B",
          "nanobody_chain": null,
          "antigen_chain": null,
          "include": null,
          "num_seq_per_target": 1,
          "sampling_temp": 0.2,
          "regions": [
            "all"
          ],
          "limit_expected_variation": false,
          "exclude_heavy": false,
          "exclude_light": false
        },
        "items": [
          {
            "pdb": "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \nATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \nEND                                                                \n"
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

        - **sequences** (*array of objects*) — Generated sequences and associated metrics:

          - **global_score** (*float*) — Mean log-likelihood over all designed positions
          - **score** (*float*) — Mean log-likelihood over the sampled region only
          - **heavy** (*string*) — Generated heavy-chain amino acid sequence
          - **light** (*string*, optional) — Generated light-chain amino acid sequence
          - **temperature** (*float*, range: 0.0–4.0) — Sampling temperature used for sequence generation
          - **mutations** (*int*) — Count of amino acid differences from the input sequence
          - **seq_recovery** (*float*, range: 0.0–1.0) — Fraction of residues identical to the input sequence

        - **logprobs** (*array of arrays of floats*, optional, shape: [sequence_length, 20]) — Per-position log-probabilities (natural log) for each amino acid

        - **logits** (*array of arrays of floats*, optional, shape: [sequence_length, 20]) — Per-position raw model logits for each amino acid

        - **pdb_posins** (*array of ints*, optional, length: sequence_length) — Residue indices including insertion codes, mapped from the input PDB

        - **pdb_chain** (*array of strings*, optional, length: sequence_length) — Chain identifiers mapped from the input PDB

        - **pdb_res** (*array of strings*, optional, length: sequence_length) — Original amino acid residues from the input PDB

        - **top_res** (*array of strings*, optional, length: sequence_length) — Highest-probability amino acid at each position

        - **perplexity** (*array of floats*, optional, length: sequence_length, range: ≥1.0) — Per-position perplexity derived from residue probabilities

        - **vocab** (*array of strings*, optional, length: 20) — Amino acid labels corresponding to the last dimension of ``logprobs`` and ``logits``

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequences": [
              {
                "global_score": 0.7118833065032959,
                "score": 0.7118833065032959,
                "heavy": "M",
                "light": "G",
                "temperature": 0.2,
                "mutations": 1,
                "seq_recovery": 0.5
              }
            ]
          }
        ]
      }


Performance
-----------

- AntiFold inference is deployed on CPU-backed instances with 2 GiB RAM, optimized for vectorized batched scoring and generation. For encoder/predictor endpoints, a single request can include up to 32 PDB structures; the generator endpoint processes 1 PDB per request with internal streaming to support large ``num_seq_per_target`` values (up to 50 000 samples).

- Compared with other inverse folding models available on BioLM (ProteinMPNN, AbMPNN, ESM-IF1), AntiFold provides higher antibody-specific design accuracy on the same PDB inputs:
  
  - CDRH3 amino acid recovery: 60% vs AbMPNN 56%, ESM-IF1 43%, ProteinMPNN 35%
  - All CDR loops: 75–84% vs AbMPNN 63–76%
  - Framework regions: 87–94% vs AbMPNN 85–89%

- Designed sequences generated via the AntiFold generator endpoint maintain backbone geometry on refolding more closely than alternative models on BioLM:
  
  - Mean CDR backbone RMSD: 0.95 Å vs AbMPNN 0.98 Å, ESM-IF1 1.01 Å, ProteinMPNN 1.03 Å (native refolded baseline 0.63 Å)

- AntiFold scores from encoder/predictor endpoints show stronger agreement with experimental antibody–antigen affinity and mutational fitness than other inverse folding models on BioLM:
  
  - Spearman correlation with deep mutational scan: 0.42 vs AbMPNN 0.32, ESM-IF1 0.33, ProteinMPNN 0.30
  - In language model–guided affinity maturation benchmarks, improved variants rank at a median of 80% vs ProteinMPNN 73%, ESM-IF1 57%, AbMPNN 55%

Applications
------------

- Antibody affinity maturation by generating backbone-constrained CDR variants from input VH/VL structures, helping therapeutic teams propose mutations that are likely to preserve the binding mode and structural fold; useful for prioritizing variants in lead-optimization campaigns; not optimal for predicting non-structural liabilities such as immunogenicity or aggregation on its own.
- Optimization of antibody humanization workflows by scoring humanized VH/VL variants for structural compatibility with a reference backbone, enabling rapid triage of humanization designs that are likely to disrupt the variable-domain structure; well suited as a structural filter alongside separate humanness or immunogenicity models rather than a replacement for them.
- Antibody and nanobody library design for display technologies by sampling structurally diverse, backbone-constrained CDR1–CDR3 sequences (including heavy-only or nanobody chains), allowing companies to construct focused libraries enriched for variants predicted to refold to the same variable-domain structure; not intended to estimate expression levels, display efficiency, or biophysical stability directly.
- Structural risk assessment of variants from deep mutational scanning or language-model–guided design by using AntiFold scores to flag mutations with low inverse-folding likelihood, supporting down-selection to variants predicted to retain variable-domain integrity and antigen binding when an antigen chain is provided; does not directly quantify specificity, off-target binding, or in vivo efficacy.
- Early computational triage of antibody and nanobody leads by scoring candidate structures (including AlphaFold/ABodyBuilder2 models) for tolerance to sequence changes across frameworks and CDRs, helping teams prioritize structurally robust scaffolds before expensive developability work; should be combined with dedicated in silico and experimental assays for stability, solubility, and manufacturability.

Limitations
-----------

- **Batch Size**: AntiFold ``predictor`` and ``encoder`` endpoints accept up to ``32`` ``items`` per request. The ``generator`` endpoint is limited to ``1`` ``item`` per request (``generate_batch_size = 1``). Design workflows that respect these limits to avoid validation errors.
- **Maximum PDB Length**: Each ``pdb`` string must be valid and no longer than ``max_pdb_str_len`` characters. Longer or malformed structures fail validation before any computation.
- **Antibody-specific scope**: AntiFold is fine-tuned for antibody variable domains and nanobodies (IMGT positions 1–128). It is not intended for arbitrary protein structures or non-antibody domains, and API predictions on such inputs may be inaccurate.
- **CDRH3 length sensitivity**: Sequence recovery and design quality are highest for typical CDRH3 lengths (≈6–9 residues) and degrade for unusually long loops (≈16+ residues). Extra structural or experimental validation is recommended when using the ``generator`` endpoint on long CDRH3s.
- **Backbone-constrained design**: AntiFold assumes the input backbone is approximately correct and explores sequence variation around that structure. It does not perform backbone redesign or de novo folding; for large conformational changes or full-structure prediction use dedicated models such as AlphaFold2 or ESMFold in your pipeline.
- **Affinity prediction context**: Affinity-related scores (e.g. ``global_score``, per-residue ``logits``/``logprobs``) are most informative when an ``antigen_chain`` is provided. Without antigen context, separating beneficial from deleterious mutations is less reliable, so combine AntiFold scores with complementary sequence-only or experimental assays.

How We Use It
-------------

AntiFold enables BioLM to target mutations in antibody variable domains that maintain the input backbone while modulating antigen-binding and developability-relevant properties. In practice, we use inverse-folding probabilities and per-residue perplexity from the encoder/predictor endpoints to score mutational scans and prioritize variants, then use the generator endpoint to sample focused CDR or framework designs under structural constraints, often in combination with protein language models and downstream predictive screens (e.g., affinity, stability, liability) to shrink the experimental search space.

- Supports affinity maturation by ranking language-model–proposed substitutions with structure-aware log-likelihoods, including optional antigen chains.
- Integrates with generative and property prediction models to iteratively refine CDRH3 and other IMGT-defined regions while preserving overall fold.

Related
-------

- ``ABodyBuilder3 Language`` – Predicts paired antibody structures from sequence, providing backbone models that AntiFold can use for inverse folding-based sequence design.
- ``ImmuneFold Antibody`` – Antibody-focused structure prediction to refold and validate AntiFold-designed sequences and assess structural stability.
- ``IgBert Paired`` – Antibody sequence language model that can score AntiFold-generated variants for repertoire-likeness, diversity and basic developability signals.
- ``AbLang-2`` – Antibody language model for post hoc filtering of AntiFold designs by humanness and developability, complementing AntiFold’s structure-constrained optimization.

References
----------

- Høie, M. H., Hummer, A., Olsen, T. H., Aguilar-Sanjuan, B., Nielsen, M., & Deane, C. M. (2024). AntiFold: Improved antibody structure-based design using inverse folding. *Bioinformatics Advances*, 5(1), vbae202. https://doi.org/10.1093/bioadv/vbae202
