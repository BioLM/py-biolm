ESM-3 Open-Small API
====================

ESM-3 Open-Small is a 1.4B-parameter multimodal protein language model specialized in sequence, structure, and function prediction through a masked-token generative approach. It supports GPU-accelerated inference for tasks such as single-pass or iterative protein structure prediction (with up to O(L²) or O(L³) time, respectively), residue-level functional annotation, and controllable protein sequence design. The service handles structural tokens, amino acid tokens, and function tokens in various combinations to enable flexible prompting, for example partial structure constraints, amino acid motifs, or function keywords. Typical usage includes creating de novo protein designs with specific residue-level motifs, analyzing the effect of mutations on a given fold, and producing single-sequence or multi-sequence outputs for biochemical screening. By combining attention-based reasoning over large protein datasets with an optional iterative refinement mode, ESM-3 Open-Small can generate protein structures and sequences over lengths up to 2048 tokens, returning predicted coordinates, pTM and pLDDT confidence scores, and multi-track logits for advanced downstream applications.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ESM-3 Open-Small.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm3-open-small",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACDEFGHIKLMNPQRSTVWY"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm3-open-small/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACDEFGHIKLMNPQRSTVWY"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            url = "https://biolm.ai/api/v3/esm3-open-small/predict/"
            headers = {
                "Authorization": "Bearer YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                "items": [
                    {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
                ]
            }

            response = requests.post(url, headers=headers, json=payload)
            JSON(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm3-open-small/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
                # Request body would go here
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm3-open-small/predict/

   Predict endpoint for ESM-3 Open-Small.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **items** (*array of objects*, min items: 1, max items: 5) — Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Unambiguous amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm3-open-small/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACDEFGHIKLMNPQRSTVWY"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **pdb** (*string*) — Model-predicted protein structure in PDB format

        - **mean_plddt** (*float*, range: 0.0–100.0) — Mean pLDDT score over all residues

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "error": true,
        "status_code": 404,
        "message": "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n<HTML><HEAD><META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=iso-8859-1\">\n<TITLE>ERROR"
      }


Encode
------

This endpoint encodes for ESM-3 Open-Small.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm3-open-small",
                action="encode",
                params={
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "sequence": "ACDEFGHIKLMNPQRSTVWY"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm3-open-small/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean"
                ]
              },
              "items": [
                {
                  "sequence": "ACDEFGHIKLMNPQRSTVWY"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            url = "https://biolm.ai/api/v3/esm3-open-small/encode/"
            headers = {
                "Authorization": "Bearer YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                "params": {"include": ["mean"]},
                "items": [{"sequence": "ACDEFGHIKLMNPQRSTVWY"}]
            }

            response = requests.post(url, headers=headers, json=payload)
            JSON(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm3-open-small/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
                # Request body would go here
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm3-open-small/encode/

   Encode endpoint for ESM-3 Open-Small.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Possible values: "mean", "per_token", "logits"


      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required*) — Protein sequence with standard amino acids

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm3-open-small/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean"
          ]
        },
        "items": [
          {
            "sequence": "ACDEFGHIKLMNPQRSTVWY"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **embeddings** (*array of floats*, size: 1280, optional) — Mean embedding vector

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "error": true,
        "status_code": 404,
        "message": "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">\n<HTML><HEAD><META HTTP-EQUIV=\"Content-Type\" CONTENT=\"text/html; charset=iso-8859-1\">\n<TITLE>ERROR"
      }


Performance
-----------

- Utilizes an NVIDIA L4 GPU for inference, providing high throughput on typical protein annotation and structure-related tasks
- Achieves sequence annotation accuracy (e.g., function keyword prediction) within 1–2% of larger transformer-based systems, while using less GPU memory
- Delivers higher precision on structure prediction tasks compared to older ESM2-based pipelines, but runs slightly slower than ESM2 due to its multimodal layers
- Exhibits similar accuracy to ESMFold on single-chain protein modeling, with mean LDDT up to two percentage points higher on independently held-out PDB benchmarks
- Maintains faster generation speeds than AlphaFold2 during structure token sampling, though AlphaFold2 can yield modestly more accurate coordinates for certain complex targets
- For nanobody modeling, performs acceptably but remains less specialized than BioLM’s custom NanobodyBuilder, which is optimized purely for CDR topologies
- Input accepted via ESM3EncodeRequest or ESM3PredictRequest JSON objects; outputs include embeddings, logits, or PDB-formatted coordinates in ESM3EncodeResponse or ESM3PredictResponse respectively
- Log-probabilities of amino acid variants can be returned via ESM3LogProbRequest to identify favorable or deleterious substitutions, with results encapsulated in ESM3LogProbResponse


Applications
------------

- High-throughput enzyme engineering for industrial processes  
  This model can generate diverse protein variants with guided structures, enabling rapid testing of enzyme active site modifications for better catalytic efficiency, thermostability, or substrate specificity. It is valuable for companies optimizing production lines that rely on specialized enzymes, though it may require additional wet-lab screening since not all generated sequences are guaranteed to fold or function properly.

- Novel protein binder design for diagnostic reagents  
  By proposing new protein structures that incorporate desired binding pockets, ESM-3 helps researchers craft sensor proteins with high affinity to specific targets. This expedites creation of biosensors for medical or environmental assays, but designs must still undergo experimental verification to confirm selectivity and stability.

- Protein-ligand interface redesign to improve catalyst specificity  
  The model can identify variant residues near a ligand-binding site and propose modifications that might alter specificity. This is useful when engineering reaction pathways or improving enzyme selectivity in pharmaceutical synthesis. Generated designs often require iterative experimental validation to refine activity.

- Combinatorial exploration of protein domains for synthetic biology  
  ESM-3 can help test divergent configurations of domains within a single protein scaffold, predicting likely fold and function. Researchers integrating metabolic or regulatory modules benefit from computationally guided architecture choices, although certain novel constructs may prove unstable in vivo and need follow-up testing.

- Gathering sequence-function insights for rational design workflows  
  By providing scoring and predicted structure outputs, ESM-3 gives clues about which mutations can shift function or stability. This guides targeted mutagenesis strategies in the biotech industry. However, the model does not fully account for complex cellular effects, so lab assays remain necessary to confirm real-world performance.


Limitations
-----------

- **Maximum Sequence Length**: Each input sequence must be ≤ 2048 amino acids, and requests cannot exceed a total batch of 5 sequences (`max_sequence_len=2048` and `batch_size=5`).
- **Resource Requirements**: The “open-small” variant runs on a GPU (L4-class) and may be slow or memory-constrained on lower-grade hardware.
- **Reduced Viral Coverage**: Training data were filtered to remove most viral sequences and certain select agents, lowering performance in tasks involving viral proteins or toxin-related queries.
- **Structure Fidelity**: Large structural rearrangements or complex 3D motifs may not be captured fully, and longer iterative protocols (or other specialized tools) might be necessary for high-accuracy designs.
- **Limited Fine-tuning**: The model’s zero-shot capabilities on exotic protein families (e.g., fusion proteins or artificially engineered motifs) can be poor without additional domain-specific data or tuning.


How We Use It
-------------

ESM-3 Open-Small enables flexible protein engineering workflows by spanning sequence analysis, structural predictions, and functional annotations within a single model. It integrates seamlessly with BioLM’s broader pipelines—such as iterative machine learning refinement and lab-based validation—to accelerate the design of enzymes, antibodies, or other proteins. This approach reduces turnaround times for research teams that combine data science, biological expertise, and standardized APIs to quickly evaluate and optimize candidate molecules.

- It supports rapid integration with existing software pipelines for multi-round designs and empirical testing.
- Its modular API facilitates large-scale protein generation, filtering, and ideation in tandem with other data science tools.


Related
-------

- ``NanoBERT`` – Lightweight language model for short biosynthetic contexts, useful for preliminary analysis before ESM-3 Open-Small designs full sequences
- ``AlphaFold2`` – Advanced structure predictor that can cross-validate ESM-3 Open-Small outputs and refine 3D conformations for higher accuracy

