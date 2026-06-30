.. _prostt5_api:

ProstT5 API
==========

ProstT5 is a bilingual protein language model capable of translating between amino acid (AA) sequences and 3Di structure tokens, enabling both sequence-to-structure (folding) and structure-to-sequence (inverse folding) tasks. ProstT5 supports two main actions: Encode (AA2fold, fold2AA) and Generate (AA2fold, fold2AA), with batch and sequence length limits as defined in the schema. BioLM provides scalable API access to ProstT5 for high-throughput protein design, structure annotation, and large-scale dataset curation.

Encode
------

This endpoint performs translation between amino acid (AA) sequences and 3Di structure tokens.

.. tab-set::

   .. tab-item:: Python (biolmai)
      :sync: sdk

      .. code-block:: python

         from biolmai import BioLM
         # AA2fold (amino acid to 3Di)
         response = BioLM(
             entity="prostt5",
             action="encode",
             params={"direction": "AA2fold"},
             items=[{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]
         )
         print(response)

         # fold2AA (3Di to amino acid)
         response = BioLM(
             entity="prostt5",
             action="encode",
             params={"direction": "fold2AA"},
             items=[{"sequence": "acdefghiklmnpqrstvwy"}]
         )
         print(response)

   .. tab-item:: Python Requests
      :sync: python

      .. code-block:: python

         import requests

         # AA2fold
         url = "https://biolm.ai/api/v3/prostt5/encode/"
         headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
         payload = {"params": {"direction": "AA2fold"}, "items": [{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]}
         resp = requests.post(url, json=payload, headers=headers)
         print(resp.json())

      .. code-block:: python

         import requests

         # fold2AA
         url = "https://biolm.ai/api/v3/prostt5/encode/"
         headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
         payload = {"params": {"direction": "fold2AA"}, "items": [{"sequence": "acdefghiklmnpqrstvwy"}]}
         resp = requests.post(url, json=payload, headers=headers)
         print(resp.json())

   .. tab-item:: R
      :sync: r

      .. code-block:: r

         # AA2fold
         library(httr)
         url <- "https://biolm.ai/api/v3/prostt5/encode/"
         headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
         body <- list(params = list(direction = "AA2fold"), items = list(list(sequence = "MKTAYIAKQRQISFVKSHFSRQ")))
         res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
         print(content(res))

         # fold2AA
         body <- list(params = list(direction = "fold2AA"), items = list(list(sequence = "acdefghiklmnpqrstvwy")))
         res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
         print(content(res))

   .. tab-item:: cURL
      :sync: curl

      .. code-block:: bash

         # AA2fold
         curl -X POST https://biolm.ai/api/v3/prostt5/encode/ \
           -H "Authorization: Token YOUR_API_KEY" \
           -H "Content-Type: application/json" \
           -d '{"params": {"direction": "AA2fold"}, "items": [{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]}'

         # fold2AA
         curl -X POST https://biolm.ai/api/v3/prostt5/encode/ \
           -H "Authorization: Token YOUR_API_KEY" \
           -H "Content-Type: application/json" \
           -d '{"params": {"direction": "fold2AA"}, "items": [{"sequence": "acdefghiklmnpqrstvwy"}]}'

.. http:post:: /api/v3/prostt5/encode

   Translation between amino acid (AA) sequences and 3Di structure tokens.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*) --- Configuration object:

        - **direction** (*string*) --- "AA2fold" or "fold2AA". Required. Selects translation direction.
      - **items** (*array of objects*, max. 16) --- List of input objects. Each object must have:

        - **sequence** (*string*, max length: 1000) --- Input sequence. For AA2fold: amino acid sequence. For fold2AA: 3Di sequence (a,c,d,e,f,g,h,i,k,l,m,n,p,q,r,s,t,v,w,y).

   **Example request**

   .. sourcecode:: http

      POST /api/v3/prostt5/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {"direction": "AA2fold"},
        "items": [
          {"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}
        ]
      }

   :statuscode 200: Successful response.
   :statuscode 400: Invalid input or parameter out of range.
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of arrays*) — For each input, a list of result objects:

        - **token** (*int*) — Token index
        - **token_str** (*string*) — 3Di token or AA
        - **score** (*float*) — Score for the translation
        - **sequence** (*string*) — Output sequence

   **Example response**

   .. code-block:: json

      {
        "results": [
          [
            {"token": 0, "token_str": "a", "score": 0.98, "sequence": "acdefghiklmnpqrstvwy"}
          ]
        ]
      }

Generate
--------

This endpoint generates sequences or structure tokens from input using ProstT5.

.. tab-set::

   .. tab-item:: Python (biolmai)
      :sync: sdk

      .. code-block:: python

         from biolmai import BioLM
         # AA2fold (amino acid to 3Di)
         response = BioLM(
             entity="prostt5",
             action="generate",
             params={"direction": "AA2fold", "temperature": 1.2, "top_p": 0.95, "top_k": 6, "repetition_penalty": 1.2, "num_samples": 1, "num_beams": 3},
             items=[{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]
         )
         print(response)

         # fold2AA (3Di to amino acid)
         response = BioLM(
             entity="prostt5",
             action="generate",
             params={"direction": "fold2AA", "temperature": 1.0, "top_p": 0.85, "top_k": 3, "repetition_penalty": 1.2, "num_samples": 1},
             items=[{"sequence": "acdefghiklmnpqrstvwy"}]
         )
         print(response)

   .. tab-item:: Python Requests
      :sync: python

      .. code-block:: python

         import requests

         url = "https://biolm.ai/api/v3/prostt5/generate/"
         headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
         payload = {"params": {"direction": "AA2fold", "temperature": 1.2, "top_p": 0.95, "top_k": 6, "repetition_penalty": 1.2, "num_samples": 1, "num_beams": 3}, "items": [{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]}
         resp = requests.post(url, json=payload, headers=headers)
         print(resp.json())

      .. code-block:: python

         import requests

         url = "https://biolm.ai/api/v3/prostt5/generate/"
         headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
         payload = {"params": {"direction": "fold2AA", "temperature": 1.0, "top_p": 0.85, "top_k": 3, "repetition_penalty": 1.2, "num_samples": 1}, "items": [{"sequence": "acdefghiklmnpqrstvwy"}]}
         resp = requests.post(url, json=payload, headers=headers)
         print(resp.json())

   .. tab-item:: R
      :sync: r

      .. code-block:: r

         # AA2fold
         library(httr)
         url <- "https://biolm.ai/api/v3/prostt5/generate/"
         headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
         body <- list(params = list(direction = "AA2fold", temperature = 1.2, top_p = 0.95, top_k = 6, repetition_penalty = 1.2, num_samples = 1, num_beams = 3), items = list(list(sequence = "MKTAYIAKQRQISFVKSHFSRQ")))
         res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
         print(content(res))

         # fold2AA
         body <- list(params = list(direction = "fold2AA", temperature = 1.0, top_p = 0.85, top_k = 3, repetition_penalty = 1.2, num_samples = 1), items = list(list(sequence = "acdefghiklmnpqrstvwy")))
         res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
         print(content(res))

   .. tab-item:: cURL
      :sync: curl

      .. code-block:: bash

         # AA2fold
         curl -X POST https://biolm.ai/api/v3/prostt5/generate/ \
           -H "Authorization: Token YOUR_API_KEY" \
           -H "Content-Type: application/json" \
           -d '{"params": {"direction": "AA2fold", "temperature": 1.2, "top_p": 0.95, "top_k": 6, "repetition_penalty": 1.2, "num_samples": 1, "num_beams": 3}, "items": [{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}]}'

         # fold2AA
         curl -X POST https://biolm.ai/api/v3/prostt5/generate/ \
           -H "Authorization: Token YOUR_API_KEY" \
           -H "Content-Type: application/json" \
           -d '{"params": {"direction": "fold2AA", "temperature": 1.0, "top_p": 0.85, "top_k": 3, "repetition_penalty": 1.2, "num_samples": 1}, "items": [{"sequence": "acdefghiklmnpqrstvwy"}]}'

.. http:post:: /api/v3/prostt5/generate/

   Generates sequences or structure tokens from input using ProstT5.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*) --- Generation parameters. Must include:

        - **direction** (*string*) --- "AA2fold" or "fold2AA". Required. Selects translation direction.
        - Other generation parameters as described below.

      - **items** (*array of objects*, max. 2) --- List of input objects. Each object must have:

        - **sequence** (*string*, max length: 512) --- Input sequence. For AA2fold: amino acid sequence. For fold2AA: 3Di sequence (a,c,d,e,f,g,h,i,k,l,m,n,p,q,r,s,t,v,w,y).

   **Example request**

   .. sourcecode:: http

      POST /api/v3/prostt5/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {"direction": "AA2fold", "temperature": 1.2, "top_p": 0.95, "top_k": 6, "repetition_penalty": 1.2, "num_samples": 1, "num_beams": 3},
        "items": [
          {"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}
        ]
      }

   :statuscode 200: Successful response.
   :statuscode 400: Invalid input or parameter out of range.
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of arrays*) — For each input, a list of result objects:

        - **sequence** (*string*) — Output sequence

   **Example response**

   .. code-block:: json

      {
        "results": [
          [
            {"sequence": "acdefghiklmnpqrstvwy"}
          ]
        ]
      }

Performance
-----------
- Supports batch processing (Encode: up to 16, Generate: up to 2 per request).
- Sequence length: up to 1000 (Encode), up to 512 (Generate).
- GPU: NVIDIA L4 (16GB RAM), 4 vCPU per request.
- Inference time: seconds per sequence on GPU; suitable for proteome-scale annotation (e.g., 1787 proteins in ~40 min on CPU, ~44s on GPU; human proteome in ~35 min on A6000 GPU).
- Encoder-only workflows (e.g., 3Di prediction via CNN) enable 10+ proteins/sec throughput on GPU.
- Embeddings and 3Di predictions benchmarked for secondary structure (Q3 up to ~90%), binding residue, conservation, and CATH/SCOPe classification tasks; remote homology detection ROC-AUC up to 0.47–0.49 (superfamily level).
- Inverse folding: ProstT5 can generate diverse sequences with similar structure (lDDT ~0.72, RMSD ~2.9, TM-score ~0.58, entropy closer to native than ProteinMPNN; see Table 2 in paper).
- Roundtrip accuracy (AA→3Di→AA) can be used to filter and improve generated sequence quality.

Applications
------------
- High-throughput protein design: Translate between sequence and structure representations for novel protein engineering, including generating diverse amino acid sequences for a given fold (inverse folding) and predicting 3Di structure tokens from sequence (folding).
- Structure annotation: Rapidly annotate large protein datasets with 3Di structure tokens, enabling proteome-scale and metagenomic-scale annotation in minutes, supporting remote homology detection at structure-comparison sensitivity (e.g., with Foldseek integration).
- Remote homology detection: Use ProstT5-predicted 3Di tokens as input to structure-based search tools (Foldseek), enabling sensitive detection of distant evolutionary relationships at orders-of-magnitude speedup over traditional structure prediction workflows.
- Antibody/nanobody/enzyme engineering: Design or analyze proteins with specific structural or functional constraints by translating between sequence and structure modalities, supporting structure-guided mutation effect prediction and candidate screening.
- Large-scale dataset curation: Annotate or generate millions of protein sequences/structures for downstream ML/AI model training, benchmarking, or structural phylogenetics, leveraging batch processing and high-throughput workflows.
- Roundtrip/Janus workflows: Combine both translation directions (AA→3Di→AA) to assess and filter generated sequences for structural fidelity, supporting advanced protein design and validation pipelines.

Limitations
-----------
- Not a general-purpose pLM; optimized for AA/3Di translation, not for all protein prediction tasks (e.g., subcellular location, function prediction may be less accurate than general pLMs).
- 3Di predictions are limited by the quality and diversity of input sequence/structure and may not capture all structural nuances; class imbalance in 3Di tokens can affect accuracy, especially for rare motifs.
- Inverse folding (fold2AA) may generate diverse but not necessarily experimentally validated or functional sequences; structural similarity is not guaranteed for all outputs.
- Sequence/structure length and batch size are limited by schema constraints (see above).
- Training data filtering (e.g., high pLDDT, short/structured proteins) may bias model toward well-structured, helical proteins and underrepresent intrinsically disordered or rare folds.
- Circularity: Some evaluation tasks (e.g., secondary structure, fold classification) may overlap with training data modalities, though practical utility is maintained.
- Model is not intended for direct experimental design without further validation; outputs should be interpreted in the context of downstream experimental or computational workflows.

How BioLM Uses ProstT5
----------------------
BioLM leverages ProstT5 for:

- Automated protein design workflows: Generate novel sequences for target folds (inverse folding), predict 3Di structure tokens for candidate sequences (folding), and combine both directions for roundtrip validation of design candidates.
- Large-scale annotation of protein datasets: Rapidly annotate proteomes or metagenomic datasets with 3Di structure tokens, enabling high-throughput remote homology detection and structure-based search (Foldseek integration) at scale.
- Antibody, nanobody, and enzyme engineering: Accelerate candidate screening and optimization by translating between sequence and structure, supporting structure-guided mutation effect prediction and design.
- Dataset curation for ML/AI: Generate or annotate millions of protein sequences/structures for downstream model training, benchmarking, or structural phylogenetics, leveraging ProstT5's speed and batch processing capabilities.
- Advanced scientific workflows: Enable Janus/roundtrip pipelines (AA→3Di→AA) to assess and filter generated sequences for structural fidelity, supporting robust protein engineering and validation.

Related
-------
- :ref:`foldseek_api` — Structure-based search tool
- :ref:`alphafold2_api` — Deep learning structure prediction for proteins

References
----------
- Heinzinger, M., Weissenow, K., Gomez Sanchez, J., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). Bilingual language model for protein sequence and structure. NAR Genomics and Bioinformatics. https://doi.org/10.1093/nargab/lqae150 