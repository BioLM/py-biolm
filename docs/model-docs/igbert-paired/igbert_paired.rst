IgBert Paired API
=================

IgBert Paired is an antibody-specific BERT language model trained on over two million naturally paired heavy/light variable region sequences from the Observed Antibody Space (OAS) dataset, after pretraining on unpaired antibody and protein sequences. The API exposes GPU-accelerated encode, generate, and log-probability endpoints for paired or unpaired inputs, supporting sequence embeddings (mean, per-residue, logits), masked-region sequence completion, and likelihood scoring for antibody engineering, affinity maturation, and therapeutic candidate prioritization workflows.

Predict
-------

Predict log probabilities (or other scores) for paired sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igbert-paired",
                action="predict",
                params={},
                items=[
                  {
                    "heavy": "EVQLVESGGGLVQ",
                    "light": "DIQMTQ",
                    "sequence": null
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igbert-paired/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "heavy": "EVQLVESGGGLVQ",
                  "light": "DIQMTQ",
                  "sequence": null
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igbert-paired/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "heavy": "EVQLVESGGGLVQ",
                      "light": "DIQMTQ",
                      "sequence": null
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igbert-paired/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  heavy = "EVQLVESGGGLVQ",
                  light = "DIQMTQ",
                  sequence = None
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igbert-paired/predict/

   Predict endpoint for IgBert Paired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to return; allowed values: "mean", "residue", "logits"


      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy chain amino acid sequence (standard amino acid codes plus allowed extended characters)

        - **light** (*string*, optional, min length: 1, max length: 256) — Light chain amino acid sequence (standard amino acid codes plus allowed extended characters)

        - **sequence** (*string*, optional, min length: 1, max length: 512) — Unpaired amino acid sequence (standard amino acid codes plus allowed extended characters)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igbert-paired/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "heavy": "EVQLVESGGGLVQ",
            "light": "DIQMTQ",
            "sequence": null
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

        - **embeddings** (*array of floats*, optional, size: 1024) — Mean embedding vector over all valid residues

        - **residue_embeddings** (*array of arrays of floats*, optional, dimensions: [sequence_length][1024]) — Per-residue embedding vectors

        - **logits** (*array of arrays of floats*, optional, dimensions: [sequence_length][vocabulary_size]) — Unnormalized prediction scores per residue position

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -34.742496490478516
          }
        ]
      }


Encode
------

Generate embeddings (mean, per-residue, etc.) from paired sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igbert-paired",
                action="encode",
                params={
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "heavy": "EVQLVESGGGLVQ",
                    "light": "DIQMTQ",
                    "sequence": null
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igbert-paired/encode/ \
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
                  "heavy": "EVQLVESGGGLVQ",
                  "light": "DIQMTQ",
                  "sequence": null
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igbert-paired/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean"
                    ]
                  },
                  "items": [
                    {
                      "heavy": "EVQLVESGGGLVQ",
                      "light": "DIQMTQ",
                      "sequence": null
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igbert-paired/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  heavy = "EVQLVESGGGLVQ",
                  light = "DIQMTQ",
                  sequence = None
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igbert-paired/encode/

   Encode endpoint for IgBert Paired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output representations to return; allowed values: "mean", "residue", "logits"

      - **items** (*array of objects*, min length: 1, max length: 32) --- Input entries:

        - **heavy** (*string*, min length: 1, max length: 256, optional) — Heavy chain amino acid sequence using the extended amino acid alphabet

        - **light** (*string*, min length: 1, max length: 256, optional) — Light chain amino acid sequence using the extended amino acid alphabet

        - **sequence** (*string*, min length: 1, max length: 512, optional) — Unpaired amino acid sequence using the extended amino acid alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igbert-paired/encode/ HTTP/1.1
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
            "heavy": "EVQLVESGGGLVQ",
            "light": "DIQMTQ",
            "sequence": null
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

        - **embeddings** (*array of floats*, size: 1024, optional) — Mean embedding vector for the encoded sequence

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 1024], optional) — Per-residue embedding vectors for the encoded sequence

        - **logits** (*array of arrays of floats*, shape: [sequence_length, vocab_size], optional) — Raw prediction logits for each token position; vocab_size is the tokenizer vocabulary size

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              -0.01601840928196907,
              -0.005099654197692871,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate new sequences from paired inputs containing mask placeholders (*)

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igbert-paired",
                action="generate",
                params={},
                items=[
                  {
                    "heavy": null,
                    "light": null,
                    "sequence": "EVQLVESGGGLVQDIQMTQ*"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igbert-paired/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "heavy": null,
                  "light": null,
                  "sequence": "EVQLVESGGGLVQDIQMTQ*"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igbert-paired/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "heavy": null,
                      "light": null,
                      "sequence": "EVQLVESGGGLVQDIQMTQ*"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igbert-paired/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  heavy = None,
                  light = None,
                  sequence = "EVQLVESGGGLVQDIQMTQ*"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igbert-paired/generate/

   Generate endpoint for IgBert Paired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to return; allowed values: "mean", "residue", "logits"

      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences for generation:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy-chain amino acid sequence using unambiguous amino acids and "*" as a placeholder
        - **light** (*string*, optional, min length: 1, max length: 256) — Light-chain amino acid sequence using unambiguous amino acids and "*" as a placeholder
        - **sequence** (*string*, optional, min length: 1, max length: 512) — Unpaired amino acid sequence using unambiguous amino acids and "*" as a placeholder

        Exactly one of the following must be provided:

        - Both **heavy** and **light**
        - **sequence** alone

        "*" must appear at least once in the concatenated **heavy** + **light** sequence or in **sequence**.

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igbert-paired/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "heavy": null,
            "light": null,
            "sequence": "EVQLVESGGGLVQDIQMTQ*"
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

        - **heavy** (*string*, optional) — Generated heavy chain amino acid sequence; length: 1–256 characters; characters: unambiguous amino acid single-letter codes (20 standard amino acids)

        - **light** (*string*, optional) — Generated light chain amino acid sequence; length: 1–256 characters; characters: unambiguous amino acid single-letter codes (20 standard amino acids)

        - **sequence** (*string*, optional) — Generated unpaired amino acid sequence; length: 1–512 characters; characters: unambiguous amino acid single-letter codes (20 standard amino acids)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "EVQLVESGGGLVQDIQMTQS"
          }
        ]
      }


Performance
-----------

- IgBert Paired runs on NVIDIA T4 GPUs with 6 GB memory per instance and uses the same 420M-parameter BERT encoder as IgBert Unpaired, configured for paired heavy–light variable regions.
- Compared with general protein language models (ProtBert, ProtT5), IgBert Paired achieves substantially higher masked-sequence recovery in antibody CDRs, especially CDRH3 (≈0.60 vs ProtBert ≈0.28 and ProtT5 ≈0.34), and improves over IgBert Unpaired across most CDR loops.
- For downstream regression from embeddings, IgBert Paired yields higher correlation with experimentally measured binding affinities (Pearson correlation up to ≈0.64;  $R^2$  up to ≈0.31) than IgBert Unpaired and antibody-agnostic models, indicating better capture of paired-chain determinants of antigen binding.
- On antibody sequence modeling metrics, IgBert Paired achieves lower pseudo-perplexity on paired variable regions (total ≈1.025) than IgBert Unpaired (≈1.031), AbLang (≈1.109), AntiBERTy (≈1.161), ProtBert (≈1.326) and ProtT5 (≈1.349), reflecting closer fit to observed antibody sequence distributions.

Applications
------------

- Sequence-level affinity maturation of paired heavy/light variable regions by ranking site-directed or combinatorial mutants using log-probability or pseudo-perplexity from the predictor endpoint, helping therapeutic antibody teams prioritise variants more likely to retain or improve antigen binding; less informative for properties dominated by Fc engineering or formulation conditions, which are outside the model’s sequence scope.
- Filtering antibody candidates by sequence naturalness (low pseudo-perplexity) using predictor scores on VH/VL pairs or unpaired V-regions, allowing discovery groups to down-select outlier sequences that may carry immunogenicity or manufacturability risk before expression testing; scores are sequence-based only and should be combined with structural models and developability assays for final decisions.
- Computing contextualised embeddings of antibody variable regions with the encoder endpoint for clustering, diversity analysis, and repertoire mining, enabling identification of convergent clonotypes or rare lineages across large NGS datasets for lead identification and lineage tracing; embeddings do not encode explicit antigen labels, so experimental binding assays remain necessary.
- In silico triage of large antibody libraries by combining generator and predictor endpoints to complete masked residues (*) in VH/VL pairs, then scoring generated variants to enrich for sequences consistent with the learned antibody distribution, reducing wet-lab screening burden; generation is most reliable for moderate local changes and may perform poorly for heavily engineered or non-immunoglobulin scaffolds.
- Local restoration of missing or low-quality residues in variable regions by passing sequences with '*' masks to the generator endpoint, improving the quality of repertoire datasets and downstream analyses such as clonal lineage reconstruction; not intended for reconstructing entire missing domains or long contiguous gaps, where predictions become uncertain and should be treated as hypotheses only.

Limitations
-----------

- **Maximum Sequence Length**: For paired inputs, each of ``heavy`` and ``light`` can be at most ``256`` amino acids. For unpaired inputs, ``sequence`` can be at most ``512`` amino acids. Requests exceeding these limits are rejected.
- **Batch Size**: The ``items`` list in ``encoder``, ``generator``, and ``predictor`` requests can contain at most ``32`` entries. Larger batches must be split across multiple requests.
- **Input Type Constraints**: Each item must be either paired (both ``heavy`` and ``light`` provided) or unpaired (``sequence`` provided). Mixing ``sequence`` with ``heavy``/``light`` in the same item, or omitting both, is invalid and will raise an error.
- **GPU Type**: IgBert endpoints are deployed on a T4-class GPU. Performance characteristics (latency, throughput) are tuned for this configuration and may differ from local runs on other accelerators.
- **Algorithmic Scope**: IgBert is trained on antibody variable-region sequences (heavy/light chains from OAS). It is not calibrated for non-antibody proteins or for properties like expression, stability, or structure where general protein models may perform better.
- **Use Case Suitability**: The API focuses on encoding (``encoder``), scoring (``predictor``), and masked-position completion (``generator`` with ``*`` placeholders). It is not intended for unconstrained sequence generation, full 3D structure prediction, or analysis of very novel formats (e.g. nanobodies with atypical annotations).

How We Use It
-------------

IgBert Paired enables end-to-end antibody engineering workflows that explicitly account for heavy–light chain coupling. We use its paired embeddings and log-probabilities to rank variants in affinity maturation campaigns, to score sequence naturalness and CDR mutations, and to guide focused library design around validated leads. Standardized, scalable API access lets us plug IgBert Paired into automated design–build–test cycles, where it complements general protein models and structure-based tools for developability and expression assessment.

- Supports joint heavy/light optimization for affinity, specificity, and sequence recovery in iterative design.
- Integrates with generative models and lab assay data to prioritize variants and refine antibody engineering strategies.

Related
-------

- ``IgT5 Paired`` – T5-based paired antibody language model trained on the same OAS dataset, useful for comparing embeddings or building ensembles with ``IgBert Paired``.
- ``IgBert Unpaired`` – IgBert variant trained on unpaired antibody sequences, suitable for encoding single chains or repertoires before pairing with ``IgBert Paired`` analyses.
- ``ABodyBuilder3 Language`` – Antibody structure prediction model that consumes language-model embeddings and can be combined with ``IgBert Paired`` encodings for sequence-to-structure workflows.
- ``AntiFold`` – Antibody structure prediction algorithm that can use ``IgBert Paired``-derived embeddings to improve structure modeling and design tasks.

References
----------

- Dreyer, F. A., & Kenlay, H. (2024). `Large scale paired antibody language models <https://zenodo.org/doi/10.5281/zenodo.10876908>`_. *Zenodo*.
- Olsen, T. H., Boyles, F., & Deane, C. M. (2022). `Observed antibody space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences <https://doi.org/10.1002/pro.4205>`_. *Protein Science*, 31(1), 141–146.

