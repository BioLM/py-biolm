nanoBERT API
============

nanoBERT is a transformer-based nanobody (VHH) language model for context-aware amino acid substitution prediction and sequence scoring. Trained on 10 million camelid nanobody sequences from INDI, it operates in a germline-agnostic manner and supports up to 154 residues per sequence and batches of up to 32 items. The API provides GPU-accelerated embeddings, masked-position infilling via "*" placeholders, and log-probability scoring for applications in nanobody design, humanization, and thermostability-focused modeling workflows.

Predict
-------

Predict log probabilities for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="nanobert",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "EVQLVESGGG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/nanobert/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "EVQLVESGGG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/nanobert/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "EVQLVESGGG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/nanobert/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "EVQLVESGGG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/nanobert/predict/

   Predict endpoint for nanoBERT.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output representations to include in the response; allowed values: "mean", "residue", "logits"

      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences:

        - **sequence** (*string*, required, min length: 1, max length: 154) — Amino acid sequence using standard unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/nanobert/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "EVQLVESGGG"
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

        - **embeddings** (*array of floats*, size: 320, optional) — Mean embedding vector for the input sequence (embedding dimension: 320)

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 320], optional) — Per-residue embedding vectors for each position in the input sequence (embedding dimension: 320)

        - **logits** (*array of arrays of floats*, shape: [sequence_length, 21], optional) — Raw prediction logits per sequence position for 21 output classes (unnormalized scores, no fixed range)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -0.002538938308134675
          }
        ]
      }


Encode
------

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="nanobert",
                action="encode",
                params={
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "sequence": "EVQLVESGGG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/nanobert/encode/ \
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
                  "sequence": "EVQLVESGGG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/nanobert/encode/"
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
                      "sequence": "EVQLVESGGG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/nanobert/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  sequence = "EVQLVESGGG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/nanobert/encode/

   Encode endpoint for nanoBERT.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output representations to return; allowed values: "mean", "residue", "logits"

      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences:

        - **sequence** (*string*, required, min length: 1, max length: 154) — Amino acid sequence using standard unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/nanobert/encode/ HTTP/1.1
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
            "sequence": "EVQLVESGGG"
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

        - **embeddings** (*array of floats*, size: 320, optional) — Mean embedding vector (length 320) returned when "mean" is included in params.include

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 320], optional) — Per-residue embedding vectors (sequence_length rows, each a length-320 float array) returned when "residue" is included in params.include

        - **logits** (*array of arrays of floats*, shape: [sequence_length, 21], optional) — Per-residue logits (sequence_length rows, each a length-21 float array) returned when "logits" is included in params.include

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              -0.29287976026535034,
              -0.00899506825953722,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate new sequences based on prompts with placeholder symbols

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="nanobert",
                action="generate",
                params={},
                items=[
                  {
                    "sequence": "EVQLVESGGG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/nanobert/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "EVQLVESGGG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/nanobert/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "EVQLVESGGG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/nanobert/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "EVQLVESGGG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/nanobert/generate/

   Generate endpoint for nanoBERT.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output types to include; allowed values are "mean", "residue", "logits"

      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences:

        - **sequence** (*string*, required, min length: 1, max length: 154) — Amino acid sequence with '*' placeholders optionally indicating positions to infill; allowed characters are standard unambiguous amino acids and '*'

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/nanobert/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "EVQLVESGGG"
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

        - **sequence** (*string*, length: 1-154 residues) — Generated nanobody amino acid sequence with '*' placeholders replaced by model predictions

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "EVQLVESGGG"
          }
        ]
      }


Performance
-----------

- nanoBERT inference is GPU-accelerated and optimized for lightweight deployment (≈14M parameters for nanoBERT_small vs. 86M for nanoBERT_big), enabling efficient sequence embedding and mutation scoring with low memory footprint (≈2 CPU cores, 2 GB RAM per worker).
- On natural llama nanobody V-regions, nanoBERT_small achieves substantially higher single-residue infilling accuracy than general protein LMs and human antibody LMs (76.9% vs. 57.4% for ESM-2 650M and ≈64–65% for human_320/human_640/AbLangHeavy), with similar performance to nanoBERT_big despite having ~6× fewer parameters.
- For highly variable CDR regions, nanoBERT_small improves nanobody-specific residue prediction over human antibody models (44.9% vs. 30.3% CDR accuracy for AbLangHeavy; CDR3: 32.3% vs. 18.8%), indicating better coverage of nanobody-specific mutational space than BioLM’s human-antibody language models.
- On therapeutic nanobody sequences, nanoBERT models reach 77.4% mean V-region accuracy, outperforming human antibody models (up to 73.9%) and ESM-2 (62.5%), and provide stronger nanobody-focused representations for downstream regression tasks (e.g., thermostability fine-tuning on NbThermo: Pearson r ≈0.5 vs. 0.39 for human_320 and ≈0.06 for randomly initialized heads).

Applications
------------

- Predicting feasible amino acid substitutions in therapeutic nanobodies by scoring masked positions with the predictor endpoint, enabling researchers to explore mutational space and prioritize variants that better match natural nanobody sequence statistics; particularly useful for antibody engineering teams working on nanobody-based therapeutics, but not optimal for canonical antibody formats or non-VHH scaffolds due to nanobody-specific training.
- Computational humanization of llama-derived nanobodies by comparing nanoBERT nativeness-style scores or positional preferences to human heavy-chain models, helping suggest framework mutations that increase human-likeness while retaining nanobody hallmark residues; valuable for biotech companies aiming to reduce immunogenicity risk of nanobody drugs, though experimental validation and orthogonal immunogenicity assessment remain essential.
- Fine-tuning nanoBERT embeddings on experimentally measured nanobody thermostability datasets (e.g., NBThermo) using the encoder endpoint to generate sequence representations, then training custom downstream models to predict stability-enhancing mutations; useful when experimental screening capacity is limited, but performance can be constrained by small or noisy training sets and requires users to implement their own fine-tuning pipelines outside the API.
- Generating nanobody-specific sequence embeddings via the encoder endpoint for downstream ML tasks such as predicting aggregation propensity or solubility, allowing bioinformatics teams to screen large VHH libraries and deprioritize candidates with unfavorable developability profiles early in discovery; embeddings are optimized for nanobody V-regions and may not generalize to canonical antibodies or unrelated protein scaffolds.
- Assessing sequence nativeness for quality control of synthetic nanobody libraries by using log probabilities from the predictor endpoint as a proxy for similarity to natural camelid repertoires, flagging sequences that deviate strongly from observed VHH distributions; useful for companies synthesizing large-scale nanobody libraries to maintain biological plausibility, but not suitable for evaluating canonical antibody, non-camelid, or non-immunoglobulin libraries.

Limitations
-----------

- **Maximum Sequence Length**: Input sequences are limited to ``154`` amino acids per item (``sequence`` field). Longer nanobody sequences must be truncated or split before submission.
- **Batch Size**: Requests are limited to a maximum of ``32`` sequences per API call (``items`` list). Larger jobs require multiple calls or external batching.
- **Sequence Type and Masking**: All endpoints only accept amino acid sequences using unambiguous residue codes; the ``generator`` endpoint additionally permits ``*`` as an infill placeholder and requires at least one ``*`` per ``sequence``. Non-standard tokens or other alphabets are rejected.
- **Domain and Training Data**: nanoBERT is trained on camelid VHH nanobody repertoires. It is not optimized for non-camelid antibodies, general proteins, or highly engineered sequences far from natural nanobody space, and may give less reliable scores or generations on such inputs.
- **Scope of Predictions**: API endpoints provide masked-sequence infilling (``generator``), sequence-level log probabilities (``predictor``), and embeddings/logits (``encoder`` with ``include`` values ``mean``, ``residue``, ``logits``). They do not compute 3D structures, binding affinities, nativeness scores, or thermostability directly; those tasks require separate models or custom fine-tuning.
- **Embeddings and Downstream Analysis**: ``mean`` and ``residue`` embeddings and per-position ``logits`` are intended as model features for downstream models. They are not pre-optimized for visualization, clustering, or zero-shot property prediction, so additional analysis (e.g., dimensionality reduction, supervised training) is typically required.

How We Use It
-------------

nanoBERT enables rapid exploration of nanobody mutational space by scoring and proposing context-aware substitutions, helping teams focus on variants that are both biologically plausible and aligned with desired developability profiles. Through standardized, scalable APIs for sequence scoring, masked infilling, and embeddings, it integrates into broader nanobody engineering workflows for humanization, thermostability optimization, and liability-aware lead refinement.

- Combines nativeness-style scoring (log probabilities) with embeddings to prioritize variants for experimental testing and downstream predictive models.
- Integrates with structure-based and developability metrics to guide multi-round design-test cycles while controlling sequence length and batch size constraints.

Related
-------

- ``NanoBodyBuilder2`` – Predicts 3D structures of nanobodies, complementing ``nanoBERT``’s sequence-based mutation scoring for end-to-end nanobody design.
- ``AbLang-2`` – Antibody-specific language model useful for comparing human antibody and nanobody sequence preferences alongside ``nanoBERT`` nativeness and mutational profiles.
- ``ABodyBuilder3 pLDDT`` – Builds antibody variable-domain structures with confidence scores; can assess structural plausibility of ``nanoBERT``-suggested nanobody variants by analogy to VH scaffolds.
- ``ImmuneFold Antibody`` – Antibody structure prediction model that can be applied to single-domain VH-like scaffolds to structurally evaluate designs informed by ``nanoBERT``.

References
----------

- Hadsund, J. T., Satława, T., Janusz, B., Shan, L., Zhou, L., Rottger, R., & Krawczyk, K. (2024). *nanoBERT: a deep learning model for gene agnostic navigation of the nanobody mutational space*. Bioinformatics. https://doi.org/10.1093/bioinformatics/btae216
