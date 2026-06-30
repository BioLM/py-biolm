CLEAN API
=========

CLEAN is a GPU-accelerated enzyme function prediction service that assigns EC numbers from amino acid sequences using contrastive learning over ESM-1b embeddings refined into a 128-dimensional, function-aware space. The API provides two endpoints: a predictor that returns ranked EC predictions with distance-based and GMM-derived confidence scores (configurable top-N and minimum confidence), and an encoder that outputs CLEAN embeddings. Typical applications include large-scale genome and metagenome annotation, pathway design, and enzyme engineering.

Predict
-------

Predict up to 5 EC numbers per sequence with at least 0.2 confidence

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="clean",
                action="predict",
                params={
                  "max_predictions": 5,
                  "min_confidence": 0.2
                },
                items=[
                  {
                    "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVK"
                  },
                  {
                    "sequence": "MKADKVKFGVEGLTYEEVRKDGDK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/clean/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "max_predictions": 5,
                "min_confidence": 0.2
              },
              "items": [
                {
                  "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVK"
                },
                {
                  "sequence": "MKADKVKFGVEGLTYEEVRKDGDK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/clean/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "max_predictions": 5,
                    "min_confidence": 0.2
                  },
                  "items": [
                    {
                      "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVK"
                    },
                    {
                      "sequence": "MKADKVKFGVEGLTYEEVRKDGDK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/clean/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                max_predictions = 5,
                min_confidence = 0.2
              ),
              items = list(
                list(
                  sequence = "MQIFVKTLTGKTITLEVEPSDTIENVK"
                ),
                list(
                  sequence = "MKADKVKFGVEGLTYEEVRKDGDK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/clean/predict/

   Predict endpoint for CLEAN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **max_predictions** (*int*, range: 1-20, default: 10) — Maximum number of EC predictions to return per sequence

        - **min_confidence** (*float*, range: 0.0-1.0, default: 0.05) — Minimum confidence score required to include a prediction in the results


      - **items** (*array of objects*, min: 1, max: 10) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 1022, required) — Amino acid sequence using the extended allowed alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/clean/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "max_predictions": 5,
          "min_confidence": 0.2
        },
        "items": [
          {
            "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVK"
          },
          {
            "sequence": "MKADKVKFGVEGLTYEEVRKDGDK"
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

        - **predictions** (*array of objects*) — Predicted EC numbers for the input sequence, ordered by distance (closest first)

          - **ec_number** (*string*) — Predicted EC number (e.g., "3.5.2.6")

          - **distance** (*float*) — Euclidean distance to the corresponding EC cluster center, unitless

          - **confidence** (*float*, range: 0.0-1.0) — GMM-based confidence score, unitless

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "predictions": [
              {
                "ec_number": "2.7.11.1",
                "distance": 6.6218,
                "confidence": 0.1642
              }
            ]
          },
          {
            "predictions": [
              {
                "ec_number": "3.4.24.50",
                "distance": 7.8765,
                "confidence": 0.0031
              }
            ]
          }
        ]
      }


Encode
------

Generate 128-dimensional CLEAN embeddings for two enzyme sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="clean",
                action="encode",
                params={},
                items=[
                  {
                    "sequence": "MKKTAIAIAVALAGFATAQA"
                  },
                  {
                    "sequence": "GHMDSNVLSKDVKAALEKAGY"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/clean/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKKTAIAIAVALAGFATAQA"
                },
                {
                  "sequence": "GHMDSNVLSKDVKAALEKAGY"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/clean/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKKTAIAIAVALAGFATAQA"
                    },
                    {
                      "sequence": "GHMDSNVLSKDVKAALEKAGY"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/clean/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKKTAIAIAVALAGFATAQA"
                ),
                list(
                  sequence = "GHMDSNVLSKDVKAALEKAGY"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/clean/encode/

   Encode endpoint for CLEAN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **items** (*array of objects*, min: 1, max: 10, required) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 1022, required) — Amino acid sequence using extended valid residue codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/clean/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKKTAIAIAVALAGFATAQA"
          },
          {
            "sequence": "GHMDSNVLSKDVKAALEKAGY"
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

        - **embedding** (*array of floats*, size: 128) — 128-dimensional CLEAN embedding for the input sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embedding": [
              0.31856825947761536,
              -0.5908583998680115,
              "... (truncated for documentation)"
            ]
          },
          {
            "embedding": [
              0.9185782074928284,
              -0.17309771478176117,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Predictive accuracy on held-out, low-identity data: on UniProt-derived tests where all sequences share <50% identity with training data, CLEAN reaches F1 ≈ 0.865 using its maximum-separation EC selection; even at a 10% identity threshold, F1 ≈ 0.67, indicating robustness beyond homology-based transfer
- Benchmark vs. other EC predictors: on the New-392 dataset (392 enzymes, 177 ECs, post-training Swiss-Prot release), CLEAN attains F1 ≈ 0.499 (precision ≈ 0.597, recall ≈ 0.481), outperforming ProteInfer (F1 ≈ 0.309) and DeepEC (F1 ≈ 0.230) under comparable evaluation; on the challenging misannotation dataset Price-149, CLEAN reaches F1 ≈ 0.495 vs. ProteInfer ≈ 0.166 and DeepEC ≈ 0.085
- Performance on rare and understudied EC numbers: on a curated validation set where each EC appears ≤5 times in training (>3000 samples, >1000 ECs), CLEAN achieves F1 ≈ 0.817, exceeding ProteInfer and DeepEC even though those models were originally trained on this set; EC-level precision and recall remain high across frequency bins, with substantially less bias toward common ECs than multilabel classification approaches
- Comparison to other BioLM models and similarity tools:
  
  - Versus general protein encoders (e.g., ESM-1b/ESM-2 available on BioLM), CLEAN’s contrastive training produces 128D embeddings where Euclidean distance tracks EC functional similarity, yielding markedly higher EC F1 than using raw ESM embeddings with simple classifiers or kNN
  - Versus sequence-similarity tools (e.g., DIAMOND on BioLM), CLEAN maintains higher precision/recall for remote homologs (<50% and down to <10% identity), promiscuous enzymes, and historically misannotated proteins, while DIAMOND can remain preferable for very high-identity, bulk annotation workloads

Applications
------------

- High-confidence EC annotation for novel and low-homology enzyme leads in discovery pipelines, enabling teams to move from metagenomic or proprietary sequencing data to actionable functional hypotheses when BLASTp and homology tools fail or disagree; CLEAN’s contrastive-learning embeddings and EC-cluster distance rankings are optimized to recover plausible EC numbers for understudied and rare functions, but they are not a replacement for wet-lab validation where regulatory or IP decisions depend on definitive mechanism-of-action data.
- Automated functional triage of large enzyme libraries (e.g., >10⁴–10⁶ variants) in directed evolution or semi-rational design campaigns, using CLEAN’s ranked EC predictions (with per-EC confidence scores and distance metrics) to filter, cluster, and prioritize sequences likely to retain desired catalytic activity or exhibit related transformations, thereby reducing experimental screening load; this is most effective when variants remain within a reasonable sequence distance of known enzymes and less suited for completely de novo, uncharacterizable folds.
- Detection and curation of misannotated or ambiguous enzymes in proprietary or public sequence collections, where CLEAN’s EC predictions can be compared against existing labels to highlight contradictions and potential corrections, helping companies clean internal knowledgebases, avoid propagating legacy database errors into downstream models, and flag candidates for targeted re-characterization rather than relying solely on automated homology-based annotations.
- Identification of candidate promiscuous enzymes with multiple EC activities to expand biocatalyst portfolios, by scanning sequence collections and using CLEAN to propose additional EC numbers with high confidence for single proteins; this supports discovery of versatile catalysts for multi-step chemoenzymatic processes and pathway shortcuts, while still requiring follow-up kinetic and substrate-scope measurements to quantify real-world utility and to downweight low-confidence secondary predictions.
- Enzyme selection for pathway design and metabolic engineering workflows, where CLEAN’s EC-aware embeddings (from the encoder endpoint) and ranked EC assignments (from the predictor endpoint) help choose plausible biocatalysts for missing steps in synthetic routes or retrobiosynthesis plans when no close homologs are known, allowing teams to assemble more complete in silico pathways from metagenomic or custom sequence datasets, while recognizing that cofactor specificity, expression feasibility, and process conditions must be evaluated with additional models or experiments beyond EC prediction alone.

Limitations
-----------

- **Maximum sequence length**: Each ``sequence`` must be a valid amino-acid string of length ``1``–``1022`` characters. Longer inputs must be truncated or pre-processed before calling ``clean``; the model is not aware of truncation and predictions on truncated sequences may lose critical functional context (for example, missing domains or termini).
- **Batch size and throughput**: Each request can contain at most ``10`` items in ``items`` for both ``CLEANPredictRequest`` and ``CLEANEncodeRequest``. Large datasets should be split into multiple batched calls; there is no cross-batch state, so users must handle any required aggregation (e.g., consensus across isoforms) themselves.
- **Prediction scope and confidence**: The model only predicts EC numbers that exist in its training set; genuinely novel enzyme functions or EC classes absent or extremely rare in UniProt may be mapped to the closest known EC and appear confident. The ``max_predictions`` (``1``–``20``, default ``10``) and ``min_confidence`` (``0.0``–``1.0``, default ``0.05``) fields in ``CLEANPredictRequestParams`` only filter which entries are returned in ``predictions``—they do not change the underlying model behavior. Users should treat low-confidence predictions (small ``confidence`` values or many ECs returned with similar ``distance``) as hypotheses, not annotations.
- **Not a general-purpose protein predictor**: CLEAN is specialized for enzyme EC annotation. It is not optimal for tasks such as protein stability prediction, binding affinity estimation, non-enzymatic functional annotation, or structure prediction; in those cases use task-specific models and, if needed, combine them with CLEAN predictions as one feature among many.
- **Embedding usage constraints**: ``CLEANEncodeResponse`` returns a single 128-dimensional ``embedding`` per ``sequence``, optimized for EC-related functional similarity. These embeddings are not guaranteed to reflect other properties (e.g., structure, localization, or expression) and may not be ideal as general-purpose protein embeddings. For applications like global protein clustering, visualization, or multitask models beyond enzyme function, consider combining CLEAN embeddings with broader protein language model embeddings.
- **Label noise and multi-function enzymes**: Because CLEAN is trained on curated but imperfect database annotations, systematic errors or inconsistencies in source EC labels can propagate into predictions. While the contrastive framework can help detect promiscuous enzymes and correct some mislabeled cases, the model may still miss rare secondary activities or over-call promiscuity. Experimental validation remains essential before using predicted EC numbers (from ``predictions[*].ec_number``) in high-impact design decisions.

How We Use It
-------------

CLEAN enables EC-aware enzyme function annotation that integrates directly into BioLM’s protein design workflows, accelerating the path from raw sequence collections to prioritized, experimentally testable candidates. Through scalable, standardized APIs, teams combine CLEAN’s function-aware embeddings and EC predictions with generative design models, structure-based scoring, and developability filters to triage metagenomic or proprietary libraries, refine legacy annotations, and surface understudied or promiscuous activities that unlock new biocatalysis and pathway engineering strategies.

- Used with BioLM’s design and stability/biophysics models, CLEAN helps define objective functions and screening criteria so wet-lab work focuses on variants with both desirable activities and manageable liabilities.  
- In iterative lab-in-the-loop campaigns, CLEAN’s embeddings and multi-EC outputs support multi-round optimization and faster convergence on enzymes suited for metabolic engineering, process development, and discovery programs.

Related
-------

- ``ESM-1b`` – Base protein language model used inside CLEAN; call directly when you need raw sequence embeddings or custom functional similarity metrics beyond EC-number prediction.
- ``ESM-2-650M`` – Newer protein language model that provides alternative embeddings; useful to compare with CLEAN’s 128D contrastive embeddings for downstream enzyme-function analyses.
- ``DIAMOND`` – Fast alignment-based sequence search; use alongside CLEAN to cross-check EC predictions at high sequence identity or to retrieve close homologs before model-based annotation.
- ``ZymCTRL`` – Enzyme-focused generative model; generate candidate enzyme sequences with ZymCTRL, then screen and annotate them with CLEAN’s EC predictions to enrich for desired activities.

References
----------

- Yu, T., Cui, H., Li, J. C., Luo, Y., Jiang, G., & Zhao, H. (2023). `Enzyme function prediction using contrastive learning <https://doi.org/10.1126/science.adf2465>`_. *Science*, 379(6639), 1358–1363.

- Yu, T., et al. (2023). `Enzyme function prediction using contrastive learning (code and data, version 1.0.0) <https://doi.org/10.5281/zenodo.7582241>`_. *Zenodo*.
