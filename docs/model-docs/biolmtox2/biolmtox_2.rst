BioLMTox-2 API
==============

BioLMTox-2 is a GPU-accelerated binary toxin classifier fine-tuned from the 650M-parameter ESM-2 model, providing toxin/not-toxin predictions and sequence embeddings directly from amino acid input without alignment or hand-crafted features. Trained on a unified dataset spanning multiple domains of life and sequence lengths, it achieves validation accuracy of 0.964 and recall of 0.984. The API supports batches of up to 16 sequences (≤2048 residues) for high-throughput biosecurity screening, therapeutic peptide design, and toxin mechanism studies.

Predict
-------

Predict toxin vs. not-toxin for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="biolmtox2",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                  },
                  {
                    "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/biolmtox2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                },
                {
                  "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
                }
              ],
              "params": {}
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/biolmtox2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                    },
                    {
                      "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
                    }
                  ],
                  "params": {}
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/biolmtox2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"
                ),
                list(
                  sequence = "GQSYFQPTNGVGYQPTNGVGYQPTN"
                )
              ),
              params = list()
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/biolmtox2/predict/

   Predict endpoint for BioLMTox-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **items** (*array of objects*, min: 1, max: 16) --- Input protein sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence containing only standard unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/biolmtox2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
          },
          {
            "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
          }
        ],
        "params": {}
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **label** (*string*, enum: "toxin", "not-toxin") — Predicted class label for the input sequence

        - **score** (*float*, range: 0.0–1.0) — Predicted probability for the returned label

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "label": "not-toxin",
            "score": 0.9999914169311523
          },
          {
            "label": "not-toxin",
            "score": 0.9999960660934448
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
                entity="biolmtox2",
                action="encode",
                params={},
                items=[
                  {
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                  },
                  {
                    "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/biolmtox2/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                },
                {
                  "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
                }
              ],
              "params": {}
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/biolmtox2/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                    },
                    {
                      "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
                    }
                  ],
                  "params": {}
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/biolmtox2/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"
                ),
                list(
                  sequence = "GQSYFQPTNGVGYQPTNGVGYQPTN"
                )
              ),
              params = list()
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/biolmtox2/encode/

   Encode endpoint for BioLMTox-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **items** (*array of objects*, min: 1, max: 16) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/biolmtox2/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
          },
          {
            "sequence": "GQSYFQPTNGVGYQPTNGVGYQPTN"
          }
        ],
        "params": {}
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **mean_representation** (*array of floats*, size: 1280) — Mean embedding vector for the input protein sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "mean_representation": [
              -0.03311070799827576,
              0.09799981117248535,
              "... (truncated for documentation)"
            ]
          },
          {
            "mean_representation": [
              -0.11304271966218948,
              0.0008354551391676068,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- **Hardware and deployment:** BioLMTox-2 is served on NVIDIA T4 GPUs with 16 GB RAM and 4 vCPUs, providing GPU-accelerated inference for batches of up to 16 protein sequences (maximum length 2048 amino acids).
- **Inference speed:** Typical latency is sub‑second per sequence for batches of 16 medium-length proteins, making BioLMTox-2 substantially faster than alignment‑based toxin classifiers (e.g., BLAST/PSI‑BLAST or PSI‑BLAST–dependent methods such as ATSE) and similar in throughput to other ESM‑2‑650M–based models hosted on BioLM.
- **Predictive performance:** On the unified BioLMTox benchmark validation set, BioLMTox-2 achieves accuracy 0.964, recall 0.984 (not‑toxin label), MCC 0.929, auROC 0.986, and auPRC 0.989. Across contemporary published benchmarks it matches or exceeds models such as ToxDL, TOXIFY, UniDL4BioPep, ToxIBTL/ToxIBTL‑VIB, and CSM‑Toxin, while underperforming ToxinPred2 by ~6.4 % accuracy on its own mixed‑toxin dataset (likely reflecting differing toxin label definitions.
- **Architecture and efficiency:** BioLMTox-2 is a fine‑tuned ESM‑2 650M transformer classifier that operates directly on single protein sequences without multiple‑sequence alignments or handcrafted physico‑chemical / graph‑based features. Compared with BioLM’s untuned ESM‑2‑650M encoders, BioLMTox‑2 provides substantially better toxin vs. not‑toxin separation in embedding space and higher classification accuracy at similar computational cost.

Applications
------------

- Rapid computational screening of engineered therapeutic proteins and peptides for predicted toxicity, enabling companies to filter large candidate libraries before expensive in vitro or in vivo studies; especially useful in high-throughput protein or peptide design cycles, while not replacing regulatory safety testing.
- Biosecurity risk assessment of novel protein sequences produced by generative design tools or directed evolution, allowing organizations to flag sequences with similarity to known toxic profiles prior to DNA synthesis or expression; helpful for mitigating dual-use concerns, but not expected to reliably identify entirely novel toxin mechanisms with no relationship to training data.
- Early-stage prioritization of peptide-based drug candidates by identifying sequences with a high predicted probability of toxicity, reducing downstream attrition and experimental load; particularly valuable for startups and small teams running iterative design–test–learn loops, and should be combined with standard toxicology assays.
- Safety evaluation of proteins expressed in agricultural biotechnology products, such as traits introduced into crop plants or microbial biocontrol agents, by providing a rapid first-pass toxin vs. not-toxin classification; can support internal risk assessments and dossier preparation, though regulatory approval still requires comprehensive experimental evidence.
- Design and screening of protein-based biomaterials (for example, scaffolds, hydrogels, or carrier proteins) by excluding sequences with high predicted toxin scores, helping teams bias early designs toward safer variants; best used as an initial computational filter that precedes more detailed biocompatibility and safety testing.

Limitations
-----------

- **Maximum Sequence Length**: Input ``sequence`` values must be between ``1`` and ``2048`` amino acids; longer proteins or peptides must be truncated before calling either ``encoder`` or ``predictor``.
- **Batch Size**: Each request to ``encoder`` or ``predictor`` must include between ``1`` and ``16`` ``items``; larger datasets must be split across multiple API calls.
- **Sequence Content Only**: BioLMTox-2 is a sequence-only classifier and embedding model. It does not use structural, evolutionary (MSA/PSSM), or physicochemical features, which may limit accuracy for highly novel, structurally unusual, or poorly annotated toxins.
- **Training Domain Bias**: The model was primarily trained on eukaryotic and bacterial proteins; performance may be reduced for sequences from underrepresented domains (e.g., archaea, viruses) or synthetic constructs far outside natural sequence distributions.
- **Output Scope**: The ``predictor`` endpoint returns a binary ``label`` (``"toxin"`` or ``"not-toxin"``) with a scalar ``score`` only; it does not provide toxin subtype, potency, target organism, or mechanism of action.
- **Homology and Similarity Edge Cases**: The model does not explicitly evaluate homologous relationships or enforce constraints on sequence similarity; predictions for proteins very close to known non-toxic homologs, or for borderline functional variants, should be interpreted with additional domain knowledge or experimental validation.

How We Use It
-------------

BioLMTox-2 enables rapid, scalable toxin risk assessment in protein engineering and therapeutic discovery pipelines by classifying candidate sequences early in design cycles and focusing wet-lab resources on safer constructs. Used together with generative design models, developability filters, and structural prediction tools, BioLMTox-2 supports iterative optimization where toxicity is screened at each round, reducing late-stage attrition and supporting internal biosecurity review for novel or low-homology sequences.

- Integrates into multi-round design workflows to score and down‑select batches of variants (up to 16 sequences, 2048 residues each per API call).
- Supports antibody, enzyme, and peptide therapeutic programs by systematically deprioritizing high-toxicity candidates before synthesis and in vivo testing.

Related
-------

- ``ESM-2 650M`` – BioLMTox-2 is fine-tuned from ESM-2 650M; use ESM-2 for compatible protein embeddings or additional feature extraction.
- ``Peptides`` – Predicts peptide bioactivity properties, complementing BioLMTox-2 by prioritizing therapeutic candidates among toxin-screened peptides.
- ``ESMFold`` – Predicts 3D structure from sequences, enabling structural analysis of proteins classified as toxins by BioLMTox-2.
- ``AlphaFold2`` – Provides high-accuracy structure prediction for deeper structural characterization of putative toxins identified by BioLMTox-2.

References
----------

- Challacombe, C. A., & Haas, N. S. (2024). *Towards a Dataset for State of the Art Protein Toxin Classification*. bioRxiv. https://doi.org/10.1101/2024.04.14.589430
