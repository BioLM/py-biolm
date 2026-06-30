IgBert Unpaired API
===================

IgBert Unpaired is an antibody-specific masked language model (MLM) trained on over two billion unpaired antibody variable-region sequences from the Observed Antibody Space (OAS) dataset. Based on a ProtBert architecture with 420M parameters and a 30-layer transformer encoder, it supports single-chain variable-region inputs up to 512 residues via the API. The service returns residue-level and mean embeddings, logits, masked-sequence recovery, and log probabilities for downstream antibody discovery and engineering workflows.

Predict
-------

Predict properties or scores for input sequences (log probability)

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igbert-unpaired",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "GHIKLMNPQRSTVWYACDEF"
                  },
                  {
                    "sequence": "NPQRSTVWYACDEFGHIKLM"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igbert-unpaired/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "GHIKLMNPQRSTVWYACDEF"
                },
                {
                  "sequence": "NPQRSTVWYACDEFGHIKLM"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igbert-unpaired/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "GHIKLMNPQRSTVWYACDEF"
                    },
                    {
                      "sequence": "NPQRSTVWYACDEFGHIKLM"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igbert-unpaired/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "GHIKLMNPQRSTVWYACDEF"
                ),
                list(
                  sequence = "NPQRSTVWYACDEFGHIKLM"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igbert-unpaired/predict/

   Predict endpoint for IgBert Unpaired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding outputs to return for each item. Allowed values: "mean", "residue", "logits".

      - **items** (*array of objects*, min: 1, max: 32) --- Input antibody sequences:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy chain amino acid sequence (validated extended amino acid codes).

        - **light** (*string*, optional, min length: 1, max length: 256) — Light chain amino acid sequence (validated extended amino acid codes).

        - **sequence** (*string*, optional, min length: 1, max length: 512) — Single unpaired amino acid sequence (validated extended amino acid codes).

        .. note::

          Exactly one of the following must be provided per item:

          - Both **heavy** and **light** sequences (paired input)
          - **sequence** alone (unpaired input)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igbert-unpaired/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "GHIKLMNPQRSTVWYACDEF"
          },
          {
            "sequence": "NPQRSTVWYACDEFGHIKLM"
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

        - **embeddings** (*array of floats*, size: 1024, optional) — Mean embedding vector for the input sequence

        - **residue_embeddings** (*array of arrays of floats*, size: [sequence_length, 1024], optional) — Per-residue embedding vectors for the input sequence

        - **logits** (*array of arrays of floats*, size: [sequence_length, vocabulary_size], optional) — Per-position logits over the model vocabulary

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -11.413089752197266
          },
          {
            "log_prob": -14.090692520141602
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
                entity="igbert-unpaired",
                action="encode",
                params={
                  "include": [
                    "mean",
                    "residue"
                  ]
                },
                items=[
                  {
                    "sequence": "ACDEFGHIKLMNPQRSTVWY"
                  },
                  {
                    "sequence": "MKTIIALSYIFCLVFADYKDDDDK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igbert-unpaired/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean",
                  "residue"
                ]
              },
              "items": [
                {
                  "sequence": "ACDEFGHIKLMNPQRSTVWY"
                },
                {
                  "sequence": "MKTIIALSYIFCLVFADYKDDDDK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igbert-unpaired/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean",
                      "residue"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "ACDEFGHIKLMNPQRSTVWY"
                    },
                    {
                      "sequence": "MKTIIALSYIFCLVFADYKDDDDK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igbert-unpaired/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean",
                  "residue"
                )
              ),
              items = list(
                list(
                  sequence = "ACDEFGHIKLMNPQRSTVWY"
                ),
                list(
                  sequence = "MKTIIALSYIFCLVFADYKDDDDK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igbert-unpaired/encode/

   Encode endpoint for IgBert Unpaired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Output types to include in response:

          - Allowed values:

            - "mean" — Mean embedding representation
            - "residue" — Per-residue embedding representations
            - "logits" — Model logits

      - **items** (*array of objects*, min: 1, max: 32) --- Input antibody sequences:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy chain amino acid sequence (standard amino acid codes plus extended set)
        - **light** (*string*, optional, min length: 1, max length: 256) — Light chain amino acid sequence (standard amino acid codes plus extended set)
        - **sequence** (*string*, optional, min length: 1, max length: 512) — Unpaired amino acid sequence (standard amino acid codes plus extended set)

        - **Constraints**:

          - Provide either both "heavy" and "light" sequences (paired input), or a single "sequence" (unpaired input)
          - Cannot provide both ("heavy", "light") and "sequence" simultaneously

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igbert-unpaired/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean",
            "residue"
          ]
        },
        "items": [
          {
            "sequence": "ACDEFGHIKLMNPQRSTVWY"
          },
          {
            "sequence": "MKTIIALSYIFCLVFADYKDDDDK"
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

        - **embeddings** (*array of floats*, size: 1024) — Mean protein embeddings
        - **residue_embeddings** (*array of arrays of floats*, size: [variable_length, 1024]) — Per-residue embeddings
        - **logits** (*array of arrays of floats*, size: [variable_length, 20]) — Logits for each amino acid position

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              0.05880357697606087,
              0.02711423859000206,
              "... (truncated for documentation)"
            ],
            "residue_embeddings": [
              [
                0.0,
                0.0,
                "... (truncated for documentation)"
              ],
              [
                -0.06851226091384888,
                -0.05662147328257561,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "embeddings": [
              -0.013091325759887695,
              -0.015862608328461647,
              "... (truncated for documentation)"
            ],
            "residue_embeddings": [
              [
                0.0,
                0.0,
                "... (truncated for documentation)"
              ],
              [
                0.009838802739977837,
                -0.04168888181447983,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate new sequences based on prompts (with '*' placeholders)

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igbert-unpaired",
                action="generate",
                params={},
                items=[
                  {
                    "sequence": "ACDEFGHIKL*NPQRSTVWY"
                  },
                  {
                    "sequence": "MKTIIALSYI*CLVFADYKDDDDK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igbert-unpaired/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACDEFGHIKL*NPQRSTVWY"
                },
                {
                  "sequence": "MKTIIALSYI*CLVFADYKDDDDK"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igbert-unpaired/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACDEFGHIKL*NPQRSTVWY"
                    },
                    {
                      "sequence": "MKTIIALSYI*CLVFADYKDDDDK"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igbert-unpaired/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACDEFGHIKL*NPQRSTVWY"
                ),
                list(
                  sequence = "MKTIIALSYI*CLVFADYKDDDDK"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igbert-unpaired/generate/

   Generate endpoint for IgBert Unpaired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to include in the response. Allowed values: "mean", "residue", "logits"

      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences with masked positions for generation:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy chain amino acid sequence using standard unambiguous codes with "*" as a placeholder, must contain at least one "*" when combined with light if both are provided
        - **light** (*string*, optional, min length: 1, max length: 256) — Light chain amino acid sequence using standard unambiguous codes with "*" as a placeholder, must contain at least one "*" when combined with heavy if both are provided
        - **sequence** (*string*, optional, min length: 1, max length: 512) — Single-chain amino acid sequence using standard unambiguous codes with at least one "*" placeholder for generation

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igbert-unpaired/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACDEFGHIKL*NPQRSTVWY"
          },
          {
            "sequence": "MKTIIALSYI*CLVFADYKDDDDK"
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

        - **embeddings** (*array of floats*, size: 1024, optional) — Mean sequence embeddings

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 1024], optional) — Per-residue embeddings

        - **logits** (*array of arrays of floats*, shape: [sequence_length, vocabulary_size], optional) — Per-position logits over the amino acid vocabulary

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **heavy** (*string*, optional) — Generated heavy chain sequence with masked positions filled

        - **light** (*string*, optional) — Generated light chain sequence with masked positions filled

        - **sequence** (*string*, optional) — Generated unpaired sequence with masked positions filled

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **log_prob** (*float*) — Log probability of the provided sequence under the model

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "ACDEFGHIKLLNPQRSTVWY"
          },
          {
            "sequence": "MKTIIALSYILCLVFADYKDDDDK"
          }
        ]
      }


Performance
-----------

- IgBert Unpaired runs on NVIDIA T4 GPUs with resources sized for up to 32 sequences per request, providing GPU-accelerated embedding generation for unpaired antibody variable regions.
- Compared with IgT5 Unpaired (3B parameters), IgBert Unpaired (420M parameters) is approximately 3–5× faster for embedding generation at similar sequence lengths, while maintaining comparable antibody-specific accuracy on sequence recovery tasks.
- Relative to general protein language models such as ProtBert, IgBert Unpaired achieves substantially higher accuracy in antibody sequence recovery, especially in hypervariable CDR loops (e.g., 90.43% vs. 65.60% accuracy for CDRH1), and yields lower pseudo-perplexity over VH/VL (1.031 vs. 1.326), indicating better modeling of antibody sequence distributions.
- On antibody binding affinity prediction benchmarks (Shanehsazzadeh et al., Warszawski et al., Koenig et al.), linear models trained on IgBert Unpaired embeddings achieve R² scores ~0.08–0.18 higher than ProtBert, while paired IgBert embeddings perform slightly better still (R² improvement of ~0.02–0.03 over IgBert Unpaired) due to access to cross-chain information.

Applications
------------

- Antibody variable-region sequence recovery for therapeutic design and lead optimization, using masked positions (e.g., unknown residues or low-quality NGS calls) in heavy or light chains to reconstruct plausible amino acids, helping restore incomplete candidates before downstream wet-lab validation.
- Binding affinity–related feature extraction from IgBert embeddings, which can be fed into in-house regression/classification models to rank antibody variants against a given antigen, reducing the number of low-quality candidates entering experimental affinity maturation campaigns. IgBert captures antibody-specific and cross-chain patterns, but it does not directly output binding affinities via the API.
- Expression-related feature extraction for engineered antibodies, where sequence-level embeddings from the encoder endpoint are used as input to custom expression or developability predictors, supporting prioritization of variants with more favorable expression profiles; general protein models may still outperform IgBert on some expression tasks, so it is best used alongside other models.
- Cross-chain representation learning for paired heavy/light design, by encoding both chains together via the paired API mode to obtain embeddings that reflect inter-chain context at the variable-region interface, aiding light-chain shuffling studies and interface-focused optimization while still requiring external models to translate embeddings into specific developability or affinity scores.
- CDR-focused sequence recovery and diversification for antibody engineering, by using the generator endpoint with masked or placeholder residues (e.g., “*”) in CDR loops to propose alternative amino acids, enabling targeted CDR exploration around an existing scaffold rather than full de novo design; generated variants should be filtered and validated with additional in silico and experimental assays.

Limitations
-----------

- **Maximum Sequence Length**: The ``heavy`` and ``light`` chain fields each support up to ``256`` amino acids, and the unpaired ``sequence`` field supports up to ``512`` amino acids. Longer antibody variable regions must be truncated or split before calling the API.
- **Batch Size**: The maximum number of items per request (``items`` list length) is ``32``. Larger datasets must be sent as multiple requests and aggregated client-side.
- **Paired vs Unpaired Inputs**: Each item must be either paired (both ``heavy`` and ``light`` set) or unpaired (only ``sequence`` set), but not both. The unpaired IgBert model does not use heavy–light pairing, so it is less suitable when cross-chain context is essential (for example, some binding affinity or developability tasks).
- **Use of Embeddings for Expression Tasks**: IgBert embeddings are tuned to antibody-specific sequence patterns and downstream tasks like sequence recovery and binding-related properties. General protein models (for example, ProtT5) can achieve better performance on broad properties such as expression level, so IgBert may not be optimal if expression prediction is your primary goal.
- **Generative and Design Workflows**: IgBert is trained with a masked language modelling objective. The ``generator`` endpoint supports in-filling around ``"*"`` placeholders, but the model is not an autoregressive designer. For large-scale *de novo* antibody design or iterative affinity maturation campaigns, encoder–decoder or causal models are often more appropriate, with IgBert used mainly for encoding, ranking, or log-probability scoring via the ``encoder`` and ``predictor`` endpoints.

How We Use It
-------------

IgBert Unpaired embeddings enable scalable assessment of antibody variable region sequences by capturing repertoire-wide patterns learned from billions of unpaired antibodies. Within BioLM workflows, these standardized embeddings feed downstream predictive and generative models to guide sequence selection, log-probability–based “naturalness” scoring, and focused in silico diversification, reducing experimental screening burden while maintaining developability constraints.

- Used to rank and filter large in silico antibody libraries by model-derived fitness or naturalness scores before synthesis.
- Combined with structure-based tools and developability predictors to balance affinity, specificity, and biophysical risk during antibody optimization cycles.

Related
-------

- ``IgBert Paired`` – Trained on concatenated heavy/light chains and exposed via the same API, providing cross-chain context that often improves embeddings and log-probability–based property modeling over IgBert Unpaired.
- ``IgT5 Unpaired`` – Antibody-specific T5 encoder that offers alternative sequence embeddings for unpaired chains, useful for comparing against IgBert Unpaired or building simple ensembles.
- ``ABodyBuilder3 Language`` – Antibody language model used to enhance structure prediction, complementary to IgBert Unpaired when linking sequence embeddings to 3D modeling workflows.
- ``AbLang-2`` – BERT-based antibody language model suited for sequence completion and representation learning, enabling direct architectural comparisons or ensemble use with IgBert Unpaired.

References
----------

- Dreyer, F. A., & Kenlay, H. (2024). `Large scale paired antibody language models <https://zenodo.org/doi/10.5281/zenodo.10876908>`_. *Zenodo*.
- Olsen, T. H., Boyles, F., & Deane, C. M. (2022). `Observed antibody space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences <https://doi.org/10.1002/pro.4205>`_. *Protein Science*, 31(1), 141–146.
- Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D., & Rost, B. (2022). `ProtTrans: Toward understanding the language of life through self-supervised learning <https://doi.org/10.1109/TPAMI.2021.3095381>`_. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10), 7112–7127.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). `BERT: Pre-training of deep bidirectional transformers for language understanding <https://arxiv.org/abs/1810.04805>`_. *arXiv preprint arXiv:1810.04805*.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). `Exploring the limits of transfer learning with a unified text-to-text transformer <https://jmlr.org/papers/v21/20-074.html>`_. *Journal of Machine Learning Research*, 21(140), 1–67.
