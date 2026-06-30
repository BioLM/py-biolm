Evo 2 1B Base API
=================

Evo 2 1B Base is a genomic language model (~1.1B parameters) trained on 1T DNA tokens from the OpenGenome2 corpus at 8,192-token context. It runs on GPU-backed infrastructure and supports three API actions over unambiguous DNA: likelihood scoring (log probability per sequence), short-range sequence generation (up to 4,096-token prompts and 4,096 new tokens), and embedding extraction from configurable layers with mean/last pooling. Typical uses include variant effect scoring, local sequence design, and feature extraction for downstream classifiers in genomics and regulatory sequence analysis.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="evo2-1b-base",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/evo2-1b-base/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/evo2-1b-base/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/evo2-1b-base/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/evo2-1b-base/predict/

   Predict endpoint for Evo 2 1B Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters for Evo2 generation:

        - **max_new_tokens** (*int*, range: 1-4096, default: 100) — Maximum number of new tokens to generate

        - **temperature** (*float*, range: ≥0.0, default: 1.0) — Sampling temperature

        - **top_k** (*int*, range: ≥1, default: 4) — Number of highest probability tokens to consider for top-k sampling

        - **top_p** (*float*, range: 0.0-1.0, default: 1.0) — Cumulative probability threshold for nucleus (top-p) sampling


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences for generation:

        - **prompt** (*string*, min length: 1, max length: 4096, required) — DNA sequence prompt of unambiguous nucleotides (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo2-1b-base/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
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

        - **embeddings** (*array of objects*) — Embedding outputs for each requested layer:

          - **layer** (*int*) — Index of the model layer used to compute this embedding

          - **mean** (*array of floats*, optional, size: [4096]) — Mean-pooled embedding vector over all sequence positions for this layer

          - **last** (*array of floats*, optional, size: [4096]) — Embedding vector at the final sequence position for this layer

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -14.87200927734375
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
                entity="evo2-1b-base",
                action="encode",
                params={
                  "embedding_layers": [
                    -2
                  ],
                  "mlp_layer": 3,
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/evo2-1b-base/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "embedding_layers": [
                  -2
                ],
                "mlp_layer": 3,
                "include": [
                  "mean"
                ]
              },
              "items": [
                {
                  "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/evo2-1b-base/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "embedding_layers": [
                      -2
                    ],
                    "mlp_layer": 3,
                    "include": [
                      "mean"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/evo2-1b-base/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                embedding_layers = list(
                  -2
                ),
                mlp_layer = 3,
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  sequence = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/evo2-1b-base/encode/

   Encode endpoint for Evo 2 1B Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters for embeddings extraction:

        - **embedding_layers** (*array of integers*, default: [-2]) — Indices of model layers from which embeddings are extracted

        - **mlp_layer** (*integer*, default: 3) — Index of the MLP layer used for embedding extraction (fixed to 3)

        - **include** (*array of strings*, default: ["mean"]) — Embedding statistics to return for each layer; allowed values: "mean", "last"


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences for embedding extraction:

        - **sequence** (*string*, min length: 1, max length: 4096, required) — DNA sequence composed of unambiguous nucleotide characters (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo2-1b-base/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "embedding_layers": [
            -2
          ],
          "mlp_layer": 3,
          "include": [
            "mean"
          ]
        },
        "items": [
          {
            "sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
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

        - **embeddings** (*array of objects*) — Embedding details for each requested layer:

          - **layer** (*int*) — Layer index used to compute this embedding (may differ from requested index, e.g., internal block index)
          - **mean** (*array of floats*, optional) — Mean-pooled embedding over all sequence positions, length depends on model variant (e.g., 1920 for 1b-base, 4096 for 7b-base)
          - **last** (*array of floats*, optional) — Embedding for the last sequence position, same length as **mean**

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              {
                "layer": 23,
                "mean": [
                  9.080395102500916e-09,
                  1.1874362826347351e-07,
                  "... (truncated for documentation)"
                ]
              }
            ]
          }
        ]
      }


Generate
--------

Generate new sequences based on prompts

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="evo2-1b-base",
                action="generate",
                params={
                  "max_new_tokens": 100,
                  "temperature": 1.0,
                  "top_k": 4,
                  "top_p": 1.0
                },
                items=[
                  {
                    "prompt": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/evo2-1b-base/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "max_new_tokens": 100,
                "temperature": 1.0,
                "top_k": 4,
                "top_p": 1.0
              },
              "items": [
                {
                  "prompt": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/evo2-1b-base/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "max_new_tokens": 100,
                    "temperature": 1.0,
                    "top_k": 4,
                    "top_p": 1.0
                  },
                  "items": [
                    {
                      "prompt": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/evo2-1b-base/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                max_new_tokens = 100,
                temperature = 1.0,
                top_k = 4,
                top_p = 1.0
              ),
              items = list(
                list(
                  prompt = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/evo2-1b-base/generate/

   Generate endpoint for Evo 2 1B Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **max_new_tokens** (*int*, range: 1-4096, default: 100) — Maximum number of bases to generate beyond each prompt

        - **temperature** (*float*, range: 0.0-∞, default: 1.0) — Sampling temperature used for generation

        - **top_k** (*int*, minimum: 1, default: 4) — Number of highest-probability tokens considered at each generation step

        - **top_p** (*float*, range: 0.0-1.0, default: 1.0) — Cumulative probability threshold for nucleus sampling

      - **items** (*array of objects*, min items: 1, max items: 1) --- Input prompts:

        - **prompt** (*string*, required, min length: 1, max length: 4096) — DNA prompt sequence containing only unambiguous nucleotide characters (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo2-1b-base/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "max_new_tokens": 100,
          "temperature": 1.0,
          "top_k": 4,
          "top_p": 1.0
        },
        "items": [
          {
            "prompt": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
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

        - **embeddings** (*array of objects*) — Embedding outputs for the requested layers

          - **layer** (*int*) — Index of the model layer used to compute this embedding

          - **mean** (*array of floats*, size: 4096, optional) — Mean-pooled embedding vector across sequence positions

          - **last** (*array of floats*, size: 4096, optional) — Embedding vector at the last sequence position


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **log_prob** (*float*, range: negative infinity to 0.0) — Log probability of the input DNA sequence under the model


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **generated** (*string*) — Generated DNA sequence tokens appended to the input prompt, length up to `max_new_tokens` (range: 1–4096 characters)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "generated": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
          }
        ]
      }


Performance
-----------

- Evo 2 1B Base is a 1.1B-parameter StripedHyena 2 DNA language model optimized for fast inference and low GPU memory use. It runs efficiently on a single NVIDIA L4 GPU and exposes three API endpoints: encoder (embeddings), predictor (sequence log-probability), and generator (DNA sequence continuation).
- Compared to Evo 2 7B Base and Evo 2 40B Base, Evo 2 1B Base offers substantially lower latency and higher throughput per dollar, making it suitable for high-throughput scoring or generation of many short genomic sequences (up to 4,096 bp per item in the API).
- Predictive performance is lower than Evo 2 7B and 40B variants on demanding tasks (e.g., zero-shot variant effect prediction on ClinVar, SpliceVarDB, BRCA1/BRCA2; deep mutational scanning and long-range regulatory modeling), but remains competitive for simpler local sequence modeling such as gene-region completion, short-context mutation scoring, and short-sequence generation.
- Relative to Evo 1 models based on the original Hyena architecture, Evo 2 1B Base benefits from StripedHyena 2’s improved training and inference efficiency, enabling better DNA likelihood calibration and higher throughput for the same hardware budget, while retaining an 8k-token pretraining context that is narrowed to 4,096 tokens in this hosted API for consistent performance.

Applications
------------

- Zero-shot scoring of single-nucleotide and small indel variants in coding and noncoding DNA using log-probability differences between reference and mutant sequences within an 8,192 bp window, enabling variant prioritization for clinical genomics and diagnostic pipelines; not designed for structural variants or very long indels and performance is weaker on eukaryotic viruses due to training-time exclusions.
- Genome-scale functional annotation by embedding or scoring long genomic fragments (up to 8,192 bp per API call) to highlight exons, introns, transcription factor binding motifs, mobile genetic elements, and other functional loci, accelerating annotation and feature discovery for microbial strain engineering and mammalian cell line development; annotations still require downstream bioinformatics and experimental validation, especially in repetitive or poorly characterized genomes.
- In silico mutational scanning of promoters, untranslated regions, and other regulatory DNA by comparing log probabilities across designed variant libraries, supporting optimization of gene expression cassettes in microbial and mammalian systems; less suitable for predicting chromatin 3D structure or higher-order epigenomic effects without additional models.
- Generative design of DNA segments conditioned on user-specified prompts using the generator endpoint, supporting diversification of microbial operons, viral-free payloads, or synthetic regulatory regions while preserving local sequence “naturalness”; generated sequences should be filtered (e.g., with Evo 2 likelihoods and domain-specific screens) and experimentally tested for function and safety, and the API’s 4,096 bp limit per request constrains single-call design length.
- Embedding-based feature extraction for downstream supervised models in applications such as gene essentiality prediction, fitness optimization of engineered strains, or task-specific variant classifiers (for example, *BRCA1/2*-like models), by using the encoder endpoint to obtain fixed-length representations; performance on any new supervised task depends on training data quality and may not match the paper’s reported results without careful model development.

Limitations
-----------

- **Maximum Sequence Length**: The Evo 2 1B Base API enforces a hard limit of ``4096`` nucleotides per ``sequence``/``prompt``. Requests with longer DNA strings will fail validation rather than being truncated.
- **Batch Size**: The ``items`` array in all Evo 2 endpoints (``encoder``, ``predictor``, ``generator``) supports a **Batch Size** of up to ``1``. Each request may contain only a single sequence for encoding, log-probability scoring, or generation.
- **GPU Type**: In hosted deployments this model is run on **GPU Type** ``L4``. Performance (throughput and latency) may differ if you self-host on other accelerators.
- **Algorithmic Scope**: Evo 2 is a nucleotide-level genomic model. The API works on unambiguous DNA strings (A/C/G/T only) and returns sequence-level scores or embeddings; it does not operate on amino acid sequences, predict 3D structures, or output protein-specific properties directly.
- **Use Case Suitability**: The 1B Base variant is optimized for short-to-mid length genomic contexts (≤ ``4096`` bp) and general-purpose embeddings/log-likelihoods. It is not the best option for million-base-pair long-context analyses, highly time-sensitive real-time applications, or tasks that require model variants specifically tuned for human clinical variant calling.
- **Training Data Coverage**: Evo 2 was trained on broad genomic data with explicit exclusion of many eukaryote-infecting viral genomes. As a result, API outputs are unreliable for human viral sequence design or viral fitness prediction, and may underperform on very niche or poorly represented taxa.

How We Use It
-------------

The Evo 2 1B Base model underpins rapid, large-scale DNA sequence evaluation in our protein and RNA engineering programs. We use its standardized APIs for log-likelihood scoring and embeddings to quantify mutational impact, prioritize candidate variants, and rapidly down-select designs before higher-cost modeling or lab work. Within end-to-end pipelines, Evo 2 1B Base serves as a fast, generalist prior over genomic context that integrates with structure-based models, task-specific variant classifiers, and generative design tools to accelerate iteration and de-risk wet-lab campaigns.

- Enables high-throughput variant scoring for early filtering of protein, RNA, and regulatory DNA designs
- Supplies sequence embeddings that can feed downstream classifiers for domain-specific fitness or pathogenicity prediction

Related
-------

- ``Evo 1.5 8k Base`` – Earlier Evo model specialized for shorter (8k) genomic contexts, useful when you need Evo-style likelihoods or embeddings with lower compute than Evo 2 1B Base.
- ``Omni-DNA 1B`` – DNA language model at a similar parameter scale, useful for cross-model benchmarking, alternative embeddings, or ensemble scoring alongside Evo 2 1B Base.
- ``DNABERT-2`` – Encoder-style DNA model providing masked-token embeddings complementary to Evo 2’s autoregressive representations, useful for comparative analyses or embedding ensembles.
- ``ESMFold`` – Structure prediction model that can be paired with Evo 2–designed coding sequences to assess whether generated proteins fold into plausible 3D structures.

References
----------

- Brixi, G., Durrant, M. G., Ku, J., Poli, M., Brockman, G., Chang, D., Gonzalez, G. A., King, S. H., Li, D. B., Merchant, A. T., Naghipourfar, M., Nguyen, E., Ricci-Tam, C., Romero, D. W., Sun, G., Taghibakshi, A., Vorontsov, A., Yang, B., Deng, M., Gorton, L., Nguyen, N., Wang, N. K., Adams, E., Baccus, S. A., Dillmann, S., Ermon, S., Guo, D., Ilango, R., Janik, K., Lu, A. X., Mehta, R., Mofrad, M. R. K., Ng, M. Y., Pannu, J., Ré, C., Schmok, J. C., St. John, J., Sullivan, J., Zhu, K., Zynda, G., Balsam, D., Collison, P., Costa, A. B., Hernandez-Boussard, T., Ho, E., Liu, M.-Y., McGrath, T., Powell, K., Burke, D. P., Goodarzi, H., Hsu, P. D., & Hie, B. L. (2024). Genome modeling and design across all domains of life with Evo 2. *bioRxiv*. https://doi.org/10.1101/2024.05.07.591036
