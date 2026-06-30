DNABERT-2 API
=============

DNABERT-2 is a GPU-accelerated Transformer encoder for genome sequence analysis that uses Byte Pair Encoding (BPE) tokenization and Attention with Linear Biases (ALiBi) to efficiently handle long, multi-species DNA sequences. The API supports batched processing of up to 10 unambiguous DNA sequences of length ≤2048, providing sequence embeddings (encoder) and per-sequence log probabilities (predictor). DNABERT-2 was pretrained on multi-species genomes and achieves state-of-the-art efficiency with ~21× fewer parameters and ~56× less GPU time than prior models.

Predict
-------

Predict log probabilities for input DNA sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dnabert2",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  },
                  {
                    "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dnabert2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                },
                {
                  "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dnabert2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                    },
                    {
                      "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dnabert2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                ),
                list(
                  sequence = "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dnabert2/predict/

   Predict endpoint for DNABERT-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 10) — Maximum number of items per request

        - **max_sequence_len** (*int*, default: 2048) — Maximum allowed length of each DNA sequence


      - **items** (*array of objects*, min: 1, max: 10) --- Input records:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — DNA sequence containing only unambiguous nucleotide bases (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dnabert2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
          },
          {
            "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
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

        - **embedding** (*array of floats*, size: 768) — Embedding vector representing the input DNA sequence

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **log_prob** (*float*, range: negative infinity to 0.0) — Log-probability of the input DNA sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -20.878572463989258
          },
          {
            "log_prob": -37.00829315185547
          }
        ]
      }


Encode
------

Generate embeddings for input DNA sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dnabert2",
                action="encode",
                params={},
                items=[
                  {
                    "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  },
                  {
                    "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dnabert2/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                },
                {
                  "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dnabert2/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                    },
                    {
                      "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dnabert2/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                ),
                list(
                  sequence = "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dnabert2/encode/

   Encode endpoint for DNABERT-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 10) — Maximum number of sequences per request (max: 10)

        - **max_sequence_len** (*int*, default: 2048) — Maximum allowed length of each DNA sequence (max: 2048)

      - **items** (*array of objects*, min: 1, max: 10) --- DNA sequences to encode:

        - **sequence** (*string*, required, min length: 1, max length: 2048) — DNA sequence containing only unambiguous nucleotides (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dnabert2/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
          },
          {
            "sequence": "GGAAAACCCCACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
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

        - **embedding** (*array of floats*) — Fixed-length DNABERT-2 embedding vector for the input DNA sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embedding": [
              -0.3722754418849945,
              -0.020519617944955826,
              "... (truncated for documentation)"
            ]
          },
          {
            "embedding": [
              -0.3600304126739502,
              -0.02206198126077652,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- DNABERT-2 inference is GPU-accelerated on NVIDIA T4 GPUs and is optimized for high-throughput DNA sequence processing via BPE tokenization and FlashAttention.
- On the GUE benchmark, DNABERT-2 improves average scores by ~6 absolute points over DNABERT and outperforms DNABERT on 23/28 datasets, while using ~3× fewer FLOPs for comparable inputs due to shorter tokenized sequences.
- Relative to Nucleotide Transformer NT-2500M-multi, DNABERT-2 attains comparable average accuracy on GUE (within ~0.1 points) while using ~21× fewer parameters and ~19–56× less GPU compute, enabling more efficient large-scale inference and fine-tuning.
- Compared to DNABERT’s overlapping k-mer tokenization, DNABERT-2’s BPE tokenizer reduces effective sequence length by ~5×, and ALiBi positional encoding plus FlashAttention further reduce memory and compute overhead, particularly for longer genomic sequences.

Applications
------------

- Identification and classification of transcription factor binding sites in genomic sequences, enabling researchers to map regulatory regions and interpret gene expression programs for target selection and validation; valuable for companies designing gene therapies or synthetic regulatory elements, though performance may decrease with very short sequences (<100 bp) where limited context is available.
- Rapid detection and differentiation of viral genome variants, such as SARS-CoV-2 lineages, allowing biotech firms to track viral evolution, support vaccine strain selection, and monitor emerging infectious threats; accuracy may be reduced if inputs include ambiguous nucleotides or substantial sequencing noise, so upstream QC is important.
- Prediction of promoter and core promoter regions in human genomes, facilitating design of synthetic gene circuits and expression vectors by locating likely transcription initiation sites; useful for optimizing gene expression in cell and gene therapy or biomanufacturing workflows, but less suitable for organisms with promoter architectures that differ strongly from the human and model training species.
- High-throughput identification of splice donor and acceptor sites in human DNA sequences, supporting annotation of alternative splicing patterns and assessment of potential splice-disrupting variants; applicable to RNA therapeutic and gene-editing design pipelines, with degraded performance expected on highly non-canonical or heavily mutated splice junctions.
- Genome-wide prediction of epigenetic marks such as histone modifications in yeast, aiding synthetic biology groups in engineering yeast strains with tuned gene expression for production strains; predictions are calibrated on yeast data and may not directly transfer to complex eukaryotic epigenomes without task-specific retraining.

Limitations
-----------

- **Maximum Sequence Length**: Each ``sequence`` in ``items`` must be at most ``2048`` nucleotides. Longer DNA regions must be split client-side before calling ``encoder`` or ``predictor``.
- **Batch Size**: Each request to ``encoder`` or ``predictor`` can include at most ``10`` sequences in ``items``. Larger datasets must be processed via multiple requests.
- DNABERT-2 uses Byte Pair Encoding (BPE) over DNA, which shortens tokenized inputs but can lose fine-grained positional detail. For very short sequences (e.g., <100 bp core promoters or short motifs), performance can be weaker than models tailored to fixed, short windows.
- The model is pre-trained on multi-species genomic data. It may underperform on highly specialized tasks that require detailed, species- or tissue-specific regulatory context unless further fine-tuned externally.
- DNABERT-2 via this API is optimized for encoding sequences (``encoder`` embeddings) and masked-language-style scoring (``predictor`` log probabilities). It is not a generative design model and is not suitable for de novo sequence generation pipelines.
- **GPU Type**: Inference runs on an NVIDIA ``T4`` GPU, which is adequate for typical research-scale workloads. Extremely high-throughput or ultra–low-latency use cases may require additional client-side batching, caching, or alternative models optimized for larger GPUs.

How We Use It
-------------

DNABERT-2 supports genome-aware protein engineering workflows by providing scalable DNA sequence embeddings and sequence-level log probabilities that help prioritize regions for experimental follow-up. Its multi-species training and efficient tokenization make it suitable for large DNA panels, enabling standardized, API-driven analysis of regulatory regions, construct designs, and variant libraries that ultimately inform enzyme and antibody optimization strategies.

- Accelerates prioritization of genomic segments that may impact expression, stability, or function of engineered proteins
- Integrates with BioLM generative and protein-focused predictive models by supplying consistent DNA-side features and uncertainty estimates for multi-round optimization

Related
-------

- ``Omni-DNA 1B`` – Large-scale DNA language model offering alternative BPE-based sequence embeddings, useful alongside ``DNABERT-2`` for comparing or ensembling genome representations.
- ``nanoBERT`` – Optimized for nanopore sequencing reads; complements ``DNABERT-2`` when you need embeddings or log-probabilities on long, noisy DNA sequences.
- ``Chai-1`` – Focused on chromatin interaction and 3D genome organization; pairs with ``DNABERT-2`` embeddings to add structural context to sequence-based analyses.
- ``Evo 2 1B Base`` – Evolution-aware DNA model that captures cross-species conservation signals, complementary to ``DNABERT-2`` for multi-species and comparative genomics tasks.

References
----------

- Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. *bioRxiv*. https://github.com/Zhihan1996/DNABERT_2
