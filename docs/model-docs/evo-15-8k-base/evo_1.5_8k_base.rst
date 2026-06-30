Evo 1.5 8k Base API
===================

Evo 1.5 8k Base is a 7B-parameter genomic language model trained on ~300B nucleotides from prokaryotic whole genomes, operating at single-nucleotide resolution with an 8,192-token context. Using a StripedHyena hybrid architecture and GPU-accelerated inference, the API supports log-probability scoring and autoregressive generation for unlabeled DNA sequences up to 4,096 bp, with batch sizes up to 2. Typical uses include mutational scoring, regulatory sequence analysis, and de novo design of coding or noncoding genomic fragments for protein engineering and synthetic biology.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="evo-15-8k-base",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ATGCATTGCGATCGTACGTG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/evo-15-8k-base/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "ATGCATTGCGATCGTACGTG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/evo-15-8k-base/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "ATGCATTGCGATCGTACGTG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/evo-15-8k-base/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "ATGCATTGCGATCGTACGTG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/evo-15-8k-base/predict/

   Predict endpoint for Evo 1.5 8k Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters for sequence generation:

        - **max_new_tokens** (*int*, range: 1-4096, default: 100) — Maximum number of new tokens to generate

        - **temperature** (*float*, range: ≥0.0, default: 0.0) — Sampling temperature

        - **top_k** (*int*, range: ≥1, default: 1) — Top-k sampling parameter

        - **top_p** (*float*, range: 0.0-1.0, default: 1.0) — Top-p (nucleus) sampling parameter

        - **prepend_bos** (*bool*, default: False) — Whether to prepend a beginning-of-sequence token to the prompt


      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences for generation:

        - **prompt** (*string*, min length: 1, max length: 4096, required) — DNA sequence using unambiguous nucleotide characters (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo-15-8k-base/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "ATGCATTGCGATCGTACGTG"
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

        - **log_prob** (*float*, range: negative infinity to 0.0) — Natural-log probability of the input DNA sequence under the Evo model

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "log_prob": -30.015625
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
                entity="evo-15-8k-base",
                action="generate",
                params={
                  "max_new_tokens": 100,
                  "temperature": 0.0,
                  "top_k": 1,
                  "top_p": 1.0,
                  "prepend_bos": false
                },
                items=[
                  {
                    "prompt": "ACTG"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/evo-15-8k-base/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "max_new_tokens": 100,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "prepend_bos": false
              },
              "items": [
                {
                  "prompt": "ACTG"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/evo-15-8k-base/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "max_new_tokens": 100,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "prepend_bos": false
                  },
                  "items": [
                    {
                      "prompt": "ACTG"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/evo-15-8k-base/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                max_new_tokens = 100,
                temperature = 0.0,
                top_k = 1,
                top_p = 1.0,
                prepend_bos = FALSE
              ),
              items = list(
                list(
                  prompt = "ACTG"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/evo-15-8k-base/generate/

   Generate endpoint for Evo 1.5 8k Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Generation parameters:

        - **max_new_tokens** (*int*, range: 1-4096, default: 100) — Maximum number of new tokens to generate

        - **temperature** (*float*, range: ≥0.0, default: 0.0) — Sampling temperature

        - **top_k** (*int*, range: ≥1, default: 1) — Top-k sampling cutoff

        - **top_p** (*float*, range: 0.0-1.0, default: 1.0) — Top-p (nucleus) sampling cutoff

        - **prepend_bos** (*bool*, default: False) — Whether to prepend a beginning-of-sequence token to the prompt


      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **prompt** (*string*, min length: 1, max length: 4096, required) — DNA sequence prompt using unambiguous nucleotide characters (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo-15-8k-base/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "max_new_tokens": 100,
          "temperature": 0.0,
          "top_k": 1,
          "top_p": 1.0,
          "prepend_bos": false
        },
        "items": [
          {
            "prompt": "ACTG"
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

        - **generated** (*string*, length: 1 to 4096 characters) — Generated DNA sequence consisting of unambiguous nucleotide characters
        - **score** (*float*) — Log-likelihood score of the generated sequence under the Evo model

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "generated": "CCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGC",
            "score": -2.4914729595184326
          }
        ]
      }


Performance
-----------

- Evo 1.5 8k Base runs on NVIDIA L4 GPUs with GPU-accelerated inference. The model uses the StripedHyena 7B-parameter architecture, which combines data-controlled convolutional (Hyena) layers with rotary self-attention for efficient long-sequence processing at single-nucleotide resolution.
- Compared to dense Transformer DNA LMs (e.g., DNABERT-2, Omni-DNA 1B), StripedHyena provides sub-quadratic time and memory complexity in sequence length, enabling faster inference and lower GPU memory usage at 8k-token context. BioLM’s deployment uses recurrent Hyena evaluation and key/value caching to further reduce latency and improve throughput for batched generation and scoring.
- On zero-shot prediction of mutational fitness, Evo 1.x models consistently achieve lower DNA-level perplexity and higher Spearman correlations than nucleotide baselines across:
  
  - Bacterial protein deep mutational scanning datasets, where Evo matches or exceeds the performance of specialized protein LMs such as ESM-2 650M and ESM-1v on *E. coli* proteins, despite being trained on genomic DNA rather than amino acid sequences.
  - Non-coding RNA fitness datasets, where Evo outperforms RNA-focused language models (e.g., RNA-FM, nanoBERT-style models) on tRNAs, rRNAs, and ribozymes.

- For regulatory and systems-level tasks, Evo substantially outperforms other nucleotide LMs from the same family on:
  
  - Zero-shot prediction of *E. coli* promoter–RBS-driven expression, with higher correlation to mRNA levels and better AUROC for protein expression than alignment-based frequency methods and other DNA LMs.
  - Zero-shot gene essentiality prediction in bacterial and phage genomes, where Evo 8k-context models (the basis for Evo 1.5 8k Base) achieve AUROCs up to ~0.8–0.86 when scoring in silico loss-of-function mutations with local genomic context.

Applications
------------

- Zero-shot prediction of mutational effects on *bacterial* protein and RNA function using log-likelihood scores from the predictor endpoint, enabling rapid in silico prioritization of beneficial mutations in industrial enzymes or bacterial RNA therapeutics. Useful for protein or RNA engineering campaigns aiming to improve stability, activity, or specificity without exhaustive screening. Not suitable for eukaryotic proteins or human therapeutic antibodies because Evo 1.5 is trained on prokaryotic genomes only.
- Generative design of novel CRISPR-Cas protein–RNA loci by sampling DNA with the generator endpoint and translating ORFs and guides downstream, allowing rapid exploration of diverse Cas9/Cas12/Cas13-like variants and their associated CRISPR arrays for genome editing tools, diagnostics, or microbial engineering. Generated designs require downstream structural or experimental validation; this base model is not the CRISPR-finetuned Evo variant, so task-specific post-filtering is important.
- In silico prioritization of essential *bacterial* genes by comparing log-likelihoods of wild-type versus in silico–mutated coding sequences (e.g., introducing premature stops) with the predictor endpoint, enabling ranking of putative antibiotic targets directly from genomic DNA. Useful for antimicrobial target discovery in bacteria and phage. Predictions are zero-shot, rely on prokaryotic training data, and may not transfer to eukaryotic genomes.
- Generative design and exploration of synthetic transposable element–like loci (e.g., IS200/IS605-style systems) by sampling contiguous DNA segments, then annotating candidate transposase and associated nuclease/RNA components with external tools. Applicable to microbial strain engineering, mobile element tool development, or chassis design with engineered insertion behavior. Functional activity, specificity, and safety must be established experimentally; this base model is not finetuned specifically on transposons.
- Generation of long bacterial DNA fragments (up to the 8k context limit per API call) with realistic local coding density and operon-like gene organization, supporting early-stage design of synthetic operons, pathways, or segments of compact genomes for biomanufacturing or bioremediation strains. Generated sequences are “genome-like” but may lack complete essential features (e.g., full tRNA and rRNA repertoires) and require manual curation, stitching of multiple generations, and iterative experimental optimization for viability.

Limitations
-----------

- **Maximum Sequence Length**: Input DNA sequences for both ``predictor`` and ``generator`` endpoints are limited to ``4096`` nucleotides via ``max_sequence_len``. Longer genomic regions must be truncated or split client-side before submission.
- **Batch Size**: The ``items`` arrays in ``EvoPredictLogProbRequest`` and ``EvoGenerateRequest`` accept at most ``2`` sequences per call (``batch_size``). Higher-throughput workloads require batching across multiple requests.
- **Model Variant and Domain**: ``EVO_1_5_8K_BASE`` (``"v1.5-8k"`` / ``"evo-1.5-8k-base"``) is pretrained only on prokaryotic genomes and prokaryotic mobile elements. It is not appropriate for eukaryotic use cases such as human variant effect prediction, mammalian regulatory analysis, or eukaryotic pathogen genomics.
- **Generative Output Scope**: ``generator`` exposes short-range sequence generation only (``max_new_tokens`` ≤ ``4096``). Although Evo can model genome-scale organization, the API cannot directly produce or evaluate megabase-scale genomes or full CRISPR/transposon systems in one call; such tasks require tiling and downstream analysis.
- **Synthetic / Out-of-Distribution Sequences**: ``predictor`` returns a single scalar ``log_prob`` per input sequence, which correlates well with mutational effects for natural-like prokaryotic sequences. For highly engineered, shuffled, or otherwise synthetic constructs that deviate strongly from natural prokaryotic distributions, these scores may be poorly calibrated.
- **Task Fit vs. Alternatives**: Evo’s StripedHyena architecture is optimized for long-context genomic modeling. For very short, single-molecule tasks (e.g., short peptides, isolated motifs) or for non-prokaryotic biology, specialized protein/RNA models or eukaryotic-focused nucleotide models are often more efficient and more accurate.

How We Use It
-------------

Evo 1.5 8k Base enables faster protein and nucleic acid design cycles by scoring and generating DNA sequences through scalable, standardized APIs. In our protein engineering and lab-in-the-loop workflows, we use Evo scores as a zero-shot proxy for mutational fitness and regulatory activity, then combine them with structural and biophysical predictors to prioritize variants for synthesis and multi-round optimization. For de novo design, Evo-guided generation provides genomically realistic coding and regulatory segments that we further filter, rank, and refine with task-specific ML models and experimental feedback.

- Supports iterative optimization of coding and regulatory sequences using genome-informed likelihood scores.
- Integrates with downstream models for structure, stability, and developability to de-risk experimental screening.

Related
-------

- ``Evo 2 1B Base`` – Successor to Evo 1.5 with larger model size and longer context, suitable when you need genome-scale modeling or design across prokaryotic and eukaryotic domains.
- ``DNABERT-2`` – Transformer-based nucleotide model that is convenient for shorter DNA segments and k-mer tokenization workflows, complementing Evo 1.5 on localized or motif-focused analyses.
- ``ESMFold`` – Protein structure prediction from amino acid sequences, useful for structurally assessing proteins implied by or derived from Evo 1.5–scored or –generated coding regions.
- ``AlphaFold2`` – High-accuracy protein structure prediction for single chains and complexes, enabling more detailed structural validation of protein designs informed by Evo 1.5 outputs.

References
----------

- Nguyen, E., Poli, M., Durrant, M. G., Thomas, A. W., Kang, B., Sullivan, J., Ng, M. Y., Lewis, A., Patel, A., Lou, A., Ermon, S., Baccus, S. A., Hernandez-Boussard, T., Ré, C., Hsu, P. D., & Hie, B. L. (2024). Sequence modeling and design from molecular to genome scale with Evo. *Science*, 386(6723), eado9336. https://doi.org/10.1126/science.ado9336

- Evo project GitHub repository: https://github.com/evo-design/evo
