ProGen2 Medium API
==================

ProGen2 Medium is a 764M-parameter autoregressive protein language model trained on ~1B sequences (UniRef90 plus metagenomic BFD30/BFD90) to model natural protein sequence distributions, generate variants, and score sequences by log-likelihood. The API exposes GPU-accelerated, batched sequence continuation from an N-terminal amino acid context (up to 512 residues) with configurable sampling (temperature, top-p) and up to 3 samples per context, supporting workflows in enzyme and antibody design, library generation, and zero-shot fitness ranking.

Generate
--------

Generate 2 protein sequence continuations from a realistic N-terminal context using the ProGen2 Medium model

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="progen2-medium",
                action="generate",
                params={
                  "temperature": 0.7,
                  "top_p": 0.92,
                  "num_samples": 2,
                  "max_length": 150
                },
                items=[
                  {
                    "context": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/progen2-medium/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.7,
                "top_p": 0.92,
                "num_samples": 2,
                "max_length": 150
              },
              "items": [
                {
                  "context": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/progen2-medium/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.7,
                    "top_p": 0.92,
                    "num_samples": 2,
                    "max_length": 150
                  },
                  "items": [
                    {
                      "context": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/progen2-medium/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.7,
                top_p = 0.92,
                num_samples = 2,
                max_length = 150
              ),
              items = list(
                list(
                  context = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/progen2-medium/generate/

   Generate endpoint for ProGen2 Medium.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Generation parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 0.8) — Sampling temperature

        - **top_p** (*float*, range: 0.0-1.0, default: 0.9) — Nucleus sampling probability

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate per input item

        - **max_length** (*int*, range: 12-512, default: 128) — Maximum total length of each generated sequence, in amino acid characters, including the input context


      - **items** (*array of objects*, min: 1, max: 1, required) --- Input items:

        - **context** (*string*, min length: 1, max length: 512, required) — Input amino acid sequence using unambiguous standard codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/progen2-medium/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.7,
          "top_p": 0.92,
          "num_samples": 2,
          "max_length": 150
        },
        "items": [
          {
            "context": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of arrays*) --- One result per input item, in the order requested:

        - **[ ]** (*array of objects*, length: ``num_samples`` per input) — Generated sequences and scores


        - **[ ].sequence** (*string*) — Generated amino acid sequence, length: 1–512 residues


        - **[ ].ll_sum** (*float*) — Sum of token log-likelihoods for the generated sequence, natural log units


        - **[ ].ll_mean** (*float*) — Mean token log-likelihood over the generated sequence, natural log units

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPIVSRIDNGEYDQLSSKEIKEYSDLTTNKRLIEVSSGHTDARMGILTEYPSTFAVKARTYDITGAHGIQKALCYGGYTFHDSDALDLFFIGCDLLWSEGDGWSIEQMHEEIYRLF",
              "ll_sum": -322.52857971191406,
              "ll_mean": -2.164621353149414
            },
            {
              "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPIVAKPESGVNDDLNGTERPVSVQIEGIDPLRVEVVHSLAKWKRLALAEYDFTVGEGLYTDMNAIRRDEDTDNIHSVYVDQWDWEKIILKEERTEDTLKAVVREIYAAFKETEKY",
              "ll_sum": -207.94718170166016,
              "ll_mean": -1.3956186175346375
            }
          ]
        ]
      }


Performance
-----------

- Model scale and hardware
- ProGen2 Medium is a 764M-parameter autoregressive protein language model trained on a universal corpus (UniRef90 + BFD30), with the same parameter count as ProGen2 OAS but a broader training distribution
- The hosted variant runs on NVIDIA T4-class GPUs (16 GB device memory), using mixed-precision decoding optimized for autoregressive generation

- Relative speed and latency
- For typical settings (``max_length`` 128, ``num_samples`` 1), ProGen2 Medium is ~1.6–1.8× faster per token than ProGen2 Large (2.7B) on the same GPU class
- It is ~1.1–1.3× slower than the antibody-specific ProGen2 OAS on short antibody-like prompts, mainly due to the larger effective vocabulary and universal training data
- End-to-end latency scales approximately linearly with requested ``max_length`` and ``num_samples``

- Modeling quality and comparison within ProGen2 family
- On held-out UniRef90+BFD30, ProGen2 Medium achieves perplexity 11.2 (test-max90) and 14.3 (test-max50), close to ProGen2 Large (11.1 / 14.4) and behind ProGen2 XLarge (9.9 / 13.9; not exposed via this endpoint)
- In practice, Medium captures most of the generative and likelihood quality of larger ProGen2 models for in-distribution proteins, with substantially lower latency and better throughput than Large/BFD90

- Zero-shot fitness and design-oriented behavior
- For narrow mutation landscapes (single/double mutants), 764M-scale ProGen2 models typically match or outperform both smaller and larger ProGen2 variants in zero-shot fitness ranking, providing a good balance between capacity and overfitting to phylogenetic biases
- For wider, low-homology, epistatic landscapes (e.g., GB1), larger ProGen2 models (Large/XLarge) can outperform Medium in top-variant discovery, but at higher inference cost
- Compared to encoder-only ESM models used purely for scoring, ProGen2 Medium tends to be more robust on sequences with indels and multi-site mutations, at the cost of higher per-sequence compute than an ESM encoder pass

Applications
------------

- Zero-shot prioritization of protein variant libraries in engineering campaigns, by scoring large in silico mutagenesis panels (substitutions and indels) and ranking variants by model likelihood as a proxy for fitness, allowing teams to reduce wet-lab screening burden and focus on the most promising candidates; particularly valuable when you already have a lead protein (e.g., enzyme, receptor, scaffold) and want to explore single- and multi-mutant neighborhoods without building task-specific supervised models
- Generative design of de novo protein sequences with realistic folds, by sampling from ProGen2 Medium under tuned temperature and nucleus sampling parameters to create diverse sequence libraries that still resemble natural structural distributions, enabling upstream library creation for downstream structural filtering (e.g., AlphaFold2) and functional assays; most useful when you need novel scaffolds or sequence diversity within a given length range and can provide at least a short N-terminal context to seed generation
- Architecture-focused diversification of protein families, by fine-tuning ProGen2 Medium on a curated set of proteins sharing a common structural class (e.g., two-layer sandwich, Rossmann-like folds) and then generating sequences that preserve the overall fold while varying surface and ligand-binding regions, allowing companies to systematically explore functional diversity within an architecture that is already manufacturable and developable in their platform; this is powerful for exploring binding pockets or stability variants but is not optimal when your training data for that architecture is extremely sparse or noisy
- Zero-shot assessment of protein fitness landscapes to guide mutagenesis strategy, by using ProGen2 Medium log-likelihoods to estimate which positions and mutation types are likely to be tolerated or beneficial before designing combinatorial libraries, helping reduce library size and bias experiments toward regions of sequence space that are enriched for functional sequences; particularly helpful for proteins with non-trivial epistasis where intuitive single-site scans fail, but less informative when the target protein is highly de novo and poorly represented in the training distribution
- Quality control and filtering of large, externally generated protein sequence sets (e.g., from other generative tools or legacy directed-evolution data) by using ProGen2 Medium to detect sequences that are strongly out-of-distribution relative to natural proteins, enabling teams to triage sequences that are likely misfolded or non-viable before synthesis and expression; this works best as a coarse filter in combination with structure prediction and biophysical models, rather than as a sole decision-maker for go/no-go calls

Limitations
-----------

- **Maximum Sequence Length**: Input ``context`` and generated sequences are limited to ``ProGen2Params.max_sequence_len`` = 512 amino acids. Requests with ``context`` longer than 512 characters or ``max_length`` > 512 will be rejected. Practical implication: very long proteins, multi-domain fusions, or concatenated constructs must be truncated or split, which can change the distribution of generated variants.
- **Batch Size and Sampling Limits**: Each request can contain at most ``ProGen2Params.batch_size`` = 1 item in ``items``, and each ``ProGen2GenerateParams`` can request at most ``num_samples`` = 3 variants per item. High-throughput library generation must therefore be done by batching requests client-side and handling rate limiting and deduplication externally.
- **Generation-Only, No Embeddings or Scoring API**: ProGen2 is exposed as an autoregressive generator only. The ``generator`` endpoint returns sequences plus log-likelihood scores (``ll_sum``, ``ll_mean``) for the generated tokens; it does *not* provide latent embeddings, structural predictions, or direct fitness scores. Tasks such as clustering, visualization, structure-based ranking, or supervised property prediction require downstream models or additional BioLM services.
- **Data- and Architecture-Driven Biases**: ProGen2 is trained on large, general protein databases and behaves as a causal language model over amino acid sequences. It is best at generating sequences that resemble natural proteins in the training distribution, but (1) it does not model explicit 3D structure, binding partners, or assay conditions; (2) it can underperform for highly specialized, poorly represented domains or for strict zero-shot fitness prediction at narrow mutational neighborhoods; and (3) it tends to reproduce dataset biases (e.g., domain composition, length, truncation patterns such as N-terminally truncated antibodies) rather than an ideal “fitness” distribution.
- **Non-Optimal Use Cases**: ProGen2 is usually *not* the best first choice when (1) you need structure-conditioned design (fixed backbone or epitope geometry); (2) you primarily need embeddings or similarity metrics for millions of sequences; (3) you require ultra-fast, large-scale screening or ranking where encoder models or lightweight fitness predictors are more efficient; or (4) you are at the very end of a design funnel where high-cost structure prediction (e.g., AlphaFold2/ESMFold) or specialized antibody/nanobody structure models provide more decision-critical information.
- **Sampling Behavior and Novelty Control**: ``temperature`` (0.0–8.0) and ``top_p`` (0.0–1.0) in ``ProGen2GenerateParams`` strongly affect quality vs. diversity. Very low ``temperature`` / low ``top_p`` can yield conservative, training-like variants with limited novelty; very high values can produce unrealistic or non-functional sequences. The API does not enforce domain-specific safe ranges beyond these hard bounds, so tuning and downstream filtering (e.g., with structure predictors or property models) are recommended, especially for large design campaigns.

How We Use It
-------------

ProGen2 Medium serves as a general-purpose backbone in protein design campaigns, where its sequence likelihoods and sequence continuations guide which variants to generate, prioritize, and send to the lab. Teams integrate its log-likelihood scores as features in downstream fitness or developability models, and use its generative capabilities to propose focused mutation sets that are then filtered by structure predictors, 3D-scoring models, and biophysical property estimators. Standardized, scalable APIs allow data scientists, ML engineers, and bioengineers to loop assay results back into the design cycle, enabling fine-tuning or targeted resampling while keeping integration effort low across existing infrastructure.

- In antibody and protein therapeutic programs, ProGen2 Medium supports in silico library design and ranking, feeding sequences into developability assessments (aggregation, solubility, stability) and other BioLM models before expression and screening.
- In enzyme and industrial protein engineering, ProGen2 Medium underlies automated proposal and scoring of variant panels, integrating with similarity search, structure-derived metrics, and thermodynamic/property predictors to align sequence choices with assay constraints and large design spaces.

Related
-------

- ``ProGen2 Large`` – Larger ProGen2 variant trained on the same data mixture; use when you need higher-capacity protein generation or stronger zero-shot fitness signals than ``ProGen2 Medium``.
- ``ProGen2 BFD90`` – ProGen2 model emphasizing metagenomic diversity; useful for exploring more diverse or out-of-distribution protein sequence space alongside ``ProGen2 Medium``.
- ``ESM-1v`` – Protein language model optimized for zero-shot variant effect prediction; pair with ``ProGen2 Medium`` by generating sequences with ProGen2 and ranking or filtering them with ESM-1v scores.
- ``ProteinMPNN`` – Structure-conditioned sequence design model; combine with ``ProGen2 Medium`` by using ProGen2 to propose sequence variations, then refining sequences on fixed backbones with ProteinMPNN.

References
----------

- Nijkamp, E., Ruffolo, J. A., Weinstein, E. N., Naik, N., & Madani, A. (2023). `ProGen2: Exploring the Boundaries of Protein Language Models <https://doi.org/10.48550/arXiv.2306.06151>`_. *arXiv preprint arXiv:2306.06151*.

