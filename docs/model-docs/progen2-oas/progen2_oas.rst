ProGen2 OAS API
===============

ProGen2 OAS is a 764M-parameter antibody-focused autoregressive protein language model trained on 554M redundancy-reduced sequences from the Observed Antibody Space (OAS) database. This API variant generates heavy-chain variable-region sequences conditioned on N-terminal germline-like VH framework contexts, using configurable temperature and nucleus (top-p) sampling and user-defined length limits up to 512 residues. It also returns per-sequence log-likelihood summaries (sum and mean) for zero-shot style fitness ranking and antibody library design workflows.

Generate
--------

Generate up to 2 antibody variable heavy chain sequences conditioned on an OAS-style VH germline framework context.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="progen2-oas",
                action="generate",
                params={
                  "temperature": 0.7,
                  "top_p": 0.92,
                  "num_samples": 2,
                  "max_length": 160
                },
                items=[
                  {
                    "context": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/progen2-oas/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.7,
                "top_p": 0.92,
                "num_samples": 2,
                "max_length": 160
              },
              "items": [
                {
                  "context": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/progen2-oas/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.7,
                    "top_p": 0.92,
                    "num_samples": 2,
                    "max_length": 160
                  },
                  "items": [
                    {
                      "context": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/progen2-oas/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.7,
                top_p = 0.92,
                num_samples = 2,
                max_length = 160
              ),
              items = list(
                list(
                  context = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/progen2-oas/generate/

   Generate endpoint for ProGen2 OAS.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Generation parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 0.8) — Sampling temperature

        - **top_p** (*float*, range: 0.0-1.0, default: 0.9) — Nucleus sampling probability

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences generated per input item

        - **max_length** (*int*, range: 12-512, default: 128) — Maximum length of each generated sequence in tokens


      - **items** (*array of objects*, min: 1, max: 1) --- Input items:

        - **context** (*string*, min length: 1, max length: 512, required) — Amino acid context sequence with unambiguous residue codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/progen2-oas/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.7,
          "top_p": 0.92,
          "num_samples": 2,
          "max_length": 160
        },
        "items": [
          {
            "context": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSA"
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

        - **[i]** (*array of objects*, length: ``num_samples``) — Generated samples for the i-th input item

          - **sequence** (*string*, length: 1–512) — Generated amino acid sequence (unambiguous amino acid alphabet)

          - **ll_sum** (*float*) — Sum of token log-likelihoods over ``sequence`` (natural logarithm units)

          - **ll_mean** (*float*) — Mean token log-likelihood over ``sequence`` (natural logarithm units)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLIGGSYYDYYYYYGMDVWGQGTTVTVSS",
              "ll_sum": -30.27536106109619,
              "ll_mean": -0.23838866502046585
            },
            {
              "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDPIVLVVYAISAFDIWGQGTMVTVSS",
              "ll_sum": -19.65195369720459,
              "ll_mean": -0.15848349779844284
            }
          ]
        ]
      }


Performance
-----------

- Model variants and parameter scales:
  
  - ``progen2-oas``: antibody-focused model aligned with PROGEN2‑medium (~764M parameters), trained on redundancy‑reduced OAS; optimized for antibody‑like generations and likelihood‑based library scoring
  - ``progen2-medium``: ~764M‑parameter universal protein model; strong trade‑off between throughput and zero‑shot fitness ranking on narrow mutational landscapes
  - ``progen2-large`` and ``progen2-bfd90``: ~2.7B‑parameter universal models; lower perplexity on natural sequences and improved performance on some wide/low‑homology fitness landscapes, with ``progen2-bfd90`` generally strongest on out‑of‑distribution benchmarks

- Hardware execution and decoding optimizations:
  
  - ``progen2-oas`` runs on 2 vCPUs with 8 GB RAM; ``progen2-medium`` uses a single 8 GB T4‑class GPU; ``progen2-large``/``progen2-bfd90`` use a single 16 GB T4‑class GPU, matching their memory footprints
  - All variants use batched autoregressive decoding with cached key/value attention states and fused attention/MLP kernels, keeping per‑token generation cost close to linear in sequence length for continuations up to 128 residues

- Comparative speed and cost vs other BioLM sequence models:
  
  - Compared with encoder‑only scorers such as ESM‑2 650M/3B, ProGen2 has higher per‑token compute due to strict autoregression, but for ≤128‑residue continuations the end‑to‑end latency is similar because it avoids repeated masked scoring of full sequences
  - Compared with larger sequence‑to‑sequence transformers such as Evo 1.5 8k Base and Evo 2 1B Base, ProGen2 is typically faster per generated residue and cheaper for large variant‑library scoring, as it is decoder‑only with smaller memory footprint and no long cross‑attention contexts

- Zero‑shot fitness and design behavior across sizes:
  
  - On narrow DMS landscapes, ~764M‑parameter models (``progen2-medium``) achieve the best average Spearman correlations (~0.50), matching or exceeding some larger encoder baselines
  - On wide or low‑homology landscapes (AAV, GFP, CM, GB1), larger variants (``progen2-large``, ``progen2-bfd90``) better recover high‑fitness tail variants, with clear gains on epistatic GB1 top‑variant discovery
  - For antibody‑related tasks, ``progen2-oas`` improves developability‑relevant statistics (aggregation, solubility) for generated libraries, while universal ProGen2 models generally achieve higher rank‑ordering accuracy for global antibody properties such as expression quality and melting temperature

Applications
------------

- Generation of diverse, human-like antibody VH sequence libraries for discovery campaigns using PROGEN2-OAS, enabling pharma and biotech teams to go beyond naïve or immunized animal repertoires while still matching natural OAS-like sequence statistics; particularly valuable for building heavy-chain variable-domain libraries with realistic CDR length patterns and framework usage, but not a drop-in replacement for target-specific panning and downstream functional screening
- In silico exploration of developability-relevant VH sequence neighborhoods (e.g., aggregation propensity, solubility, stability proxies) by generating batches of PROGEN2-OAS VH variants around a lead framework/CDR context and ranking them with external developability predictors, allowing therapeutic teams to de-risk liabilities (such as hydrophobic CDR patches) before cell-line development; useful for improving expression and solubility, but not guaranteed to preserve antigen binding without experimental validation
- Zero-shot prioritization of VH variants in affinity maturation or library-mining campaigns by using PROGEN2-OAS log-likelihood scores (ll_sum or ll_mean) to rank candidate CDR and framework mutations, helping teams triage large mutational sets down to tractable panels for wet-lab screening; particularly useful when DMS data are limited, though the model does not see antigen context and therefore cannot replace structure-based or binding-assay-driven design
- Rapid generation of species- and isotype-adapted VH variants (e.g., humanization-like workflows) by conditioning PROGEN2-OAS on tailored N-terminal framework prompts reflecting species and chain type, enabling CROs and biotech companies to steer variable domains toward human-like repertoires while maintaining key CDR motifs; effective for moving sequences toward human-like space, but final liabilities and immunogenicity still require orthogonal in silico and experimental assessment
- Construction of realistic synthetic VH benchmarking sets for internal analytics and ML tooling by sampling large, non-redundant repertoires from PROGEN2-OAS that mirror OAS diversity, giving bioinformatics and data science teams high-quality test beds for annotation, clustering, paratope prediction, and sequence-analytics pipelines without relying solely on proprietary or patient-derived datasets

Limitations
-----------

- **Maximum sequence length**: Input ``context`` must be between 1 and ``512`` amino acids (``min_length=1``, ``max_length=512``). Longer sequences are rejected; truncate or window sequences client-side before calling the ``generator`` endpoint. Generated ``sequence`` outputs are also capped by ``max_length`` (``ge=12``, ``le=512``), so you cannot autoregress beyond position 512 in a single request.
- **Batch size and sampling limits**: Each ``generator`` request can include at most ``1`` item in ``items`` (``max_items=1``). Each item can request at most ``3`` generated sequences via ``num_samples`` (``ge=1``, ``le=3``). Large libraries must be created via many (possibly parallel) API calls; this is not a “millions of sequences per call” service.
- **Generation controls and scores**: ``temperature`` (``0.0–8.0``) and ``top_p`` (``0.0–1.0``) only control diversity of amino acids in the output ``sequence``. They do not enforce binding, stability, developability, or other biophysical constraints. ``ll_sum`` and ``ll_mean`` are log-likelihood scores under the ProGen2-OAS language model and are useful only for *relative* ranking within the same ``model_type="oas"`` and identical ``params``, not as calibrated fitness, affinity, or liability predictions.
- **Antibody- and data-specific bias**: ``model_type="oas"`` is trained solely on OAS antibody variable fragments and reproduces that distribution, including artifacts like frequent N-terminal truncations in the training data. It is not a general-purpose protein generator and is unsuitable for non-antibody proteins, non-Ig folds, or highly engineered scaffolds far outside immune-repertoire space.
- **Use cases where ProGen2 OAS is not optimal**: This API exposes a single-chain, sequence-only causal language model. It does not return embeddings, structures, paired heavy–light designs, or multi-chain complexes. It is not the right choice when you need structure prediction, sequence embeddings for clustering/visualization, conditioning on antigen or backbone structure, or strict control over global properties (e.g., solubility, aggregation) without downstream filters or models.
- **Scientific and zero-shot limitations**: Although ProGen2 variants can correlate with fitness on some benchmarks, ``ll_mean`` from the ``oas`` model does not reliably predict experimental fitness, affinity, or developability, especially for out-of-distribution antibodies, aggressive mutational scans, or antigen-specific optimization. For tasks like final candidate selection, epistatic landscape exploration, or joint structure–sequence design, plan to combine this API with specialized models and experimental screening.

How We Use It
-------------

ProGen2 OAS enables rapid in silico exploration of antibody VH sequence space as a generative engine within closed-loop discovery and optimization campaigns. Standardized APIs generate up to three VH variants per framework context, which teams then route through structure prediction (e.g., IgFold, AlphaFold-based models), developability scoring (aggregation, solubility, charge, liabilities), and zero-shot fitness predictors to filter, rank, and select sequences for synthesis. By wiring ProGen2 OAS into assay data pipelines, LIMS, and sequence analytics tools, experimental data from each round can be fed back into campaign-specific models to reduce experimental burden and systematically move toward antibodies with improved binding, stability, and manufacturability profiles.

- Integrated with other BioLM generative and predictive models to co-optimize sequence novelty, fitness, and developability for antibody leads.
- Used in multi-round, lab-in-the-loop campaigns where batched ProGen2 OAS generations are programmatically scored, triaged, and advanced to synthesis through standardized API-driven pipelines.

Related
-------

- ``ProGen2 Large`` – General-purpose ProGen2 model trained on diverse proteins; useful to benchmark antibody-focused ProGen2 OAS generations against broader protein designs and likelihoods.
- ``ESM-1v`` – Zero-shot variant effect predictor that complements ProGen2 OAS by ranking antibody variants for likely fitness preservation or improvement.
- ``IgT5 Unpaired`` – Antibody-specific sequence model for unpaired chains that can be combined with ProGen2 OAS to propose and refine VH variable region designs.
- ``ImmuneFold Antibody`` – Antibody structure prediction model to assess structural plausibility and basic developability of VH sequences generated with ProGen2 OAS.

References
----------

- Nijkamp, E., Ruffolo, J., Weinstein, E. N., Naik, N., & Madani, A. (2023). *ProGen2: Exploring the Boundaries of Protein Language Models*. arXiv:2309.16590. https://doi.org/10.48550/arXiv.2309.16590
