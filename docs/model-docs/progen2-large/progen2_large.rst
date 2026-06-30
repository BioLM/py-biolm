ProGen2 Large API
=================

ProGen2 Large is a 2.7B-parameter autoregressive protein language model for sequence continuation and likelihood-based scoring of amino acid sequences. The API accepts raw protein sequences up to 512 residues and can generate up to 3 continuations per input, with configurable temperature, top_p, and max_length (≤128 tokens), returning per-completion log-likelihoods (sum and mean). This GPU-accelerated service is suited to zero-shot, likelihood-based ranking for mutational scans, library design, and prioritizing protein variants.

Generate
--------

Generate up to two protein sequences continuing from a short signal peptide context using the ProGen2 Large model

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="progen2-large",
                action="generate",
                params={
                  "temperature": 0.7,
                  "top_p": 0.92,
                  "num_samples": 2,
                  "max_length": 80
                },
                items=[
                  {
                    "context": "MKTFFVVALATLLASASAA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/progen2-large/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.7,
                "top_p": 0.92,
                "num_samples": 2,
                "max_length": 80
              },
              "items": [
                {
                  "context": "MKTFFVVALATLLASASAA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/progen2-large/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.7,
                    "top_p": 0.92,
                    "num_samples": 2,
                    "max_length": 80
                  },
                  "items": [
                    {
                      "context": "MKTFFVVALATLLASASAA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/progen2-large/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.7,
                top_p = 0.92,
                num_samples = 2,
                max_length = 80
              ),
              items = list(
                list(
                  context = "MKTFFVVALATLLASASAA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/progen2-large/generate/

   Generate endpoint for ProGen2 Large.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Generation parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 0.8) — Sampling temperature for sequence generation

        - **top_p** (*float*, range: 0.0-1.0, default: 0.9) — Nucleus sampling cumulative probability threshold

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate per input item

        - **max_length** (*int*, range: 12-512, default: 128) — Maximum total sequence length in tokens, including the input context and generated continuation


      - **items** (*array of objects*, min: 1, max: 1) --- Input records:

        - **context** (*string*, min length: 1, max length: 512, required) — Amino acid context sequence using unambiguous residue codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/progen2-large/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.7,
          "top_p": 0.92,
          "num_samples": 2,
          "max_length": 80
        },
        "items": [
          {
            "context": "MKTFFVVALATLLASASAA"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of arrays of objects*) --- One result per input item, in the order requested:

        - **[ ]** (*array of objects*, length: num_samples) — Generated samples for a single input item

          - **sequence** (*string*, length: 1–512) — Generated amino acid sequence, unaligned
          - **ll_sum** (*float*) — Log-likelihood of ``sequence`` under the selected ProGen2 model, unnormalized sum over tokens, natural log units (ln)
          - **ll_mean** (*float*) — Log-likelihood of ``sequence`` under the selected ProGen2 model, mean per token, natural log units (ln)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKTFFVVALATLLASASAASIEAEGDEELFLELFENESSSSQFARLVSESDYQKAVSFLDLNLRVKVTRPKELMGVVAR",
              "ll_sum": -177.1235809326172,
              "ll_mean": -2.2420706748962402
            },
            {
              "sequence": "MKTFFVVALATLLASASAATRSLRYADSAAANDAPAAAAASSGASGASGASGASGASGASGASGASGASGASGASGASG",
              "ll_sum": -113.85159301757812,
              "ll_mean": -1.4411593675613403
            }
          ]
        ]
      }


Performance
-----------

- Model family and deployment
  - ProGen2 Large and BFD90 are ~2.7B‑parameter autoregressive transformers as in the ProGen2 paper, deployed with T4‑class GPU acceleration and mixed‑precision decoding for throughput
  - Medium (764M) and OAS (~764M, antibody‑specific) variants are smaller and typically faster per request, with lower capacity for broad, low‑homology sequence modeling than Large/BFD90

- Predictive performance and fitness ranking
  - On held‑out UniRef‑like protein sets, Large/BFD90 achieve perplexity ≈11.1, better density estimation than Medium/OAS and close to the 6.4B xlarge models
  - Zero‑shot fitness on narrow landscapes (single/double mutants) peaks at the 764M size (average Spearman ρ ≈0.50) and is slightly lower for Large/xlarge, so Medium can rank near‑wild‑type variants as well as or better than Large/BFD90 at lower cost
  - On wide, low‑homology landscapes (e.g., AAV, GFP, GB1), Large/BFD90 track xlarge and outperform smaller variants, enriching for high‑fitness variants many mutations away from known sequences

- Comparison to other BioLM sequence‑only models
  - Versus ESM‑2 (8M–3B), ProGen2 Large/BFD90 generally yield lower perplexity on UniRef‑like data and stronger zero‑shot fitness on many deep mutational scanning benchmarks, with per‑call latency similar to ESM‑2 650M–3B and higher than ESM‑2 35M/150M
  - Versus Evo 1.5 / Evo 2, Evo models are more parameter‑efficient and offer better controllability/conditioning, but ProGen2 Large/BFD90 often match or exceed them in zero‑shot fitness on classic benchmarks, especially for difficult, low‑homology edits
  - Versus ESM‑1v, ProGen2 Large/BFD90 typically provide equal or better zero‑shot fitness ranking on FLIP narrow and wide tasks, particularly when indels are present, benefiting from the autoregressive objective

- Structural realism and end‑to‑end pipelines
  - Sequences sampled from ProGen2 xlarge fold with high confidence (median pLDDT ~74, median TM‑score ~0.9 to nearest PDB template); ProGen2 Large/BFD90 show similar fold realism with slightly lower structural similarity but comparable sequence‑level diversity
  - In design pipelines such as “ProGen2 → structure prediction (e.g., AlphaFold2/Chai‑1) → sequence‑to‑structure design (e.g., ProteinMPNN/LigandMPNN)”, ProGen2 contributes little to wall‑clock time relative to structure prediction but substantially improves which candidates are worth folding, and explores unconstrained sequence space more efficiently than structure‑conditioned generators when no backbone is fixed

Applications
------------

- Protein variant scoring to prioritize candidates before wet-lab screening, using ProGen2 Large log-likelihoods (``ll_sum`` / ``ll_mean``) from the ``generator`` endpoint to rank mutational libraries (e.g. stability or expression variants of a therapeutic enzyme or receptor), reducing the number of low-fitness constructs sent to expression while still exploring diverse sequence space; particularly useful when you have limited assay capacity but many possible single or combinatorial mutations
- Zero-shot fitness estimation on mutational landscapes that are several edits away from any known sequence for challenging proteins such as gene therapy capsid proteins or low-homology industrial biocatalysts, using ProGen2 Large likelihood as a proxy for viability to triage which regions of sequence space are worth synthesizing; especially valuable when MSAs are shallow or unreliable and alignment-based methods underperform, but less optimal when you have rich family-specific structural/functional data that could support specialized models
- Generative library design for protein engineering campaigns, where ProGen2 Large (via the ``generator`` endpoint with tunable ``temperature`` / ``top_p`` and ``max_length`` up to 512 aa) is used to sample full-length variants around a starting scaffold (e.g. new versions of a known enzyme or binding domain), enabling construction of DNA libraries that expand beyond natural diversity while maintaining plausible foldability; best used in conjunction with downstream filters (structure prediction, developability models, domain knowledge) rather than as a standalone “one-shot” design tool
- Focused generation of proteins with a shared structural architecture, by combining ProGen2 Large with fine-tuning or conditioning workflows on curated sets of proteins sharing a known fold class (e.g. two-layer sandwich CATH 3.30 domains) and then using the API to generate sequences that preserve global architecture but diversify local functional regions such as ligand-binding pockets, supporting campaigns that need new binders or catalysts within a defined scaffold family; less appropriate when the target function is tightly dependent on a specific active-site geometry that cannot be inferred from sequence statistics alone
- Developability-aware protein library refinement for therapeutic and industrial proteins, where teams first generate a broad library with ProGen2 Large and then use the returned likelihoods to re-rank or filter variants for better expression, solubility, or manufacturability profiles (e.g. selecting top X% of model-ranked variants for CHO expression or microbial fermentation), reducing attrition from biophysical liabilities; this can improve hit quality but does not replace dedicated, assay-trained developability or immunogenicity models when those are available

Limitations
-----------

- **Maximum sequence length and valid input**: Input ``context`` and generated tokens together are limited to ``ProGen2Params.max_sequence_len`` = 512 amino acids. For each item, the requested ``max_length`` plus the ``context`` length must be ≤512. Each ``context`` must be 1–512 characters long and pass ``validate_aa_unambiguous`` (standard unambiguous amino acids only; no custom tokens, gaps, or non-protein characters).
- **Batch size and sampling limits**: ``ProGen2Params.batch_size`` = 1, so each request may contain at most 1 item in ``items``. For that item, ``num_samples`` is limited to 1–3. Larger ``max_length`` with higher ``num_samples`` increases latency and cost. ``temperature`` (0.0–8.0) and ``top_p`` (0.0–1.0) are exposed directly; extreme settings (very high ``temperature`` or very low ``top_p``) typically produce unstable or low-quality sequences.
- **Model type constraints and hardware**: ``ProGen2ModelTypes.MEDIUM``, ``LARGE``, and ``BFD90`` run on a single T4-class **GPU Type** and are trained on broad protein databases for general protein generation and scoring. ``ProGen2ModelTypes.OAS`` runs on CPU only and is trained on antibody repertoire data; it is specialized for antibody-like sequences and is not recommended for generic enzyme, receptor, or arbitrary protein design.
- **Causal LM behavior and outputs**: ProGen2 is a left-to-right causal language model that performs next-token prediction only. It does not return embeddings, per-residue features, or 3D structure. The output fields ``ll_sum`` and ``ll_mean`` are log-likelihood scores under the language model, not calibrated measures of fitness, stability, binding, or expression, and should not be used as direct surrogates for these properties without validation.
- **Fitness prediction and data-distribution mismatch**: While ``ll_sum`` / ``ll_mean`` can correlate with experimental fitness on some landscapes, performance depends strongly on how similar your proteins are to the training distribution. Larger variants (e.g. those backing ``LARGE`` / ``BFD90``) do not always yield better zero-shot fitness ranking than mid-sized models and may perform worse on narrow, near-natural mutational scans; for large-scale variant ranking, simpler or faster scoring models may be more appropriate.
- **Use cases where ProGen2 is not optimal**: This API is not designed for (a) structure-conditioned design from a known backbone, (b) explicitly 3D-aware scoring of detailed active sites, conformational changes, or metal coordination, (c) generation of proteins longer than 512 amino acids (multi-domain or multi-chain designs), or (d) fine-grained developability optimization (aggregation, solubility, polyspecificity) where specialized predictive or structure-based models generally perform better. ProGen2 is best used as a sequence generator or coarse scorer within a broader design and evaluation pipeline.

How We Use It
-------------

ProGen2 Large serves as a generative backbone for protein engineering campaigns, enabling rapid exploration of local sequence space around a target while keeping data and compute workflows manageable across teams. In typical use, it proposes diverse variants conditioned on a seed sequence via standardized generation APIs, and those candidates are then evaluated and filtered with separate sequence- and structure-based predictors (e.g., AlphaFold2-derived metrics, biophysical property models, antibody- or enzyme-focused fitness estimators). Experimental readouts from each design–build–test cycle can then inform downstream ranking models or narrower ProGen2 variants, so that subsequent generations better align with assay objectives such as activity, binding, stability, and expression.

- For enzyme and binder campaigns, ProGen2 Large establishes an initial, diverse variant pool, while dedicated fitness and developability models rank candidates under constraints like charge, size, aggregation risk, and manufacturability.  
- In antibody optimization, ProGen2-based libraries integrate with repertoire-specific models and developability filters to construct focused panels for affinity maturation and lead refinement with fewer experimental rounds.

Related
-------

- ``ProGen2 Medium`` – Smaller ProGen2 variant with the same architecture and training objective; useful when you want faster, cheaper sequence generation while maintaining similar behavior to ProGen2 Large.
- ``ProGen2 BFD90`` – ProGen2 model trained with a metagenomic-enriched dataset; useful alongside ProGen2 Large when comparing log-likelihoods under different underlying training distributions.
- ``ESM-1v`` – Zero-shot fitness prediction–oriented protein language model; can provide an independent likelihood-based score for variants generated or scored with ProGen2 Large.
- ``ProteinMPNN`` – Structure-conditioned sequence design model; can be used after ProGen2 Large to redesign or refine sequences around a given backbone while preserving the desired fold.

References
----------

- Nijkamp, E., Ruffolo, J., Weinstein, E. N., Naik, N., & Madani, A. (2023). `ProGen2: Exploring the Boundaries of Protein Language Models <https://doi.org/10.48550/arXiv.2306.06106>`_. *arXiv preprint arXiv:2306.06106*.

