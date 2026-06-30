ProGen2 BFD90 API
=================

ProGen2 BFD90 is a 2.7B-parameter autoregressive protein language model trained on ~1B sequences from UniRef90 and a 90% identity–clustered BFD metagenomic dataset, emphasizing broader metagenomic coverage than standard ProGen2. The API exposes GPU-accelerated de novo amino-acid sequence continuation from an N-terminal context (up to 512 residues per request), using configurable temperature and nucleus (top-p) sampling to generate up to 3 variants per input. Typical uses include generative library design, zero-shot fitness prioritization, and diversity filtering in protein engineering campaigns.

Generate
--------

Generate up to 2 protein sequence continuations from a short enzyme-like N-terminal context using the ProGen2 BFD90 model.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="progen2-bfd90",
                action="generate",
                params={
                  "temperature": 0.7,
                  "top_p": 0.9,
                  "num_samples": 2,
                  "max_length": 150
                },
                items=[
                  {
                    "context": "MKTAYIAKQRQISFVKSHFSRQ"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/progen2-bfd90/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_samples": 2,
                "max_length": 150
              },
              "items": [
                {
                  "context": "MKTAYIAKQRQISFVKSHFSRQ"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/progen2-bfd90/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_samples": 2,
                    "max_length": 150
                  },
                  "items": [
                    {
                      "context": "MKTAYIAKQRQISFVKSHFSRQ"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/progen2-bfd90/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.7,
                top_p = 0.9,
                num_samples = 2,
                max_length = 150
              ),
              items = list(
                list(
                  context = "MKTAYIAKQRQISFVKSHFSRQ"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/progen2-bfd90/generate/

   Generate endpoint for ProGen2 BFD90.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Generation configuration parameters:

        - **temperature** (*float*, range: 0.0-8.0, default: 0.8) — Sampling temperature

        - **top_p** (*float*, range: 0.0-1.0, default: 0.9) — Nucleus sampling probability

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences generated per input item

        - **max_length** (*int*, range: 12-512, default: 128) — Maximum number of tokens in each generated sequence


      - **items** (*array of objects*, min: 1, max: 1) --- Input specification for each generation request:

        - **context** (*string*, min length: 1, max length: 512, required) — Input amino acid sequence using unambiguous residue codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/progen2-bfd90/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.7,
          "top_p": 0.9,
          "num_samples": 2,
          "max_length": 150
        },
        "items": [
          {
            "context": "MKTAYIAKQRQISFVKSHFSRQ"
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

        - (*inner array of objects*, length: ``num_samples`` per input) — Generated sequences and log-likelihood scores

          - **sequence** (*string*) — Generated amino acid sequence including the provided context, length: 1–512 residues

          - **ll_sum** (*float*) — Sum of per-token log-likelihoods for the generated sequence under the selected ProGen2 model, in natural log units

          - **ll_mean** (*float*) — Mean per-token log-likelihood for the generated sequence under the selected ProGen2 model, in natural log units

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERVARLEGASASTRDLGQAYSSPEGFTALEGKLNIPQSAKANRQQYFEALARNRVDFLRATVVANELQQPTETEEKILAFRNECAFEKERYSDEVWAAWMRLLKKGEPQYRHITRYKRRALSRLG",
              "ll_sum": -278.47631072998047,
              "ll_mean": -1.8689684867858887
            },
            {
              "sequence": "MKTAYIAKQRQISFVKSHFSRQPIAVVGLSCLFPKARDLRQYWQNIVSAVDCITDVPPDRWDVADYYDPDPSAPDKTYGRRGGFIPQIDFNPMEFGIPPNTLEATDTSQLLSLLVARQVLEDAGYGEGRAFDRERTSVILGVTGALELV",
              "ll_sum": -192.94214248657227,
              "ll_mean": -1.2949137389659882
            }
          ]
        ]
      }


Performance
-----------

- Model characteristics and deployment
  - ProGen2 BFD90 is a 2.7B-parameter autoregressive transformer trained on Uniref90 plus a high-redundancy BFD90 metagenomic dataset, and on BioLM runs on NVIDIA T4-class GPUs with 4 vCPUs and 16 GB RAM per replica
  - Rotary positional encodings and optimized attention implementations enable efficient left-to-right decoding with cached key/value states, so per-token cost decreases after the initial context pass
  - Batched decoding across multiple samples per input (up to 3 per request item) reuses the same cached hidden states, improving effective throughput for library generation and likelihood scoring

- Predictive and generative performance relative to other ProGen2 variants
  - Compared to ProGen2 Large (2.7B, Uniref90+BFD30), BFD90 achieves slightly lower perplexity on identity-reduced evolutionary test sets (e.g., test-max90: 12.6 vs. 12.7) and modestly better zero-shot performance on metagenomic-style and low-homology landscapes (higher AAV AUC, higher GB1 top-100 average), reflecting its stronger metagenomic bias
  - Relative to ProGen2 Medium (764M), BFD90 improves robustness on wider, epistatic, and indel-rich landscapes but smaller models can match or slightly outperform it on narrow, near–wild-type DMS-style landscapes due to beneficial underfitting of phylogenetic artifacts
  - Compared to ProGen2 OAS (antibody-specific 764M), BFD90 yields higher average correlations on general fitness benchmarks (expression, stability, broad DMS) and antibody “general” properties (e.g., higher average ρ for expression and Tm), while OAS remains superior for matching fine-grained antibody repertoire statistics

- Comparison to other BioLM sequence models
  - Versus encoder-only masked language models such as ESM-2, ProGen2 BFD90 provides direct autoregressive log-likelihoods for continuations, enabling straightforward zero-shot fitness estimation and sampling-based design; ESM-style models are stronger for embeddings and property prediction but require additional scaffolding to generate sequences and to handle long insertions/deletions
  - Compared to structure-conditioned models like ProteinMPNN and LigandMPNN, BFD90 does not require 3D structural inputs or graph construction and is therefore more efficient when only sequence information is available, though structure-based models generally achieve higher accuracy on fixed-backbone recovery and local packing metrics

- Zero-shot fitness and out-of-distribution behavior
  - Within the ProGen2 family, zero-shot fitness performance on narrow landscapes peaks at 764M parameters, but BFD90 remains within the same performance band while improving on wider or low-homology tasks (e.g., better GB1 top-100 averages than the Uniref90+BFD30 2.7B model)
  - On indel-rich benchmarks, the fully autoregressive architecture used by BFD90 handles insertions and deletions more stably than masking-based encoders such as ESM-1v, which rely on heuristic scoring over masked positions rather than explicit left-to-right likelihoods

Applications
------------

- Protein variant scoring for industrial enzyme campaigns, using ProGen2 BFD90 log-likelihoods to rank mutation libraries (single-point and combinatorial) before wet-lab screening, reducing experimental load by focusing on variants that are more consistent with evolutionarily plausible, foldable sequences; particularly useful when you already have a lead scaffold and moderate assay throughput, but not optimal as a sole decision-maker when you have hard biophysical constraints that are not reflected in the training distribution
- Zero-shot fitness-style prioritization of low-homology or highly mutated protein designs by comparing ProGen2 BFD90 log-likelihoods across candidates, helping teams differentiate likely-functional from non-functional sequences in “wide” landscapes (high edit distance from wild type) and triage de novo designs, scaffold hops, or metagenomic hits for follow-up characterization; valuable when MSAs are shallow or unreliable, though performance can be weaker for very well-studied families where specialized, family-specific models or rich experimental datasets are available
- Generative expansion of protein families for library design, by using the generator endpoint to sample ProGen2 BFD90 continuations from an N-terminal context of a known sequence or domain architecture to produce diverse, plausible variants that remain within a desired fold class (e.g., CATH 3.30 two-layer sandwich) for directed evolution, stability engineering, or substrate scope broadening; most effective when combined with downstream structural prediction and application-specific filters (e.g., stability, aggregation, or activity models), rather than used as an unconstrained generator
- Developability-aware protein library pruning, where ProGen2 BFD90 log-likelihoods are used as a universal prior to filter large, synthetically generated protein sets (e.g., recombination libraries, codon-optimized variants) by removing sequences with extremely low model probability that are more likely to be misfolded or non-expressing, thereby improving the “hit rate” for expression, solubility, and stable folding in microbial or mammalian expression systems; this is helpful upstream of high-throughput screening but does not replace explicit developability models or experimental tests
- In silico risk assessment for sequence edits in production biologics (non-antibody proteins), using ProGen2 BFD90 to compare log-likelihoods of proposed manufacturing or IP-driven sequence changes (e.g., removal of PTM sites, introduction of tags, minor sequence humanization of non-antibody scaffolds) and flag edits that move the sequence into very low-probability regions of protein space, guiding regulatory and CMC teams toward safer, more conservative modifications while still requiring orthogonal stability and activity assessments

Limitations
-----------

- **Maximum sequence length**: Input ``context`` and generated ``sequence`` outputs are limited to ``ProGen2Params.max_sequence_len`` (``512``) amino acids. Requests with ``context`` longer than ``512`` or ``max_length`` > ``512`` are rejected. The model cannot reason about residues beyond this window, which can be limiting for very long multi-domain or fusion proteins.
- **Batch size and sampling limits**: Each request can contain at most ``ProGen2Params.batch_size`` (``1``) item in ``items``, and each item can generate at most ``num_samples=3`` sequences. Large combinatorial libraries therefore require client-side batching and orchestration; workflows needing high-throughput scoring are better served by simpler scoring models or higher-throughput generative models.
- **Input / output semantics**: ``context`` must be a non-empty, unambiguous amino-acid sequence (standard letters only, validated by ``validate_aa_unambiguous``) and is treated purely as an autoregressive prefix, not as a structured constraint (no masks, positional constraints, or structure/ligand input). Outputs are plain ``sequence`` strings with log-likelihood scores (``ll_sum``, ``ll_mean``) that reflect model preference over the training distribution, not experimental fitness, structure quality, or developability.
- **Generative language model only**: ProGen2 BFD90 is an autoregressive protein language model trained on Uniref90+BFD90 sequences. The API only supports sequence generation and log-likelihood scoring; it does not provide embeddings, structures, residue-level annotations, or backbone-conditioned design. Tasks like structure prediction, structure-conditioned design, clustering, or visualization require separate embedding or structure models.
- **Fitness prediction and generalization limits**: Although ``ll_mean`` can correlate with fitness on some benchmarks, correlations are landscape- and family-dependent and degrade for wide mutational scans, low-homology regions, or highly novel scaffolds. It should not be used as a stand-alone oracle for stability, binding affinity, immunogenicity, or other safety-critical properties; combine with dedicated stability/structure/developability models and experimental validation.
- **Model-family and domain coverage**: Different ProGen2 variants are biased to their training sets. The BFD90 variant is a universal model, not antibody-specific; for antibody CDR design, paratope-focused optimization, or strict control of immune-repertoire-like properties, specialized antibody models (e.g. ``oas``) or structure-based antibody tools are typically more appropriate.

How We Use It
-------------

ProGen2 BFD90 enables organizations to shift from random mutagenesis to systematic, model-driven protein engineering by acting as the generative core in multi-round design–build–test–learn cycles. Teams use the API to propose diverse, biophysically realistic variants around starting scaffolds, then connect these libraries to structure predictors like AlphaFold2, in silico developability and stability screens, and zero-shot fitness models for filtering and ranking before synthesis. Training on a broad UniRef90+BFD90 mixture makes the model suitable for low-homology enzyme programs and therapeutic protein work, while standardized, scalable API calls support integration into workflow engines, LIMS, and ML pipelines.

- In enzyme and industrial protein programs, ProGen2 BFD90 accelerates local sequence exploration around desired functions, with downstream stability, activity, and manufacturability models refining candidates into synthesis-ready shortlists matched to assay capacity.  
- In antibody and therapeutic protein campaigns, it integrates with repertoire-focused models, developability predictors, and experimental binding data to iteratively propose variants that balance affinity, expression, and biophysical risk, lowering the number of wet-lab cycles needed to identify leads.

Related
-------

- ``ProGen2 Large`` – Same model family as ProGen2 BFD90 but trained on a different sequence mix; useful for comparing how metagenomic data depth affects sequence generation and likelihood-based scoring of variants.
- ``ESM-1v`` – Zero-shot mutation effect predictor that complements ProGen2 BFD90 by independently scoring designed variants for fitness without additional training.
- ``ProteinMPNN`` – Structure-conditioned sequence designer that pairs well with ProGen2 BFD90 by refining or redesigning ProGen2-generated sequences for specified backbone structures.
- ``ESMFold`` – Fast structure prediction model that can be used after ProGen2 BFD90 to assess foldability and structural novelty of generated sequences directly from sequence.

References
----------

- Nijkamp, E., Ruffolo, J., Weinstein, E. N., Naik, N., & Madani, A. (2023). `ProGen2: Exploring the Boundaries of Protein Language Models <https://arxiv.org/abs/2306.06155>`_. *arXiv preprint arXiv:2306.06155*.

