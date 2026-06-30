TEMPRO 650M API
===============

TEMPRO 650M predicts nanobody (VHH/sdAb) melting temperature (Tm, °C) directly from amino acid sequence using ESM-2 t33_650M embeddings (≈650M parameters, ~2.4 GB) and a DNN regressor trained on 567 curated sequences (NbThermo plus internal, 80/20 split). The API accepts 100–160 aa protein sequences, performs GPU-accelerated batch inference (up to 8 sequences per request), and returns per-sequence Tm. Tailored to single-domain antibodies (~120 aa typical), it supports library triage, variant ranking, and stability gating in design loops.

Predict
-------

Predict melting temperature for input nanobody sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="tempro-650m",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSFNWIHWVRQAPGKGLEYKMIAASSSVGGTPYYADSVEGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCV"
                  },
                  {
                    "sequence": "QVQLQESGGGSVQAGGSLRLSCTASGFNIHKNYLAHWFRQAPGKEREGVAALSPAGGTPYYADSVKGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCARR"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/tempro-650m/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSFNWIHWVRQAPGKGLEYKMIAASSSVGGTPYYADSVEGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCV"
                },
                {
                  "sequence": "QVQLQESGGGSVQAGGSLRLSCTASGFNIHKNYLAHWFRQAPGKEREGVAALSPAGGTPYYADSVKGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCARR"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/tempro-650m/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSFNWIHWVRQAPGKGLEYKMIAASSSVGGTPYYADSVEGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCV"
                    },
                    {
                      "sequence": "QVQLQESGGGSVQAGGSLRLSCTASGFNIHKNYLAHWFRQAPGKEREGVAALSPAGGTPYYADSVKGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCARR"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/tempro-650m/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSFNWIHWVRQAPGKGLEYKMIAASSSVGGTPYYADSVEGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCV"
                ),
                list(
                  sequence = "QVQLQESGGGSVQAGGSLRLSCTASGFNIHKNYLAHWFRQAPGKEREGVAALSPAGGTPYYADSVKGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCARR"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/tempro-650m/predict/

   Predict endpoint for TEMPRO 650M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*integer*, default: 8) — Maximum number of items allowed in a single request

        - **min_sequence_len** (*integer*, default: 100) — Minimum allowed length of each protein sequence

        - **max_sequence_len** (*integer*, default: 160) — Maximum allowed length of each protein sequence


      - **items** (*array of objects*, min length: 1, max length: 8) --- Input sequences:

        - **sequence** (*string*, min length: 100, max length: 160, required) — Protein sequence using supported amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/tempro-650m/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSFNWIHWVRQAPGKGLEYKMIAASSSVGGTPYYADSVEGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCV"
          },
          {
            "sequence": "QVQLQESGGGSVQAGGSLRLSCTASGFNIHKNYLAHWFRQAPGKEREGVAALSPAGGTPYYADSVKGRFTISRDNAKNTVYLQMNSLRPEDTAVYYCARR"
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

        - **tm** (*float*, typical range: 40.0–95.0 °C) — Predicted melting temperature in Celsius

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "tm": 91.82323455810547
          },
          {
            "tm": 73.70567321777344
          }
        ]
      }


Performance
-----------

- Throughput and latency (end-to-end, including ESM-2 650M embedding and regression head)
  - NVIDIA H100 80GB: 8-sequence batch completes in ~0.30–0.60 s (p95 ~0.75 s); 12–22 sequences/s sustained
  - NVIDIA A100 40GB: 8-sequence batch completes in ~0.60–1.10 s (p95 ~1.40 s); 8–14 sequences/s sustained
  - NVIDIA A10G 24GB: 8-sequence batch completes in ~1.20–1.80 s (p95 ~2.0 s); 4–7 sequences/s sustained

- Memory and compute characteristics
  - ESM-2 650M encoder weights ~2.4 GB; total VRAM use for an 8×160 aa batch typically ~4–6 GB
  - Mixed-precision FP16 kernels and fused attention reduce activation memory by ~30–40% vs FP32 and improve latency by ~20–30%

- Predictive accuracy on nanobody thermostability
  - On NbThermo-like held-out nanobody data, models using 650M embeddings achieve MAE in the mid–single-digit °C range for in-distribution sdAbs
  - External nanobody validation not seen during training yields R² ≈ 0.25 for 650M embeddings vs ≈ 0.58 for 3B and ≈ 0.67 for 15B, consistent with scaling of embedding capacity for stability prediction
  - Compared with general thermostability tools (e.g., ProTDet, DeepStabP), TEMPRO models preserve non-zero correlation on nanobody-only sets where generic predictors are near-random

- Relative performance within the TEMPRO family
  - TEMPRO 650M is typically 2–3× faster per batch than TEMPRO 3B and fits comfortably on 8–12 GB GPUs, at the cost of higher error; 3B roughly doubles explained variance vs 650M on external nanobody validation (R² ~0.58 vs ~0.25)
  - TEMPRO 15B serves as a higher-capacity research reference (e.g., literature reports MAE ~4.0 °C, RMSE ~5.7 °C on held-out nanobody test sets) but is substantially slower and more resource-intensive than 650M

Applications
------------

- High-throughput triage of VHH sequence libraries before expression: use TEMPRO 650M to rapidly score large VHH/nanobody candidate sets (from phage/yeast display, DNA synthesis, or generative design) and filter out sequences with low predicted Tm to reduce downstream assay load; this accelerates hit discovery by focusing expression and biophysics on stability-favored clones (e.g., drop sequences predicted <60–65 °C when room-temperature stability is required); best for coarse, at-scale filtering where speed matters; the API expects single-domain nanobody-length sequences (100–160 aa) and may be less reliable for scFv/IgG, VHH-Fc fusions, or heavily tagged constructs, so confirm with DSF/DSC
- Stability gating during affinity maturation and humanization: rank-order proposed VHH variants by predicted Tm to maintain or improve thermostability while exploring affinity-improving substitutions, enabling design cycles that enforce a minimum stability bar (e.g., keep ≥70 °C for subcutaneous delivery profiles); valuable for program teams balancing potency with developability; TEMPRO 650M provides sequence-level Tm regression only (no residue-level attribution), so use as a ranking heuristic and validate experimentally; for very tight or nuanced ranking, consider higher-capacity TEMPRO models via BioLM
- Developability risk assessment prior to scale-up and formulation: combine predicted Tm from TEMPRO 650M with orthogonal in silico metrics to flag nanobody leads at risk of low thermal robustness before investing in upstream cell line development and downstream fill–finish; helps decide whether to prioritize stabilization strategies or alternative leads (e.g., avoid scaling candidates predicted <55–60 °C when RT logistics are expected); predictions reflect sequence-intrinsic thermostability and do not account for buffer/excipients, pH, formulation, or glycosylation, so formulation effects must be confirmed in wet lab
- Change-control and QC for sequence edits across the engineering pipeline: quickly compare predicted Tm between a reference VHH and edited variants (signal peptides removed, purification tags altered, back-translation differences that change amino-acid sequence) to catch stability regressions before ordering DNA or scheduling biophysical panels; practical for vendor oversight and tech-transfer checkpoints; the model is calibrated on single-domain nanobody sequences (~12–15 kDa, 100–160 aa), so multi-domain fusions, long linkers, or out-of-range lengths can reduce reliability
- Indication and use-case alignment for target product profiles: use TEMPRO 650M predictions to align nanobody leads with intended deployment (e.g., prioritize higher Tm for field-deployable diagnostics or RT-stable reagents; tolerate moderate Tm for cold-chain biotherapeutics with rapid turnaround); enables early portfolio decisions without exhaustive stress studies; not a substitute for real-time and accelerated stability studies, which remain necessary for regulatory filings and final CMC decisions

Limitations
-----------

- **Input/Output contract**: ``items`` must contain 1–8 entries (see **Batching limits** below). Each entry requires a ``sequence`` of amino acids. The response ``results`` returns one object per input with a single float ``tm`` (predicted melting temperature in Celsius). No confidence interval or uncertainty estimate is provided.
- **Sequence limits**: **Minimum Sequence Length** 100 aa and **Maximum Sequence Length** 160 aa. Requests with ``sequence`` outside 100–160 amino acids are rejected. Only sequences composed of the 20 canonical amino acids are supported; non-standard residues, unknown characters, tags/linkers, or fusion partners should be removed before submission.
- **Batching limits**: **Batch Size** up to 8 sequences per request via ``items``. Larger workloads must be split client-side. Ordering is preserved 1:1 between ``items`` and ``results``.
- **Scope and generalization**: Trained on single-domain camelid VHH nanobodies (sdAbs). Predictions for full-length antibodies, scFvs, multi-domain constructs, shark VNARs, or non-antibody proteins are out-of-distribution and may be inaccurate even if they fall within the 100–160 aa length window.
- **Model performance caveats (650M variant)**: This endpoint uses embeddings from the ESM-2 650M model. External validation of the 650M-based TEMPRO variant showed modest fit to experimental data (e.g., for an independent INDI nanobody set, ``R² ≈ 0.25`` and larger errors for very low or very high experimental Tm values). Use outputs primarily for relative ranking and always confirm final designs experimentally.
- **Context not modeled**: Predictions are based on sequence alone and do not incorporate buffer conditions, pH, concentration, assay method, engineered disulfides, formulation, glycosylation/PTMs, oligomeric state, or Fc/fusion effects. This endpoint is best suited for mid/late-stage in silico triage of cleaned VHH domains, not for early ultra-high-throughput screening or constructs where domain or formulation context dominates Tm.

How We Use It
-------------

TEMPRO 650M enables fast, sequence-only nanobody Tm estimation that integrates into BioLM design–make–test–learn workflows, where it is used to prioritize thermostable variants before synthesis, guide affinity maturation, and enforce developability gates. Via standardized APIs, TEMPRO scores are combined with structure-derived metrics (AlphaFold2 pLDDT, NetSurfP-3.0), physicochemical filters (charge/pI, hydrophobicity, cysteine/disulfide patterns), solubility and aggregation predictors, and generative design models to accelerate downselection. In practice, teams deploy TEMPRO 650M for early, high-throughput triage and cost control on large nanobody libraries, then re-rank shortlists with larger TEMPRO variants (3B/15B) or project-calibrated models before ordering, reducing experimental cycles and improving lead quality.

- Rapid triage at scale: score hundreds of thousands to tens of millions of nanobody-length sequences, gate by Tm thresholds, and focus synthesis on thermostable designs.
- Design constraints and risk reduction: combine Tm with developability filters and regional structure confidence to preserve stable frameworks while exploring CDR diversity.
- Closed-loop optimization: route prioritized variants to assay, feed DSC/DSF results back through the API, and escalate from 650M to higher-capacity models for final ranking and campaign decisions.

Related
-------

- ``TEMPRO 3B`` – Higher‑capacity TEMPRO variant using ESM‑2 3B embeddings; run alongside 650M to ensemble or sanity‑check nanobody Tm estimates.
- ``ESM-2 650M`` – Provides the sequence embeddings TEMPRO 650M is trained on; precompute embeddings to speed large‑batch Tm prediction and reuse across design iterations.
- ``Peptides`` – Computes physicochemical descriptors (aliphatic index, GRAVY, charge, etc.) used in TEMPRO feature ablations; useful for interpreting Tm drivers and pre/post‑filtering candidate sequences.
- ``ABodyBuilder3 pLDDT`` – Antibody‑focused structure prediction with regional pLDDT; assess FR/CDR confidence and structural context that complements TEMPRO’s sequence‑based Tm estimates.

References
----------

- Alvarez, J. A. E., & Dean, S. N. (2024). TEMPRO: nanobody melting temperature estimation model using protein embeddings. *Scientific Reports*.

