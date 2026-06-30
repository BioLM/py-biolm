TEMPRO 3B API
=============

TEMPRO 3B predicts nanobody (VHH) melting temperature (Tm, °C) directly from amino acid sequence using ESM-2 t36_3B UR50D per-residue embeddings (2560-d) and a deep neural network regressor trained on 567 nanobodies. The service performs single or batched GPU-accelerated inference for 100–160 AA sequences, returning one Tm per input. On held-out nanobody data, the ESM-3B model achieved ~4.2 °C MAE, and external validation reported R² ≈ 0.58 versus experimental Tm. Typical uses include variant ranking, library triage, thermostability screening, and design workflows.

Predict
-------

Predict melting temperature for input nanobody sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="tempro-3b",
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

            curl -X POST https://biolm.ai/api/v3/tempro-3b/predict/ \
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

            url = "https://biolm.ai/api/v3/tempro-3b/predict/"
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

            url <- "https://biolm.ai/api/v3/tempro-3b/predict/"
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

.. http:post:: /api/v3/tempro-3b/predict/

   Predict endpoint for TEMPRO 3B.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 8) — Maximum number of sequences accepted per request

        - **min_sequence_len** (*int*, default: 100) — Minimum allowed sequence length in amino acids

        - **max_sequence_len** (*int*, default: 160) — Maximum allowed sequence length in amino acids


      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **sequence** (*string*, length: 100–160, required) — Protein sequence using supported amino acid codes (100–160 residues)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/tempro-3b/predict/ HTTP/1.1
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

        - **tm** (*float*) — Predicted melting temperature in Celsius

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "tm": 76.50244140625
          },
          {
            "tm": 75.36508178710938
          }
        ]
      }


Performance
-----------

- Model variant and pipeline: TEMPRO 3B uses ESM-2 3B (UR50) sequence embeddings followed by a compact feed-forward regressor; embeddings are computed on-the-fly and streamed directly into the DNN (no 3D structure prediction), enabling low-latency, GPU-accelerated execution
- Predictive accuracy: on the internal nanobody hold-out set, the 3B variant achieves MAE ≈ 4.2 °C and RMSE ≈ 5.8–6.0 °C; on the external INDI nanobody set, it attains R² ≈ 0.58 (vs ≈ 0.67 for TEMPRO 15B and ≈ 0.25 for TEMPRO 650M), indicating strong correlation to experimental Tm
- Hardware, latency, and scaling: deployments are optimized for NVIDIA A100 80 GB and L4/L40S GPUs using mixed-precision (FP16) with fused attention kernels; dynamic batching and graph capture keep p95 latency low and throughput stable under high concurrency, with per-GPU throughput scaling linearly across additional GPUs
- Comparative efficiency within TEMPRO: TEMPRO 3B is within ~0.2 °C MAE of TEMPRO 15B on internal benchmarks (≈ 4.24 °C vs 4.03 °C) while being roughly 4–6× faster and substantially cheaper to run; compared to TEMPRO 650M, it offers modestly lower error on internal data and markedly better external generalization (R² +0.33 on INDI), providing the best accuracy-per-millisecond trade-off in the TEMPRO family

Applications
------------

- High-throughput triage of nanobody libraries from display or AI generation: run TEMPRO 3B on thousands to millions of VHH sequences (100–160 amino acids) to prioritize candidates above a thermostability gate (for example, Tm ≥ 65–70 °C) before expression and DSC/CD. This reduces wet-lab screening cost and cycle time by focusing on molecules more likely to survive handling, transport, and downstream processing. Best used for relative ranking; absolute Tm can deviate by several degrees and depends on assay buffer and conditions.
- Variant ranking for thermostabilization campaigns: enumerate targeted point mutations or small combinatorial sets (for example, CDR1 scans or FR2–FR3 cysteine-introducing designs) and score each variant with TEMPRO 3B to pick a compact, high-confidence subset for synthesis. This enables rapid design–build–test loops that improve Tm while maintaining binding through parallel affinity assays. TEMPRO 3B does not predict affinity, expression, or solubility and is not a generative model; use it to rank your designed variants by predicted Tm.
- Developability gating for therapeutic and diagnostic nanobodies: apply predicted Tm as an early developability filter to reduce late-stage attrition—select leads more likely to tolerate room-temperature handling, shipping, and storage for systemic therapeutics, imaging agents, or point-of-care diagnostics. This helps formulation teams focus on inherently more stable scaffolds. Not a substitute for forced-degradation, aggregation, or viscosity studies; Tm is one dimension of developability and does not capture immunogenicity risk.
- Resource prioritization for structural modeling and biophysical assays: use TEMPRO 3B scores to decide which sequences warrant structure prediction, cloning, and biophysical characterization, enabling “fewer, better” experiments. Teams can schedule DSC only on top-ranked sequence sets and defer lower-confidence designs. Predictions are for isolated VHH domains; stability may shift after adding tags, linkers, or conjugations and should be rechecked post-modification.
- Stability-aware assembly of multi-format VHH constructs: select monomer VHH building blocks with higher predicted Tm for biparatopic binders, multivalents, CAR binders, or Fc fusions to reduce the risk of low overall thermal margins. This increases the chance that the assembled construct remains stable under manufacturing and storage constraints. TEMPRO 3B estimates Tm of single-domain VHHs from sequence; fusion context, inter-domain contacts, and formulation effects are not modeled and require empirical verification.

Limitations
-----------

- **API I/O constraints**: Input ``items`` is a list of objects each with a single ``sequence`` (string). Allowed ``items`` length is 1–8 (``batch_size`` = 8). Each ``sequence`` must be 100–160 amino acids (``min_sequence_len`` = 100, ``max_sequence_len`` = 160) and use only the 20 standard amino acids; any non-canonical or ambiguous residues are rejected by ``validate_aa_extended``. Output ``results`` is a list aligned 1:1 with ``items``; each entry contains a single ``tm`` (float, °C). No confidence intervals, mutation scanning, per-residue attributions, or intermediate features are returned
- **Domain specificity**: The model is trained specifically on camelid VHH/sdAb nanobody sequences. Reliability degrades for other antibody formats (scFv, Fab, full IgG), shark VNAR, multimeric fusions, or sequences including long signal peptides, tags, or linkers. Such sequences may satisfy the ``sequence`` length constraint but remain out-of-distribution and less accurate
- **Assay/context dependence**: Experimental melting temperatures depend on buffer composition, pH, concentration, scan rate, and assay modality. The predicted ``tm`` is context-agnostic and best used for relative ranking of candidates measured under comparable conditions, not as an absolute specification, release criterion, or cross-assay comparison
- **Feature simplification vs. paper**: The published TEMPRO framework combines ESM embeddings with additional inputs (NetSurfP3, AlphaFold2 pLDDT per region, physicochemical features) and reports mean absolute error of ~4 °C on held-out data. This API exposes a simplified sequence-to-``tm`` predictor based on protein language model embeddings and does not accept or return structural models, regional pLDDT, or other per-feature contributions; structure- or formulation-driven effects (e.g., specific disulfides, CDR conformations) may therefore be under-modeled
- **Out-of-distribution sequences**: Highly synthetic or adversarial inputs (unusual cysteine/disulfide patterns, extreme net charge or hydrophobicity, long low-complexity or homopolymer segments) that still pass length and alphabet checks can yield degraded accuracy, especially when extrapolating beyond the nanobody Tm range represented in the training data (~46–88 °C in published validations). Non-natural residues are rejected outright by the ``sequence`` validator
- **Throughput and decision fit**: Each call is limited to at most 8 sequences (``items`` ≤ ``batch_size``); large libraries require client-side batching and aggregation. The model returns a single scalar ``tm`` per sequence and does not directly estimate ΔΔTm between variants. For fine-grained mutation prioritization, developability risk assessment, or final go/no-go decisions where small Tm changes matter, this API should be combined with orthogonal in silico models and experimental measurements

How We Use It
-------------

BioLM uses TEMPRO 3B as a fast, sequence-only nanobody Tm estimator to gate design candidates, reduce synthesis load, and shorten optimization cycles. In antibody and nanobody campaigns, TEMPRO 3B supports early-stage triage and ranking across large variant libraries, then its scores are combined with masked language model proposals, AlphaFold/ESMFold structure QC, NetSurfP3-derived features, MAESTRO ΔΔG, and developability metrics (charge, hydrophobicity, liabilities) to enforce thermostability thresholds aligned to storage, formulation, and CMC requirements. Standardized, scalable APIs enable batch scoring of up to 8 nanobody-length sequences (100–160 amino acids), program-specific Tm decision rules, and direct push of Tm annotations into LIMS/ELN, accelerating lab-in-the-loop campaigns from initial scaffolds through affinity maturation and humanization.

- Primary applications: stability-aware ranking of CDR variants and humanization candidates; guided mutational scanning to raise Tm while preserving binding; portfolio-level thermostability risk screening prior to synthesis.
- Integration pattern: TEMPRO 3B for high-throughput, sequence-only triage, followed by higher-capacity ensembles (e.g., ESM-15B-based models) at down-selection, with outputs feeding directly into design-of-experiments plans and procurement queues.

Related
-------

- ``TEMPRO 650M`` – Lightweight TEMPRO variant using smaller ESM-2 embeddings; useful for rapid, large-scale Tm screening and prioritizing sequences before running TEMPRO 3B.
- ``NanoBodyBuilder2`` – Builds VHH structures and numbering from sequences; use after Tm prediction to inspect FR/CDR geometry, disulfides, and design stabilizing mutations.
- ``Peptides`` – Computes sequence-level physicochemical descriptors (e.g., aliphatic index, GRAVY, instability) that underlie TEMPRO’s feature set; aids in interpreting Tm predictions and basic developability filtering.

References
----------

- Alvarez, J. A. E., & Dean, S. N. (2024). OPEN TEMPRO: nanobody melting temperature estimation model using protein embeddings. *Scientific Reports*.
