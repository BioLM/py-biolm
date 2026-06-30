SPURS API
=========

SPURS predicts protein thermostability changes (ΔΔG, kcal/mol) for all single amino acid substitutions in a protein in a single forward pass. It rewires an ESM protein language model with structure priors from ProteinMPNN via Adapter attention, conditioning on a wild-type sequence and an experimental or predicted single-chain structure (PDB or mmCIF, length ≤1024). The API returns either an L×20 ΔΔG matrix or aggregated ΔΔG values for user-specified single or multi-residue mutations. GPU-accelerated inference supports stability-aware design, mutation ranking, variant triage, and low-N fitness modeling workflows.

Predict
-------

Predict ΔΔG for specific mutations  or return a full single-mutation ΔΔG matrix when mutations are omitted (items[2]). Each item must supply either PDB or mmCIF structure content and a single-letter chain_id.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="spurs",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "M",
                    "pdb": "ATOM      1  N   MET A   1      11.104  13.207   2.022  1.00 20.00           N\nEND",
                    "chain_id": "A",
                    "mutations": [
                      "M1I"
                    ]
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/spurs/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "M",
                  "pdb": "ATOM      1  N   MET A   1      11.104  13.207   2.022  1.00 20.00           N\nEND",
                  "chain_id": "A",
                  "mutations": [
                    "M1I"
                  ]
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/spurs/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "M",
                      "pdb": "ATOM      1  N   MET A   1      11.104  13.207   2.022  1.00 20.00           N\nEND",
                      "chain_id": "A",
                      "mutations": [
                        "M1I"
                      ]
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/spurs/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "M",
                  pdb = "ATOM      1  N   MET A   1      11.104  13.207   2.022  1.00 20.00           N
            END",
                  chain_id = "A",
                  mutations = list(
                    "M1I"
                  )
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/spurs/predict/

   Predict endpoint for SPURS.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Model configuration parameters:

        - **batch_size** (*integer*, default: 4) — Maximum number of items allowed in a single request

        - **max_sequence_len** (*integer*, default: 1024) — Maximum allowed length of the input protein sequence per item


      - **items** (*array of objects*, min length: 1, max length: 4) --- List of request entries:

        - **sequence** (*string*, required, min length: 1, max length: 1024) — Protein sequence using 20 canonical amino acid codes

        - **pdb** (*string*, optional, min length: 1, max length: 2000000) — PDB format structure content; required if **cif** is not provided

        - **cif** (*string*, optional, min length: 1, max length: 2000000) — mmCIF format structure content; required if **pdb** is not provided

        - **chain_id** (*string*, default: "A", min length: 1, max length: 1) — Single-letter chain identifier within the provided structure

        - **mutations** (*array of strings*, optional, min length: 1) — List of single mutations in the format "<WT><position><MT>" with 1-indexed positions using canonical amino acid codes; omit this field to request a full single-mutant saturation matrix

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/spurs/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "M",
            "pdb": "ATOM      1  N   MET A   1      11.104  13.207   2.022  1.00 20.00           N\nEND",
            "chain_id": "A",
            "mutations": [
              "M1I"
            ]
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

        - **mutations** (*array of strings*, optional) — Mutations evaluated for this result item, formatted "<WT><position><MT>" with 1-indexed positions; null when a full saturation mutagenesis matrix is returned.


        - **ddG** (*float*, optional) — Predicted ΔΔG in kcal/mol for the requested mutation set.


        - **ddG_contributions** (*object*, optional) — Map from mutation string to predicted ΔΔG in kcal/mol for each mutation in a multi-mutation request.


        - **ddG_matrix** (*object*, optional) — Saturation mutagenesis ΔΔG matrix when no mutation list is requested:

          - **values** (*array of arrays of floats*, shape: (sequence_length, 20)) — Predicted ΔΔG values in kcal/mol for each sequence position (rows) and amino acid (columns).


          - **residue_axis** (*array of strings*, size: sequence_length) — Wild-type amino acid label for each matrix row.


          - **amino_acid_axis** (*array of strings*, size: 20) — Amino acid labels for matrix columns in canonical 20-letter alphabet order.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "mutations": [
              "M1I"
            ],
            "ddG": -0.15379896759986877
          }
        ]
      }


Performance
-----------

- Inference complexity and throughput
  - Single forward pass produces an L×20 ΔΔG grid for all single substitutions (O(1) in the number of mutants), unlike per‑mutant models that require O(L×20) passes
  - Structure features are computed once per structure payload and reused across all substitutions, avoiding repeated graph construction and structure encoding
  - Implementation is optimized for high-throughput saturation mutagenesis and multi-protein batches via GPU micro-batching and fused attention/graph kernels in mixed precision

- Hardware and numerical precision
  - Optimized for NVIDIA data-center GPUs (A100 40/80 GB, H100 80 GB) and validated on A40/A10 within the 1–1024 residue sequence limit defined in the SPURS schema
  - Uses FP16/BF16 mixed precision with FP32-safe layer norms and reductions to maintain numerical stability over dense L×20 mutation grids
  - Joint ProteinMPNN backbone graph preprocessing and ESM2-650M sequence encoding are memory-optimized to maintain high SM occupancy on A100/H100

- Predictive accuracy vs related BioLM stability models
  - On the Megascale test split, SPURS improves Spearman correlation by ≈0.05–0.07 over structure-only ThermoMPNN-style baselines and over fine‑tuned ESM or ProteinMPNN used alone
  - Across FireProt(HF), Ssym, S8754, S2648, S4346, S571, and related benchmarks, SPURS matches or exceeds the best published structure-only or sequence-only models on most datasets and generalizes to ΔT\ :sub:`m` prediction despite being trained on ΔΔG
  - On the human Domainome dataset, SPURS attains correlation ≈0.54 vs ≈0.49 for ThermoMPNN, improving variant ranking across hundreds of human domains

- Comparative behavior vs other BioLM stability predictors
  - Relative to Stability Oracle–style structure-based graph transformers, SPURS requires one pass per protein to obtain the full L×20 saturation profile, yielding ≈10\ :sup:`4` fewer passes for a 500-residue domain at comparable structure awareness
  - Compared with sequence-only models such as TemBERTure or ESM2 zero-shot scores, SPURS is more computationally intensive per protein but is more appropriate for saturation mutagenesis workloads and reduces false-positive stabilizers in buried/core positions by conditioning on structure
  - For coarse triage of very large variant libraries without structures, TemBERTure/ESM2 remain faster; SPURS is best suited as a second-stage, structure-informed reranker when higher stability-ranking fidelity is required

Applications
------------

- High-throughput stabilizing-mutation discovery for industrial enzymes by running single-pass ΔΔG scans of all 20 substitutions per position on user-supplied structures (PDB/mmCIF via the API), enabling rapid shortlisting for thermostabilization, solvent tolerance, and shelf-life improvements while reducing wet-lab screening; for example, scanning lipases, esterases, or transaminases to pick top N stabilizing edits before building combinatorial libraries; note that SPURS is trained on single substitutions and predicts thermodynamic stability (ΔΔG) rather than catalytic turnover, solvent effects, or epistasis, so multi-mutation designs, formulation effects, and process-specific conditions require experimental validation, and predictions on low-confidence structures or chains with unresolved/disordered regions should be treated cautiously
- Antibody and nanobody developability optimization by prioritizing framework or periphery CDR substitutions predicted to reduce destabilization while preserving binding paratopes, supporting higher expression, better thermal stability, and improved manufacturability for IgG, scFv, and VHH formats when structures are supplied via PDB/mmCIF or modeled externally; for example, ranking stabilizing edits in the VL/VH frameworks after humanization to recover Tm and reduce aggregation risk; limitations include single-chain inference for the specified chain_id (quaternary assembly, glycosylation, Fc engineering, and aggregation pathways are not explicitly modeled) and no direct prediction of affinity changes
- Recombinant construct rescue for difficult-to-express proteins by proposing stabilizing point mutations that improve folding and soluble yield in hosts like E. coli or CHO, reducing the need for large directed-evolution libraries; for example, applying SPURS to experimentally determined or externally predicted structures for cytokines or soluble domains of membrane proteins to select stabilizing edits prior to expression screens; not optimal for targets where secretion, trafficking, codon usage, or signal peptides are the primary bottlenecks rather than folding stability of the structured domain provided as PDB/mmCIF
- Functional hotspot triage in engineering campaigns by combining SPURS ΔΔG predictions (from the API’s single-mutation matrices) with external protein language model fitness scores to compute residual-based “function scores” that flag residues where fitness loss is larger than stability effects alone, helping teams avoid mutating active or binding sites while concentrating stabilizing edits in more tolerant regions; for example, mapping SH3 peptide-binding pockets or metal-binding residues in dehydrogenases before saturation mutagenesis; this analysis is prioritization guidance rather than definitive annotation and depends on reasonable evolutionary signal in the sequence family and a reliable structure for the chain being evaluated
- Low-N fitness modeling augmentation by feeding SPURS-predicted ΔΔG as an additional feature alongside one-hot sequence and protein language model likelihoods in supervised models trained on 48–240 labeled variants, improving ranking accuracy for assays that are partially stability-linked (expression, abundance, organismal fitness); for example, boosting first-round predictive models in enzyme activity screens or membrane transporter expression assays; gains may be limited when the target property is weakly coupled to folding stability (e.g., subtle specificity or allosteric changes), and feature utility should be validated per assay and per structural model used as SPURS input

Limitations
-----------

- **Maximum sequence length** and **Batch Size**: Each ``sequence`` must be composed of 20-letter canonical amino acids and be ``<= 1024`` residues long (``SpursParams.max_sequence_len``). The ``items`` array in ``SpursPredictRequest`` must contain between ``1`` and ``4`` entries (``SpursParams.batch_size = 4``). Each item uses a single-letter ``chain_id`` and is treated as a single-chain prediction.
- Required structure input and single-chain context: Each item must include either ``pdb`` or ``cif`` structure content; the API does not perform any structure prediction. Only the specified ``chain_id`` is modeled; other chains, ligands, membranes, solvent, and cofactors are ignored, so interface-, complex-, or ligand-driven stability effects may be under-estimated.
- Mutation inputs and encoding: ``mutations`` (when provided) must be a non-empty list of single–amino-acid substitutions in the format ``<WT><position><MT>`` (e.g., ``M3L``) with 1-indexed positions. WT/MT must both be from the 20-letter canonical alphabet and the wild-type letter must match the ``sequence`` at that position. Non-canonical residues, insertions, deletions, ambiguous codes, and other mutation grammars are not supported.
- Output conventions and additivity: ``ddG`` values are returned in kcal/mol with sign convention ΔΔG = ΔG(mutant) − ΔG(wild type); negative values correspond to predicted stabilizing substitutions. When ``mutations`` is omitted, ``ddG_matrix.values`` has shape ``(sequence_length, 20)``, with rows ordered by ``residue_axis`` (wild-type positions) and columns by ``amino_acid_axis`` (canonical alphabet). For multiple requested mutations, ``ddG`` is computed as the sum of per-mutation ``ddG_contributions``; higher-order epistasis is not explicitly modeled.
- Scientific/algorithmic limitations: Predictions are conditioned on the provided static structure and can degrade with low-confidence, missing, or misfolded regions, or when large conformational changes, cofactors, membranes, or complexes are important for stability. SPURS is trained on single substitutions and thermostability ΔΔG; it is not calibrated for absolute ΔG, experimental ΔT\ :sub:`m` under specific assay conditions, pH/temperature dependence, post-translational modifications, or non-canonical amino acids.
- When SPURS is not optimal: Tasks dominated by binding or activity rather than intrinsic stability, design of multi-chain interfaces or complexes, or ultra-high-throughput triage of large variant libraries without reliable structures may be better served by sequence-only or specialized structural models. In typical workflows, SPURS is most appropriate once a reasonable structure for the target chain is available and when ranking point mutations by predicted stability is the main objective.

How We Use It
-------------

SPURS enables rapid, structure-aware stability scoring that we apply across discovery pipelines to de-risk designs before synthesis. Via the API, we generate full single-mutation ∆∆G matrices in one pass per protein, combine these with protein language model fitness scores (e.g., ESM1v) to prioritize functional regions, and feed SPURS-derived stability priors into low-N fitness models and multi-objective ranking alongside 3D energy and developability metrics. When experimental structures are unavailable, teams submit AlphaFold models through the same interface, standardizing triage and enabling scalable, API-driven loops between in silico design and lab validation.

- Stability-aware design: integrates with generative proposals (e.g., ESM-based design, ProteinMPNN, diffusion models) to enrich stabilizing variants, shrink site-saturation/combinatorial libraries, and set constraints for multi-mutation exploration based on single-mutation ∆∆G.
- Function and developability mapping: combines SPURS ∆∆G landscapes with pLM-based fitness to highlight activity interfaces while filtering variants by charge, hydrophobicity, and liability heuristics for manufacturability and expression.

Related
-------

- ``AlphaFold2`` – Provides 3D structures when no experimental model is available; use its backbone coordinates as SPURS input and optionally mask low-confidence regions (e.g., by pLDDT) before scoring stability
- ``ESM-1v`` – Zero-shot variant effect model; fit a stability–fitness curve using SPURS ∆∆G and ESM-1v scores, then use residuals to highlight functionally important or constrained sites
- ``ESM-IF1`` – Structure-conditioned sequence model; compare or combine its sequence likelihoods with SPURS ∆∆G to cross-check interface- or core-sensitive mutation effects
- ``TemBERTure Regression`` – Sequence-only melting temperature regressor; pair its ∆Tm predictions with SPURS ∆∆G to jointly rank stabilizing or thermostable designs

References
----------

- Li, Z., & Luo, Y. (2025). Rewiring protein sequence and structure generative models to enhance protein stability prediction. *bioRxiv*. https://doi.org/10.1101/2025.02.13.638154
