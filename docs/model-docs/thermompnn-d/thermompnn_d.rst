ThermoMPNN-D API
================

ThermoMPNN-D is a structure-based neural network for predicting protein stability changes (ΔΔG, kcal/mol) for single and double point mutations from input PDB structures. It extends ThermoMPNN with a Siamese architecture using ProteinMPNN node, edge, and sequence embeddings to model epistatic interactions between residue pairs. The API provides GPU-accelerated inference on one structure at a time (sequence length ≤1024) in single, additive, or epistatic modes, supporting SSM scans and targeted ranking of stabilizing single and double mutants.

Predict
-------

Single-mutation ddG prediction on a specified chain for a small 3-residue structure. Returns only mutations with ddG <= -0.5 kcal/mol.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="thermompnn-d",
                action="predict",
                params={
                  "mode": "single",
                  "chain": "A",
                  "distance": 5.0,
                  "threshold": -0.5
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   MET A   1      11.104  13.207   5.678  1.00 20.00           N\nATOM      2  CA  MET A   1      12.560  13.350   5.432  1.00 20.00           C\nATOM      3  C   MET A   1      13.014  14.795   5.102  1.00 20.00           C\nATOM      4  N   ALA A   2      14.350  15.050   4.950  1.00 20.00           N\nATOM      5  CA  ALA A   2      14.910  16.420   4.650  1.00 20.00           C\nATOM      6  C   ALA A   2      16.410  16.320   4.350  1.00 20.00           C\nATOM      7  N   GLY A   3      17.010  17.640   4.150  1.00 20.00           N\nATOM      8  CA  GLY A   3      18.450  17.770   3.880  1.00 20.00           C\nATOM      9  C   GLY A   3      18.840  19.210   3.520  1.00 20.00           C\nTER\nEND\n",
                    "mutations": [
                      "M1A",
                      "A2V",
                      "G3L"
                    ]
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/thermompnn-d/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "mode": "single",
                "chain": "A",
                "distance": 5.0,
                "threshold": -0.5
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   MET A   1      11.104  13.207   5.678  1.00 20.00           N\nATOM      2  CA  MET A   1      12.560  13.350   5.432  1.00 20.00           C\nATOM      3  C   MET A   1      13.014  14.795   5.102  1.00 20.00           C\nATOM      4  N   ALA A   2      14.350  15.050   4.950  1.00 20.00           N\nATOM      5  CA  ALA A   2      14.910  16.420   4.650  1.00 20.00           C\nATOM      6  C   ALA A   2      16.410  16.320   4.350  1.00 20.00           C\nATOM      7  N   GLY A   3      17.010  17.640   4.150  1.00 20.00           N\nATOM      8  CA  GLY A   3      18.450  17.770   3.880  1.00 20.00           C\nATOM      9  C   GLY A   3      18.840  19.210   3.520  1.00 20.00           C\nTER\nEND\n",
                  "mutations": [
                    "M1A",
                    "A2V",
                    "G3L"
                  ]
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/thermompnn-d/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "mode": "single",
                    "chain": "A",
                    "distance": 5.0,
                    "threshold": -0.5
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   MET A   1      11.104  13.207   5.678  1.00 20.00           N\nATOM      2  CA  MET A   1      12.560  13.350   5.432  1.00 20.00           C\nATOM      3  C   MET A   1      13.014  14.795   5.102  1.00 20.00           C\nATOM      4  N   ALA A   2      14.350  15.050   4.950  1.00 20.00           N\nATOM      5  CA  ALA A   2      14.910  16.420   4.650  1.00 20.00           C\nATOM      6  C   ALA A   2      16.410  16.320   4.350  1.00 20.00           C\nATOM      7  N   GLY A   3      17.010  17.640   4.150  1.00 20.00           N\nATOM      8  CA  GLY A   3      18.450  17.770   3.880  1.00 20.00           C\nATOM      9  C   GLY A   3      18.840  19.210   3.520  1.00 20.00           C\nTER\nEND\n",
                      "mutations": [
                        "M1A",
                        "A2V",
                        "G3L"
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

            url <- "https://biolm.ai/api/v3/thermompnn-d/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                mode = "single",
                chain = "A",
                distance = 5.0,
                threshold = -0.5
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   MET A   1      11.104  13.207   5.678  1.00 20.00           N
            ATOM      2  CA  MET A   1      12.560  13.350   5.432  1.00 20.00           C
            ATOM      3  C   MET A   1      13.014  14.795   5.102  1.00 20.00           C
            ATOM      4  N   ALA A   2      14.350  15.050   4.950  1.00 20.00           N
            ATOM      5  CA  ALA A   2      14.910  16.420   4.650  1.00 20.00           C
            ATOM      6  C   ALA A   2      16.410  16.320   4.350  1.00 20.00           C
            ATOM      7  N   GLY A   3      17.010  17.640   4.150  1.00 20.00           N
            ATOM      8  CA  GLY A   3      18.450  17.770   3.880  1.00 20.00           C
            ATOM      9  C   GLY A   3      18.840  19.210   3.520  1.00 20.00           C
            TER
            END
            ",
                  mutations = list(
                    "M1A",
                    "A2V",
                    "G3L"
                  )
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/thermompnn-d/predict/

   Predict endpoint for ThermoMPNN-D.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Prediction configuration:

        - **mode** (*string*, enum: ["single", "additive", "epistatic"], default: "single") — Prediction mode used to interpret and score the provided mutations

        - **chain** (*string*, optional) — Chain identifier to use for prediction; if null, the first chain in the PDB is used

        - **distance** (*float*, range: 0.0–∞, default: 5.0) — Minimum CA–CA distance in Angstroms used when filtering double mutations

        - **threshold** (*float*, default: -0.5) — ddG threshold in kcal/mol used to filter returned mutations


      - **items** (*array of objects*, min: 1, max: 1, required) --- Prediction inputs:

        - **pdb** (*string*, min length: 1, max length: max_pdb_str_len, required) — Protein structure in PDB format

        - **mutations** (*array of strings*, optional) — Mutation strings; if null, a site-saturation mutagenesis scan is performed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/thermompnn-d/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "mode": "single",
          "chain": "A",
          "distance": 5.0,
          "threshold": -0.5
        },
        "items": [
          {
            "pdb": "ATOM      1  N   MET A   1      11.104  13.207   5.678  1.00 20.00           N\nATOM      2  CA  MET A   1      12.560  13.350   5.432  1.00 20.00           C\nATOM      3  C   MET A   1      13.014  14.795   5.102  1.00 20.00           C\nATOM      4  N   ALA A   2      14.350  15.050   4.950  1.00 20.00           N\nATOM      5  CA  ALA A   2      14.910  16.420   4.650  1.00 20.00           C\nATOM      6  C   ALA A   2      16.410  16.320   4.350  1.00 20.00           C\nATOM      7  N   GLY A   3      17.010  17.640   4.150  1.00 20.00           N\nATOM      8  CA  GLY A   3      18.450  17.770   3.880  1.00 20.00           C\nATOM      9  C   GLY A   3      18.840  19.210   3.520  1.00 20.00           C\nTER\nEND\n",
            "mutations": [
              "M1A",
              "A2V",
              "G3L"
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

        - **mutation** (*string*) — Mutation identifier in single or double format (e.g., "A100V" or "A100V:B200L")

        - **position** (*int*, optional, 1-indexed) — Residue index for single mutations

        - **position1** (*int*, optional, 1-indexed) — First residue index for double mutations

        - **position2** (*int*, optional, 1-indexed) — Second residue index for double mutations

        - **wildtype** (*string*, optional, length: 1, alphabet: "ACDEFGHIKLMNPQRSTVWYX") — Wildtype amino acid for single mutations

        - **wildtype1** (*string*, optional, length: 1, alphabet: "ACDEFGHIKLMNPQRSTVWYX") — First wildtype amino acid for double mutations

        - **wildtype2** (*string*, optional, length: 1, alphabet: "ACDEFGHIKLMNPQRSTVWYX") — Second wildtype amino acid for double mutations

        - **mutation_aa** (*string*, optional, length: 1, alphabet: "ACDEFGHIKLMNPQRSTVWYX") — Mutant amino acid for single mutations

        - **mutation_aa1** (*string*, optional, length: 1, alphabet: "ACDEFGHIKLMNPQRSTVWYX") — First mutant amino acid for double mutations

        - **mutation_aa2** (*string*, optional, length: 1, alphabet: "ACDEFGHIKLMNPQRSTVWYX") — Second mutant amino acid for double mutations

        - **ddg** (*float*, units: kcal/mol) — Predicted change in free energy (ddG)

        - **distance** (*float*, optional, units: Ångström) — CA–CA distance between mutated residues for double mutations

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          null
        ]
      }


Performance
-----------

- Model family and scope
  - ThermoMPNN-D is a ThermoMPNN-derived, ProteinMPNN-based graph neural network with a Siamese aggregation head specialized for order-invariant double point mutation ΔΔG prediction (epistasis-aware)
  - The deployed API exposes three inference regimes (single, additive, epistatic); epistatic mode uses the full ThermoMPNN-D head, whereas additive mode reuses single-mutant ThermoMPNN-style predictions summed over the two sites
- Comparative predictive accuracy on double mutants
  - On Megascale-D, additive ThermoMPNN (sum of two single-mutant ΔΔG predictions) achieves Spearman correlation coefficients (SCC) ≈ 0.59 for double mutants, while ThermoMPNN-D ensembles reach SCC ≈ 0.54–0.55; this is consistent with most epistasis-aware heads not improving global rank metrics over strong single-mutant baselines
  - On PTMUL-D, ThermoMPNN-D ensembles reach SCC ≈ 0.59, slightly ahead of many physics-based (Rosetta, FoldX) and sequence-only baselines, and competitive with or better than the additive ThermoMPNN baseline depending on training split
  - Across deep mutational scans where fitness is only a proxy for stability, all methods yield lower correlations (SCC ≈ 0.2–0.4); ThermoMPNN-D remains competitive but is sometimes slightly behind specialized AlphaFold-based pipelines such as Mutate Everything
- Epistasis, distance dependence, and stabilizing double mutants
  - For bulk double-mutant ranking, ThermoMPNN-D behaves close to an additive regime, similar to other state-of-the-art models; additive single-mutant sums often match or slightly outperform explicit epistatic heads on global SCC
  - ThermoMPNN-D’s epistasis-aware head provides its clearest benefit on stabilizing double mutants: for variants with ΔΔG ≤ -0.5 kcal/mol, it improves Matthews Correlation Coefficient over additive ThermoMPNN (≈ 0.19 vs 0.17 on Megascale-D and ≈ 0.38 vs 0.28 on PTMUL-D), enriching genuinely stabilizing pairs
  - The API’s ``distance`` parameter mirrors the model’s inductive bias that strong epistatic couplings are local in 3D space; for distal pairs, predictions tend toward additive behavior, where additive ThermoMPNN generally provides similar ranking at slightly lower computational cost
- Position within BioLM’s stability model suite and calibration
  - Relative to ThermoMPNN (single-mutant) and ES M2StabP (sequence/structure-based stability), ThermoMPNN-D is the preferred BioLM model when explicitly modeling double mutants and discovering synergistic stabilizing pairs that are spatially proximal
  - For large-scale single-mutant scans or when epistasis is not the primary concern, ThermoMPNN in single or additive mode is typically more efficient and slightly more accurate on global correlation metrics
  - ThermoMPNN-D outputs ΔΔG in kcal/mol calibrated on Megascale-D and PTMUL-D; rank-based metrics (e.g., Spearman ≈ 0.5–0.6 on curated double-mutant stability datasets) are more reliable than absolute magnitudes, and very strong epistatic coupling energies may be systematically underestimated across models

Applications
------------

- Prioritizing double-mutant stability scans in industrial protein engineering campaigns (for example, thermostabilizing a manufacturing biocatalyst), by ranking candidate double point mutations that ThermoMPNN-D predicts as more stabilizing than a simple additive single-mutant baseline, allowing teams to focus wet-lab screening on variants with potentially favorable epistatic couplings rather than exhaustively testing all combinations; note that in many cases overall double-mutant accuracy is similar to additive models, so results should be treated as a prioritization signal rather than a hard filter
- Designing stabilizing mutation pairs for developability optimization of recombinant protein therapeutics (such as Fc-fusion proteins or cytokines), by proposing double substitutions that are predicted to improve folding stability while accounting for non-additive interactions learned from structure-based graph embeddings, helping reduce aggregation and formulation risk when moving from research-grade to GMP manufacturing; users should be aware that performance on strongly destabilizing variants is closer to additive baselines, so experimental confirmation remains essential
- Engineering robust double-mutant backbones for high-temperature or harsh-process biocatalysts in industrial bioprocessing, by using ThermoMPNN-D in epistatic mode to score user-specified mutation pairs or site-saturation scans on a given structure, identifying double mutants that may jointly offset destabilizing substitutions added for activity or specificity improvements, supporting iterative design cycles where activity-enhancing changes are “paid for” with stabilizing pairs rather than assuming independent single-mutation effects
- Interpreting epistatic liability in variant panels for diagnostic or safety assessment of protein-based products (for example, characterizing non-additive stability effects in concurrent variants observed in cell-line evolution or viral contaminant proteins in biologics manufacturing), by comparing ThermoMPNN-D epistatic double-mutant ddG predictions to the API’s additive mode on the same residue pairs to flag double mutants whose stability is unexpectedly high or low and may alter clearance, persistence, or manufacturability
- Supporting generative protein design pipelines that propose multiple concurrent mutations, by using ThermoMPNN-D as an epistasis-aware stability filter after generative sequence models or combinatorial design, enabling ranking and down-selection of designed variants where pairwise residue couplings are predicted to preserve or improve stability on a supplied structure, while relying on the API’s single or additive modes when only independent single-site substitutions are being considered

Limitations
-----------

- **Maximum sequence length and batch size**: ThermoMPNN-D accepts exactly one structure per request (``items`` must have length 1) and supports chains up to ``max_sequence_len`` = 1024 residues. There is no server-side batching beyond ``batch_size`` = 1, so any parallelism or large mutation sets must be split across multiple requests. Multi-chain complexes or very long proteins may need to be truncated or scored per-chain via ``chain``, which can miss inter-chain contacts.
- **Mode-specific mutation handling**: The ``mode`` parameter controls how the ``mutations`` list is interpreted. ``"single"`` expects single-site strings of the form ``"WT{position}MUT"`` (e.g. ``"A100V"``). ``"additive"`` and ``"epistatic"`` expect exactly two sites per string in the form ``"WT1{pos1}MUT1:WT2{pos2}MUT2"`` (e.g. ``"A100V:B200L"``). Mixed single/double formats, more than two sites per mutation string, or higher-order variants (>2 mutations) are rejected.
- **Local, structure-based predictions with distance filtering**: All predictions use the supplied ``pdb`` (validated and truncated to ``max_pdb_str_len``) and are sensitive to backbone quality, residue numbering, and the selected ``chain``. The ``distance`` parameter (Cα–Cα cutoff in Å) is applied only for double mutations in ``"additive"`` or ``"epistatic"`` modes: double mutations with separation greater than ``distance`` are filtered out and will not appear in the response. Misaligned indices, missing residues, or low-quality structures can lead to unreliable ``ddg`` estimates.
- **Limited epistasis capture and scope of ``ddg``**: Benchmarks from the ThermoMPNN-D paper show that overall double-mutant accuracy is often similar to a simple ``"additive"`` sum of single-mutant predictions; the main gain is improved detection of rare stabilizing double mutants. The ``ddg`` output is a predicted change in thermodynamic stability (ΔΔG in kcal/mol) only; it is not a validated predictor of activity, expression level, binding affinity, or organism-level fitness, and correlations to deep mutational scan fitness measurements are modest.
- **Result filtering and coverage**: By default, only mutations with ``ddg`` ≤ ``threshold`` (default ``threshold`` = -0.5 kcal/mol) are returned. This is optimized for stabilizing or near-neutral variants; to analyze destabilizing mutations or obtain full coverage, users must raise ``threshold`` (e.g. to 100). For double mutations, the combination of ``distance`` and ``threshold`` can exclude many candidate variants from the response.
- **Use cases where alternative models are preferable**: ThermoMPNN-D does not generate sequences (it only scores provided ``mutations`` or performs SSM scans) and is not optimized for very large libraries (e.g. millions of designs) due to ``batch_size`` = 1 and structure dependence. Sequence-only stability proxies, fast per-residue scoring, or generative design models are typically better suited for early-stage, high-throughput workflows, with ThermoMPNN-D reserved for structure-aware triage of smaller candidate sets focused on stability.

How We Use It
-------------

ThermoMPNN-D enables teams to prioritize stabilizing double mutations by explicitly modeling epistatic interactions instead of relying only on additive DDG estimates from single-mutant predictors. In practice, it integrates as a focused double-mutation gate in protein engineering workflows: organizations first use single-mutation stability models (e.g., ThermoMPNN in ``single`` or ``additive`` mode), protein language models, or generative design tools to propose variants, then apply ThermoMPNN-D in ``epistatic`` mode to rank double mutants most likely to improve stability for downstream build–test–learn cycles.

- Acts as a refinement stage after generative or combinatorial library design, filtering to nearby double mutants (via the ``distance`` parameter) that are most promising for stabilization.
- Complements structure-based tools such as ProteinMPNN and AlphaFold-derived designs by providing an epistasis-aware stability score that helps select smaller, higher-value experimental panels.

Related
-------

- ``ThermoMPNN`` – Single-mutation DDG predictor using the same ProteinMPNN backbone and feature space; provides the additive double-mutant baseline to compare against ``ThermoMPNN-D`` epistatic predictions.
- ``ESM2StabP`` – Sequence-only protein stability predictor that complements structure-based ``ThermoMPNN-D`` when no reliable 3D model is available, or to cross-check DDG trends.
- ``ProteinMPNN`` – Structure-conditioned sequence-design GNN that supplies the residue and edge embeddings used in ``ThermoMPNN-D``; can propose stabilized variants that are then re-scored for epistasis-aware evaluation.
- ``ESM-IF1`` – Inverse folding model for sampling alternative local sequences around mutation sites, whose single and double mutants can be quantitatively ranked with ``ThermoMPNN-D`` for stability impact.

References
----------

- Dieckhaus, H., & Kuhlman, B. (2025). Protein stability models fail to capture epistatic interactions of double point mutations. *Manuscript in preparation.*
