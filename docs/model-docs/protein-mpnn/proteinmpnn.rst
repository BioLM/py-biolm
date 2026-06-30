ProteinMPNN API
===============

ProteinMPNN is a graph-based inverse folding model for fixed-backbone protein sequence design, supporting single- and multi-chain complexes, symmetry tying, and order-agnostic decoding. Given atomic protein backbones (PDB), it generates diverse amino acid sequences in ~1–2 s per 100 residues on GPU, with typical native sequence recovery ~50–55% on native and designed monomers/oligomers. The API supports batched design, temperature-controlled sampling, amino acid biases/omissions, per-residue constraints, symmetry coupling, and log-probability scores for ranking candidates in enzyme, binder, nanoparticle, and stability redesign workflows.

Generate
--------

This endpoint gensats for ProteinMPNN.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="protein-mpnn",
                action="generate",
                params={
                  "temperature": 0.1,
                  "fixed_residues": [],
                  "redesigned_residues": [],
                  "bias_AA": {},
                  "bias_AA_per_residue": {},
                  "omit_AA": "",
                  "omit_AA_per_residue": {},
                  "symmetry_residues": [],
                  "symmetry_weights": [],
                  "homo_oligomer": false,
                  "chains_to_design": [],
                  "parse_these_chains_only": [],
                  "parse_atoms_with_zero_occupancy": false,
                  "number_of_batches": 1,
                  "batch_size": 1,
                  "repack_everything": false,
                  "pack_side_chains": false,
                  "number_of_packs_per_design": 1,
                  "sc_num_samples": 16,
                  "sc_num_denoising_steps": 3,
                  "force_hetatm": false,
                  "pack_with_ligand_context": true,
                  "fasta_seq_separation": ":",
                  "file_ending": "",
                  "zero_indexed": 0,
                  "pdb_path": null,
                  "redesigned_residues_multi": null,
                  "fixed_residues_multi": null,
                  "bias_AA_per_residue_multi": null,
                  "omit_AA_per_residue_multi": null,
                  "save_stats": null,
                  "verbose": true,
                  "ligand_mpnn_use_side_chain_context": null,
                  "ligand_mpnn_use_atom_context": true,
                  "ligand_mpnn_cutoff_for_score": 8.0,
                  "global_transmembrane_label": "soluble",
                  "transmembrane_buried": null,
                  "transmembrane_interface": null
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND                                                                           \n"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/protein-mpnn/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.1,
                "fixed_residues": [],
                "redesigned_residues": [],
                "bias_AA": {},
                "bias_AA_per_residue": {},
                "omit_AA": "",
                "omit_AA_per_residue": {},
                "symmetry_residues": [],
                "symmetry_weights": [],
                "homo_oligomer": false,
                "chains_to_design": [],
                "parse_these_chains_only": [],
                "parse_atoms_with_zero_occupancy": false,
                "number_of_batches": 1,
                "batch_size": 1,
                "repack_everything": false,
                "pack_side_chains": false,
                "number_of_packs_per_design": 1,
                "sc_num_samples": 16,
                "sc_num_denoising_steps": 3,
                "force_hetatm": false,
                "pack_with_ligand_context": true,
                "fasta_seq_separation": ":",
                "file_ending": "",
                "zero_indexed": 0,
                "pdb_path": null,
                "redesigned_residues_multi": null,
                "fixed_residues_multi": null,
                "bias_AA_per_residue_multi": null,
                "omit_AA_per_residue_multi": null,
                "save_stats": null,
                "verbose": true,
                "ligand_mpnn_use_side_chain_context": null,
                "ligand_mpnn_use_atom_context": true,
                "ligand_mpnn_cutoff_for_score": 8.0,
                "global_transmembrane_label": "soluble",
                "transmembrane_buried": null,
                "transmembrane_interface": null
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND                                                                           \n"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/protein-mpnn/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.1,
                    "fixed_residues": [],
                    "redesigned_residues": [],
                    "bias_AA": {},
                    "bias_AA_per_residue": {},
                    "omit_AA": "",
                    "omit_AA_per_residue": {},
                    "symmetry_residues": [],
                    "symmetry_weights": [],
                    "homo_oligomer": false,
                    "chains_to_design": [],
                    "parse_these_chains_only": [],
                    "parse_atoms_with_zero_occupancy": false,
                    "number_of_batches": 1,
                    "batch_size": 1,
                    "repack_everything": false,
                    "pack_side_chains": false,
                    "number_of_packs_per_design": 1,
                    "sc_num_samples": 16,
                    "sc_num_denoising_steps": 3,
                    "force_hetatm": false,
                    "pack_with_ligand_context": true,
                    "fasta_seq_separation": ":",
                    "file_ending": "",
                    "zero_indexed": 0,
                    "pdb_path": null,
                    "redesigned_residues_multi": null,
                    "fixed_residues_multi": null,
                    "bias_AA_per_residue_multi": null,
                    "omit_AA_per_residue_multi": null,
                    "save_stats": null,
                    "verbose": true,
                    "ligand_mpnn_use_side_chain_context": null,
                    "ligand_mpnn_use_atom_context": true,
                    "ligand_mpnn_cutoff_for_score": 8.0,
                    "global_transmembrane_label": "soluble",
                    "transmembrane_buried": null,
                    "transmembrane_interface": null
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND                                                                           \n"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/protein-mpnn/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.1,
                fixed_residues = list(),
                redesigned_residues = list(),
                bias_AA = list(),
                bias_AA_per_residue = list(),
                omit_AA = "",
                omit_AA_per_residue = list(),
                symmetry_residues = list(),
                symmetry_weights = list(),
                homo_oligomer = FALSE,
                chains_to_design = list(),
                parse_these_chains_only = list(),
                parse_atoms_with_zero_occupancy = FALSE,
                number_of_batches = 1,
                batch_size = 1,
                repack_everything = FALSE,
                pack_side_chains = FALSE,
                number_of_packs_per_design = 1,
                sc_num_samples = 16,
                sc_num_denoising_steps = 3,
                force_hetatm = FALSE,
                pack_with_ligand_context = TRUE,
                fasta_seq_separation = ":",
                file_ending = "",
                zero_indexed = 0,
                pdb_path = None,
                redesigned_residues_multi = None,
                fixed_residues_multi = None,
                bias_AA_per_residue_multi = None,
                omit_AA_per_residue_multi = None,
                save_stats = None,
                verbose = TRUE,
                ligand_mpnn_use_side_chain_context = None,
                ligand_mpnn_use_atom_context = TRUE,
                ligand_mpnn_cutoff_for_score = 8.0,
                global_transmembrane_label = "soluble",
                transmembrane_buried = None,
                transmembrane_interface = None
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  
            ATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  
            ATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  
            ATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  
            ATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  
            TER       6      ALA A   1                                                      
            END                                                                           
            "
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/protein-mpnn/generate/

   Generate endpoint for ProteinMPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **temperature** (*float*, default: 0.1) — Sampling temperature

        - **fixed_residues** (*array of strings*, default: []) — Residue identifiers to keep fixed

        - **redesigned_residues** (*array of strings*, default: []) — Residue identifiers to redesign

        - **bias_AA** (*object*, default: {}) — Per–amino acid bias; keys are single-letter amino acid codes, values are floats

        - **bias_AA_per_residue** (*object*, default: {}) — Per-residue amino acid bias; keys are residue identifiers, values are objects mapping single-letter amino acid codes to floats

        - **omit_AA** (*string*, default: "") — Concatenated single-letter amino acid codes to omit globally

        - **omit_AA_per_residue** (*object*, default: {}) — Per-residue amino acids to omit; keys are residue identifiers, values are strings of single-letter amino acid codes

        - **symmetry_residues** (*array of arrays of strings*, default: []) — Groups of residue identifiers constrained together

        - **symmetry_weights** (*array of arrays of floats*, default: []) — Weights corresponding to symmetry_residues groups

        - **homo_oligomer** (*boolean*, default: False) — Homooligomer design flag

        - **chains_to_design** (*array of strings*, default: []) — Chain IDs to design

        - **parse_these_chains_only** (*array of strings*, default: []) — Chain IDs to parse from the PDB

        - **parse_atoms_with_zero_occupancy** (*boolean*, default: False) — Include atoms with zero occupancy

        - **number_of_batches** (*int*, range: 1–1, default: 1) — Number of batches

        - **batch_size** (*int*, range: 1–2, default: 1) — Number of designs per batch

        - **repack_everything** (*boolean*, optional, default: False) — Repack all side chains flag

        - **pack_side_chains** (*boolean*, optional, default: False) — Enable side-chain packing

        - **number_of_packs_per_design** (*int*, optional, range: 1–8, default: 1) — Number of side-chain packing runs per design

        - **sc_num_samples** (*int*, optional, range: 1–64, default: 16) — Number of side-chain samples per design

        - **sc_num_denoising_steps** (*int*, optional, range: 1–10, default: 3) — Number of side-chain denoising steps

        - **force_hetatm** (*boolean*, optional, default: False) — Treat HETATM records as ligands

        - **pack_with_ligand_context** (*boolean*, optional, default: True) — Include ligand context when packing side chains

        - **fasta_seq_separation** (*string*, default: ":") — Separator string for FASTA sequences

        - **file_ending** (*string*, default: "") — Output file suffix

        - **zero_indexed** (*int*, default: 0) — Residue index numbering mode flag

        - **pdb_path** (*null*, fixed) — Unused field

        - **redesigned_residues_multi** (*null*, fixed) — Unused field

        - **fixed_residues_multi** (*null*, fixed) — Unused field

        - **bias_AA_per_residue_multi** (*null*, fixed) — Unused field

        - **omit_AA_per_residue_multi** (*null*, fixed) — Unused field

        - **save_stats** (*null*, fixed) — Unused field

        - **verbose** (*boolean*, default: True) — Verbose logging flag

        - **ligand_mpnn_use_side_chain_context** (*null*, fixed) — Unused field

        - **ligand_mpnn_use_atom_context** (*boolean*, optional, default: True) — Ligand-aware atom context flag

        - **ligand_mpnn_cutoff_for_score** (*float*, optional, default: 8.0) — Distance cutoff for ligand scoring in angstroms

        - **global_transmembrane_label** (*string*, optional, allowed: "membrane", "soluble", default: "soluble") — Global transmembrane label

        - **transmembrane_buried** (*array of strings*, optional, default: None) — Residue identifiers labeled as buried transmembrane

        - **transmembrane_interface** (*array of strings*, optional, default: None) — Residue identifiers labeled as transmembrane interface


      - **items** (*array of objects*, min: 1, max: 1) --- Input structures:

        - **pdb** (*string*, required, min length: 1, max length: max_pdb_str_len) — PDB-formatted structure text

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/protein-mpnn/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.1,
          "fixed_residues": [],
          "redesigned_residues": [],
          "bias_AA": {},
          "bias_AA_per_residue": {},
          "omit_AA": "",
          "omit_AA_per_residue": {},
          "symmetry_residues": [],
          "symmetry_weights": [],
          "homo_oligomer": false,
          "chains_to_design": [],
          "parse_these_chains_only": [],
          "parse_atoms_with_zero_occupancy": false,
          "number_of_batches": 1,
          "batch_size": 1,
          "repack_everything": false,
          "pack_side_chains": false,
          "number_of_packs_per_design": 1,
          "sc_num_samples": 16,
          "sc_num_denoising_steps": 3,
          "force_hetatm": false,
          "pack_with_ligand_context": true,
          "fasta_seq_separation": ":",
          "file_ending": "",
          "zero_indexed": 0,
          "pdb_path": null,
          "redesigned_residues_multi": null,
          "fixed_residues_multi": null,
          "bias_AA_per_residue_multi": null,
          "omit_AA_per_residue_multi": null,
          "save_stats": null,
          "verbose": true,
          "ligand_mpnn_use_side_chain_context": null,
          "ligand_mpnn_use_atom_context": true,
          "ligand_mpnn_cutoff_for_score": 8.0,
          "global_transmembrane_label": "soluble",
          "transmembrane_buried": null,
          "transmembrane_interface": null
        },
        "items": [
          {
            "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND                                                                           \n"
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

        - **sequence** (*string*) — Designed amino acid sequence using 20 standard residues

        - **pdb** (*string*) — PDB-formatted structure corresponding to the design backbone, possibly including remarks and modified occupancies

        - **overall_confidence** (*float*) — Unnormalized model score for the designed sequence, higher values indicate higher confidence

        - **ligand_confidence** (*float*) — Unnormalized model score for ligand or interface compatibility, higher values indicate higher confidence

        - **seq_rec** (*float*, range: 0.0–1.0) — Sequence recovery fraction

        - **log_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-residue log-probabilities over 20 amino acids, one inner array per residue position

        - **sampling_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-residue sampling probabilities over 20 amino acids, normalized per position

        - **pdb_packed** (*object*, optional) — Packed structure outputs for side-chain models keyed by packing context name

          - **<key>** (*string*) — Packed structure in PDB format for the given packing context

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "G",
            "pdb": "REMARK Selection '(backbone) and ...occupancy > 0))'\nATOM      1  N   GLY A   1      11.104  13.207  11.947  1.00  0.30           N  \nATOM      2  CA  GLY A   1      12.560  13.051  11.824  1.00  0.30... (truncated for documentation)",
            "overall_confidence": 0.3011,
            "ligand_confidence": 0.3011,
            "seq_rec": 0.0,
            "log_probs": [
              [
                -2.3079421520233154,
                -4.612211227416992,
                "... (truncated for documentation)"
              ]
            ],
            "sampling_probs": [
              [
                1.5495932530029677e-05,
                1.5237160279918545e-15,
                "... (truncated for documentation)"
              ]
            ]
          }
        ]
      }


Performance
-----------

- Core design accuracy (fixed-backbone inverse folding):
  - On held-out native monomer backbones, ProteinMPNN recovers ~50–52% of wild‑type residues, versus ~33% for Rosetta fixed‑backbone design (≈15–20 absolute percentage‑point improvement) with ~200× lower per‑target compute cost than the original Rosetta CPU benchmarks (1.2 s vs 258.8 s for 100 residues in the paper)
  - Across 690 monomers, 732 homomers, and 98 heteromers, median sequence recovery is ~52%, ~55%, and ~51% respectively, with 90–95% recovery in buried core positions
- Multichain and symmetry-aware performance:
  - Symmetry-aware decoding with logit averaging across equivalent positions improves homomer median recovery by ~2–3 percentage points over unconstrained decoding and yields higher structural consistency under AlphaFold2/ESMFold on assemblies
  - For symmetric oligomers and designed nanoparticles, experimental benchmarks show substantially higher rates of correct oligomerization and assembly than Rosetta-designed sequences on the same backbones (e.g., ~28% vs 0% correct oligomerization for C5/C6 cyclic oligomers; successful rescue of failed tetrahedral nanoparticles with crystal structures within ~1.2 Å Cα RMSD)
- Robustness, downstream foldability, and comparison to other BioLM models:
  - Noise-trained ProteinMPNN variants (up to ~0.3 Å backbone noise) generate sequences that map more reliably to the target backbone under single-sequence AlphaFold2/ESMFold than un-noised models, producing ~2–3× more designs above high lDDT‑Cα thresholds at the cost of a small drop in native sequence recovery
  - Relative to general inverse-folding transformers (e.g., ESM‑IF1, ProstT5 Fold2AA), ProteinMPNN achieves higher native sequence recovery, especially in buried/core residues, and performs better on large or symmetric assemblies because it encodes explicit chain identity and symmetry‑averaged logits
  - Compared to unconditional sequence LMs used with structure-prediction filters (e.g., ESM‑1v, ProGen2, Boltz‑2), ProteinMPNN typically yields 5–10× more sequences predicted (by AlphaFold2/ESMFold) to fold to the exact target backbone per unit of downstream structure‑prediction compute
- Throughput and ranking in BioLM workflows:
  - ProteinMPNN is substantially lighter than structure-prediction models provided by BioLM (AlphaFold2, ESMFold, ProstT5 AA2Fold): redesign of a 100‑residue monomer is typically >100× faster and far less GPU‑intensive than a full AlphaFold2 run, and several‑fold faster than an ESMFold forward pass, enabling tens to hundreds of designs per backbone before expensive validation
  - Average per‑residue log‑probability correlates strongly with native sequence recovery and downstream AlphaFold2/ESMFold confidence, so ProteinMPNN scores can be used to rank and prune large design sets, reducing the number of heavy structure‑prediction calls needed per successful design relative to ranking solely by AlphaFold2/ESMFold metrics

Applications
------------

- Fixed–backbone sequence design to improve stability and expression of industrial proteins and biologics from an experimentally determined or AI-predicted 3D backbone, enabling rapid redesign of poorly expressed or unstable candidates (for example, detergent-compatible proteases or Fc-fusion scaffolds) without re-running expensive structure modeling; less appropriate when no reliable backbone is available or when large backbone rearrangements are required
- Symmetry-aware sequence design for self-assembling protein nanoparticles and multimeric scaffolds, where ProteinMPNN can tie symmetric interface residues across chains via ``symmetry_residues`` to generate consistent sequences for Cn, Dn, or polyhedral assemblies, allowing vaccine and delivery-platform companies to iterate on tetrahedral or cyclic cage designs while keeping the scaffold geometry fixed
- Interface redesign for protein–protein binders and adaptor proteins by mixing fixed and designable regions on multi-chain backbones (for example, target chains in ``fixed_residues``, binder chains in ``chains_to_design``) to improve binding affinity, specificity, or solubility for pre-modeled complexes such as receptor–ligand or cytokine–receptor pairs; most useful when the backbone pose is reasonably correct and less suitable when interface geometry is uncertain
- Backbone-locked library generation for display technologies (phage, yeast, mRNA display), where a single validated backbone (for example, a de novo scaffold around a small-molecule pocket or a stabilized repeat protein) is resampled at higher ``temperature`` to produce diverse, structure-consistent sequences, enabling focused libraries enriched for correctly folding variants rather than random mutagenesis around one wild-type sequence
- In silico stability-focused re-encoding of existing manufacturing workhorses (such as secreted proteases, binding scaffolds, or diagnostic proteins) by redesigning non-functional regions while keeping catalytic or binding residues in ``fixed_residues``, to improve expression yield, thermostability, or aggregation resistance prior to DNA synthesis and cell-line development; this is most effective when functional residues and backbone are well characterized and does not replace downstream developability profiling or cell-based testing

Limitations
-----------

- **Maximum sequence length and PDB size**: ProteinMPNN only designs residues present in the input PDB and is limited to backbones where each chain has at most ``MPNNParams.max_sequence_len = 1024`` residues. Longer chains must be truncated or split before submission. The request body enforces a single PDB per item (``MPNNGenerateRequest.items`` has ``min_items=1``, ``max_items=1``) and each ``MPNNGenerateRequestItem`` has a single PDB string (``pdb``), so very large complexes may need to be decomposed into smaller design problems.
- **Batching and throughput constraints**: Each request can generate a limited number of designs. ``params.number_of_batches`` is constrained to ``1 <= number_of_batches <= MPNNParams.num_batches (1)`` and ``params.batch_size`` to ``1 <= batch_size <= MPNNParams.batch_size (2)``, so at most two sequences are generated per request for a given PDB. This API is not intended for exhaustive large-scale sampling; for very high-throughput sequence exploration, a faster, lower-fidelity generator (e.g. language-model–based) is often better as a first pass, with ProteinMPNN used for focused redesign.
- **Backbone-dependence and structure quality**: ProteinMPNN is a fixed-backbone inverse folding model; it assumes the input PDB backbone is close to the desired folded state and does not optimize or relax coordinates. It is robust to modest noise (trained with Gaussian backbone noise) but will faithfully encode whatever geometry is provided, including artifacts (e.g. strained loops, clashes, unrealistic hallucinations). For highly uncertain backbones or early-stage backbone exploration, upstream backbone-generative or physics-based modeling is recommended before calling this API.
- **What the model optimizes (and what it does not)**: Outputs are optimized for compatibility of amino-acid identities with the given backbone and local environment, not for binding affinity, catalytic activity, immunogenicity, or expression in a specific host. Model types such as ``MPNNModelTypes.SOLUBLE``, ``MPNNModelTypes.GLOBAL_LABEL_MEMBRANE``, ``MPNNModelTypes.PER_RESIDUE_LABEL_MEMBRANE``, ``MPNNModelTypes.LIGAND``, and ``MPNNModelTypes.SIDE_CHAIN`` bias sequences toward solubility, membrane context, ligand context, or side-chain packing, but do not replace task-specific scoring (e.g. docking, binding-energy calculations, epitope prediction). For strong functional constraints, ProteinMPNN should be one stage in a broader design–filter–rank pipeline.
- **Symmetry, residue specification, and interface design limits**: Residue-level controls such as ``fixed_residues``, ``redesigned_residues``, ``bias_AA_per_residue``, ``omit_AA_per_residue``, ``symmetry_residues``, ``transmembrane_buried``, and ``transmembrane_interface`` must reference residues present in the input PDB; invalid chain IDs or out-of-range residue numbers cause validation errors. Symmetry is enforced only where explicitly specified via ``symmetry_residues`` and optional ``symmetry_weights``; the model does not infer symmetry or assembly stoichiometry from the PDB. For complex, multi-interface designs (e.g. large heteromeric assemblies or diffuse PPI networks), additional interface-specific modeling beyond this API is typically required.
- **Sampling diversity and downstream compatibility**: Diversity is controlled primarily via ``params.temperature`` and amino-acid bias/omit options (``bias_AA``, ``bias_AA_per_residue``, ``omit_AA``, ``omit_AA_per_residue``). Higher temperatures increase sequence diversity but usually lower model log-probability (``overall_confidence``) and experimental success rates. The API does not produce embeddings or learned feature vectors; responses contain sequences (``sequence``) plus per-position probabilities/log-probabilities (``sampling_probs``, ``log_probs``). For representation learning (clustering, visualization, downstream ML), combine ProteinMPNN with a separate embedding model (e.g. ESM, ProtT5).

How We Use It
-------------

BioLM uses ProteinMPNN as a core sequence-design engine within iterative protein engineering workflows, where scalable APIs enable teams to move directly from structural hypotheses (experimental or predicted) to diverse sequence panels for enzymes, antibodies, membrane proteins, ligand-binding domains, and multi-component assemblies. ProteinMPNN-generated libraries feed into structure prediction (e.g., AlphaFold/RoseTTAFold), backbone-generation tools, sequence-embedding models, and downstream developability predictors so that binder design, nanoparticle assembly, interface optimization, and stability tuning can be run as end-to-end, lab-in-the-loop campaigns rather than isolated modeling steps.

- ProteinMPNN designs are routinely combined with BioLM predictive models for stability, expression, solubility, immunogenicity, and biophysical properties, enabling automatic ranking and triage of candidates before synthesis.  
- The API-centric design supports multi-round optimization campaigns (e.g., affinity maturation, thermostability tuning, solubility rescue) by integrating with existing data-science pipelines, LIMS, and experimental feedback loops without one-off tooling for each project.

Related
-------

- ``ESM-IF1`` – Structure-conditioned inverse folding model that, like ProteinMPNN, designs sequences for fixed backbones; useful as an alternative design engine or for consensus designs on the same structure.
- ``LigandMPNN`` – Extends ProteinMPNN to protein–ligand complexes; use when you need sequences that both fold to a backbone and preserve a small-molecule binding site.
- ``Soluble ProteinMPNN`` – ProteinMPNN variant trained on soluble proteins; apply to the same backbones when you want designs biased toward better expression and solubility.
- ``ESMFold`` – Sequence-to-structure predictor frequently used after ProteinMPNN to rapidly check whether designed sequences are predicted to adopt the intended backbone.

References
----------

- Dauparas, J., Anishchenko, I., Bennett, N., Bai, H., Ragotte, R. J., Milles, L. F., Wicky, B. I. M., Courbet, A., de Haas, R. J., Bethel, N., Leung, P. J. Y., Huddy, T. F., Pellock, S., Tischer, D., Chan, F., Koepnick, B., Nguyen, H., Kang, A., Sankaran, B., Bera, A. K., King, N. P., & Baker, D. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49–56. https://doi.org/10.1126/science.add2187
