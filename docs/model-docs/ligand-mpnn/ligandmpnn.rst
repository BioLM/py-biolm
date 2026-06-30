LigandMPNN API
==============

LigandMPNN is a GPU-accelerated, structure-conditioned protein sequence design model that incorporates explicit atomic context from small molecules, nucleic acids, metals, and optional sidechain coordinates. Given fixed backbone and ligand coordinates in a PDB input, the API generates amino acid sequences in an autoregressive manner, with per-residue probabilities, overall and ligand-contact confidence scores, and optional sidechain repacking. Typical uses include active-site remodeling, small-molecule binder and sensor optimization, and redesign of complexes around bound ligands.

Generate
--------

This endpoint gensats for LigandMPNN.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="ligand-mpnn",
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
                  "ligand_mpnn_cutoff_for_score": 8.0
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

            curl -X POST https://biolm.ai/api/v3/ligand-mpnn/generate/ \
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
                "ligand_mpnn_cutoff_for_score": 8.0
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

            url = "https://biolm.ai/api/v3/ligand-mpnn/generate/"
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
                    "ligand_mpnn_cutoff_for_score": 8.0
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

            url <- "https://biolm.ai/api/v3/ligand-mpnn/generate/"
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
                ligand_mpnn_cutoff_for_score = 8.0
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

.. http:post:: /api/v3/ligand-mpnn/generate/

   Generate endpoint for LigandMPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **temperature** (*float*, default: 0.1) — Sampling temperature

        - **fixed_residues** (*array of strings*, default: []) — Residue identifiers to keep fixed; each entry formatted as [ChainID][ResidueNumber][OptionalInsertionCode]

        - **redesigned_residues** (*array of strings*, default: []) — Residue identifiers to redesign; each entry formatted as [ChainID][ResidueNumber][OptionalInsertionCode]

        - **bias_AA** (*object*, default: {}) — Global per–amino-acid logit biases; keys are single-letter amino acid codes, values are floats

        - **bias_AA_per_residue** (*object*, default: {}) — Per-residue amino-acid logit biases; keys are residue specs [ChainID][ResidueNumber][OptionalInsertionCode], values are objects mapping single-letter amino acid codes to floats

        - **omit_AA** (*string*, default: "") — Concatenated single-letter amino acid codes to globally omit

        - **omit_AA_per_residue** (*object*, default: {}) — Per-residue amino acids to omit; keys are residue specs [ChainID][ResidueNumber][OptionalInsertionCode], values are strings of single-letter amino acid codes

        - **symmetry_residues** (*array of arrays of strings*, default: []) — Groups of symmetry-linked residues; each inner array contains residue specs [ChainID][ResidueNumber][OptionalInsertionCode]

        - **symmetry_weights** (*array of arrays of floats*, default: []) — Per-residue symmetry weights; each inner array must match the length of the corresponding entry in symmetry_residues

        - **homo_oligomer** (*boolean*, default: False) — Flag indicating homo-oligomer sequence sharing across chains

        - **chains_to_design** (*array of strings*, default: []) — Chain identifiers to include in sequence design; each entry must match a chain ID present in the PDB

        - **parse_these_chains_only** (*array of strings*, default: []) — Chain identifiers to parse from the PDB; each entry must match a chain ID present in the PDB

        - **parse_atoms_with_zero_occupancy** (*boolean*, default: False) — Whether to include atoms with zero occupancy from the PDB when parsing

        - **number_of_batches** (*int*, range: 1–1, default: 1) — Number of batches of designs to generate per request

        - **batch_size** (*int*, range: 1–2, default: 1) — Number of designs to generate per batch

        - **repack_everything** (*boolean*, default: False) — Whether to repack all residues when side-chain packing is enabled

        - **pack_side_chains** (*boolean*, default: False) — Whether to run side-chain packing and include side-chain-packed outputs

        - **number_of_packs_per_design** (*int*, range: 1–8, default: 1) — Number of independent side-chain packing runs per design

        - **sc_num_samples** (*int*, range: 1–64, default: 16) — Number of side-chain samples per packing run

        - **sc_num_denoising_steps** (*int*, range: 1–10, default: 3) — Number of denoising steps used in side-chain sampling

        - **force_hetatm** (*boolean*, default: False) — Whether to treat all HETATM records as ligand or context atoms for side-chain packing

        - **pack_with_ligand_context** (*boolean*, default: True) — Whether to include ligand atomic context during side-chain packing

        - **fasta_seq_separation** (*string*, default: ":") — Separator string used between chain sequences when writing FASTA-formatted sequences

        - **file_ending** (*string*, default: "") — Optional suffix string associated with output file naming

        - **zero_indexed** (*int*, default: 0) — Flag indicating residue index origin in associated filenames

        - **pdb_path** (*null*, fixed) — Unused; must be null

        - **redesigned_residues_multi** (*null*, fixed) — Unused; must be null

        - **fixed_residues_multi** (*null*, fixed) — Unused; must be null

        - **bias_AA_per_residue_multi** (*null*, fixed) — Unused; must be null

        - **omit_AA_per_residue_multi** (*null*, fixed) — Unused; must be null

        - **save_stats** (*null*, fixed) — Unused; must be null

        - **verbose** (*boolean*, default: True) — Verbosity flag for internal logging

        - **ligand_mpnn_use_side_chain_context** (*null*, fixed) — Unused; must be null

        - **ligand_mpnn_use_atom_context** (*boolean*, default: True) — Whether to use ligand atomic context in the design computation

        - **ligand_mpnn_cutoff_for_score** (*float*, default: 8.0) — Distance cutoff in ångströms for including ligand atoms in scoring


      - **items** (*array of objects*, min: 1, max: 1, required) --- Input structures:

        - **pdb** (*string*, required, min length: 1, max length: max_pdb_str_len) — PDB-formatted structure text containing ATOM and optional HETATM records

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/ligand-mpnn/generate/ HTTP/1.1
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
          "ligand_mpnn_cutoff_for_score": 8.0
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

        - **sequence** (*string*, length ≤ 1,024) — Designed amino-acid sequence

        - **pdb** (*string*) — Designed structure in PDB format

        - **overall_confidence** (*float*) — Mean model confidence score over redesigned residues, range: 0.0–100.0

        - **ligand_confidence** (*float*) — Mean model confidence score over residues within the ligand/context cutoff, range: 0.0–100.0

        - **seq_rec** (*float*, range: 0.0–1.0) — Native sequence recovery fraction over redesigned residues

        - **log_probs** (*array of arrays of floats*, shape: [L, 21]) — Autoregressive per-position log-probabilities over 21 amino-acid types

          - Inner array (*float[21]*) — log-probabilities from ``log_softmax`` over 21 residue types for one designed position

        - **sampling_probs** (*array of arrays of floats*, shape: [L, 21]) — Per-position categorical sampling probabilities over 21 amino-acid types

          - Inner array (*float[21]*, sum: 1.0) — Softmax probabilities used for temperature-based sampling at one position

        - **pdb_packed** (*object*, optional) — Sidechain-packed structures by design index for side_chain model outputs

          - **<design_id>** (*string*) — Sidechain-packed structure for one design instance in PDB format

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "T",
            "pdb": "REMARK Selection '(backbone) and ...occupancy > 0))'\nATOM      1  N   THR A   1      11.104  13.207  11.947  1.00  0.12           N  \nATOM      2  CA  THR A   1      12.560  13.051  11.824  1.00  0.12... (truncated for documentation)",
            "overall_confidence": 0.1179,
            "ligand_confidence": 1.0,
            "seq_rec": 0.0,
            "log_probs": [
              [
                -2.8629157543182373,
                -4.322875499725342,
                "... (truncated for documentation)"
              ]
            ],
            "sampling_probs": [
              [
                0.0002876336802728474,
                1.3131530585130946e-10,
                "... (truncated for documentation)"
              ]
            ]
          }
        ]
      }


Performance
-----------

- Model complexity and runtime:
  
  - LigandMPNN uses the same message-passing backbone as ProteinMPNN with additional protein–ligand and intraligand graph encoders, increasing parameter count from ~1.7M to ~2.6M
  - Despite this, decoding remains linear in protein length; CPU benchmarks from the reference implementation scale from ~0.6 s (ProteinMPNN) to ~0.9 s (LigandMPNN) per 100 residues, so BioLM’s deployment has only a small constant-factor overhead relative to ProteinMPNN
  - Within the MPNN family (ProteinMPNN, soluble/membrane variants, LigandMPNN), LigandMPNN is slightly slower per residue but still far lighter than structure-prediction models such as AlphaFold2 or ESMFold, which require MSAs or large sequence databases

- Accuracy at ligand-contact sites (versus ProteinMPNN and Rosetta, held-out PDB complexes):
  
  - Small-molecule contacts (side-chain atoms within 5 Å of non-protein atoms): native sequence recovery ≈ 63.3 % for LigandMPNN versus ≈ 50.4 % for both ProteinMPNN and Rosetta (genpot)
  - Nucleotide contacts: ≈ 50.5 % for LigandMPNN, versus ≈ 34.0 % for ProteinMPNN and ≈ 35.2 % for Rosetta with a DNA-optimized energy function
  - Metal contacts: ≈ 77.5 % for LigandMPNN, versus ≈ 40.6 % for ProteinMPNN and ≈ 36.0 % for a Rosetta metal-optimized protocol; away from ligand-contact positions, LigandMPNN’s global sequence recovery is comparable to ProteinMPNN

- Side-chain packing performance near ligands (within 5 Å of context atoms):
  
  - χ1 recovery (torsion within 10° of crystal) improves consistently over Rosetta and over a protein-only MPNN side-chain model: small-molecule sites ≈ 86.1 % (LigandMPNN) vs 83.3 % (Protein-only MPNN) vs 76.0 % (Rosetta); nucleotide sites ≈ 71.4 % vs 65.6 % vs 66.2 %; metal sites ≈ 79.3 % vs 76.7 % vs 68.6 %
  - For deeper torsions (χ3, χ4), all methods degrade in accuracy, but LigandMPNN still outperforms Rosetta and slightly improves on protein-only MPNN near ligands
  - Ligand-aware packing in the API leverages this model, enabling more reliable evaluation and repacking of residues around small molecules, nucleic acids and metals than ProteinMPNN-based packing alone

- Relative efficiency and deployment behavior:
  
  - Compared to Rosetta-based sequence-design and packing protocols, LigandMPNN inference avoids combinatorial Monte Carlo searches and is ≈ 250× faster on similar CPUs in the reference implementation; BioLM’s GPU-backed deployment preserves this large speed advantage at scale
  - Within BioLM pipelines, LigandMPNN is typically a minor contributor to end-to-end latency: it is significantly cheaper and faster per call than structure-prediction models (AlphaFold2, ESMFold, NanobodyBuilder2) while providing 10–40 percentage points higher native sequence recovery in ligand-contact regions than ProteinMPNN
  - Training with ~0.1 Å Gaussian coordinate noise improves robustness to non-crystallographic backbones (e.g., RFdiffusion-like designs, AlphaFold2/ESMFold hallucinations), so performance degrades less on generated structures than methods that assume ideal crystallographic geometry

Applications
------------

- Structure-based optimization of small-molecule binders around fixed ligand poses, using LigandMPNN to redesign residues within ~5 Å of a bound compound to improve shape complementarity, hydrogen bonding, and pocket pre-organization; this is valuable for biotech and platform companies that already have scaffold backbones (for example from RFdiffusion, RoseTTAFold All-Atom, or Rosetta) and need to rapidly generate and prioritize binder variants for specific drugs, metabolites, or imaging agents without hand-tuning force-field parameters
- Design and maturation of protein-based small-molecule sensors, where LigandMPNN is used to refine or re-pack binding sites around pre-positioned ligands to improve affinity and specificity while maintaining a given global fold; this is useful for engineering biosensor proteins for diagnostics, process monitoring, or cell-based reporting systems, since the model explicitly accounts for non-protein atoms and predicts sidechain conformations rather than relying on purely backbone-based inverse folding
- Sequence design of protein–DNA interfaces for sequence-specific DNA binders, by conditioning on the atomic coordinates of a target DNA duplex and a candidate protein backbone to generate residues that recognize bases in the major groove; this enables industrial teams building programmable DNA-binding domains (for synthetic regulation, genome targeting, or DNA diagnostics) to move beyond generic DNA affinity and toward more sequence-selective binders with experimentally validated design patterns, while still requiring downstream structure prediction and experimental validation for off-target assessment
- Context-aware redesign of metal-binding sites (for example Zn, Fe, or other transition metals), leveraging LigandMPNN’s explicit encoding of metal identity and geometry to propose coordinating residues and sidechain conformations around an already positioned metal ion; this is useful for stabilizing metal-dependent scaffolds (such as catalytic or structural metal sites) in bioprocess-compatible formats, but it is not optimal for elements that are extremely rare or absent in PDB training data, where additional physics-based modeling or element mapping is required
- Rescue and affinity improvement of existing structure-based designs that show weak or no binding, by feeding in the original backbone and ligand pose and using LigandMPNN to locally respecify binding-site residues and sidechains; this is particularly valuable for teams with legacy Rosetta or in-house designs that partially failed in the lab, allowing systematic sequence-level optimization and sidechain repacking around the ligand at GPU-accelerated throughput instead of re-running expensive Monte Carlo packing or manually re-engineering pockets

Limitations
-----------

- **Sequence length and input size**: Each design request is limited to backbones with at most ``1024`` residues per chain (``MPNNParams.max_sequence_len``). Very large assemblies (for example, multi‑megadalton complexes) may need to be split and designed in pieces. The request payload ``items`` must contain valid PDB text in ``items[0].pdb`` (only one item is allowed per request for LigandMPNN); malformed coordinates or missing backbone atoms (N, CA, C, O) will cause validation to fail.
- **Batching and throughput**: LigandMPNN design is optimized for small batches. The top‑level parameters ``number_of_batches`` and ``batch_size`` are both capped at ``1`` (``MPNNParams.num_batches`` and ``MPNNParams.batch_size``), so each API call can only generate one batch of one design per input structure. The ``LigandMPNNGenerateRequest.items`` list may contain at most ``MPNNParams.items_batch_size`` PDBs (``1``) per call; to design larger libraries you must parallelize across multiple API calls.
- **Ligand and atom context assumptions**: LigandMPNN expects small molecules, nucleotides, metals, or selected side chains encoded as ``HETATM`` (or side‑chain atoms when ``ligand_mpnn_use_atom_context`` / ``force_hetatm`` are used) with reasonable geometry near the binding site. Only the closest 25 ligand atoms per residue are used internally, so very large ligands (for example, long DNA or RNA segments) may be only partially “seen” at each position. Unusual elements that are rare or absent in the PDB training data should be mapped to chemically similar elements before submission; otherwise, scores and designs around those atoms can be unreliable.
- **Design scope and control**: Residue‑level control via ``fixed_residues``, ``redesigned_residues``, ``symmetry_residues``, and chain selection via ``chains_to_design`` / ``parse_these_chains_only`` is validated against the input PDB: chain IDs and residue indices must exist and lie within the detected chain lengths, or the request will fail. The model is sequence‑design only: it assumes the input backbone (and ligand coordinates) are fixed and close to a realizable structure, and it does not relax or repair poor backbones or clashes; for backbone generation or large‑scale conformational search, diffusion‑based or structure‑prediction models are more appropriate.
- **Scoring and side‑chain modeling**: The API returns per‑residue probabilities and overall confidence scores (for example, ``overall_confidence``, ``ligand_confidence``, ``log_probs``, ``sampling_probs``), but these are not physical binding free energies and should not be interpreted as quantitative ΔG values. The optional side‑chain packing model (controlled via ``pack_side_chains`` / ``repack_everything`` and related ``sc_*`` parameters) improves local packing but still struggles with higher‑order torsions (chi3/chi4) and cannot replace full all‑atom refinement when exact hydrogen‑bonding networks or metal coordination geometry are critical.
- **Non‑optimal use cases**: LigandMPNN is specialized for sequence design on pre‑specified backbones with local ligand/atom context. It is not the best choice when you primarily need: (a) de novo backbone generation (use backbone diffusion or structure‑hallucination models instead), (b) high‑throughput ranking of millions of sequences without explicit structures (sequence‑ or embedding‑based models scale better), (c) global stability optimization without nonprotein context (standard ProteinMPNN or sequence‑based fitness models are typically more efficient), or (d) detailed, physics‑based binding energy decomposition (classical docking or Rosetta‑style all‑atom energy functions are more appropriate downstream of LigandMPNN designs).

How We Use It
-------------

LigandMPNN enables structure-based sequence design around small molecules, nucleotides and metals as a programmable service, turning docked or experimentally determined complexes into ranked sequence hypotheses that are ready for synthesis and testing. In BioLM workflows it integrates upstream with backbone-generation and docking (for example RFdiffusion-style backbones, ligand docking or curated crystal structures) and downstream with sequence-embedding, stability and developability predictors and experimental design loops. Standardized, GPU-backed APIs allow teams to run scalable campaigns where candidate binding pockets are generated, sequences are locally redesigned around the ligand with explicit atomic context and optional sidechain repacking, then filtered and prioritized under project-specific constraints such as manufacturability, IP considerations or assay cost.

- Integrates with other structure-generation, docking and scoring models to form end-to-end binder, sensor and enzyme design workflows.
- Supports iterative design–build–test–learn cycles where experimental data refines which ligand-contact variants and pockets are explored in subsequent rounds.

Related
-------

- ``ProteinMPNN`` – Backbone-only sequence design model that LigandMPNN extends; useful as a faster or simpler alternative when no nonprotein context (ligands, metals, nucleic acids) is present.
- ``SolubleMPNN`` – ProteinMPNN variant biased toward soluble designs; can be used instead of LigandMPNN when optimizing solubility for ligand-free backbones.
- ``ESMFold`` – Structure prediction from sequence; commonly used downstream of LigandMPNN to check that ligand-conditioned designs still fold to the intended backbone.
- ``AlphaFold2`` – High-accuracy structure prediction for designed complexes; can be paired with LigandMPNN designs to assess global folding and binding-site geometry.

References
----------

- Dauparas, J., Lee, G. R., Pecoraro, R., An, L., Anishchenko, I., Glasscock, C., & Baker, D. (2025). Atomic context-conditioned protein sequence design using LigandMPNN. *Nature Methods*. https://doi.org/10.1038/s41592-025-02626-1
