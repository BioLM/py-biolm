Global Label Membrane MPNN API
==============================

Global Label Membrane MPNN is a ProteinMPNN variant trained to generate protein sequences consistent with membrane protein–like topologies while allowing a global transmembrane vs soluble label. Given backbone structures as PDB input, it produces redesigned sequences conditioned on a user-specified global label (``membrane`` or ``soluble``) and optional residue constraints, with up to 1 item and batch size 1 per request. The service returns sequences, redesigned PDBs, and per-position log and sampling probabilities for downstream protein design workflows.

Generate
--------

This endpoint gensats for Global Label Membrane MPNN.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="global-label-membrane-mpnn",
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
                  "global_transmembrane_label": "soluble"
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

            curl -X POST https://biolm.ai/api/v3/global-label-membrane-mpnn/generate/ \
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
                "global_transmembrane_label": "soluble"
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

            url = "https://biolm.ai/api/v3/global-label-membrane-mpnn/generate/"
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
                    "global_transmembrane_label": "soluble"
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

            url <- "https://biolm.ai/api/v3/global-label-membrane-mpnn/generate/"
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
                global_transmembrane_label = "soluble"
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

.. http:post:: /api/v3/global-label-membrane-mpnn/generate/

   Generate endpoint for Global Label Membrane MPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **temperature** (*float*, default: 0.1) — Sampling temperature

        - **fixed_residues** (*array of strings*, default: [], each value format: [ChainID][ResidueNumber][OptionalInsertionCode]) — Residue positions that must remain unchanged, validated against the input PDB

        - **redesigned_residues** (*array of strings*, default: [], each value format: [ChainID][ResidueNumber][OptionalInsertionCode]) — Residue positions that must be redesigned, validated against the input PDB

        - **bias_AA** (*object*, default: {}, keys: single-letter amino acid codes, values: floats) — Global amino acid sampling bias per residue type

        - **bias_AA_per_residue** (*object*, default: {}, keys: residue specifications [ChainID][ResidueNumber][OptionalInsertionCode], values: objects with keys as single-letter amino acid codes and values as floats) — Amino acid sampling bias for specific residues

        - **omit_AA** (*string*, default: "", characters: single-letter amino acid codes) — Amino acid types globally disallowed at all positions

        - **omit_AA_per_residue** (*object*, default: {}, keys: residue specifications [ChainID][ResidueNumber][OptionalInsertionCode], values: strings of single-letter amino acid codes) — Amino acid types disallowed at specific residues

        - **symmetry_residues** (*array of arrays of strings*, default: [], each inner string format: [ChainID][ResidueNumber][OptionalInsertionCode]) — Groups of residues constrained to share symmetry definitions, validated against the input PDB

        - **symmetry_weights** (*array of arrays of floats*, default: []) — Per-residue weights corresponding to each group in symmetry_residues

        - **homo_oligomer** (*boolean*, default: False) — Flag indicating homo-oligomer design mode

        - **chains_to_design** (*array of strings*, default: []) — Chain IDs to be designed, each chain validated to exist in the input PDB

        - **parse_these_chains_only** (*array of strings*, default: []) — Chain IDs to be parsed from the input PDB, each validated to exist in the input PDB

        - **parse_atoms_with_zero_occupancy** (*boolean*, default: False) — Flag for including atoms with zero occupancy from the input PDB

        - **number_of_batches** (*int*, range: 1-1, default: 1) — Number of batches to generate per request

        - **batch_size** (*int*, range: 1-2, default: 1) — Number of sequences generated per batch

        - **repack_everything** (*boolean*, optional, default: False) — Flag for repacking all side chains when using the side-chain model

        - **pack_side_chains** (*boolean*, optional, default: False) — Flag for enabling side-chain packing when using the side-chain model

        - **number_of_packs_per_design** (*int*, optional, range: 1-8, default: 1) — Number of side-chain packing runs per design

        - **sc_num_samples** (*int*, optional, range: 1-64, default: 16) — Number of side-chain model samples per design

        - **sc_num_denoising_steps** (*int*, optional, range: 1-10, default: 3) — Number of denoising steps for side-chain sampling

        - **force_hetatm** (*boolean*, optional, default: False) — Flag for forcing inclusion of HETATM records from the input PDB

        - **pack_with_ligand_context** (*boolean*, optional, default: True) — Flag for packing side chains with ligand context when applicable

        - **fasta_seq_separation** (*string*, default: ":") — Separator used when concatenating sequences in FASTA-formatted outputs

        - **file_ending** (*string*, default: "") — Auxiliary file ending field, not used by the model

        - **zero_indexed** (*int*, default: 0) — Indexing mode flag for residue numbering

        - **pdb_path** (*null*, fixed: null) — Unused path field, always null

        - **redesigned_residues_multi** (*null*, fixed: null) — Unused multi-chain redesign field, always null

        - **fixed_residues_multi** (*null*, fixed: null) — Unused multi-chain fixed residue field, always null

        - **bias_AA_per_residue_multi** (*null*, fixed: null) — Unused multi-chain per-residue bias field, always null

        - **omit_AA_per_residue_multi** (*null*, fixed: null) — Unused multi-chain per-residue omit field, always null

        - **save_stats** (*null*, fixed: null) — Unused statistics saving field, always null

        - **verbose** (*boolean*, default: True) — Verbosity flag for processing

        - **ligand_mpnn_use_side_chain_context** (*null*, fixed: null) — Unused ligand side-chain context field, always null

        - **ligand_mpnn_use_atom_context** (*boolean*, optional, default: True) — Flag for using ligand atom context when ligand-specific scoring is enabled

        - **ligand_mpnn_cutoff_for_score** (*float*, optional, default: 8.0) — Distance cutoff in Å for ligand-related scoring

        - **global_transmembrane_label** (*string*, optional, allowed: "membrane" or "soluble", default: "soluble") — Global transmembrane classification label applied to the structure

        - **transmembrane_buried** (*array of strings*, optional, default: null, each value format: [ChainID][ResidueNumber][OptionalInsertionCode]) — Residues labeled as transmembrane buried, validated against the input PDB

        - **transmembrane_interface** (*array of strings*, optional, default: null, each value format: [ChainID][ResidueNumber][OptionalInsertionCode]) — Residues labeled as transmembrane interface, validated against the input PDB


      - **items** (*array of objects*, min: 1, max: 1) --- Input structures:

        - **pdb** (*string*, required, min length: 1, max length: max_pdb_str_len) — PDB-formatted structure text containing ATOM and/or HETATM records, validated for formatting and consistency

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/global-label-membrane-mpnn/generate/ HTTP/1.1
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
          "global_transmembrane_label": "soluble"
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

        - **sequence** (*string*) — Designed amino acid sequence using unambiguous 20-letter alphabet, length ≤ 1024 residues

        - **pdb** (*string*) — Protein backbone coordinates in PDB format corresponding to the designed sequence

        - **overall_confidence** (*float*) — Model confidence score for the designed sequence–structure pair, unitless

        - **ligand_confidence** (*float*) — Model confidence score for ligand-environment compatibility, unitless

        - **seq_rec** (*float*, range: 0.0–1.0) — Sequence recovery fraction relative to the input backbone reference

        - **log_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-residue log-probabilities over the 20 amino acids, where L is the sequence length

        - **sampling_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-residue sampling probabilities over the 20 amino acids, where L is the sequence length, each inner array summing to 1.0

        - **pdb_packed** (*object*, optional) — Side-chain–packed structures for side-chain models:

          - **<chain_id>** (*string*) — Side-chain–packed PDB string for the specified chain identifier

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "R",
            "pdb": "REMARK Selection '(backbone) and ...occupancy > 0))'\nATOM      1  N   ARG A   1      11.104  13.207  11.947  1.00  0.12           N  \nATOM      2  CA  ARG A   1      12.560  13.051  11.824  1.00  0.12... (truncated for documentation)",
            "overall_confidence": 0.1174,
            "ligand_confidence": 0.1174,
            "seq_rec": 0.0,
            "log_probs": [
              [
                -2.4706003665924072,
                -5.006504058837891,
                "... (truncated for documentation)"
              ]
            ],
            "sampling_probs": [
              [
                0.0280643068253994,
                2.721860551616745e-13,
                "... (truncated for documentation)"
              ]
            ]
          }
        ]
      }


Performance
-----------

- Model family and conditioning behavior:
  
  - Global Label Membrane MPNN is a ProteinMPNN variant trained with a single binary conditioning variable (``global_transmembrane_label="membrane"`` or ``"soluble"``) that modulates global environment-specific residue distributions at essentially the same computational cost as base ProteinMPNN.
  - Relative to base ProteinMPNN (trained on mixed environments without conditioning), the global label allows steering the generative distribution toward membrane-like or soluble-like compositions without retraining, reducing downstream rejection from gross environment mismatches (e.g., over-hydrophobic soluble designs on membrane-like topologies).

- Performance vs Per-Residue Label Membrane MPNN:
  
  - Global Label Membrane MPNN is typically faster and simpler to configure because it applies a single global label and does not process per-residue transmembrane/interface tensors or associated consistency checks.
  - The per-residue variant can enforce more detailed environment patterns at specific positions (e.g., buried or interface residues in mixed environments) but incurs extra overhead from parsing and validating the additional residue-level annotations.

- Performance vs Soluble ProteinMPNN and base ProteinMPNN:
  
  - On backbones that are clearly soluble with no membrane-like features, Soluble ProteinMPNN usually provides slightly better soluble-expression outcomes, as it is explicitly trained on soluble-only data.
  - On membrane-native backbones (e.g., GPCR-, rhomboid-, claudin-like folds), Global Label Membrane MPNN is more robust than Soluble ProteinMPNN and base ProteinMPNN when switching between membrane-like and soluble-like design regimes, maintaining membrane-appropriate or solubilized surface hydrophobicity profiles according to the chosen global label.

- Hardware, runtime, and scaling in BioLM pipelines:
  
  - Global Label Membrane MPNN uses the same lightweight, CPU-efficient inference engine and resource footprint as other BioLM MPNN variants (no GPU required; similar memory and runtime characteristics across base, soluble, and membrane-conditioned checkpoints).
  - Compared with BioLM’s structure-prediction models (e.g., AlphaFold2, ESMFold, Chai-1) and backbone-generation diffusion models, Global Label Membrane MPNN is substantially faster and less resource-intensive because it operates on fixed backbones; in typical pipelines that pair AlphaFold2 backbone generation with MPNN design, the MPNN step is a minor fraction of end-to-end wall-clock time.

Applications
------------

- Design of soluble GPCR-like scaffolds for ligand screening and SAR campaigns, enabling teams to port the canonical 7TM-like topology into highly stable, monomeric, water-soluble proteins that can be expressed in standard systems; this is valuable when native GPCRs are too unstable or aggregation-prone for high-throughput biophysics (SPR, BLI, MST) or fragment screening, while still preserving the overall fold geometry needed to position small-molecule or peptide-binding sites
- Generation of soluble analogues of other complex membrane-like folds (e.g., rhomboid-like and claudin-like topologies) to support target validation, mechanistic studies, and structure-based hit optimization, by providing crystallography- and cryo-EM–ready scaffolds that approximate the native fold without detergents or lipid systems; this is particularly useful for companies building platform programs around historically “undruggable” membrane targets but needing robust structural surrogates for medicinal chemistry and binder discovery
- Creation of custom soluble receptor surrogates for high-throughput binder discovery (display libraries, DNA-encoded libraries, or in silico docking), where the model is used to design membrane-protein–like backbones and Global Label Membrane MPNN-derived sequences that present native-like helices and loops in a soluble or membrane-labeled context; this helps reduce assay artefacts and false negatives caused by misfolded or micelle-embedded native receptors, but is not optimal when exact native membrane environment effects (lipid dependence, oligomerization, allosteric lipids) are the primary concern
- Expansion of proprietary scaffold libraries with non-natural, membrane-derived folds for next-generation biologics and molecular glues, allowing R&D teams to explore sequence space with very low homology to natural proteins while maintaining experimentally validated designability (high thermal stability, monomeric behavior); this is valuable for IP differentiation and manufacturability, though downstream functionalization (e.g., engineering binding pockets or allosteric sites) still requires additional design cycles and wet-lab selection
- Platform integration for automated backbone + sequence design campaigns targeting specific membrane topologies (e.g., customer GPCR or transporter structures) to generate panels of soluble or membrane-labeled analogues with tunable surface properties (hydrophobicity, charge, epitope exposure) and global “membrane vs soluble” conditioning, enabling CRO/biotech groups to stand up scalable design–build–test loops; this is most effective when high-quality structural templates of the membrane target exist and the goal is robust surrogates rather than faithful recapitulation of full native signaling function

Limitations
-----------

- **Maximum sequence length and input size**: Backbones longer than 1024 residues per chain are not supported. Any designable chain extracted from the input ``pdb`` must therefore have ``<= 1024`` distinct residue positions; longer chains will fail validation.
- **Batching and concurrency limits**: Each request can include at most one ``items`` entry (one ``pdb`` backbone; ``min_items=1``, ``max_items=1``) and generates up to ``batch_size=2`` sequences per call with ``number_of_batches=1``. For higher throughput, you must parallelize across multiple API calls at the application level.
- **Backbone quality and labeling constraints**: The model does not modify or relax backbone coordinates in ``pdb``; it only designs sequences conditioned on the provided structure and the global membrane label (``global_transmembrane_label="membrane"`` or ``"soluble"``). Poorly resolved, highly strained, or non-biological backbones will typically yield low-confidence sequences, and the global label cannot enforce fine-grained topology or per-helix environment constraints.
- **Scope of membrane vs soluble control**: ``GlobalMembraneMPNNGenerateParams`` exposes only a single global tag (``global_transmembrane_label``) and does not support per-region placement of membrane vs solvent; use the per-residue membrane model (``per_residue_label_membrane`` / ``ResidueMembraneMPNNGenerateRequest``) when you need differential treatment of transmembrane, interface, and soluble segments. This global model is not suitable for detailed tuning of interface/buried residues in multi-pass or mixed-topology membrane proteins.
- **Algorithmic and functional limitations**: Global Label Membrane MPNN optimizes local sequence compatibility with a fixed backbone and environment label; it does not guarantee correct folding, stability, binding, or activity in vitro/in vivo, and it does not perform structure prediction, dynamics, or ligand design. As in AF2seq–MPNN workflows, complex membrane folds, long-range allostery, and functional motifs (e.g. GPCR signaling micro-switches, catalytic dyads) may require additional structure prediction, filtering, and experimental screening.
- **Non-optimal use cases**: This model is not the best choice when you (1) need backbone generation or redesign (use a generative backbone model or AF2-based design instead), (2) require explicit solubilization of membrane folds into soluble analogues (the ``soluble`` MPNN variant or a dedicated soluble-design pipeline is more appropriate), (3) must co-design protein–ligand or protein–protein interfaces (use the ``ligand`` or ``side_chain`` MPNN variants with appropriate context), or (4) need sequence embeddings for clustering/visualization (use a sequence- or structure-embedding model rather than this conditional sequence generator).

How We Use It
-------------

Global Label Membrane MPNN enables rapid redesign of membrane-derived backbones into either soluble or membrane-like sequence families, which we then integrate with structure predictors, backbone generators, and downstream property models in standardized, API-driven discovery loops. In practice, we use it to generate and compare soluble analogs and membrane-tuned variants of GPCRs, rhomboid proteases, claudins, and other transmembrane folds, prioritize them with BioLM stability, developability, binding, and biophysics predictors, and couple the best designs into automated sequence–structure–property campaigns that support target de-risking, scaffold discovery, and feasibility assessment for therapeutics, diagnostics, and enzyme engineering.

- Integrates with backbone generators (e.g., AF2-based or diffusion models) and other BioLM predictors to build scalable redesign–ranking pipelines for membrane-derived scaffolds.  
- Enables programmatic creation of soluble surrogates and membrane-optimized variants, improving assay design, manufacturability readouts, and early platform fit decisions.

Related
-------

- ``Soluble ProteinMPNN`` – Soluble-only variant of ProteinMPNN used in the original AF2seq–MPNNsol pipeline; useful for comparing explicit membrane labeling in Global Label Membrane MPNN against purely soluble-biased sequence design on the same backbone.
- ``Per-Residue Label Membrane MPNN`` – Adds residue-level membrane/solvent labels instead of a single global tag, enabling fine-grained control of which positions are treated as transmembrane vs solvent-exposed.
- ``ProteinMPNN`` – General-purpose backbone-conditioned sequence design model; use alongside Global Label Membrane MPNN to benchmark how global membrane labels alter sequence preferences and surface hydrophobicity.
- ``LigandMPNN`` – Extends MPNN-style design to protein–ligand complexes; after using Global Label Membrane MPNN for membrane-aware scaffold design, apply LigandMPNN to optimize binding pockets around bound small molecules or cofactors.

References
----------

- Goverde, C. A., Pacesa, M., Dornfeld, L. J., Georgeon, S., Rosset, S., Dauparas, J., Schellhaas, C., Kozlov, S., Baker, D., Ovchinnikov, S., & Correia, B. E. (2023). Computational design of soluble analogues of integral membrane protein structures. *bioRxiv*. https://doi.org/10.1101/2023.05.09.540044
