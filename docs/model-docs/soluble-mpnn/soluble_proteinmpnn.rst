Soluble ProteinMPNN API
=======================

Soluble ProteinMPNN is a structure-conditioned sequence design service for generating soluble analogs of integral membrane protein folds. Given a single backbone in PDB format (one structure per request), it applies a ProteinMPNN model retrained on soluble structures to redesign residues, lowering surface hydrophobicity while preserving the target topology. The API exposes residue-level constraints, per-residue amino acid biases/omissions, symmetry specifications, and soluble/membrane labeling, supporting design of GPCR-like and other complex folds for enzyme engineering, ligand binding, and biophysical studies.

Generate
--------

This endpoint gensats for Soluble ProteinMPNN.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="soluble-mpnn",
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

            curl -X POST https://biolm.ai/api/v3/soluble-mpnn/generate/ \
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

            url = "https://biolm.ai/api/v3/soluble-mpnn/generate/"
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

            url <- "https://biolm.ai/api/v3/soluble-mpnn/generate/"
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

.. http:post:: /api/v3/soluble-mpnn/generate/

   Generate endpoint for Soluble ProteinMPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **temperature** (*float*, default: 0.1) — Sampling temperature

        - **fixed_residues** (*array of strings*, default: []) — Residue specifications kept fixed during sequence generation

        - **redesigned_residues** (*array of strings*, default: []) — Residue specifications that are required to change during sequence generation

        - **bias_AA** (*object*, default: {}) — Global amino acid sampling bias; keys are single-letter amino acid codes, values are float bias weights

        - **bias_AA_per_residue** (*object*, default: {}) — Per-residue amino acid sampling bias; keys are residue specifications, values are objects mapping single-letter amino acid codes to float bias weights

        - **omit_AA** (*string*, default: "") — String of single-letter amino acid codes globally excluded from sampling

        - **omit_AA_per_residue** (*object*, default: {}) — Per-residue amino acid exclusions; keys are residue specifications, values are strings of single-letter amino acid codes

        - **symmetry_residues** (*array of arrays of strings*, default: []) — Groups of residue specifications constrained together for symmetry

        - **symmetry_weights** (*array of arrays of floats*, default: []) — Weight values associated with each group in symmetry_residues

        - **homo_oligomer** (*boolean*, default: False) — Homooligomer design mode flag

        - **chains_to_design** (*array of strings*, default: []) — Chain identifiers to include in design

        - **parse_these_chains_only** (*array of strings*, default: []) — Chain identifiers to restrict PDB parsing to

        - **parse_atoms_with_zero_occupancy** (*boolean*, default: False) — Whether atoms with zero occupancy are included when parsing the PDB

        - **number_of_batches** (*int*, range: 1-1, default: 1) — Number of batches per request

        - **batch_size** (*int*, range: 1-2, default: 1) — Number of designs generated per batch

        - **repack_everything** (*boolean*, default: False) — Whether all residues are repacked in side-chain modeling

        - **pack_side_chains** (*boolean*, default: False) — Whether side-chain packing is enabled

        - **number_of_packs_per_design** (*int*, range: 1-8, default: 1) — Number of side-chain packing runs per design

        - **sc_num_samples** (*int*, range: 1-64, default: 16) — Number of side-chain sampling trajectories

        - **sc_num_denoising_steps** (*int*, range: 1-10, default: 3) — Number of denoising steps in side-chain sampling

        - **force_hetatm** (*boolean*, default: False) — Whether HETATM records are included in side-chain modeling

        - **pack_with_ligand_context** (*boolean*, default: True) — Whether side-chain packing uses ligand context

        - **fasta_seq_separation** (*string*, default: ":") — Separator used when concatenating sequences in FASTA-formatted outputs

        - **file_ending** (*string*, default: "") — File suffix used internally

        - **zero_indexed** (*int*, default: 0) — Residue index origin flag

        - **pdb_path** (*null*, fixed: null) — Unused field

        - **redesigned_residues_multi** (*null*, fixed: null) — Unused field

        - **fixed_residues_multi** (*null*, fixed: null) — Unused field

        - **bias_AA_per_residue_multi** (*null*, fixed: null) — Unused field

        - **omit_AA_per_residue_multi** (*null*, fixed: null) — Unused field

        - **save_stats** (*null*, fixed: null) — Unused field

        - **verbose** (*boolean*, default: True) — Logging verbosity flag

        - **ligand_mpnn_use_side_chain_context** (*null*, fixed: null) — Unused field

        - **ligand_mpnn_use_atom_context** (*boolean*, default: True) — Whether ligand-aware scoring uses atom-level context

        - **ligand_mpnn_cutoff_for_score** (*float*, default: 8.0) — Distance cutoff in angstroms for ligand scoring

        - **global_transmembrane_label** (*string*, allowed: "membrane" | "soluble", default: "soluble") — Global transmembrane environment label applied to all residues

        - **transmembrane_buried** (*array of strings*, default: null) — Residue specifications labeled as buried transmembrane

        - **transmembrane_interface** (*array of strings*, default: null) — Residue specifications labeled as transmembrane interface


      - **items** (*array of objects*, min: 1, max: 1, required) --- Input structures:

        - **pdb** (*string*, min length: 1, max length: max_pdb_str_len, required) — PDB-formatted structure string containing ATOM and/or HETATM records

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/soluble-mpnn/generate/ HTTP/1.1
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

        - **sequence** (*string*) — Designed amino acid sequence using single-letter codes, length L

        - **pdb** (*string*) — Designed structure in PDB format corresponding to ``sequence``

        - **overall_confidence** (*float*) — Model confidence score for the design, unitless

        - **ligand_confidence** (*float*) — Model confidence score related to ligand or atom context, unitless

        - **seq_rec** (*float*) — Sequence recovery percentage relative to the input, range: 0.0–100.0

        - **log_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-position log-probabilities for 20 amino acids, values in natural logarithm space

        - **sampling_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-position sampling probabilities for 20 amino acids, each inner array sums to 1.0

        - **pdb_packed** (*object*, optional) — Side-chain repacked structures for side_chain models

          - **<key>** (*string*) — Context identifier such as a chain ID or design label

          - **<value>** (*string*) — Repacked structure in PDB format for the corresponding key

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "G",
            "pdb": "REMARK Selection '(backbone) and ...occupancy > 0))'\nATOM      1  N   GLY A   1      11.104  13.207  11.947  1.00  0.38           N  \nATOM      2  CA  GLY A   1      12.560  13.051  11.824  1.00  0.38... (truncated for documentation)",
            "overall_confidence": 0.3815,
            "ligand_confidence": 0.3815,
            "seq_rec": 0.0,
            "log_probs": [
              [
                -2.184384822845459,
                -4.591545581817627,
                "... (truncated for documentation)"
              ]
            ],
            "sampling_probs": [
              [
                4.99277894050465e-06,
                1.7545898396612687e-16,
                "... (truncated for documentation)"
              ]
            ]
          }
        ]
      }


Performance
-----------

- Model behavior and training:
  - Soluble ProteinMPNN is a ProteinMPNN variant retrained on a PDB subset filtered to exclude annotated transmembrane structures, biasing designs toward soluble-like surface composition while retaining core packing and fold-conditional behavior comparable to the base ProteinMPNN on soluble backbones.
- Backbone‑conditional structural accuracy:
  - For experimentally validated AF2seq → Soluble ProteinMPNN designs from Goverde et al. (e.g., CLF_4, RPF_9, GLF_18/32, TBF_24), design models vs X‑ray structures show backbone RMSD\ :sub:`Cα` ≈ 0.73–1.05 Å and full‑atom RMSD ≈ 1.3–2.1 Å, indicating high side‑chain and global‑fold fidelity under fixed‑backbone design conditions.
- Solubility bias and experimental outcomes vs standard ProteinMPNN:
  - Relative to generic ProteinMPNN, the soluble variant reduces the fraction of apolar residues on solvent‑exposed surfaces for membrane‑like or amphipathic backbones, shifting surface hydrophobicity toward naturally soluble profiles and supporting higher empirical rates of soluble, monomeric, often hyperthermostable designs (many with T\ :sub:`m` > 90 °C) in AF2seq → Soluble ProteinMPNN pipelines.
- Performance relative to other BioLM MPNN‑family models:
  - On generic soluble backbones, fixed‑backbone design metrics (e.g., sequence recovery, AF2/RoseTTAFold reprediction confidence) are similar to base ProteinMPNN, while on membrane‑like topologies the soluble model empirically yields a better trade‑off between maintaining the target fold and lowering surface hydrophobicity, and is typically not the runtime bottleneck compared with downstream structure‑prediction models in full design–validation workflows.

Applications
------------

- Designing soluble GPCR-like scaffolds for high-throughput ligand screening and SAR campaigns by generating water-soluble 7TM-like helical bundles that preserve overall GPCR topology while removing membrane-facing hydrophobics; this enables pharma and biotech teams to run fragment screens, SPR/biophysical assays, or display-based selections using robust, easily expressed scaffolds instead of native membrane GPCRs, with the important caveat that these designs are structural surrogates and are not expected to reproduce native signaling or G protein coupling
- Engineering soluble analogues of other membrane folds (for example, claudin-like and rhomboid-like helices) as tractable surrogates for difficult membrane targets, enabling biophysical characterization, structural studies, and binder discovery against GPCR-adjacent or junction-forming topologies in standard aqueous expression systems; this is useful when native membrane proteins are unstable, aggregate in detergents, or require lipid environments that slow down lead discovery, noting that catalytic or transport activities of the original membrane proteins are typically not preserved
- Creation of de novo soluble helical scaffolds with predefined complex topologies for small-molecule and peptide binder design by using Soluble ProteinMPNN to generate low-hydrophobicity surfaces on integral-membrane-like backbones while maintaining long-range packing; this supports industrial platforms that need panels of stable, monomeric, non-natural scaffolds (low similarity to natural sequences) to host binding pockets, switchable elements, or conformational sensors directed at intracellular signaling components, rather than reproducing known natural folds
- Development of soluble “mock targets” for assay development, QC, and manufacturability screening by designing highly thermostable soluble variants of topologies normally found only in the membrane that are compatible with bacterial or yeast expression; this lets companies de-risk downstream screening and analytical workflows before committing to cell-based or membrane reconstitution assays, while acknowledging that these analogues approximate the structural environment but not full native activity
- Integration into automated protein engineering pipelines for fold-space exploration and IP generation, where Soluble ProteinMPNN is used to systematically sample sequence space around integral membrane-like topologies under soluble constraints (reduced surface hydrophobicity, cysteines masked by default), enabling creation of proprietary, highly stable scaffolds with novel sequences and folds; this is valuable for organizations building platform IP, but not optimal when preserving native membrane protein function is the primary objective

Limitations
-----------

- **Maximum sequence length and structure size**: Each input ``pdb`` is limited to backbones where every designed chain has at most ``MPNNParams.max_sequence_len = 1024`` residues. Very large complexes or concatenated multi-domain models that exceed this per-chain limit are not supported and should be split into smaller design units before calling the API.
- **Batching and throughput constraints**: The request-wide ``params.batch_size`` and ``params.number_of_batches`` are capped at ``MPNNParams.batch_size = 2`` and ``MPNNParams.num_batches = 1`` respectively, and ``items`` is limited to a single ``MPNNGenerateRequestItem`` per call (``min_items=1``, ``max_items=1``). This API is optimized for iterative design on a small number of backbones per request rather than bulk redesign of large structure libraries in a single call.
- **Backbone quality and topology dependence**: Soluble ProteinMPNN assumes the input ``pdb`` already represents a physically reasonable soluble backbone; it does not repair distorted structures, add missing regions, or change global topology. It works best on high-quality, globular backbones and may perform poorly on highly flexible, low-confidence, or poorly packed models. For generating or refining backbones (e.g., AF2- or diffusion-based backbone sampling), other algorithms should be used upstream before sequence design.
- **Solubility-focused sequence priors**: This model is trained on soluble proteins (``MPNNModelTypes.SOLUBLE``) and biases surface composition away from transmembrane-like hydrophobicity. It is not suitable when you want to preserve native membrane-protein surface properties (e.g., for embedding in lipid bilayers); in those cases, use other ProteinMPNN variants (such as ``MPNNModelTypes.PROTEIN`` or membrane-labeled models) rather than this soluble design API.
- **Design scope and labels**: Residue-level control is limited to what you specify in ``params.fixed_residues``, ``params.redesigned_residues``, ``params.bias_AA``, and related per-residue options (such as ``params.bias_AA_per_residue``, ``params.omit_AA``, and ``params.omit_AA_per_residue``). The soluble model does not consume or enforce per-residue membrane labels (``transmembrane_buried``, ``transmembrane_interface``), so it is not optimal when you need fine-grained control over membrane vs. solvent exposure at individual positions.
- **Use cases where other models are preferable**: Soluble ProteinMPNN is not an embedding model, does not predict structures, and does not generate entirely new folds. It is suboptimal for: (1) large-scale screening where sequence ranking or structure prediction is the bottleneck; (2) tasks requiring evolutionary signal or fitness prediction; and (3) design of explicitly membrane-embedded proteins or complexes, where diffusion-based backbone generators, structure-prediction-guided design pipelines, or membrane-aware ProteinMPNN variants are generally more appropriate.

How We Use It
-------------

Soluble ProteinMPNN enables redesign of complex or membrane-derived backbones into soluble, expressible proteins that can be evaluated in silico alongside structure prediction models, 3D scoring, and downstream property predictors. Through scalable, standardized APIs, teams integrate soluble sequence redesign into loops that explore GPCR-like and other membrane folds as soluble scaffolds, generate soluble analogues of challenging targets, and iteratively filter redesigned variants for developability, stability, and biophysical profiles before synthesis.

- Integrates with backbone-generating and structure-validation models (e.g., AlphaFold-style workflows) to expand accessible soluble fold space for ligand binding, enzymology, and display systems.  
- Connects to BioLM predictive tools (such as structure-derived metrics, charge and hydrophobicity profiles, and thermostability proxies) so soluble variants can be triaged, prioritized, and tracked across multi-round engineering campaigns.

Related
-------

- ``AlphaFold2`` – Predicts 3D structures for Soluble ProteinMPNN-designed sequences, enabling in silico assessment of fold accuracy and backbone–sequence iteration workflows.
- ``ESM-IF1`` – Provides an alternative structure-conditioned sequence design model for the same backbone, useful for comparison or ensemble design alongside Soluble ProteinMPNN.
- ``ESMFold`` – Performs fast single-sequence structure prediction for redesigned proteins, enabling quick screening of Soluble ProteinMPNN outputs before more expensive modeling or experiments.
- ``ProteinMPNN`` – Backbone-conditioned sequence design model trained on mixed soluble/membrane data; serves as a baseline for comparing solubility-focused redesigns produced by Soluble ProteinMPNN.

References
----------

- Goverde, C. A., Pacesa, M., Dornfeld, L. J., Georgeon, S., Rosset, S., Dauparas, J., Schellhaas, C., Kozlov, S., Baker, D., Ovchinnikov, S., & Correia, B. E. (2023). Computational design of soluble analogues of integral membrane protein structures. *bioRxiv*. https://doi.org/10.1101/2023.05.09.540044
