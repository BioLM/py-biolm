Per-Residue Label Membrane MPNN API
===================================

Per-Residue Label Membrane MPNN is a CPU-accelerated variant of ProteinMPNN that designs amino acid sequences on fixed backbones while explicitly modeling membrane environments at per-residue resolution. Given a PDB structure, it samples sequences up to 1,024 residues per chain, with optional constraints on fixed/redesigned positions, residue-level transmembrane (buried/interface) labels, symmetry, and amino acid biases or omissions. Typical uses include engineering membrane proteins with region-specific solubility, stability, or interface properties.

Generate
--------

This endpoint gensats for Per-Residue Label Membrane MPNN.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="per-residue-label-membrane-mpnn",
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

            curl -X POST https://biolm.ai/api/v3/per-residue-label-membrane-mpnn/generate/ \
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

            url = "https://biolm.ai/api/v3/per-residue-label-membrane-mpnn/generate/"
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

            url <- "https://biolm.ai/api/v3/per-residue-label-membrane-mpnn/generate/"
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

.. http:post:: /api/v3/per-residue-label-membrane-mpnn/generate/

   Generate endpoint for Per-Residue Label Membrane MPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters for sequence generation:

        - **temperature** (*float*, default: 0.1) — Sampling temperature

        - **fixed_residues** (*array of strings*, default: []) — Residue specifications to keep fixed

        - **redesigned_residues** (*array of strings*, default: []) — Residue specifications to redesign

        - **bias_AA** (*object*, default: {}) — Per–amino-acid sampling bias; keys are single-letter amino acid codes, values are floats

        - **bias_AA_per_residue** (*object*, default: {}) — Per-residue amino-acid sampling bias; keys are residue specifications, values are objects mapping single-letter amino acid codes to floats

        - **omit_AA** (*string*, default: "") — Concatenated single-letter amino acid codes to omit globally

        - **omit_AA_per_residue** (*object*, default: {}) — Per-residue amino acids to omit; keys are residue specifications, values are strings of single-letter amino acid codes

        - **symmetry_residues** (*array of arrays of strings*, default: []) — Groups of residue specifications constrained by symmetry

        - **symmetry_weights** (*array of arrays of floats*, default: []) — Symmetry weights corresponding to ``symmetry_residues`` groups

        - **homo_oligomer** (*boolean*, default: False) — Homo-oligomer flag

        - **chains_to_design** (*array of strings*, default: []) — Chain IDs to include in design

        - **parse_these_chains_only** (*array of strings*, default: []) — Chain IDs to parse from the input PDB

        - **parse_atoms_with_zero_occupancy** (*boolean*, default: False) — Include atoms with zero occupancy

        - **number_of_batches** (*int*, range: 1-1, default: 1) — Number of batches to generate

        - **batch_size** (*int*, range: 1-2, default: 1) — Number of designs per batch

        - **repack_everything** (*boolean*, default: False, optional) — Side-chain repacking flag

        - **pack_side_chains** (*boolean*, default: False, optional) — Enable side-chain packing

        - **number_of_packs_per_design** (*int*, range: 1-8, default: 1, optional) — Number of packing runs per design

        - **sc_num_samples** (*int*, range: 1-64, default: 16, optional) — Number of side-chain samples per design

        - **sc_num_denoising_steps** (*int*, range: 1-10, default: 3, optional) — Number of denoising steps for side-chain sampling

        - **force_hetatm** (*boolean*, default: False, optional) — Force inclusion of HETATM records

        - **pack_with_ligand_context** (*boolean*, default: True, optional) — Use ligand context for packing

        - **fasta_seq_separation** (*string*, default: ":", optional) — FASTA sequence separation character

        - **file_ending** (*string*, default: "", optional) — File ending tag

        - **zero_indexed** (*int*, default: 0, optional) — Residue indexing mode

        - **pdb_path** (*null*, fixed) — Unused field

        - **redesigned_residues_multi** (*null*, fixed) — Unused field

        - **fixed_residues_multi** (*null*, fixed) — Unused field

        - **bias_AA_per_residue_multi** (*null*, fixed) — Unused field

        - **omit_AA_per_residue_multi** (*null*, fixed) — Unused field

        - **save_stats** (*null*, fixed) — Unused field

        - **verbose** (*boolean*, default: True, optional) — Verbosity flag

        - **ligand_mpnn_use_side_chain_context** (*null*, fixed) — Unused field

        - **ligand_mpnn_use_atom_context** (*boolean*, default: True, optional) — Use atom-level context for ligand-aware models

        - **ligand_mpnn_cutoff_for_score** (*float*, default: 8.0, optional) — Distance cutoff for ligand scoring

        - **global_transmembrane_label** (*string*, allowed: "membrane", "soluble", default: "soluble", optional) — Global transmembrane label

        - **transmembrane_buried** (*array of strings*, default: null, optional) — Residue specifications labeled as buried transmembrane

        - **transmembrane_interface** (*array of strings*, default: null, optional) — Residue specifications labeled as transmembrane interface


      - **items** (*array of objects*, min: 1, max: 1, required) --- Input structures:

        - **pdb** (*string*, min length: 1, max length: ``max_pdb_str_len``, required) — PDB-format structure string validated for syntax and content

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/per-residue-label-membrane-mpnn/generate/ HTTP/1.1
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

        - **sequence** (*string*) — Designed amino acid sequence in 1-letter codes

        - **pdb** (*string*) — PDB-formatted structure associated with the designed sequence, including ATOM/HETATM records

        - **overall_confidence** (*float*) — Design confidence score, dimensionless

        - **ligand_confidence** (*float*) — Ligand-context confidence score, dimensionless

        - **seq_rec** (*float*, range: 0.0–100.0) — Sequence recovery percentage, units: percent

        - **log_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-position log-probabilities over 20 amino acids, where L is sequence length

        - **sampling_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-position sampling probabilities over 20 amino acids, where L is sequence length


      - **results** (*array of objects, side-chain variant*) --- One result per input item, in the order requested:

        - **sequence** (*string*) — Designed amino acid sequence in 1-letter codes

        - **pdb** (*string*) — PDB-formatted structure associated with the designed sequence, including ATOM/HETATM records

        - **overall_confidence** (*float*) — Design confidence score, dimensionless

        - **ligand_confidence** (*float*) — Ligand-context confidence score, dimensionless

        - **seq_rec** (*float*, range: 0.0–100.0) — Sequence recovery percentage, units: percent

        - **log_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-position log-probabilities over 20 amino acids, where L is sequence length

        - **sampling_probs** (*array of arrays of floats*, shape: [L, 20]) — Per-position sampling probabilities over 20 amino acids, where L is sequence length

        - **pdb_packed** (*object*) — Side-chain–repacked structures per chain

          - **<chain_id>** (*string*) — Repacked structure for the given chain identifier in PDB format

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "V",
            "pdb": "REMARK Selection '(backbone) and ...occupancy > 0))'\nATOM      1  N   VAL A   1      11.104  13.207  11.947  1.00  0.11           N  \nATOM      2  CA  VAL A   1      12.560  13.051  11.824  1.00  0.11... (truncated for documentation)",
            "overall_confidence": 0.1101,
            "ligand_confidence": 0.1101,
            "seq_rec": 0.0,
            "log_probs": [
              [
                -2.52133846282959,
                -4.525322914123535,
                "... (truncated for documentation)"
              ]
            ],
            "sampling_probs": [
              [
                0.017263982445001602,
                3.419389077441437e-11,
                "... (truncated for documentation)"
              ]
            ]
          }
        ]
      }


Performance
-----------

- Model architecture and conditioning:
  
  - Variant of ``ProteinMPNN`` sharing the same graph neural network core, with additional per-residue conditioning channels for ``transmembrane_buried`` and ``transmembrane_interface`` annotations
  - Per-residue labels are validated against the input backbone and propagated through the graph, adding <10% computational overhead relative to base ``ProteinMPNN`` for typical protein sizes

- Relative speed and throughput:
  
  - Inference speed is effectively the same order as ``ProteinMPNN`` and ``Soluble ProteinMPNN`` on the same backbone, with very similar memory footprint and FLOP usage
  - Substantially faster and cheaper than structure-prediction–based redesign workflows (for example, running ``AlphaFold2`` or ``ESMFold`` on many sequence variants for the same backbone)

- Comparative design behavior:
  
  - Compared to ``Global Label Membrane MPNN``, per-residue labeling yields higher effective design accuracy at membrane interfaces (more realistic enrichment of interface aromatics/charges and buried hydrophobics), at the cost of a modest runtime increase
  - Compared to base ``ProteinMPNN`` on membrane-exposed regions, per-residue conditioning reduces over-solubilization and better recovers hydrophobic cores and amphipathic patterns in predicted transmembrane segments
  - Compared to ``Soluble ProteinMPNN`` on backbones containing membrane-like segments, it better preserves membrane-adapted packing in buried regions while allowing explicit solubilization or reweighting at user-labeled interface positions

- Scoring, calibration, and deployment characteristics:
  
  - ``overall_confidence`` is directly comparable to scores from ``ProteinMPNN`` and ``Soluble ProteinMPNN`` and can be used to rank sequences within or across design runs
  - ``sampling_probs`` and ``log_probs`` are typically sharper in membrane-buried segments than for soluble-only models, reflecting stronger learned constraints in lipid-exposed cores
  - Runs efficiently on the same GPU configurations as other MPNN-family models; when co-deployed with ``ProteinMPNN`` and ``LigandMPNN``, per-GPU throughput is effectively identical, and CPU-only execution remains feasible for small systems though GPU is preferred for large design campaigns

Applications
------------

- Rapid assessment of transmembrane vs soluble exposure in designed proteins by assigning per-residue membrane/soluble labels to 3D backbones, enabling protein engineers to see which positions the Membrane MPNN model treats as membrane-facing vs solvent-facing and prioritize mutations or redesign of problematic patches for expression, aggregation resistance, and formulation robustness
- Design of soluble analogues of membrane protein scaffolds for screening and mechanistic assays by mapping per-residue membrane propensities on GPCRs, ion channels, or transporters and then programmatically inverting hydrophobic/hydrophilic patterns at positions labeled as membrane-exposed, enabling panel designs for high-throughput ligand screening or structural biology without detergents or reconstitution (not intended to preserve native signaling or transport function outside membranes)
- Targeted re-embedding and re-parameterization of membrane-facing residues in computational protein redesign pipelines by using per-residue labels as constraints or weights when calling Membrane MPNN for sequence optimization, so residues classified as membrane-buried retain appropriate hydrophobic character while loop or extracellular residues are diversified for stability, manufacturability, or binding interface engineering in GPCR-like or rhomboid-like scaffolds
- Automated quality control of generative backbone designs for complex helical bundles and β-barrels by running the per-residue label model on candidate structures to detect inconsistent patterns (for example, polar residues predicted as membrane-exposed across large surfaces, or strongly membrane-like segments in an intended soluble scaffold), helping teams filter or flag designs before downstream modeling, synthesis, and expression; not optimal for very small or highly disordered proteins where membrane vs soluble segmentation is intrinsically ambiguous
- Integration of membrane-region annotations into large-scale ML-driven protein engineering workflows where organizations maintain libraries of real or designed membrane proteins (such as receptors, transporters, or membrane enzymes) and use per-residue labels to drive feature engineering (for example, region-specific mutational constraints or region-aware stability predictors), improving robustness and interpretability of design–build–test–learn cycles across diverse membrane topologies

Limitations
-----------

- **Maximum sequence length**: Backbones with more than 1024 residues per chain are not supported. Any PDB input where a chain exceeds this will be rejected because ``max_sequence_len`` is fixed at ``1024`` for this model.
- **Batching and throughput limits**: Each request may contain at most one design item (``items`` has ``max_items=1``), and sequence generation is limited to ``batch_size <= 2`` and ``number_of_batches <= 1``. This API is therefore optimized for per‑structure design, not for very high‑throughput screening of hundreds of backbones in a single call.
- **Membrane label specification**: Per‑residue membrane labeling must be provided as residue identifiers (for example, ``A15`` or ``B120A``) in the ``transmembrane_buried`` and ``transmembrane_interface`` lists. These identifiers must correspond to existing chains and residue indices in the uploaded ``pdb``; labels outside the detected chain lengths, invalid chain IDs, or malformed residue strings will cause validation errors and no design will be produced.
- **Backbone‑fixed design only**: The model assumes the input ``pdb`` backbone is already a suitable target structure. It does not relax the backbone, redesign topology, or assess foldability in a membrane vs. soluble context. Non‑physical backbones, misaligned transmembrane segments, or unrealistic per‑residue labels can yield low‑quality or misleading sequences even if the API call succeeds.
- **Membrane‑aware, not generic**: This checkpoint (``per_residue_label_membrane``) is specialized for designs where residue‑level transmembrane vs. interface annotations matter (for example, tuning exposed vs. buried residues across a bilayer). For generic soluble design, ligand‑context design, or global membrane tagging tasks, the ``soluble``, ``protein``, ``ligand``, or ``global_label_membrane`` variants are typically more appropriate and computationally simpler.
- **No guarantee of experimental behavior**: Output sequences (``sequence``, ``log_probs``, ``sampling_probs`` and the redesigned ``pdb``) are optimized for compatibility with the provided backbone and labels, not for expression yield, aggregation resistance, binding, or activity in vitro or in vivo. For applications such as enzyme design, GPCR‑like receptor engineering, or solubilizing complex membrane folds, this model should be used within a broader design and validation pipeline (for example, downstream structure prediction, stability filters, and experimental screening).

How We Use It
-------------

Per-Residue Label Membrane MPNN enables data teams and protein engineers to redesign membrane-derived folds into soluble or otherwise context-optimized variants as a standardized API step within larger ML-driven design pipelines. By integrating residue-level membrane/solvent labeling with structure-aware sequence design, it complements backbone generation (e.g., AF2- or diffusion-based designs), structure prediction, and sequence encoders, so teams can systematically tune surface hydrophobicity, burial, and interface exposure on complex topologies such as GPCR-, rhomboid-, or claudin-like folds. Typical use cases include converting integral membrane backbones into soluble surrogates for biophysics and screening, or re-optimizing per-residue environments on designed receptors to improve expression, manufacturability, or assay compatibility, with redesigned panels flowing directly into iterative wet-lab testing.

- Integrates with generative backbone models, AlphaFold-like predictors, and sequence-embedding services to form automated design–predict–filter loops for membrane and membrane-derived targets.  
- Supports multi-objective campaigns (stability, solubility, manufacturability, epitope exposure) by enabling per-residue-aware redesigns that can be ranked and combined with downstream structure-, sequence-, and physics-based scoring before experimental screening.

Related
-------

- ``Soluble ProteinMPNN`` – Backbone-conditioned sequence design model trained only on soluble proteins; combine with per-residue membrane labels to redesign transmembrane regions for aqueous environments or to reduce surface hydrophobicity.
- ``Global Label Membrane MPNN`` – Uses a single global membrane/soluble label to bias designs; complements per-residue labels for enforcing overall membrane compatibility while still allowing local, residue-level control.
- ``ProteinMPNN`` – General-purpose backbone-conditioned sequence design model underlying the membrane label variants; use per-residue labels to guide or filter ProteinMPNN designs toward desired burial or membrane exposure patterns.
- ``LigandMPNN`` – Ligand-aware MPNN variant for designing binding sites; pair with per-residue membrane labels to tune pocket residues in membrane-like environments while keeping non-binding surfaces compatible with solvent exposure.

References
----------

- Goverde, C. A., Pacesa, M., Dornfeld, L. J., Georgeon, S., Rosset, S., Dauparas, J., Schellhaas, C., Kozlov, S., Baker, D., Ovchinnikov, S., & Correia, B. E. (2023). Computational design of soluble analogues of integral membrane protein structures. *bioRxiv*. https://doi.org/10.1101/2023.05.09.540044
