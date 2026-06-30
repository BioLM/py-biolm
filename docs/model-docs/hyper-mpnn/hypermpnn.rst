HyperMPNN API
=============

HyperMPNN is a retrained ProteinMPNN-style inverse folding model optimized on AlphaFold-predicted hyperthermophilic proteomes to design protein sequences with thermostability-associated amino acid biases (more apolar cores, more positively charged surfaces). The API accepts fixed backbones as single- or multi-chain PDB strings and returns designed sequences, global scores, and per-residue log-probabilities, supporting batched, CPU-based inference for high-throughput redesign of enzymes, scaffolds, and protein nanoparticles beyond mesophilic baselines.

Generate
--------

HyperMPNN-style ProteinMPNN request to generate thermostable sequence designs for a single-chain soluble protein, with chain A biased toward hydrophobic and charged residues typical of hyperthermophiles, explicit fixed/redesigned residues, symmetry constraints, and per-residue transmembrane annotations.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="hyper-mpnn",
                action="generate",
                params={
                  "temperature": 0.3,
                  "fixed_residues": [
                    "A3",
                    "A4",
                    "A5"
                  ],
                  "redesigned_residues": [
                    "A8",
                    "A9",
                    "A10"
                  ],
                  "bias_AA": {
                    "A": 0.5,
                    "V": 0.4,
                    "L": 0.4,
                    "E": -0.2,
                    "D": -0.2
                  },
                  "bias_AA_per_residue": {
                    "A7": {
                      "K": 0.8,
                      "R": 0.7
                    },
                    "A9": {
                      "I": 0.6,
                      "V": 0.6
                    }
                  },
                  "omit_AA": "C",
                  "omit_AA_per_residue": {
                    "A6": "P",
                    "A10": "G"
                  },
                  "symmetry_residues": [
                    [
                      "A2",
                      "A9"
                    ],
                    [
                      "A3",
                      "A8"
                    ]
                  ],
                  "symmetry_weights": [
                    [
                      1.0,
                      1.0
                    ],
                    [
                      0.8,
                      0.8
                    ]
                  ],
                  "homo_oligomer": false,
                  "chains_to_design": [
                    "A"
                  ],
                  "parse_these_chains_only": [
                    "A"
                  ],
                  "parse_atoms_with_zero_occupancy": false,
                  "number_of_batches": 1,
                  "batch_size": 2,
                  "repack_everything": false,
                  "pack_side_chains": false,
                  "number_of_packs_per_design": 2,
                  "sc_num_samples": 16,
                  "sc_num_denoising_steps": 3,
                  "force_hetatm": false,
                  "pack_with_ligand_context": false,
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
                  "transmembrane_buried": [
                    "A2",
                    "A3"
                  ],
                  "transmembrane_interface": [
                    "A8",
                    "A9"
                  ]
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N  \nATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00 20.00           C  \nATOM      3  C   MET A   1      12.100   9.500  11.000  1.00 20.00           C  \nATOM      4  O   MET A   1      11.900   8.300  10.800  1.00 20.00           O  \nATOM      5  N   ALA A   2      13.100  10.000  11.800  1.00 20.00           N  \nATOM      6  CA  ALA A   2      14.100   9.200  12.500  1.00 20.00           C  \nATOM      7  C   ALA A   2      15.300   8.900  11.600  1.00 20.00           C  \nATOM      8  O   ALA A   2      16.200   8.100  11.900  1.00 20.00           O  \nATOM      9  N   LYS A   3      15.300   9.600  10.500  1.00 20.00           N  \nATOM     10  CA  LYS A   3      16.400   9.400   9.600  1.00 20.00           C  \nATOM     11  C   LYS A   3      17.700   9.900  10.200  1.00 20.00           C  \nATOM     12  O   LYS A   3      18.700   9.200   9.900  1.00 20.00           O  \nATOM     13  N   GLU A   4      17.700  11.000  11.000  1.00 20.00           N  \nATOM     14  CA  GLU A   4      18.800  11.600  11.800  1.00 20.00           C  \nATOM     15  C   GLU A   4      18.300  12.100  13.200  1.00 20.00           C  \nATOM     16  O   GLU A   4      19.000  12.900  13.800  1.00 20.00           O  \nATOM     17  N   ILE A   5      17.100  11.700  13.700  1.00 20.00           N  \nATOM     18  CA  ILE A   5      16.500  12.000  15.000  1.00 20.00           C  \nATOM     19  C   ILE A   5      16.900  10.900  16.000  1.00 20.00           C  \nATOM     20  O   ILE A   5      17.400  11.100  17.100  1.00 20.00           O  \nATOM     21  N   ARG A   6      16.700   9.700  15.600  1.00 20.00           N  \nATOM     22  CA  ARG A   6      17.000   8.600  16.500  1.00 20.00           C  \nATOM     23  C   ARG A   6      18.500   8.600  16.800  1.00 20.00           C  \nATOM     24  O   ARG A   6      19.000   7.700  17.400  1.00 20.00           O  \nATOM     25  N   VAL A   7      19.200   9.600  16.400  1.00 20.00           N  \nATOM     26  CA  VAL A   7      20.600   9.700  16.700  1.00 20.00           C  \nATOM     27  C   VAL A   7      21.200   8.500  17.500  1.00 20.00           C  \nATOM     28  O   VAL A   7      22.400   8.500  17.700  1.00 20.00           O  \nATOM     29  N   LYS A   8      20.400   7.600  18.100  1.00 20.00           N  \nATOM     30  CA  LYS A   8      20.800   6.400  18.900  1.00 20.00           C  \nATOM     31  C   LYS A   8      21.000   5.200  18.000  1.00 20.00           C  \nATOM     32  O   LYS A   8      21.900   4.400  18.300  1.00 20.00           O  \nATOM     33  N   LEU A   9      20.100   5.000  17.000  1.00 20.00           N  \nATOM     34  CA  LEU A   9      20.100   3.900  16.000  1.00 20.00           C  \nATOM     35  C   LEU A   9      21.400   3.500  15.300  1.00 20.00           C  \nATOM     36  O   LEU A   9      21.600   2.300  15.100  1.00 20.00           O  \nATOM     37  N   GLU A  10      22.200   4.400  14.900  1.00 20.00           N  \nATOM     38  CA  GLU A  10      23.500   4.200  14.300  1.00 20.00           C  \nATOM     39  C   GLU A  10      24.500   4.900  15.200  1.00 20.00           C  \nATOM     40  O   GLU A  10      25.700   4.700  15.000  1.00 20.00           O  \nTER      41      GLU A  10                                                              \nEND                                                                             "
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/hyper-mpnn/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.3,
                "fixed_residues": [
                  "A3",
                  "A4",
                  "A5"
                ],
                "redesigned_residues": [
                  "A8",
                  "A9",
                  "A10"
                ],
                "bias_AA": {
                  "A": 0.5,
                  "V": 0.4,
                  "L": 0.4,
                  "E": -0.2,
                  "D": -0.2
                },
                "bias_AA_per_residue": {
                  "A7": {
                    "K": 0.8,
                    "R": 0.7
                  },
                  "A9": {
                    "I": 0.6,
                    "V": 0.6
                  }
                },
                "omit_AA": "C",
                "omit_AA_per_residue": {
                  "A6": "P",
                  "A10": "G"
                },
                "symmetry_residues": [
                  [
                    "A2",
                    "A9"
                  ],
                  [
                    "A3",
                    "A8"
                  ]
                ],
                "symmetry_weights": [
                  [
                    1.0,
                    1.0
                  ],
                  [
                    0.8,
                    0.8
                  ]
                ],
                "homo_oligomer": false,
                "chains_to_design": [
                  "A"
                ],
                "parse_these_chains_only": [
                  "A"
                ],
                "parse_atoms_with_zero_occupancy": false,
                "number_of_batches": 1,
                "batch_size": 2,
                "repack_everything": false,
                "pack_side_chains": false,
                "number_of_packs_per_design": 2,
                "sc_num_samples": 16,
                "sc_num_denoising_steps": 3,
                "force_hetatm": false,
                "pack_with_ligand_context": false,
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
                "transmembrane_buried": [
                  "A2",
                  "A3"
                ],
                "transmembrane_interface": [
                  "A8",
                  "A9"
                ]
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N  \nATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00 20.00           C  \nATOM      3  C   MET A   1      12.100   9.500  11.000  1.00 20.00           C  \nATOM      4  O   MET A   1      11.900   8.300  10.800  1.00 20.00           O  \nATOM      5  N   ALA A   2      13.100  10.000  11.800  1.00 20.00           N  \nATOM      6  CA  ALA A   2      14.100   9.200  12.500  1.00 20.00           C  \nATOM      7  C   ALA A   2      15.300   8.900  11.600  1.00 20.00           C  \nATOM      8  O   ALA A   2      16.200   8.100  11.900  1.00 20.00           O  \nATOM      9  N   LYS A   3      15.300   9.600  10.500  1.00 20.00           N  \nATOM     10  CA  LYS A   3      16.400   9.400   9.600  1.00 20.00           C  \nATOM     11  C   LYS A   3      17.700   9.900  10.200  1.00 20.00           C  \nATOM     12  O   LYS A   3      18.700   9.200   9.900  1.00 20.00           O  \nATOM     13  N   GLU A   4      17.700  11.000  11.000  1.00 20.00           N  \nATOM     14  CA  GLU A   4      18.800  11.600  11.800  1.00 20.00           C  \nATOM     15  C   GLU A   4      18.300  12.100  13.200  1.00 20.00           C  \nATOM     16  O   GLU A   4      19.000  12.900  13.800  1.00 20.00           O  \nATOM     17  N   ILE A   5      17.100  11.700  13.700  1.00 20.00           N  \nATOM     18  CA  ILE A   5      16.500  12.000  15.000  1.00 20.00           C  \nATOM     19  C   ILE A   5      16.900  10.900  16.000  1.00 20.00           C  \nATOM     20  O   ILE A   5      17.400  11.100  17.100  1.00 20.00           O  \nATOM     21  N   ARG A   6      16.700   9.700  15.600  1.00 20.00           N  \nATOM     22  CA  ARG A   6      17.000   8.600  16.500  1.00 20.00           C  \nATOM     23  C   ARG A   6      18.500   8.600  16.800  1.00 20.00           C  \nATOM     24  O   ARG A   6      19.000   7.700  17.400  1.00 20.00           O  \nATOM     25  N   VAL A   7      19.200   9.600  16.400  1.00 20.00           N  \nATOM     26  CA  VAL A   7      20.600   9.700  16.700  1.00 20.00           C  \nATOM     27  C   VAL A   7      21.200   8.500  17.500  1.00 20.00           C  \nATOM     28  O   VAL A   7      22.400   8.500  17.700  1.00 20.00           O  \nATOM     29  N   LYS A   8      20.400   7.600  18.100  1.00 20.00           N  \nATOM     30  CA  LYS A   8      20.800   6.400  18.900  1.00 20.00           C  \nATOM     31  C   LYS A   8      21.000   5.200  18.000  1.00 20.00           C  \nATOM     32  O   LYS A   8      21.900   4.400  18.300  1.00 20.00           O  \nATOM     33  N   LEU A   9      20.100   5.000  17.000  1.00 20.00           N  \nATOM     34  CA  LEU A   9      20.100   3.900  16.000  1.00 20.00           C  \nATOM     35  C   LEU A   9      21.400   3.500  15.300  1.00 20.00           C  \nATOM     36  O   LEU A   9      21.600   2.300  15.100  1.00 20.00           O  \nATOM     37  N   GLU A  10      22.200   4.400  14.900  1.00 20.00           N  \nATOM     38  CA  GLU A  10      23.500   4.200  14.300  1.00 20.00           C  \nATOM     39  C   GLU A  10      24.500   4.900  15.200  1.00 20.00           C  \nATOM     40  O   GLU A  10      25.700   4.700  15.000  1.00 20.00           O  \nTER      41      GLU A  10                                                              \nEND                                                                             "
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/hyper-mpnn/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.3,
                    "fixed_residues": [
                      "A3",
                      "A4",
                      "A5"
                    ],
                    "redesigned_residues": [
                      "A8",
                      "A9",
                      "A10"
                    ],
                    "bias_AA": {
                      "A": 0.5,
                      "V": 0.4,
                      "L": 0.4,
                      "E": -0.2,
                      "D": -0.2
                    },
                    "bias_AA_per_residue": {
                      "A7": {
                        "K": 0.8,
                        "R": 0.7
                      },
                      "A9": {
                        "I": 0.6,
                        "V": 0.6
                      }
                    },
                    "omit_AA": "C",
                    "omit_AA_per_residue": {
                      "A6": "P",
                      "A10": "G"
                    },
                    "symmetry_residues": [
                      [
                        "A2",
                        "A9"
                      ],
                      [
                        "A3",
                        "A8"
                      ]
                    ],
                    "symmetry_weights": [
                      [
                        1.0,
                        1.0
                      ],
                      [
                        0.8,
                        0.8
                      ]
                    ],
                    "homo_oligomer": false,
                    "chains_to_design": [
                      "A"
                    ],
                    "parse_these_chains_only": [
                      "A"
                    ],
                    "parse_atoms_with_zero_occupancy": false,
                    "number_of_batches": 1,
                    "batch_size": 2,
                    "repack_everything": false,
                    "pack_side_chains": false,
                    "number_of_packs_per_design": 2,
                    "sc_num_samples": 16,
                    "sc_num_denoising_steps": 3,
                    "force_hetatm": false,
                    "pack_with_ligand_context": false,
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
                    "transmembrane_buried": [
                      "A2",
                      "A3"
                    ],
                    "transmembrane_interface": [
                      "A8",
                      "A9"
                    ]
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N  \nATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00 20.00           C  \nATOM      3  C   MET A   1      12.100   9.500  11.000  1.00 20.00           C  \nATOM      4  O   MET A   1      11.900   8.300  10.800  1.00 20.00           O  \nATOM      5  N   ALA A   2      13.100  10.000  11.800  1.00 20.00           N  \nATOM      6  CA  ALA A   2      14.100   9.200  12.500  1.00 20.00           C  \nATOM      7  C   ALA A   2      15.300   8.900  11.600  1.00 20.00           C  \nATOM      8  O   ALA A   2      16.200   8.100  11.900  1.00 20.00           O  \nATOM      9  N   LYS A   3      15.300   9.600  10.500  1.00 20.00           N  \nATOM     10  CA  LYS A   3      16.400   9.400   9.600  1.00 20.00           C  \nATOM     11  C   LYS A   3      17.700   9.900  10.200  1.00 20.00           C  \nATOM     12  O   LYS A   3      18.700   9.200   9.900  1.00 20.00           O  \nATOM     13  N   GLU A   4      17.700  11.000  11.000  1.00 20.00           N  \nATOM     14  CA  GLU A   4      18.800  11.600  11.800  1.00 20.00           C  \nATOM     15  C   GLU A   4      18.300  12.100  13.200  1.00 20.00           C  \nATOM     16  O   GLU A   4      19.000  12.900  13.800  1.00 20.00           O  \nATOM     17  N   ILE A   5      17.100  11.700  13.700  1.00 20.00           N  \nATOM     18  CA  ILE A   5      16.500  12.000  15.000  1.00 20.00           C  \nATOM     19  C   ILE A   5      16.900  10.900  16.000  1.00 20.00           C  \nATOM     20  O   ILE A   5      17.400  11.100  17.100  1.00 20.00           O  \nATOM     21  N   ARG A   6      16.700   9.700  15.600  1.00 20.00           N  \nATOM     22  CA  ARG A   6      17.000   8.600  16.500  1.00 20.00           C  \nATOM     23  C   ARG A   6      18.500   8.600  16.800  1.00 20.00           C  \nATOM     24  O   ARG A   6      19.000   7.700  17.400  1.00 20.00           O  \nATOM     25  N   VAL A   7      19.200   9.600  16.400  1.00 20.00           N  \nATOM     26  CA  VAL A   7      20.600   9.700  16.700  1.00 20.00           C  \nATOM     27  C   VAL A   7      21.200   8.500  17.500  1.00 20.00           C  \nATOM     28  O   VAL A   7      22.400   8.500  17.700  1.00 20.00           O  \nATOM     29  N   LYS A   8      20.400   7.600  18.100  1.00 20.00           N  \nATOM     30  CA  LYS A   8      20.800   6.400  18.900  1.00 20.00           C  \nATOM     31  C   LYS A   8      21.000   5.200  18.000  1.00 20.00           C  \nATOM     32  O   LYS A   8      21.900   4.400  18.300  1.00 20.00           O  \nATOM     33  N   LEU A   9      20.100   5.000  17.000  1.00 20.00           N  \nATOM     34  CA  LEU A   9      20.100   3.900  16.000  1.00 20.00           C  \nATOM     35  C   LEU A   9      21.400   3.500  15.300  1.00 20.00           C  \nATOM     36  O   LEU A   9      21.600   2.300  15.100  1.00 20.00           O  \nATOM     37  N   GLU A  10      22.200   4.400  14.900  1.00 20.00           N  \nATOM     38  CA  GLU A  10      23.500   4.200  14.300  1.00 20.00           C  \nATOM     39  C   GLU A  10      24.500   4.900  15.200  1.00 20.00           C  \nATOM     40  O   GLU A  10      25.700   4.700  15.000  1.00 20.00           O  \nTER      41      GLU A  10                                                              \nEND                                                                             "
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/hyper-mpnn/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.3,
                fixed_residues = list(
                  "A3",
                  "A4",
                  "A5"
                ),
                redesigned_residues = list(
                  "A8",
                  "A9",
                  "A10"
                ),
                bias_AA = list(
                  A = 0.5,
                  V = 0.4,
                  L = 0.4,
                  E = -0.2,
                  D = -0.2
                ),
                bias_AA_per_residue = list(
                  A7 = list(
                    K = 0.8,
                    R = 0.7
                  ),
                  A9 = list(
                    I = 0.6,
                    V = 0.6
                  )
                ),
                omit_AA = "C",
                omit_AA_per_residue = list(
                  A6 = "P",
                  A10 = "G"
                ),
                symmetry_residues = list(
                  list(
                    "A2",
                    "A9"
                  ),
                  list(
                    "A3",
                    "A8"
                  )
                ),
                symmetry_weights = list(
                  list(
                    1.0,
                    1.0
                  ),
                  list(
                    0.8,
                    0.8
                  )
                ),
                homo_oligomer = FALSE,
                chains_to_design = list(
                  "A"
                ),
                parse_these_chains_only = list(
                  "A"
                ),
                parse_atoms_with_zero_occupancy = FALSE,
                number_of_batches = 1,
                batch_size = 2,
                repack_everything = FALSE,
                pack_side_chains = FALSE,
                number_of_packs_per_design = 2,
                sc_num_samples = 16,
                sc_num_denoising_steps = 3,
                force_hetatm = FALSE,
                pack_with_ligand_context = FALSE,
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
                transmembrane_buried = list(
                  "A2",
                  "A3"
                ),
                transmembrane_interface = list(
                  "A8",
                  "A9"
                )
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N  
            ATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00 20.00           C  
            ATOM      3  C   MET A   1      12.100   9.500  11.000  1.00 20.00           C  
            ATOM      4  O   MET A   1      11.900   8.300  10.800  1.00 20.00           O  
            ATOM      5  N   ALA A   2      13.100  10.000  11.800  1.00 20.00           N  
            ATOM      6  CA  ALA A   2      14.100   9.200  12.500  1.00 20.00           C  
            ATOM      7  C   ALA A   2      15.300   8.900  11.600  1.00 20.00           C  
            ATOM      8  O   ALA A   2      16.200   8.100  11.900  1.00 20.00           O  
            ATOM      9  N   LYS A   3      15.300   9.600  10.500  1.00 20.00           N  
            ATOM     10  CA  LYS A   3      16.400   9.400   9.600  1.00 20.00           C  
            ATOM     11  C   LYS A   3      17.700   9.900  10.200  1.00 20.00           C  
            ATOM     12  O   LYS A   3      18.700   9.200   9.900  1.00 20.00           O  
            ATOM     13  N   GLU A   4      17.700  11.000  11.000  1.00 20.00           N  
            ATOM     14  CA  GLU A   4      18.800  11.600  11.800  1.00 20.00           C  
            ATOM     15  C   GLU A   4      18.300  12.100  13.200  1.00 20.00           C  
            ATOM     16  O   GLU A   4      19.000  12.900  13.800  1.00 20.00           O  
            ATOM     17  N   ILE A   5      17.100  11.700  13.700  1.00 20.00           N  
            ATOM     18  CA  ILE A   5      16.500  12.000  15.000  1.00 20.00           C  
            ATOM     19  C   ILE A   5      16.900  10.900  16.000  1.00 20.00           C  
            ATOM     20  O   ILE A   5      17.400  11.100  17.100  1.00 20.00           O  
            ATOM     21  N   ARG A   6      16.700   9.700  15.600  1.00 20.00           N  
            ATOM     22  CA  ARG A   6      17.000   8.600  16.500  1.00 20.00           C  
            ATOM     23  C   ARG A   6      18.500   8.600  16.800  1.00 20.00           C  
            ATOM     24  O   ARG A   6      19.000   7.700  17.400  1.00 20.00           O  
            ATOM     25  N   VAL A   7      19.200   9.600  16.400  1.00 20.00           N  
            ATOM     26  CA  VAL A   7      20.600   9.700  16.700  1.00 20.00           C  
            ATOM     27  C   VAL A   7      21.200   8.500  17.500  1.00 20.00           C  
            ATOM     28  O   VAL A   7      22.400   8.500  17.700  1.00 20.00           O  
            ATOM     29  N   LYS A   8      20.400   7.600  18.100  1.00 20.00           N  
            ATOM     30  CA  LYS A   8      20.800   6.400  18.900  1.00 20.00           C  
            ATOM     31  C   LYS A   8      21.000   5.200  18.000  1.00 20.00           C  
            ATOM     32  O   LYS A   8      21.900   4.400  18.300  1.00 20.00           O  
            ATOM     33  N   LEU A   9      20.100   5.000  17.000  1.00 20.00           N  
            ATOM     34  CA  LEU A   9      20.100   3.900  16.000  1.00 20.00           C  
            ATOM     35  C   LEU A   9      21.400   3.500  15.300  1.00 20.00           C  
            ATOM     36  O   LEU A   9      21.600   2.300  15.100  1.00 20.00           O  
            ATOM     37  N   GLU A  10      22.200   4.400  14.900  1.00 20.00           N  
            ATOM     38  CA  GLU A  10      23.500   4.200  14.300  1.00 20.00           C  
            ATOM     39  C   GLU A  10      24.500   4.900  15.200  1.00 20.00           C  
            ATOM     40  O   GLU A  10      25.700   4.700  15.000  1.00 20.00           O  
            TER      41      GLU A  10                                                              
            END                                                                             "
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/hyper-mpnn/generate/

   Generate endpoint for HyperMPNN.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters for HyperMPNN generation:

        - **temperature** (*float*, default: 0.1) — Sampling temperature

        - **fixed_residues** (*array of strings*, default: empty array) — Residue specification strings "[ChainID][ResidueNumber][OptionalInsertionCode]" to keep fixed; each must exist in the input PDB

        - **redesigned_residues** (*array of strings*, default: empty array) — Residue specification strings "[ChainID][ResidueNumber][OptionalInsertionCode]" to redesign; each must exist in the input PDB

        - **bias_AA** (*object*, default: empty object) — Global amino-acid bias map (keys: single-letter unambiguous amino-acid codes, values: float weights)

        - **bias_AA_per_residue** (*object*, default: empty object) — Per-residue amino-acid bias map:

          - **<residue_spec>** (*object*) — Key is a residue specification string "[ChainID][ResidueNumber][OptionalInsertionCode]" that must exist in the input PDB

            - **<aa_code>** (*float*) — Key is a single-letter unambiguous amino-acid code, value is a float weight

        - **omit_AA** (*string*, default: "", allowed chars: unambiguous amino-acid codes) — Global string of amino acids to omit

        - **omit_AA_per_residue** (*object*, default: empty object) — Per-residue amino acids to omit:

          - **<residue_spec>** (*string*) — Residue specification string "[ChainID][ResidueNumber][OptionalInsertionCode]" that must exist in the input PDB, mapped to a string of unambiguous amino-acid codes

        - **symmetry_residues** (*array of arrays of strings*, default: empty array) — Groups of residue specification strings "[ChainID][ResidueNumber][OptionalInsertionCode]" that must exist in the input PDB

        - **symmetry_weights** (*array of arrays of floats*, default: empty array) — Symmetry weights corresponding to groups in symmetry_residues; each inner array length must match the corresponding symmetry_residues group

        - **homo_oligomer** (*boolean*, default: false) — Homo-oligomer design flag

        - **chains_to_design** (*array of strings*, default: empty array) — Chain IDs to design; each must exist in the input PDB

        - **parse_these_chains_only** (*array of strings*, default: empty array) — Chain IDs to parse from the input PDB; each must exist in the input PDB

        - **parse_atoms_with_zero_occupancy** (*boolean*, default: false) — Whether atoms with zero occupancy are parsed from the PDB

        - **number_of_batches** (*int*, range: 1-48, default: 1) — Number of design batches

        - **batch_size** (*int*, range: 1-1000, default: 1) — Number of designs per batch

        - **repack_everything** (*boolean or null*, default: false) — Whether all residues are repacked in side-chain mode

        - **pack_side_chains** (*boolean or null*, default: false) — Whether side chains are packed in side-chain mode

        - **number_of_packs_per_design** (*int or null*, range: 1-8, default: 1) — Number of packing runs per design in side-chain mode

        - **sc_num_samples** (*int or null*, range: 1-64, default: 16) — Number of side-chain samples in side-chain denoising

        - **sc_num_denoising_steps** (*int or null*, range: 1-10, default: 3) — Number of denoising steps in side-chain mode

        - **force_hetatm** (*boolean or null*, default: false) — Whether to include HETATM records during parsing

        - **pack_with_ligand_context** (*boolean or null*, default: true) — Whether to pack using ligand context when ligands are present

        - **fasta_seq_separation** (*string*, default: ":") — Separator string used for concatenating FASTA sequences

        - **file_ending** (*string*, default: "") — File ending label string

        - **zero_indexed** (*int*, default: 0) — Residue indexing mode flag

        - **pdb_path** (*null*, fixed: null) — Unused field; always null

        - **redesigned_residues_multi** (*null*, fixed: null) — Unused field; always null

        - **fixed_residues_multi** (*null*, fixed: null) — Unused field; always null

        - **bias_AA_per_residue_multi** (*null*, fixed: null) — Unused field; always null

        - **omit_AA_per_residue_multi** (*null*, fixed: null) — Unused field; always null

        - **save_stats** (*null*, fixed: null) — Unused field; always null

        - **verbose** (*boolean*, default: true) — Verbosity flag for internal processing

        - **ligand_mpnn_use_side_chain_context** (*null*, fixed: null) — Unused field; always null

        - **ligand_mpnn_use_atom_context** (*boolean or null*, default: true) — LigandMPNN atom context flag

        - **ligand_mpnn_cutoff_for_score** (*float or null*, default: 8.0) — LigandMPNN distance cutoff in Å for scoring

        - **global_transmembrane_label** (*string or null*, allowed: "membrane", "soluble", default: "soluble") — Global transmembrane label

        - **transmembrane_buried** (*array of strings or null*, default: null) — Residue specification strings "[ChainID][ResidueNumber][OptionalInsertionCode]" for buried transmembrane residues; each must exist in the input PDB

        - **transmembrane_interface** (*array of strings or null*, default: null) — Residue specification strings "[ChainID][ResidueNumber][OptionalInsertionCode]" for transmembrane interface residues; each must exist in the input PDB


      - **items** (*array of objects*, min items: 1, max items: 1) --- Input structures:

        - **pdb** (*string*, min length: 1, max length: max_pdb_str_len, required) — PDB content string containing ATOM and/or HETATM records validated on input

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/hyper-mpnn/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.3,
          "fixed_residues": [
            "A3",
            "A4",
            "A5"
          ],
          "redesigned_residues": [
            "A8",
            "A9",
            "A10"
          ],
          "bias_AA": {
            "A": 0.5,
            "V": 0.4,
            "L": 0.4,
            "E": -0.2,
            "D": -0.2
          },
          "bias_AA_per_residue": {
            "A7": {
              "K": 0.8,
              "R": 0.7
            },
            "A9": {
              "I": 0.6,
              "V": 0.6
            }
          },
          "omit_AA": "C",
          "omit_AA_per_residue": {
            "A6": "P",
            "A10": "G"
          },
          "symmetry_residues": [
            [
              "A2",
              "A9"
            ],
            [
              "A3",
              "A8"
            ]
          ],
          "symmetry_weights": [
            [
              1.0,
              1.0
            ],
            [
              0.8,
              0.8
            ]
          ],
          "homo_oligomer": false,
          "chains_to_design": [
            "A"
          ],
          "parse_these_chains_only": [
            "A"
          ],
          "parse_atoms_with_zero_occupancy": false,
          "number_of_batches": 1,
          "batch_size": 2,
          "repack_everything": false,
          "pack_side_chains": false,
          "number_of_packs_per_design": 2,
          "sc_num_samples": 16,
          "sc_num_denoising_steps": 3,
          "force_hetatm": false,
          "pack_with_ligand_context": false,
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
          "transmembrane_buried": [
            "A2",
            "A3"
          ],
          "transmembrane_interface": [
            "A8",
            "A9"
          ]
        },
        "items": [
          {
            "pdb": "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N  \nATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00 20.00           C  \nATOM      3  C   MET A   1      12.100   9.500  11.000  1.00 20.00           C  \nATOM      4  O   MET A   1      11.900   8.300  10.800  1.00 20.00           O  \nATOM      5  N   ALA A   2      13.100  10.000  11.800  1.00 20.00           N  \nATOM      6  CA  ALA A   2      14.100   9.200  12.500  1.00 20.00           C  \nATOM      7  C   ALA A   2      15.300   8.900  11.600  1.00 20.00           C  \nATOM      8  O   ALA A   2      16.200   8.100  11.900  1.00 20.00           O  \nATOM      9  N   LYS A   3      15.300   9.600  10.500  1.00 20.00           N  \nATOM     10  CA  LYS A   3      16.400   9.400   9.600  1.00 20.00           C  \nATOM     11  C   LYS A   3      17.700   9.900  10.200  1.00 20.00           C  \nATOM     12  O   LYS A   3      18.700   9.200   9.900  1.00 20.00           O  \nATOM     13  N   GLU A   4      17.700  11.000  11.000  1.00 20.00           N  \nATOM     14  CA  GLU A   4      18.800  11.600  11.800  1.00 20.00           C  \nATOM     15  C   GLU A   4      18.300  12.100  13.200  1.00 20.00           C  \nATOM     16  O   GLU A   4      19.000  12.900  13.800  1.00 20.00           O  \nATOM     17  N   ILE A   5      17.100  11.700  13.700  1.00 20.00           N  \nATOM     18  CA  ILE A   5      16.500  12.000  15.000  1.00 20.00           C  \nATOM     19  C   ILE A   5      16.900  10.900  16.000  1.00 20.00           C  \nATOM     20  O   ILE A   5      17.400  11.100  17.100  1.00 20.00           O  \nATOM     21  N   ARG A   6      16.700   9.700  15.600  1.00 20.00           N  \nATOM     22  CA  ARG A   6      17.000   8.600  16.500  1.00 20.00           C  \nATOM     23  C   ARG A   6      18.500   8.600  16.800  1.00 20.00           C  \nATOM     24  O   ARG A   6      19.000   7.700  17.400  1.00 20.00           O  \nATOM     25  N   VAL A   7      19.200   9.600  16.400  1.00 20.00           N  \nATOM     26  CA  VAL A   7      20.600   9.700  16.700  1.00 20.00           C  \nATOM     27  C   VAL A   7      21.200   8.500  17.500  1.00 20.00           C  \nATOM     28  O   VAL A   7      22.400   8.500  17.700  1.00 20.00           O  \nATOM     29  N   LYS A   8      20.400   7.600  18.100  1.00 20.00           N  \nATOM     30  CA  LYS A   8      20.800   6.400  18.900  1.00 20.00           C  \nATOM     31  C   LYS A   8      21.000   5.200  18.000  1.00 20.00           C  \nATOM     32  O   LYS A   8      21.900   4.400  18.300  1.00 20.00           O  \nATOM     33  N   LEU A   9      20.100   5.000  17.000  1.00 20.00           N  \nATOM     34  CA  LEU A   9      20.100   3.900  16.000  1.00 20.00           C  \nATOM     35  C   LEU A   9      21.400   3.500  15.300  1.00 20.00           C  \nATOM     36  O   LEU A   9      21.600   2.300  15.100  1.00 20.00           O  \nATOM     37  N   GLU A  10      22.200   4.400  14.900  1.00 20.00           N  \nATOM     38  CA  GLU A  10      23.500   4.200  14.300  1.00 20.00           C  \nATOM     39  C   GLU A  10      24.500   4.900  15.200  1.00 20.00           C  \nATOM     40  O   GLU A  10      25.700   4.700  15.000  1.00 20.00           O  \nTER      41      GLU A  10                                                              \nEND                                                                             "
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

        - **sequence** (*string*) — Designed amino acid sequence using single-letter residue codes, length ≤ 1024

        - **pdb** (*string*) — Designed structure in PDB format text, including REMARK and ATOM/HETATM records

        - **overall_confidence** (*float*) — Scalar design confidence score parsed from a JSON number or numeric string

        - **ligand_confidence** (*float*) — Scalar ligand-environment confidence score parsed from a JSON number or numeric string

        - **seq_rec** (*float*) — Scalar sequence recovery metric parsed from a JSON number or numeric string

        - **log_probs** (*array of arrays of floats*) — Per-position log-probabilities over the residue vocabulary; shape: [L, V], where L = length of ``sequence`` and V = number of residue categories; each inner value parsed as float

        - **sampling_probs** (*array of arrays of floats*) — Per-position sampling probabilities over the residue vocabulary; shape: [L, V], where L = length of ``sequence`` and V = number of residue categories; each inner value parsed as float

        - **pdb_packed** (*object*, optional) — Side-chain–packed structures from side-chain models keyed by identifier:

          - **<chain_or_model_id>** (*string*) — Packed PDB content for the given chain or packing context

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence": "MAKEIRVKAR",
            "pdb": "REMARK Selection ''(backbone) and ...ccupancy > 0)))''\nATOM      1  N   MET A   1      10.000  10.000  10.000  1.00  0.00           N  \nATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00  0.... (truncated for documentation)",
            "overall_confidence": 0.1755,
            "ligand_confidence": 0.1755,
            "seq_rec": 0.3333,
            "log_probs": [
              [
                -0.0,
                -0.0,
                "... (truncated for documentation)"
              ],
              [
                -0.0,
                -0.0,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sampling_probs": [
              [
                0.0,
                0.0,
                "... (truncated for documentation)"
              ],
              [
                0.0,
                0.0,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence": "MAKEIRVKAR",
            "pdb": "REMARK Selection '(backbone) and ...ccupancy > 0)))'\nATOM      1  N   MET A   1      10.000  10.000  10.000  1.00  0.00           N  \nATOM      2  CA  MET A   1      11.200  10.500  10.300  1.00  0.00... (truncated for documentation)",
            "overall_confidence": 0.1755,
            "ligand_confidence": 0.1755,
            "seq_rec": 0.3333,
            "log_probs": [
              [
                -0.0,
                -0.0,
                "... (truncated for documentation)"
              ],
              [
                -0.0,
                -0.0,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sampling_probs": [
              [
                0.0,
                0.0,
                "... (truncated for documentation)"
              ],
              [
                0.0,
                0.0,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Computational characteristics relative to ProteinMPNN:
  
  - HyperMPNN uses the same message-passing architecture, depth, and width as ProteinMPNN, retrained on ~29k AlphaFold structures from hyperthermophiles; per-residue inference cost and memory footprint are effectively identical on the same hardware
  - BioLM deploys HyperMPNN with the same optimized implementation stack as other MPNN variants, so throughput, latency scaling with protein length, and cost per redesigned residue match ProteinMPNN and LigandMPNN

- Thermostability-targeted design behavior:
  
  - When redesigning mesophilic proteins, HyperMPNN shifts surface composition toward hyperthermophile-like profiles (more apolar and positively charged residues, fewer polar uncharged), whereas ProteinMPNN tends toward alanine/glutamate/lysine-rich mesophile-like surfaces
  - In protein cores, HyperMPNN increases apolar content by ~4–5% relative to mesophilic references, closely matching native hyperthermophile cores; ProteinMPNN designs deviate from this pattern
  - For non-hyperthermophilic inputs (e.g., *E. coli*), HyperMPNN redesigns typically move median net charge into the positive regime (≈ +3), while ProteinMPNN produces negatively charged designs, opposite to the charge trend associated with thermostability in hyperthermophiles

- Structural-physics and sequence-modeling performance:
  
  - HyperMPNN approximately doubles the median number of salt bridges per redesigned *E. coli* protein (≈17 vs ≈9 for ProteinMPNN) while maintaining native-like backbone geometry, enabling more extensive electrostatic networks without introducing non-native compaction or expansion
  - Designed sequences preserve contact order and radius of gyration distributions comparable to both mesophilic and hyperthermophilic natives, indicating that HyperMPNN alters sequence-level stability determinants without distorting global fold topology
  - On its held-out hyperthermophile test set, HyperMPNN attains perplexity ≈ 5.18 and per-residue recovery ≈ 0.48, matching ProteinMPNN’s accuracy; relative to ProteinMPNN, users can expect similar native-sequence recovery but with a systematic bias toward thermostable, hyperthermophile-like compositions

- Experimental performance and workflow integration:
  
  - In redesigns of the I53-50B pentameric nanoparticle component (parent Tm ~65°C), a HyperMPNN consensus design remained folded to at least 95°C by CD, representing a ≥30°C apparent stability increase and demonstrating that the learned composition shift can translate into large experimental Tm gains
  - Compared with a ProteinMPNN design on the same backbone that is also stable to 95°C but strongly negatively charged (≈ −7.9), the HyperMPNN design (net charge ≈ +2.0) better reproduces hyperthermophile-like surface electrostatics, which can be advantageous when targeting positively charged lumen-facing surfaces
  - Within multi-model BioLM pipelines (e.g., HyperMPNN for global redesign followed by structure prediction and supervised stability scoring), HyperMPNN typically contributes only a small fraction of total runtime while materially enriching for high-Tm candidates relative to starting from ProteinMPNN or language-model-based sequence generators

Applications
------------

- Thermostabilization of existing industrial enzymes given a reliable 3D structure (experimental or AlphaFold2), by using HyperMPNN (``model_type="hyper"``) to redesign non-catalytic residues toward hyperthermophile-like amino acid composition, enabling operation at higher process temperatures (e.g., shifting a 50–60°C enzyme toward 80–95°C use) and reducing contamination risk and cleaning costs; activity and solubility still require experimental verification
- Design of highly heat-stable self-assembling protein nanoparticles and carriers (e.g., I53-50-like scaffolds) by restructuring surface and core residues while preserving assembly interfaces using HyperMPNN’s structure-conditioned sequence model, allowing vaccine and biologic delivery systems that maintain integrity after prolonged exposure to elevated temperatures and cold-chain interruptions; requires an oligomer model built into a single PDB for design
- Pre-screening of stabilizing sequence variants before wet-lab campaigns in protein engineering pipelines, by using HyperMPNN via the ``generator`` endpoint to generate thermostable sequence panels that are then filtered with secondary in silico models (e.g., ddG predictors, activity models) and assays, reducing the number of variants that need to be constructed while focusing on mutation patterns consistent with hyperthermophilic stability strategies
- Retrofitting mesophilic biocatalysts for high-temperature manufacturing steps where substrate solubility or reaction rates are limiting, via structure-guided redesign of the scaffold (keeping active-site residues in ``fixed_residues``) to incorporate apolar core packing and charged surface patterns learned from hyperthermophiles; users should expect that catalytic performance, expression levels, and solubility may change and must be re-optimized experimentally
- Early-stage stability risk mitigation for protein-based therapeutics and diagnostic reagents (e.g., binding scaffolds, cytokine mimetics) by generating HyperMPNN designs biased toward higher melting temperatures, providing alternative sequence options when formulation, storage, or transport conditions are expected to exceed typical room-temperature stability thresholds; not optimal for targets lacking a reliable monomeric or oligomeric structure model or where function depends on finely tuned conformational dynamics near physiological temperature

Limitations
-----------

- **Maximum sequence length**: HyperMPNN inherits the MPNN architectural limit of at most ``1024`` residues per chain as parsed from the input ``pdb`` string. Chains with more than ``1024`` residues are not supported and must be truncated or split before calling the API.
- **Batching and throughput**: ``HyperMPNNGenerateRequest.items`` must contain exactly ``1`` structure (``min_items=1``, ``max_items=1``). Within ``params``, ``batch_size`` must satisfy ``1 <= batch_size <= 1000`` and ``number_of_batches`` must satisfy ``1 <= number_of_batches <= 48``. Very large design campaigns should be split across multiple API requests by the client.
- **Structure and residue specification requirements**: The ``pdb`` field must be a valid PDB-formatted string (``ATOM``/``HETATM`` records with consistent chain and residue numbering) and is validated by ``validate_pdb``. Residue-level options such as ``fixed_residues``, ``redesigned_residues``, ``bias_AA_per_residue``, ``omit_AA_per_residue``, ``symmetry_residues``, ``transmembrane_buried``, and ``transmembrane_interface`` use the ``[ChainID][ResidueNumber][OptionalInsertionCode]`` syntax (for example ``A10``, ``B52A``) and must refer to residues present in the uploaded structure; invalid chain IDs or out-of-range residue numbers will raise validation errors.
- **Thermostability-focused design bias**: HyperMPNN is a retrained ProteinMPNN variant biased toward the amino acid composition of hyperthermophilic proteins (increased apolar core and positively charged surface residues). It is intended for stabilizing well-structured proteins and is not optimized for preserving native-like flexibility, fine-tuning low-temperature activity, or reproducing organism-specific mesophilic amino acid compositions. For neutral or organism-matched designs, other MPNN variants or protein language models may be more suitable.
- **Backbone dependence and design scope**: The model assumes a reasonably accurate backbone (experimental or high-confidence predicted). It does not fix large backbone errors, change topology, or assess foldability from sequence alone; it only designs sequences on the provided structure. HyperMPNN generally introduces many mutations to embed hyperthermophile-like patterns and is therefore less appropriate than single-mutation ``ddG`` predictors for local scanning or minimal-change engineering.
- **Domain and expression context**: Training data are hyperthermophilic and predominantly prokaryotic. The strong bias toward high-temperature stability can reduce folding or expression efficiency in mesophilic hosts such as *E. coli*. HyperMPNN is not ideal for mammalian-only folds with no hyperthermophilic analogs, highly disordered or IDR-rich targets, membrane proteins (use membrane-aware MPNN variants instead), or applications where maintaining wild-type expression at low or moderate temperature is the primary goal.

How We Use It
-------------

HyperMPNN enables thermostability-focused sequence redesign as a configurable step within broader protein engineering programs, integrating seamlessly with structure prediction, sequence embeddings, and developability filters to increase the likelihood of higher melting temperatures for enzymes, antibodies, and protein nanoparticles. Using the same standardized MPNN API schema, teams can combine HyperMPNN designs with folding models (e.g., AlphaFold2-based services), language-model embeddings, biophysical scoring (charge, salt bridges, radius of gyration), and client assay data to route large variant sets through a consistent design → filter → rank → test → iterate loop, turning thermostability from an ad hoc consideration into a scalable design objective that improves hit quality and shortens optimization cycles.

- In enzyme and industrial biocatalyst projects, HyperMPNN redesigns are evaluated alongside activity and solubility predictors to select variants that balance elevated operating temperatures with process-relevant performance.  
- In vaccine and nanoparticle applications, HyperMPNN complements interface design, antigen display, and manufacturability assessments, supporting thermostable carriers that better tolerate distribution constraints without adding extra assay rounds.

Related
-------

- ``ProteinMPNN`` – Base backbone-conditioned sequence design model that HyperMPNN retrains on hyperthermophilic structures; serves as a mesophilic or “neutral” baseline or for designs where extreme thermostability is not required.
- ``ThermoMPNN`` – Supervised stability change (ddG) predictor using ProteinMPNN-derived features; can score HyperMPNN-designed variants to estimate stability shifts and prioritize designs.
- ``ESM2StabP`` – Sequence-based protein stability predictor; complements HyperMPNN by enabling fast, structure-free screening of designed sequences for global stability trends.
- ``TemBERTure Regression`` – Sequence-based optimal temperature regression model; can be applied downstream of HyperMPNN to estimate changes in preferred operating temperature of redesigned proteins.

References
----------

- Ertelt, M., Schlegel, P., Beining, M., Kaysser, L., Meiler, J., & Schoeder, C. T. (2024). HyperMPNN – A general strategy to design thermostable proteins learned from hyperthermophiles. Preprint and code available at https://github.com/meilerlab/HyperMPNN.
