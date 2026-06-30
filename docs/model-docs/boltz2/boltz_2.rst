Boltz-2 API
===========

Boltz-2 is a GPU-accelerated, structure- and affinity-aware biomolecular co-folding model for proteins, nucleic acids, and small-molecule ligands. It predicts 3D complex structures with per-complex and per-chain confidence metrics (pLDDT, pTM, ipTM, PAE, PDE), and optionally returns single and pairwise embeddings. For ligands, it outputs a binary binding likelihood and a continuous IC50-like affinity value on a log10 scale, with optional molecular-weight correction. The API supports multi-entity complexes and optional bond, pocket, and contact constraints for hypothesis-driven modeling.

Predict
-------

Description of what this endpoint does

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="boltz2",
                action="predict",
                params={
                  "recycling_steps": 3,
                  "sampling_steps": 20,
                  "diffusion_samples": 1,
                  "step_scale": 1.638,
                  "seed": 42,
                  "potentials": true,
                  "include": [
                    "pae",
                    "pde"
                  ],
                  "max_msa_seqs": 4096,
                  "subsample_msa": true,
                  "num_subsampled_msa": 512,
                  "affinity_mw_correction": true,
                  "sampling_steps_affinity": 50,
                  "diffusion_samples_affinity": 5,
                  "affinity": {
                    "binder": "L"
                  }
                },
                items=[
                  {
                    "molecules": [
                      {
                        "id": "A",
                        "type": "protein",
                        "sequence": "MKTWVPEITQG",
                        "modifications": [
                          {
                            "position": 5,
                            "ccd": "PDP"
                          }
                        ]
                      },
                      {
                        "id": "L",
                        "type": "ligand",
                        "smiles": "CC(C)C(=O)Nc1ccccc1"
                      }
                    ],
                    "constraints": [
                      {
                        "pocket": {
                          "binder": "L",
                          "contacts": [
                            [
                              "A",
                              6
                            ],
                            [
                              "A",
                              7
                            ],
                            [
                              "A",
                              9
                            ]
                          ],
                          "max_distance": 5.0
                        }
                      }
                    ]
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/boltz2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "molecules": [
                    {
                      "id": "A",
                      "type": "protein",
                      "sequence": "MKTWVPEITQG",
                      "modifications": [
                        {
                          "position": 5,
                          "ccd": "PDP"
                        }
                      ]
                    },
                    {
                      "id": "L",
                      "type": "ligand",
                      "smiles": "CC(C)C(=O)Nc1ccccc1"
                    }
                  ],
                  "constraints": [
                    {
                      "pocket": {
                        "binder": "L",
                        "contacts": [
                          [
                            "A",
                            6
                          ],
                          [
                            "A",
                            7
                          ],
                          [
                            "A",
                            9
                          ]
                        ],
                        "max_distance": 5.0
                      }
                    }
                  ]
                }
              ],
              "params": {
                "recycling_steps": 3,
                "sampling_steps": 20,
                "diffusion_samples": 1,
                "step_scale": 1.638,
                "seed": 42,
                "potentials": true,
                "include": [
                  "pae",
                  "pde"
                ],
                "max_msa_seqs": 4096,
                "subsample_msa": true,
                "num_subsampled_msa": 512,
                "affinity_mw_correction": true,
                "sampling_steps_affinity": 50,
                "diffusion_samples_affinity": 5,
                "affinity": {
                  "binder": "L"
                }
              }
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/boltz2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "molecules": [
                        {
                          "id": "A",
                          "type": "protein",
                          "sequence": "MKTWVPEITQG",
                          "modifications": [
                            {
                              "position": 5,
                              "ccd": "PDP"
                            }
                          ]
                        },
                        {
                          "id": "L",
                          "type": "ligand",
                          "smiles": "CC(C)C(=O)Nc1ccccc1"
                        }
                      ],
                      "constraints": [
                        {
                          "pocket": {
                            "binder": "L",
                            "contacts": [
                              [
                                "A",
                                6
                              ],
                              [
                                "A",
                                7
                              ],
                              [
                                "A",
                                9
                              ]
                            ],
                            "max_distance": 5.0
                          }
                        }
                      ]
                    }
                  ],
                  "params": {
                    "recycling_steps": 3,
                    "sampling_steps": 20,
                    "diffusion_samples": 1,
                    "step_scale": 1.638,
                    "seed": 42,
                    "potentials": true,
                    "include": [
                      "pae",
                      "pde"
                    ],
                    "max_msa_seqs": 4096,
                    "subsample_msa": true,
                    "num_subsampled_msa": 512,
                    "affinity_mw_correction": true,
                    "sampling_steps_affinity": 50,
                    "diffusion_samples_affinity": 5,
                    "affinity": {
                      "binder": "L"
                    }
                  }
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/boltz2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  molecules = list(
                    list(
                      id = "A",
                      type = "protein",
                      sequence = "MKTWVPEITQG",
                      modifications = list(
                        list(
                          position = 5,
                          ccd = "PDP"
                        )
                      )
                    ),
                    list(
                      id = "L",
                      type = "ligand",
                      smiles = "CC(C)C(=O)Nc1ccccc1"
                    )
                  ),
                  constraints = list(
                    list(
                      pocket = list(
                        binder = "L",
                        contacts = list(
                          list(
                            "A",
                            6
                          ),
                          list(
                            "A",
                            7
                          ),
                          list(
                            "A",
                            9
                          )
                        ),
                        max_distance = 5.0
                      )
                    )
                  )
                )
              ),
              params = list(
                recycling_steps = 3,
                sampling_steps = 20,
                diffusion_samples = 1,
                step_scale = 1.638,
                seed = 42,
                potentials = TRUE,
                include = list(
                  "pae",
                  "pde"
                ),
                max_msa_seqs = 4096,
                subsample_msa = TRUE,
                num_subsampled_msa = 512,
                affinity_mw_correction = TRUE,
                sampling_steps_affinity = 50,
                diffusion_samples_affinity = 5,
                affinity = list(
                  binder = "L"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/boltz2/predict/

   Predict endpoint for Boltz-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **recycling_steps** (*integer*, default: 3, range: 1–10) — Number of recycling iterations used during structure prediction

        - **sampling_steps** (*integer*, default: 20, range: 1–200) — Number of diffusion sampling steps used during structure prediction

        - **diffusion_samples** (*integer*, default: 1, range: 1–10) — Number of structure samples to generate per item

        - **step_scale** (*float*, default: 1.638, range: 0.1–10.0) — Scaling factor applied to the diffusion step schedule

        - **seed** (*integer*, default: 42, optional) — Random seed for sampling

        - **potentials** (*boolean*, default: true) — Whether to enable physics-based steering potentials during inference

        - **include** (*array of strings*, default: []) — Additional outputs to return; allowed values: "pae", "pde", "plddt", "embeddings"

        - **max_msa_seqs** (*integer*, default: 8192, range: 1–32768) — Maximum number of MSA sequences used for prediction

        - **subsample_msa** (*boolean*, default: false) — Whether to subsample MSA sequences

        - **num_subsampled_msa** (*integer*, default: 1024, range: 1–8192) — Number of MSA sequences to use if subsample_msa is true

        - **affinity_mw_correction** (*boolean*, default: false) — Whether to apply molecular-weight-based correction to the affinity_pred_value output

        - **sampling_steps_affinity** (*integer*, default: 200, range: 1–200) — Number of sampling steps used for affinity prediction

        - **diffusion_samples_affinity** (*integer*, default: 5, range: 1–50) — Number of diffusion samples used for affinity prediction

        - **affinity** (*object*, optional) — Affinity prediction parameters:

          - **binder** (*string*, required) — Chain ID used as the binder for affinity computation; must match a chain ID from items.molecules.id


      - **items** (*array of objects*, min length: 1, max length: 1) --- Input data:

        - **molecules** (*array of objects*, min length: 1) — Molecular entities included in the complex:

          - **id** (*string or array of strings*, required) — Chain identifier or list of chain identifiers for the entity

          - **type** (*string*, required; one of: "protein", "dna", "rna", "ligand") — Entity type

          - **sequence** (*string*, required if type is "protein", "dna", or "rna"; must be omitted if type is "ligand") — Linear polymer sequence

          - **smiles** (*string*, optional; allowed only if type is "ligand") — Ligand structure as a SMILES string

          - **ccd** (*string*, optional; allowed only if type is "ligand") — Ligand CCD identifier

          - **alignment** (*object*, optional; allowed only if type is "protein") — Multiple sequence alignment data by database:

            - **mgnify** (*string*, optional) — Alignment content for the "mgnify" database

            - **small_bfd** (*string*, optional) — Alignment content for the "small_bfd" database

            - **uniref90** (*string*, optional) — Alignment content for the "uniref90" database

          - **modifications** (*array of objects*, optional; allowed only if type is "protein", "dna", or "rna") — Site-specific polymer modifications:

            - **position** (*integer*, required) — 1-based residue index of the modified position

            - **ccd** (*string*, required) — CCD identifier for the modified residue or nucleotide

          - **cyclic** (*boolean*, default: false) — Whether the polymer is cyclic

        - **constraints** (*array of objects*, optional) — Structural constraints for the complex; each element must define at least one of bond, pocket, or contact:

          - **bond** (*object*, optional) — Covalent bond constraint between two atoms:

            - **atom1** (*array*, required) — Atom specification as [chain_id, residue_index_or_atom, ...]; first element must be a chain ID present in molecules.id

            - **atom2** (*array*, required) — Atom specification as [chain_id, residue_index_or_atom, ...]; first element must be a chain ID present in molecules.id

          - **pocket** (*object*, optional) — Pocket constraint specification:

            - **binder** (*string*, required) — Chain ID of the binder; must match a value in molecules.id and must not be an array

            - **contacts** (*array of arrays*, required) — List of pocket residue or atom specifications as [[chain_id, residue_index_or_atom, ...], ...]; each chain_id must match a value in molecules.id

            - **max_distance** (*float*, optional) — Maximum allowed distance in angstroms between the binder and pocket contacts

          - **contact** (*object*, optional) — Distance constraint between two tokens:

            - **token1** (*array*, required) — Token specification as [chain_id, residue_index_or_atom, ...]; first element must be a chain ID present in molecules.id

            - **token2** (*array*, required) — Token specification as [chain_id, residue_index_or_atom, ...]; first element must be a chain ID present in molecules.id

            - **max_distance** (*float*, required) — Maximum allowed distance in angstroms between token1 and token2

        - **templates** (*array of objects*, optional) — Template structure definitions used during prediction:

          - **cif** (*string*, required) — Template structure in CIF format; must be non-empty

          - **chain_id** (*string or array of strings*, optional) — Chain ID or list of chain IDs mapped to this template; each value must match a molecules.id chain ID

          - **template_id** (*string or array of strings*, optional) — Template identifier or list of identifiers

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/boltz2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "molecules": [
              {
                "id": "A",
                "type": "protein",
                "sequence": "MKTWVPEITQG",
                "modifications": [
                  {
                    "position": 5,
                    "ccd": "PDP"
                  }
                ]
              },
              {
                "id": "L",
                "type": "ligand",
                "smiles": "CC(C)C(=O)Nc1ccccc1"
              }
            ],
            "constraints": [
              {
                "pocket": {
                  "binder": "L",
                  "contacts": [
                    [
                      "A",
                      6
                    ],
                    [
                      "A",
                      7
                    ],
                    [
                      "A",
                      9
                    ]
                  ],
                  "max_distance": 5.0
                }
              }
            ]
          }
        ],
        "params": {
          "recycling_steps": 3,
          "sampling_steps": 20,
          "diffusion_samples": 1,
          "step_scale": 1.638,
          "seed": 42,
          "potentials": true,
          "include": [
            "pae",
            "pde"
          ],
          "max_msa_seqs": 4096,
          "subsample_msa": true,
          "num_subsampled_msa": 512,
          "affinity_mw_correction": true,
          "sampling_steps_affinity": 50,
          "diffusion_samples_affinity": 5,
          "affinity": {
            "binder": "L"
          }
        }
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **cif** (*string*) — Predicted 3D structure in mmCIF format

        - **plddt** (*array of float*, size: [num_tokens], range: 0..100, optional) — Per-token predicted LDDT values

        - **pae** (*array of float*, size: [num_tokens, num_tokens], optional) — Pairwise predicted aligned error matrix

        - **pde** (*array of float*, size: [num_tokens, num_tokens], optional) — Pairwise predicted distance error matrix

        - **embeddings** (*object*, optional) — Token and pairwise representations

          - **s** (*array of float*, shape: [num_tokens, embedding_dim]) — Single-token embeddings

          - **z** (*array of float*, shape: [num_tokens, num_tokens, embedding_dim]) — Pairwise token embeddings

        - **confidence** (*object*) — Confidence metrics for the predicted structure

          - **confidence_score** (*float*) — Overall confidence score for the prediction

          - **ptm** (*float*) — Predicted TM-score over all chains

          - **iptm** (*float*) — Predicted interface TM-score over all chains

          - **ligand_iptm** (*float*) — Predicted interface TM-score for ligand chains

          - **protein_iptm** (*float*) — Predicted interface TM-score for protein chains

          - **complex_plddt** (*float*, range: 0..100) — Mean pLDDT over all tokens

          - **complex_iplddt** (*float*, range: 0..100) — Mean pLDDT over interface tokens

          - **complex_pde** (*float*) — Mean PDE over all token pairs

          - **complex_ipde** (*float*) — Mean PDE over interface token pairs

          - **chains_ptm** (*object*) — Per-chain predicted TM-scores (keys are chain indices as strings, values are floats)

          - **pair_chains_iptm** (*object*) — Nested interface TM-scores per chain pair (outer and inner keys are chain indices as strings, values are floats)

        - **affinity** (*object*, optional) — Binding affinity predictions from the ensemble

          - **affinity_pred_value** (*float*) — Ensemble binding affinity estimate on log10-scale

          - **affinity_probability_binary** (*float*, range: 0..1) — Ensemble binding likelihood

          - **affinity_pred_value1** (*float*) — First ensemble model binding affinity estimate on log10-scale

          - **affinity_probability_binary1** (*float*, range: 0..1) — First ensemble model binding likelihood

          - **affinity_pred_value2** (*float*) — Second ensemble model binding affinity estimate on log10-scale

          - **affinity_probability_binary2** (*float*, range: 0..1) — Second ensemble model binding likelihood

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "cif": "data_model\n_entry.id model\n_struct.entry_id model\n_struct.pdbx_model_details .\n_struct.pdbx_structure_determination_methodology computational\n_struct.title .\n_audit_conform.dict_location https://raw.g... (truncated for documentation)",
            "confidence": {
              "confidence_score": 0.6288325190544128,
              "ptm": 0.4148341715335846,
              "iptm": 0.22942186892032623,
              "ligand_iptm": 0.22942186892032623,
              "protein_iptm": 0.0,
              "complex_plddt": 0.7286851406097412,
              "complex_iplddt": 0.6597349643707275,
              "complex_pde": 0.4406202733516693,
              "complex_ipde": 0.8638767004013062,
              "chains_ptm": {
                "0": 0.5614914894104004,
                "1": 0.7034032940864563
              },
              "pair_chains_iptm": {
                "0": {
                  "0": 0.5614914894104004,
                  "1": 0.10003116726875305
                },
                "1": {
                  "0": 0.22942186892032623,
                  "1": 0.7034032940864563
                }
              }
            },
            "affinity": {
              "affinity_pred_value": 2.3326456546783447,
              "affinity_probability_binary": 0.18777765333652496,
              "affinity_pred_value1": 2.656043529510498,
              "affinity_probability_binary1": 1.56500973389484e-05,
              "affinity_pred_value2": 1.589586615562439,
              "affinity_probability_binary2": 0.3755396604537964
            }
          }
        ]
      }


Performance
-----------

- Constraints and resource profile
  - Total sequence budget: up to 1,024 tokens per request across all molecules (proteins/DNA/RNA per residue; ligands per atom), matching ``max_sequence_len`` in the Boltz schema
  - GPU targets: NVIDIA A100 80 GB and H100 80 GB; H100 typically achieves lower latency for default Boltz‑2 pose+affinity settings (higher ``sampling_steps_affinity`` and ``diffusion_samples_affinity``)
  - Memory use: ~20–30 GB for protein–ligand complexes near 1,024 tokens with default ``diffusion_samples`` / ``diffusion_samples_affinity``; scales roughly linearly with token count and number of diffusion samples

- Runtime and throughput (end‑to‑end GPU time; excludes network and queueing)
  - Pose‑only quick pass (``sampling_steps=20``, ``diffusion_samples=1``): ~3–8 s per complex on H100; ~6–12 s on A100
  - Default pose+affinity mode (``sampling_steps_affinity=200``, ``diffusion_samples_affinity=5``): ~18–25 s per complex on H100; ~30–45 s on A100
  - Typical throughput in default pose+affinity mode: ~140–200 complexes per H100‑hour; ~80–120 per A100‑hour, scaling near‑linearly with the number of GPUs for independent jobs

- Predictive accuracy and trade‑offs
  - Affinity (using API ensemble outputs ``affinity_pred_value`` / ``affinity_probability_binary``)
    - FEP+ 4‑target subset (CDK2, TYK2, JNK1, P38): Pearson R≈0.66, pairwise MAE≈0.85 kcal/mol; ≳1,000× lower wall‑clock per analogue than ABFE/FEP protocols
    - OpenFE subset (876 complexes): Pearson R≈0.62, pairwise MAE≈0.93 kcal/mol
    - MF‑PCBA (hit discovery, via ``affinity_probability_binary``): AP≈0.025, EF@0.5%≈18.4, AUROC≈0.81; substantially better early enrichment than Chemgauss4 docking (EF@0.5%≈2.0)
  - Structure and dynamics (from CIF + confidence outputs)
    - PDB 2024–2025: competitive with other open co‑folding models; generally improved over Boltz‑1, with a remaining gap to AlphaFold3 on some antibody–antigen complexes
    - MD‑like local dynamics: stronger RMSF correlations than Boltz‑1 and comparable to MD emulators (e.g., BioEmu, AlphaFlow) on held‑out MD datasets when using multiple diffusion samples

- Relative performance and deployment optimizations
  - Compared to other BioLM structural models
    - ESMFold / AlphaFold2: better for fast monomer folding; they do not co‑fold small molecules or expose affinity heads, so screening workflows typically require separate docking/FEP stages and higher end‑to‑end latency than a single Boltz‑2 pose+affinity API call
    - Chai‑1 / ProteinX‑class co‑folders: similar wall‑times for pose generation under comparable sampling settings; Boltz‑2 additionally exposes an affinity head (binary + IC50‑like regression) with state‑of‑the‑art MF‑PCBA enrichment via the API
  - Compared to docking and FEP
    - Docking (Chemgauss4/FRED): CPU‑seconds per ligand but markedly weaker early enrichment on MF‑PCBA; Boltz‑2 uses GPU‑seconds yet yields far better active ranking via ``affinity_probability_binary``
    - FEP/ABFE (OpenFE, FEP+): higher ultimate accuracy on well‑curated series but orders of magnitude slower; Boltz‑2 approaches their rank‑correlation on public benchmarks and is typically used as a fast pre‑filter
  - System‑level optimizations in the hosted deployment
    - Mixed‑precision trunk (bfloat16) and optimized triangle attention kernels reduce memory and improve throughput near the 1,024‑token limit
    - Pocket‑centric affinity computation restricts PairFormer attention to protein–ligand and intra‑ligand interactions, reducing pair feature volume by >5× relative to whole‑complex processing
    - MSA/template alignment and feature caching at the service layer amortize cost across campaigns, so large ligand libraries scale predominantly with the number of unique protein targets rather than total complex count

Applications
------------

- High-throughput virtual screening with structure-based affinity scoring: Use Boltz-2’s binding likelihood and IC50-like affinity prediction (``affinity_probability_binary``, ``affinity_pred_value``) to triage large purchasable libraries on GPUs, enriching true actives beyond docking and ipTM heuristics and enabling inexpensive early down-selection before wet-lab or ABFE/FEP confirmation; this approaches FEP-like ranking quality at ~20 seconds per ligand and showed strong enrichment (e.g., MF-PCBA EF@0.5% ≈ 18.4) and prospective TYK2 success, but results depend on correct pocket/state identification and are less reliable when cofactors, ions, or critical waters are essential and not modeled
- Hit-to-lead and lead optimization ranking within congeneric series: Prioritize analogs by predicted affinity differences (``affinity_pred_value`` with optional molecular-weight correction via ``affinity_mw_correction``) to guide SAR cycles on tractable targets (e.g., kinases CDK2, TYK2, JNK1, p38), with performance approaching FEP on public benchmarks (Pearson R ≈ 0.66 on a 4-target FEP subset) at >1000× lower cost, helping medicinal chemists focus synthesis on the most promising modifications; this is not optimal when the binding mode is misassigned, the affinity crop misses long-range interactions, or for target classes that historically require custom preparation (e.g., some GPCRs) without additional care
- Generative design scoring loop for synthesizable lead discovery: Integrate Boltz-2 as the reward signal inside a synthesis-aware generator to score proposed ligands via the same affinity outputs used for screening (``affinity_probability_binary``, ``affinity_pred_value``), so you can explore diverse, make-on-demand, high-scoring binders beyond fixed libraries while keeping chemistry practical; this accelerates design–make–test cycles and yielded strong prospective TYK2 candidates validated by ABFE in silico, but guardrails are needed to prevent reward hacking (e.g., model ensembling, PAINS/med-chem filters, and secondary physics or experimental validation)
- Constraint- and template-steered pose modeling to test binding hypotheses: Apply pocket/contact constraints and multimeric templates (via Boltz-2’s ``constraints`` and ``templates`` inputs) to bias poses toward specific binding modes, enabling fragment-growing directions, warhead placement in covalent campaigns, or site-directed mutagenesis planning in proteins where pocket geometry drives selectivity; this improves hypothesis testing without retraining, yet remains sensitive to incorrect assumptions—overly aggressive constraints can yield plausible-looking but non-physical poses, so use confidence metrics and, where needed, physics-based relaxation downstream
- Protein conformational ensemble modeling for pocket-state selection: Use method conditioning together with multiple diffusion samples (via ``diffusion_samples`` and ``sampling_steps`` in the request parameters) to generate protein conformational ensembles for structure-enabled campaigns, such as selecting open/closed states for SBDD, disambiguating induced-fit risks, or prioritizing constructs for crystallography and cryo-EM; this helps match screening to the relevant biological state and showed improved RMSF correlations versus prior releases, but it is not a substitute for full MD—coverage of rare states and cofactor-dependent rearrangements can be limited without additional experimental or simulation data

Limitations
-----------

- **Batch Size** and **Maximum sequence length**: The ``items`` list supports at most 1 entry (``BoltzModelParams.batch_size = 1``). Total tokenized length across all ``molecules`` in an item must be ≤ ``max_sequence_len = 1024`` (proteins/DNA/RNA tokenize at residue/nucleotide level; ligands at atom level). Longer systems must be cropped or split across requests. Runtime increases with sampling hyperparameters: ``recycling_steps`` (default ``3``, range ``1–10``), ``sampling_steps`` (default ``20``, ``1–200``), ``diffusion_samples`` (default ``1``, ``1–10``); affinity-specific settings use ``sampling_steps_affinity = 200`` (``1–200``) and ``diffusion_samples_affinity = 5`` (``1–50``) by default. Setting ``potentials = False`` disables physical steering and can yield less realistic poses but does not change these limits
- Input typing and validations: Each element of ``molecules`` is a ``BoltzEntity``. For ``type`` in ``"protein"``/``"dna"``/``"rna"``, ``sequence`` is required; for ``"ligand"``, ``sequence`` must be omitted and at least one of ``smiles`` or ``ccd`` is required. ``alignment`` is only allowed for proteins and must use keys ``"mgnify"``, ``"small_bfd"``, or ``"uniref90"``. ``modifications`` are only valid for protein/DNA/RNA entities. Optional ``templates`` require non-empty ``cif`` content and any ``chain_id`` they reference must match a chain ID in ``molecules``. Optional ``constraints`` entries must include at least one of ``bond``, ``pocket``, or ``contact``; ``pocket.binder`` supports only a single chain; ``bond`` is only supported between CCD ligands and canonical residues; all chain IDs referenced in any constraint must exist in ``molecules``
- Affinity I/O and scope: Affinity predictions are returned only when ``params.affinity`` is set with a valid ``binder`` chain ID present in ``molecules``; otherwise the ``affinity`` block is omitted from ``results``. Outputs include ``affinity_pred_value`` (a log\ :sub:`10` IC50-like value in μM units, on an assay-dependent scale) and ``affinity_probability_binary``, plus per-ensemble-member fields (``affinity_pred_value1``, ``affinity_pred_value2``, ``affinity_probability_binary1``, ``affinity_probability_binary2``). Enabling ``affinity_mw_correction = True`` applies a molecular-weight-based calibration to ``affinity_pred_value``, which can change cross-assay comparability. In practice, affinity quality depends on the predicted 3D protein–ligand pose; incorrect pocket choice, mis-docking, or wrong protein conformational state will degrade predictions. The affinity head does not explicitly account for cofactors, ions, water, or multimeric partners, and the fixed pocket crop can truncate long-range or allosteric interactions
- Structure/dynamics caveats: Boltz-2 improves physical plausibility via ``potentials`` but can still miss large induced-fit motions, long-range allostery, and very large complexes. MD method conditioning improves local flexibility estimates (for example, RMSF-like behavior), but it is not a substitute for running molecular dynamics and is trained on a finite set of MD systems. Antibody–antigen interfaces are improved over Boltz-1 but can underperform highly tuned antibody-specific or proprietary models on challenging, unseen antigens
- When Boltz-2 is not the best first tool: For ultra-large early-stage screening (≫10^6 compounds) under tight compute or budget constraints, simple ligand filters, sequence-based methods, or docking may be more appropriate initially; Boltz-2 is better suited for structure-aware re-ranking and lead optimization on narrowed libraries. Projects that need tightly calibrated absolute ΔG across narrow congeneric series, or especially difficult targets (for example some GPCRs), may still require ABFE/FEP protocols for final ranking. Do not use structure confidence metrics such as ``confidence.iptm`` or ``confidence.ptm`` as proxies for binding affinity; pose or structure confidence is not reliably predictive of binding strength
- Output size and throughput: Requesting optional outputs via ``include`` increases compute time and payload size. ``include = ["embeddings"]`` returns high-dimensional embedding tensors ``embeddings.s`` (per token) and ``embeddings.z`` (pairwise), and ``include = ["pae","pde","plddt"]`` returns per-token or pairwise matrices that scale quadratically with token count. For high-throughput workloads, omit large extras where possible. The ``cif`` in ``results`` contains only the top-ranked structural sample; increasing ``diffusion_samples`` or ``sampling_steps`` increases latency but not response size

How We Use It
-------------

BioLM uses Boltz-2 as a standardized, affinity- and structure-aware scoring layer in design–make–test–learn campaigns, enabling structure-based ranking of large small-molecule and biologic libraries from hit discovery through lead optimization. We co-fold protein–ligand or protein–binder complexes via the API, optionally apply pocket and contact constraints to explore binding modes, and use Boltz-2’s binder likelihood and IC50-like affinity values (via the affinity binder chain parameter) to prioritize compounds within assay series. These scores integrate directly with our generative chemistries (for example GFlowNet-based ligand design), protein design workflows (such as MLM-guided interface mutagenesis), and downstream developability/ADMET filters, and can be combined with physics-based ABFE/FEP when projects require higher-resolution triage.

- Lead optimization and de novo design: Boltz-2-guided ranking focuses synthetic effort on diverse, synthesizable compounds that satisfy medicinal chemistry constraints, while selective cross-checks with physics-based methods reduce overall screening cost and iteration time.
- Antibody and enzyme programs: Interface-aware complex structures and affinity-linked metrics from Boltz-2 support CDR or active-site variant selection, alongside sequence embeddings and biophysical property screens, to drive multi-round maturation with consistent, API-driven scoring.

Related
-------

- ``Chai-1`` – Alternative co-folding model for protein–ligand and multimer complexes; use to generate complementary poses or multimeric templates that can guide ``Boltz-2`` via template conditioning.
- ``ESMFold`` – Fast monomer folding to generate apo/domain protein structures that serve as templates or help ``Boltz-2`` localize binding pockets when no complex structure is available.
- ``ESM-IF1`` – Inverse folding on ``Boltz-2`` protein–ligand poses to propose sequence mutations that improve pocket complementarity or remove clashes for affinity optimization.
- ``Biotite PDB RMSD`` – Compute RMSD between ``Boltz-2``-predicted complexes and reference structures (or across samples) to rank poses and assess conformational stability.

References
----------

- Passaro et al. (2025). *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction* (`bioRxiv preprint <https://doi.org/10.1101/2025.06.14.659707>`_). Code and weights: `<https://github.com/jwohlwend/boltz>`_.
