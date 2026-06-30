Chai-1 API
==========

Chai-1 is a GPU-accelerated, multi-modal biomolecular structure prediction model that infers 3D structures of proteins, nucleic acids, ligands, and complexes from sequence and SMILES inputs. The API accepts up to 5 molecules per request (proteins ≤1024 residues; DNA/RNA ≤3072 bases; ligands ≤128 characters) with optional per-protein MSAs from UniRef90, MGnify, or Small BFD. It returns CIF-format coordinates plus optional pLDDT and PAE scores, enabling single-sequence and MSA-informed workflows for drug discovery and protein/nucleic acid engineering.

Predict
-------

Predict 3D structures and optional confidence metrics with Chai-1

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="chai1",
                action="predict",
                params={
                  "num_trunk_recycles": 4,
                  "num_diffusion_timesteps": 180,
                  "num_diffn_samples": 1,
                  "use_esm_embeddings": true,
                  "seed": 42,
                  "include": []
                },
                items=[
                  {
                    "molecules": [
                      {
                        "name": "TestProtein",
                        "type": "protein",
                        "sequence": "MAAASNDENERK"
                      }
                    ]
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/chai1/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "num_trunk_recycles": 4,
                "num_diffusion_timesteps": 180,
                "num_diffn_samples": 1,
                "use_esm_embeddings": true,
                "seed": 42,
                "include": []
              },
              "items": [
                {
                  "molecules": [
                    {
                      "name": "TestProtein",
                      "type": "protein",
                      "sequence": "MAAASNDENERK"
                    }
                  ]
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/chai1/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "num_trunk_recycles": 4,
                    "num_diffusion_timesteps": 180,
                    "num_diffn_samples": 1,
                    "use_esm_embeddings": true,
                    "seed": 42,
                    "include": []
                  },
                  "items": [
                    {
                      "molecules": [
                        {
                          "name": "TestProtein",
                          "type": "protein",
                          "sequence": "MAAASNDENERK"
                        }
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

            url <- "https://biolm.ai/api/v3/chai1/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                num_trunk_recycles = 4,
                num_diffusion_timesteps = 180,
                num_diffn_samples = 1,
                use_esm_embeddings = TRUE,
                seed = 42,
                include = list()
              ),
              items = list(
                list(
                  molecules = list(
                    list(
                      name = "TestProtein",
                      type = "protein",
                      sequence = "MAAASNDENERK"
                    )
                  )
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/chai1/predict/

   Predict endpoint for Chai-1.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **num_trunk_recycles** (*int*, range: 1-10, default: 3) — Number of trunk recycles

        - **num_diffusion_timesteps** (*int*, range: 50-200, default: 200) — Number of diffusion timesteps

        - **num_diffn_samples** (*int*, range: 1-5, default: 1) — Number of diffusion samples per input item

        - **use_esm_embeddings** (*bool*, default: True) — Whether to use ESM embeddings

        - **seed** (*int*, default: 42) — Random seed for stochastic components

        - **include** (*array of strings*, allowed: ["pae", "plddt"], default: []) — Per-residue or pairwise score types to request (always overridden to [])


      - **items** (*array of objects*, min: 1, max: 1) --- Input items for prediction:

        - **molecules** (*array of objects*, min: 1, max: 5) — Molecules in the complex:

          - **name** (*string*, required) — Molecule identifier

          - **type** (*string*, allowed: "protein", "RNA", "DNA", "ligand", "polymer_hybrid", "water", "unknown", required) — Molecule type

          - **sequence** (*string*, optional) — Primary sequence; proteins (max length: 1024, unambiguous amino acids), DNA/RNA (max length: 3072, unambiguous bases), ligands (max length: 128)

          - **smiles** (*string*, optional) — SMILES representation for ligand molecules

          - **alignment** (*object*, optional) — Multiple sequence alignment data for protein molecules:

            - **mgnify** (*string*, optional) — Encoded alignment content from the "mgnify" database

            - **small_bfd** (*string*, optional) — Encoded alignment content from the "small_bfd" database

            - **uniref90** (*string*, optional) — Encoded alignment content from the "uniref90" database

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/chai1/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "num_trunk_recycles": 4,
          "num_diffusion_timesteps": 180,
          "num_diffn_samples": 1,
          "use_esm_embeddings": true,
          "seed": 42,
          "include": []
        },
        "items": [
          {
            "molecules": [
              {
                "name": "TestProtein",
                "type": "protein",
                "sequence": "MAAASNDENERK"
              }
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

      - **results** (*array of arrays of objects*) --- One result per input item, in the order requested:

        - **cif** (*string*) — Predicted structure in mmCIF format

        - **pae** (*array of arrays of floats*, optional) — Predicted aligned error matrix with shape [N, N], where N is the number of residues in the predicted model

        - **plddt** (*array of floats*, optional) — Per-residue confidence scores with length N, where N is the number of residues in the predicted model

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "cif": "data_model\n_entry.id model\n_struct.entry_id model\n_struct.pdbx_model_details .\n_struct.pdbx_structure_determination_methodology computational\n_struct.title 'Chai-1 predicted structure'\n_audit_conform.... (truncated for documentation)"
            }
          ]
        ]
      }


Performance
-----------

- Chai-1 inference is hosted on NVIDIA A100 80GB GPUs, matching the reference hardware in the technical report, and runs in a BioLM-optimized pipeline for multi-entity complexes (proteins, nucleic acids, ligands). Batch size is fixed to 1 per request item, with up to 5 molecules per complex.

- Structural accuracy is state-of-the-art across key benchmarks and generally exceeds other BioLM structural predictors on interaction tasks:
  
  - Protein–ligand (PoseBusters): 77% ligand RMSD success (RMSD < 2 Å), comparable to AlphaFold3 (76%) and substantially higher than RoseTTAFold All-Atom (42%).
  - Protein–protein interfaces: 75.1% DockQ success (DockQ > 0.23), outperforming AlphaFold-Multimer 2.3 (67.7%); for antibody–protein interfaces, 52.9% vs. 38.0% DockQ success.
  - Protein monomers (CASP15): average LDDT 0.849 vs. 0.843 for AlphaFold-Multimer 2.3 and 0.801 for ESM3.

- Single-sequence mode (no MSAs, using protein language model embeddings) maintains high accuracy and reduces upstream alignment cost, and outperforms other BioLM-hosted single-sequence predictors under comparable conditions:
  
  - Protein–protein DockQ success (no MSA): 69.8% (Chai-1) vs. 67.7% (AlphaFold-Multimer 2.3 with MSAs).
  - Antibody–protein DockQ success (no MSA): 47.9% (Chai-1) vs. 38.0% (AlphaFold-Multimer 2.3 with MSAs).

- Confidence metrics and geometry are suitable for downstream ranking without extra post-processing, which simplifies large-scale screening workflows compared to some other models:
  
  - ipTM, pLDDT, and PAE correlate strongly with true structure quality across protein–protein, protein–ligand, and protein–nucleic acid benchmarks, enabling reliable selection of higher-quality predictions.
  - Predicted complexes show low inter-molecular clash rates on PoseBusters, so BioLM does not apply additional clash penalties when ranking Chai-1 samples, unlike typical workflows for some other models (e.g., AlphaFold3).

Applications
------------

- Protein–small molecule complex prediction to prioritize medicinal chemistry leads; Chai-1 models ligand binding directly from protein sequence and ligand SMILES, helping pharma and biotech teams triage virtual screening hits and rationalize SAR, while users should visually inspect poses, especially for highly flexible ligands or very large assemblies where pose accuracy can decrease.
- Antibody–antigen interface modeling to support therapeutic antibody optimization and affinity maturation; useful for mapping paratope–epitope contacts and comparing design variants from sequence, with accuracy further improved when users incorporate externally derived epitope or contact restraints, but very high-resolution predictions still typically require experimental follow-up.
- Multimeric protein complex structure prediction to guide protein engineering and synthetic biology; enables assessment of hetero-oligomeric interfaces or scaffolds when experimental structures are unavailable, including single-sequence mode without MSAs for faster iteration, though relative chain orientations can occasionally be mis-modeled without external structural or biophysical constraints.
- Single-sequence protein structure prediction for rapid exploration of immunological and other highly variable protein design spaces; allows high-throughput in silico screening of sequence variants directly from FASTA without requiring MSAs, suitable for iterative design cycles, but generally underperforms full-MSA runs when rich evolutionary information is available.
- Protein–nucleic acid (DNA/RNA) complex modeling to support gene-editing systems and RNA therapeutics; enables prediction of CRISPR-associated protein–DNA interfaces or RNA-binding protein complexes from sequence inputs, although accuracy is typically lower than for protein-only targets and can often be improved by combining predictions with nucleic acid-specific data or experimental constraints.

Limitations
-----------

- **Maximum Sequence Length**: Protein ``sequence`` values are limited to ``1024`` residues, RNA/DNA ``sequence`` values to ``3072`` bases (``Chai1Params.max_rna_dna_len``), and ligand ``sequence`` values to ``128`` characters (``Chai1Params.max_ligand_len``). Inputs exceeding these limits are rejected.
- **Batch Size and Items**: ``batch_size`` is fixed at ``1`` (``Chai1Params.batch_size``). Each ``Chai1PredictRequest`` may contain only one ``items`` entry, and each ``items[0].molecules`` list may include at most ``5`` molecules (``Chai1Params.max_fasta_entries``). ``molecules`` must be non-empty.
- **Entity Types and Alignments**: ``Chai1Molecule.type`` must be one of ``"protein"``, ``"RNA"``, ``"DNA"``, ``"ligand"``, ``"polymer_hybrid"``, ``"water"``, or ``"unknown"``. The ``alignment`` field is only valid for ``"protein"`` molecules; providing it for other types will fail validation.
- **Modified Residues and Non‑standard Chemistry**: Chai-1 predictions are sensitive to modified residues and other non-standard components. Removing modifications or mapping them to canonical residues or generic tokens can substantially change predicted structures; treat modified and unmodified inputs as distinct systems.
- **Complex Assembly Accuracy**: Chai-1 may correctly fold individual chains but mis-position them relative to one another, especially for large complexes or when no experimental restraints (e.g. contact or pocket constraints) are provided. For unconstrained global docking of very large assemblies, additional modeling or filtering is often required.
- **Use Cases Where Other Models May Be Preferable**: For very high-throughput single-chain folding without MSAs or templates (e.g. ranking millions of designed monomers), simpler single-sequence models such as ESMFold can be faster and cheaper. For detailed, unconstrained antibody–antigen interface design requiring many high-accuracy poses, specialized antibody tools or AF-Multimer-style workflows may still be necessary.

How We Use It
-------------

BioLM uses Chai-1 as a core structure-prediction component to rapidly evaluate 3D configurations of proteins, nucleic acids, and ligands within design campaigns. Standardized, scalable API calls generate CIF structures and calibrated confidence metrics (pLDDT, PAE) that we integrate with sequence embeddings, biophysical filters, and generative models to prioritize variants for synthesis, guide multi-round antibody maturation, and refine protein–ligand or antibody–antigen interfaces, including cases informed by experimental constraints such as mapped epitopes or cross-linking data.

- Integrates with BioLM’s generative design and ranking workflows to shorten protein engineering and antibody optimization cycles.
- Uses Chai-1 confidence metrics alongside thermodynamic- and sequence-derived properties to focus lab effort on high-value candidates.

Related
-------

- ``AlphaFold2`` – High-accuracy protein monomer and complex prediction; useful as a baseline or secondary method to compare against Chai-1 protein-only results.
- ``ESMFold`` – Single-sequence protein folding; practical for rapid protein-only predictions without MSAs, and a point of comparison to Chai-1's single-sequence mode.
- ``ESM-IF1`` – Protein–protein interface prediction; can be used to sanity-check or prioritize interfaces within Chai-1 multimer predictions.
- ``ABodyBuilder3 pLDDT`` – Antibody structure prediction; complements Chai-1 for antibody-only modeling and for refining antibody chains in antibody–antigen complexes.

References
----------

- Chai Discovery team (2024). *Chai-1: Decoding the molecular interactions of life*. bioRxiv. `https://doi.org/10.1101/2024.10.10.615955`
