AlphaFold2 API
==============

AlphaFold2 is a GPU-accelerated protein structure prediction model that infers atomic 3D coordinates from amino acid sequences using MSAs and structural templates, with typical backbone accuracies near 1 Å r.m.s.d.\ :sub:`95` on CASP14 targets. The BioLM API exposes encoder and predictor endpoints: the encoder builds MSAs/templates from configurable databases (MGnify, small BFD, UniRef90) using MMseqs2, and the predictor runs end-to-end folding with up to 512 residues and 4,000 MSA sequences, returning full-length PDB models suitable for large-scale structure annotation and protein engineering workflows.

Predict
-------

Predicts 3D structure for a protein sequence

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="alphafold2",
                action="predict",
                params={
                  "databases": [
                    "mgnify",
                    "small_bfd",
                    "uniref90"
                  ],
                  "predictions_per_model": 1,
                  "relax": "none",
                  "return_templates": true,
                  "msa_iterations": 1,
                  "max_msa_sequences": 1000,
                  "algorithm": "mmseqs2"
                },
                items=[
                  {
                    "sequence": "MAAAAAAGAGPEMVRGQVFDVGPR"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/alphafold2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "MAAAAAAGAGPEMVRGQVFDVGPR"
                }
              ],
              "params": {
                "databases": [
                  "mgnify",
                  "small_bfd",
                  "uniref90"
                ],
                "predictions_per_model": 1,
                "relax": "none",
                "return_templates": true,
                "msa_iterations": 1,
                "max_msa_sequences": 1000,
                "algorithm": "mmseqs2"
              }
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/alphafold2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "MAAAAAAGAGPEMVRGQVFDVGPR"
                    }
                  ],
                  "params": {
                    "databases": [
                      "mgnify",
                      "small_bfd",
                      "uniref90"
                    ],
                    "predictions_per_model": 1,
                    "relax": "none",
                    "return_templates": true,
                    "msa_iterations": 1,
                    "max_msa_sequences": 1000,
                    "algorithm": "mmseqs2"
                  }
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/alphafold2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "MAAAAAAGAGPEMVRGQVFDVGPR"
                )
              ),
              params = list(
                databases = list(
                  "mgnify",
                  "small_bfd",
                  "uniref90"
                ),
                predictions_per_model = 1,
                relax = "none",
                return_templates = TRUE,
                msa_iterations = 1,
                max_msa_sequences = 1000,
                algorithm = "mmseqs2"
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/alphafold2/predict/

   Predict endpoint for AlphaFold2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*) --- Configuration parameters:

        - **databases** (*array of strings*, default: ["mgnify", "small_bfd", "uniref90"]) — Sequence databases to search; allowed values: "small_bfd", "mgnify", "uniref90"

        - **predictions_per_model** (*integer*, default: 1, range: 1–8) — Number of structure predictions to generate per internal model

        - **relax** (*string*, default: "none") — Post-processing relaxation mode; allowed values: "all", "best", "none"

        - **return_templates** (*boolean*, default: true) — Whether to include template information in the internal processing

        - **msa_iterations** (*integer*, default: 1, range: 1–5) — Number of multiple sequence alignment refinement iterations

        - **max_msa_sequences** (*integer*, optional, default: null, range: 1–4000) — Maximum number of sequences to retain in the multiple sequence alignment

        - **algorithm** (*string*, optional, default: "mmseqs2") — MSA search algorithm; allowed value: "mmseqs2"


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 512) — Protein sequence using extended amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/alphafold2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "MAAAAAAGAGPEMVRGQVFDVGPR"
          }
        ],
        "params": {
          "databases": [
            "mgnify",
            "small_bfd",
            "uniref90"
          ],
          "predictions_per_model": 1,
          "relax": "none",
          "return_templates": true,
          "msa_iterations": 1,
          "max_msa_sequences": 1000,
          "algorithm": "mmseqs2"
        }
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **pdbs** (*array of strings*, size: 1–8) — Predicted protein structures in PDB format, one entry per generated model per input sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdbs": [
              "ATOM      1  N   MET A   1      24.392  -8.460 -14.399  1.00 47.22           N  \nATOM      2  CA  MET A   1      24.158  -8.618 -12.966  1.00 47.22           C  \nATOM      3  C   MET A   1      23.267... (truncated for documentation)",
              "ATOM      1  N   MET A   1      16.309  -0.252 -21.014  1.00 46.71           N  \nATOM      2  CA  MET A   1      15.302  -1.265 -20.707  1.00 46.71           C  \nATOM      3  C   MET A   1      14.836... (truncated for documentation)",
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Encode
------

Get MSAs for a protein sequence - compatible with AlphaFold2, Chai1, and other models

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="alphafold2",
                action="encode",
                params={
                  "databases": [
                    "mgnify",
                    "small_bfd",
                    "uniref90"
                  ],
                  "return_templates": false,
                  "msa_iterations": 2,
                  "max_msa_sequences": 1000,
                  "algorithm": "mmseqs2"
                },
                items=[
                  {
                    "sequence": "YDNVNKVRVAIKKISPFEHQGQVDVTYAMK"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/alphafold2/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "YDNVNKVRVAIKKISPFEHQGQVDVTYAMK"
                }
              ],
              "params": {
                "databases": [
                  "mgnify",
                  "small_bfd",
                  "uniref90"
                ],
                "return_templates": false,
                "msa_iterations": 2,
                "max_msa_sequences": 1000,
                "algorithm": "mmseqs2"
              }
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/alphafold2/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "YDNVNKVRVAIKKISPFEHQGQVDVTYAMK"
                    }
                  ],
                  "params": {
                    "databases": [
                      "mgnify",
                      "small_bfd",
                      "uniref90"
                    ],
                    "return_templates": false,
                    "msa_iterations": 2,
                    "max_msa_sequences": 1000,
                    "algorithm": "mmseqs2"
                  }
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/alphafold2/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "YDNVNKVRVAIKKISPFEHQGQVDVTYAMK"
                )
              ),
              params = list(
                databases = list(
                  "mgnify",
                  "small_bfd",
                  "uniref90"
                ),
                return_templates = FALSE,
                msa_iterations = 2,
                max_msa_sequences = 1000,
                algorithm = "mmseqs2"
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/alphafold2/encode/

   Encode endpoint for AlphaFold2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **databases** (*array of strings*, default: ["mgnify", "small_bfd", "uniref90"], possible values: "small_bfd", "mgnify", "uniref90") — Databases used to build multiple sequence alignments

        - **return_templates** (*boolean*, optional, default: true) — Whether to include template search results

        - **msa_iterations** (*integer*, range: 1..5, default: 1) — Number of iterations for the MSA search

        - **max_msa_sequences** (*integer*, range: 1..4000, optional, default: null) — Upper limit on the number of sequences kept in the MSA

        - **algorithm** (*string*, optional, default: "mmseqs2", possible values: "mmseqs2") — Algorithm used for the MSA search


      - **items** (*array of objects*, min items: 1, max items: 1) --- Input data:

        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using supported amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/alphafold2/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "YDNVNKVRVAIKKISPFEHQGQVDVTYAMK"
          }
        ],
        "params": {
          "databases": [
            "mgnify",
            "small_bfd",
            "uniref90"
          ],
          "return_templates": false,
          "msa_iterations": 2,
          "max_msa_sequences": 1000,
          "algorithm": "mmseqs2"
        }
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **alignments** (*object*) — Alignment data grouped by database

          - **small_bfd** (*array of strings*, optional) — Serialized MSA data from the small_bfd database, including header, aligned sequences, and format tag

          - **mgnify** (*array of strings*, optional) — Serialized MSA data from the mgnify database, including header, aligned sequences, and format tag

          - **uniref90** (*array of strings*, optional) — Serialized MSA data from the uniref90 database, including header, aligned sequences, and format tag


        - **templates** (*array of objects*, optional) — Template hit records, empty if no templates are returned

          - **index** (*integer*) — Template index within the hit list

          - **name** (*string*) — Template identifier

          - **aligned_cols** (*integer*) — Number of aligned residue columns in the hit

          - **sum_probs** (*float*) — Sum of per-column alignment probabilities for the hit

          - **query** (*string*) — Query subsequence used in the alignment

          - **hit_sequence** (*string*) — Template subsequence aligned to the query

          - **indices_query** (*array of integers*) — 0-based or 1-based query residue positions included in the alignment

          - **indices_hit** (*array of integers*) — 0-based or 1-based template residue positions included in the alignment

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "alignments": {
              "small_bfd": [
                "small_bfd",
                ">DNHYJGOIDVCTWIW\nYDNVNKVRVAIKKISPFEHQGQVDVTYAMK\n",
                "a3m"
              ],
              "mgnify": [
                "mgnify",
                ">CZPVQDXKUAWVQDB\nYDNVNKVRVAIKKISPFEHQGQVDVTYAMK\n",
                "a3m"
              ],
              "uniref90": [
                "uniref90",
                ">GCAAJQDKIOCJTJR\nYDNVNKVRVAIKKISPFEHQGQVDVTYAMK\n",
                "a3m"
              ]
            },
            "templates": []
          }
        ]
      }


Performance
-----------

- AlphaFold2 achieves near-experimental backbone accuracy on challenging targets, with a median Cα r.m.s.d.\ :sub:`95` of ~0.96 Å on CASP14 domains, versus ~2.8 Å for the next-best method; BioLM’s deployment typically retains this accuracy profile for monomeric and homomeric proteins within the supported sequence length (≤512 residues).
- Side-chain conformations are highly reliable when the backbone is accurate, with median all-atom r.m.s.d.\ :sub:`95` ≈1.5 Å compared to ≈3.5 Å for alternative methods, and predicted local-distance difference test (pLDDT) closely tracks true lDDT-Cα, providing a robust per-residue confidence metric for downstream filtering.
- Compared with ESMFold on BioLM, AlphaFold2 is substantially more accurate at atomic detail and for complex or novel folds but is slower and more computationally expensive per prediction; conversely, AlphaFold2 underperforms specialized nanobody models such as NanoBodyBuilder2 on single-domain nanobody targets.
- BioLM’s AlphaFold2 pipeline uses MMseqs2-based MSA generation (faster than Jackhmmer-based workflows) and GPU-accelerated inference on 3 × NVIDIA A100 80 GB GPUs with iterative recycling and equivariant attention, enabling high-accuracy predictions even for deep MSAs (optimal gains up to ~30–100 effective MSA sequences, with diminishing returns beyond).

Applications
------------

- Rapid prediction of monomeric protein 3D structures directly from amino acid sequences, enabling faster iteration in protein engineering programs by reducing dependence on X-ray, NMR, or cryo-EM for initial structural hypotheses.
- High-resolution backbone and side-chain modeling to inform rational mutagenesis, supporting stability engineering, active-site redesign, and manufacturability assessments for industrial enzymes and other commercial proteins.
- Structural characterization of novel or poorly annotated protein domains, including sequences without close structural homologs, to de-risk synthetic biology designs such as biosensors, binders, and protein-based materials.
- In silico triage of designed or variant protein sequences by predicting which are likely to fold into well-ordered, single-chain structures, allowing teams to prioritize a small set of candidates for experimental expression and biophysical screening.
- Estimation of putative protein–protein interaction surfaces for homomeric assemblies or fused domains within a single polypeptide, useful for designing linkers and multi-domain constructs, with the caveat that accuracy decreases when function depends strongly on heteromeric partners, ligands, or other cofactors not modeled by the API.

Limitations
-----------

- **Maximum Sequence Length**: The AlphaFold2 API accepts protein sequences up to ``512`` amino acids via the ``sequence`` field. Longer proteins must be truncated or segmented and modeled separately.
- **Batch Size**: **Batch Size** is limited to ``1`` sequence per request (``items`` list). To process multiple sequences, submit separate ``encoder`` or ``predictor`` requests.
- **MSA Depth and Quality**: Prediction accuracy depends strongly on the depth and quality of the MSA. Accuracy degrades when the median alignment depth falls below ~30 effective sequences. The API allows up to ``4000`` MSA sequences (via ``max_msa_sequences``); setting this too low or using shallow databases may reduce accuracy.
- **Protein Complexes and Context**: This AlphaFold2 configuration predicts single-chain structures. It generally performs well on monomers and homomers, but is less reliable for chains whose native fold depends on extensive heterotypic contacts in larger complexes or on specific ligands, ions, or cofactors that are not explicitly modeled.
- **Computational Cost and Runtime**: AlphaFold2 is computationally intensive. Sequences near the ``512`` residue limit, higher ``msa_iterations`` (1–5), and using multiple ``databases`` can substantially increase wall-clock time; long jobs may run close to the API timeout of 3 hours.
- **Structural Relaxation**: Enabling ``relax`` (``all`` or ``best``) in ``AF2NIMPredictParams`` adds an Amber-based post-processing step that improves stereochemistry but typically does not change global accuracy. Relaxation increases runtime and is usually only needed for downstream modeling or visualization.

How We Use It
-------------

BioLM uses AlphaFold2 as a structural oracle within protein design campaigns, generating atomic-resolution models and confidence scores that guide sequence selection, risk assessment, and experiment planning across enzyme engineering, antibody optimization, and de novo design. Standardized prediction and MSA-encoding APIs integrate these structures with sequence language models, docking, and property predictors, enabling scalable in silico screening and iterative lab-in-the-loop optimization while prioritizing variants with reliable local pLDDT-supported geometry.

- Provides structure-aware features to refine predictive models, improve enrichment of desired biophysical properties, and de-risk experimental rounds.
- Integrates with generative design and structure-based property predictors to shorten design cycles and focus synthesis on the most promising protein candidates.

Related
-------

- ``ESMFold`` – Fast single-sequence protein structure prediction; useful for rapid screening or when deep MSAs for AlphaFold2 are not available.
- ``ABodyBuilder3 pLDDT`` – Antibody-focused structure prediction with per-residue confidence; preferred over AlphaFold2 for variable regions and antibody developability workflows.
- ``ImmuneFold Antibody`` – Specialized antibody and immune receptor modeling; complements AlphaFold2 when modeling antibody–antigen interfaces or immune-specific scaffolds.
- ``ProstT5 AA2Fold`` – Sequence-to-structure generation leveraging protein language modeling; supports exploration of designed variants and comparative analysis of AlphaFold2 predictions.

References
----------

- Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S. A. A., Ballard, A. J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., Back, T., Petersen, S., Reiman, D., Clancy, E., Zielinski, M., Steinegger, M., Pacholska, M., Berghammer, T., Bodenstein, S., Silver, D., Vinyals, O., Senior, A. W., Kavukcuoglu, K., Kohli, P., & Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589. https://doi.org/10.1038/s41586-021-03819-2
