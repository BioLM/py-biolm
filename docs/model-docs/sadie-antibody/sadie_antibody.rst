Sadie Antibody API
==================

Sadie Antibody (SADIE) provides AIRR-compliant BCR/TCR annotation and antibody numbering, assigning V/J genes, chain type, CDR and framework regions, and per-region identities and scores directly from amino acid sequences. The API supports configurable numbering schemes (Chothia, IMGT, Kabat) and region definitions (including ABM, contact, SCDR), with batch processing of up to 8 sequences of length ≤2048. Results are returned as structured tables suitable for repertoire analysis, clonotyping, lineage tracing, and therapeutic antibody design workflows.

Predict
-------

Predict properties or scores for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="sadie-antibody",
                action="predict",
                params={
                  "region_assign": "imgt",
                  "scheme": "chothia",
                  "scfv": false,
                  "allowed_chain": [
                    "H",
                    "K",
                    "L"
                  ]
                },
                items=[
                  {
                    "sequence": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/sadie-antibody/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "region_assign": "imgt",
                "scheme": "chothia",
                "scfv": false,
                "allowed_chain": [
                  "H",
                  "K",
                  "L"
                ]
              },
              "items": [
                {
                  "sequence": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/sadie-antibody/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "region_assign": "imgt",
                    "scheme": "chothia",
                    "scfv": false,
                    "allowed_chain": [
                      "H",
                      "K",
                      "L"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/sadie-antibody/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                region_assign = "imgt",
                scheme = "chothia",
                scfv = FALSE,
                allowed_chain = list(
                  "H",
                  "K",
                  "L"
                )
              ),
              items = list(
                list(
                  sequence = "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/sadie-antibody/predict/

   Predict endpoint for Sadie Antibody.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **region_assign** (*enum: [imgt, kabat, chothia, abm, contact, scdr]*, default: "imgt", optional) — Region definition used for framework and CDR segmentation

        - **scheme** (*enum: [imgt, kabat, chothia]*, default: "chothia", optional) — Numbering scheme applied to sequences

        - **scfv** (*boolean*, default: false) — Whether to allow single-chain Fv sequences

        - **allowed_chain** (*array of strings*, default: ["H", "K", "L"]) — Chain types to include; each element must be one of ["L", "H", "K", "A", "B", "G", "D"]


      - **items** (*array of objects*, min: 1, max: 8, required) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2048, required) — Amino acid sequence using extended amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/sadie-antibody/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "region_assign": "imgt",
          "scheme": "chothia",
          "scfv": false,
          "allowed_chain": [
            "H",
            "K",
            "L"
          ]
        },
        "items": [
          {
            "sequence": "QVQLVQSGAEVKKPGASVKVSCKVSGYTSPTTIHWVRQAPGKGLEWMGGISPYRGDTIYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYCARDGYSSGYYGMDVWGQGTTVTVSS"
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

        - **domain_no** (*int*) --- Zero-based index of the matched antibody domain

        - **hmm_species** (*string*) --- Species label from HMM alignment (e.g. "human")

        - **chain_type** (*string*) --- Single-letter chain type inferred by HMM (e.g. "H", "K", "L")

        - **e_value** (*float*) --- HMM alignment E-value

        - **score** (*float*) --- HMM alignment score

        - **identity_species** (*string*) --- Species with highest sequence identity

        - **v_gene** (*string*) --- Top V gene call, including allele (e.g. "IGHV1-24*01")

        - **v_identity** (*float*) --- V gene identity as a fraction (0.0–1.0)

        - **j_gene** (*string*) --- Top J gene call, including allele (e.g. "IGHJ6*01")

        - **j_identity** (*float*) --- J gene identity as a fraction (0.0–1.0)

        - **Chain** (*string*) --- Single-letter chain label matching the numbered chain

        - **Numbering** (*array of ints*, length ≤ 2048) --- Residue numbering assigned to each input position

        - **Insertion** (*array of strings*, length ≤ 2048) --- Insertion codes for each numbered position

        - **scheme** (*string*) --- Applied numbering scheme (e.g. "chothia", "kabat", "imgt")

        - **region_definition** (*string*) --- Applied CDR/framework region definition (e.g. "imgt", "kabat")

        - **fwr1_aa_gaps** (*string*) --- FWR1 amino acid sequence including internal gaps

        - **fwr1_aa_no_gaps** (*string*) --- FWR1 amino acid sequence with gaps removed

        - **cdr1_aa_gaps** (*string*) --- CDR1 amino acid sequence including internal gaps

        - **cdr1_aa_no_gaps** (*string*) --- CDR1 amino acid sequence with gaps removed

        - **fwr2_aa_gaps** (*string*) --- FWR2 amino acid sequence including internal gaps

        - **fwr2_aa_no_gaps** (*string*) --- FWR2 amino acid sequence with gaps removed

        - **cdr2_aa_gaps** (*string*) --- CDR2 amino acid sequence including internal gaps

        - **cdr2_aa_no_gaps** (*string*) --- CDR2 amino acid sequence with gaps removed

        - **fwr3_aa_gaps** (*string*) --- FWR3 amino acid sequence including internal gaps

        - **fwr3_aa_no_gaps** (*string*) --- FWR3 amino acid sequence with gaps removed

        - **cdr3_aa_gaps** (*string*) --- CDR3 amino acid sequence including internal gaps

        - **cdr3_aa_no_gaps** (*string*) --- CDR3 amino acid sequence with gaps removed

        - **fwr4_aa_gaps** (*string*) --- FWR4 amino acid sequence including internal gaps

        - **fwr4_aa_no_gaps** (*string*) --- FWR4 amino acid sequence with gaps removed

        - **leader** (*string*) --- Amino acids before the aligned domain

        - **follow** (*string*) --- Amino acids after the aligned domain

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "domain_no": 0,
            "hmm_species": "human",
            "chain_type": "H",
            "e_value": 0.0,
            "score": 176.25,
            "identity_species": "human",
            "v_gene": "IGHV1-24*01",
            "v_identity": 0.87,
            "j_gene": "IGHJ6*01",
            "j_identity": 1.0,
            "Chain": "H",
            "Numbering": [
              1,
              2,
              "... (truncated for documentation)"
            ],
            "Insertion": [
              "",
              "",
              "... (truncated for documentation)"
            ],
            "scheme": "chothia",
            "region_definition": "imgt",
            "fwr1_aa_gaps": "QVQLVQSGAEVKKPGASVKVSCKVS",
            "fwr1_aa_no_gaps": "QVQLVQSGAEVKKPGASVKVSCKVS",
            "cdr1_aa_gaps": "GYTSP-TT",
            "cdr1_aa_no_gaps": "GYTSPTT",
            "fwr2_aa_gaps": "IHWVRQAPGKGLEWMGG",
            "fwr2_aa_no_gaps": "IHWVRQAPGKGLEWMGG",
            "cdr2_aa_gaps": "ISPYRGDT",
            "cdr2_aa_no_gaps": "ISPYRGDT",
            "fwr3_aa_gaps": "IYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYC",
            "fwr3_aa_no_gaps": "IYAQKFQGRVTMTEDTSTDTAYMELSSLKSEDTAVYYC",
            "cdr3_aa_gaps": "ARDGYSSGYYGMDV",
            "cdr3_aa_no_gaps": "ARDGYSSGYYGMDV",
            "fwr4_aa_gaps": "WGQGTTVTVSS",
            "fwr4_aa_no_gaps": "WGQGTTVTVSS",
            "leader": "",
            "follow": ""
          }
        ]
      }


Performance
-----------

- CPU-optimized HMM-based implementation for antibody numbering and region annotation, deployed on lightweight containers (0.125 vCPU, 1 GB RAM per worker), allowing many concurrent requests without GPU dependency.
- Throughput and latency are substantially better than BLAST-based tools (e.g., IgBLAST) for V/J assignment and region delineation, and typically faster than ANARCI-style HMM pipelines at comparable accuracy due to streamlined SADIE integration and minimal I/O overhead.
- Within the BioLM portfolio, Sadie Antibody is among the most efficient sequence-analysis APIs per CPU minute: markedly cheaper and lower-latency than large protein language model–based annotation workflows, while providing similar or better alignment- and region-level accuracy for standard antibody engineering tasks.

Applications
------------

- Antibody sequence annotation to identify CDRs and framework regions, enabling systematic engineering workflows such as affinity maturation, de-immunization, and epitope mapping by precisely localizing sequence changes to functional regions
- Clustering antibodies by CDR amino acid similarity using SADIE-compatible outputs to group related clones or lineages in large discovery campaigns, supporting repertoire down-selection and hit expansion; clustering works best within reasonably related sequence sets and may be less informative for highly divergent repertoires
- AIRR-compliant annotation outputs that interoperate with downstream immunoinformatics tools (e.g., immcantation pipelines, AIRR-compliant databases), simplifying integration of SADIE-based workflows into existing discovery, developability, and clinical sequence analysis pipelines
- Antibody variable region numbering with common schemes (Chothia, Kabat, IMGT) and region definitions (e.g., IMGT, Kabat, Chothia, ABM, contact, SCDR) to standardize residue indexing across internal and external datasets, facilitating structural modeling, mutational scanning analysis, IP comparisons, and regulatory documentation; numbering is intended for conventional immunoglobulin chains and may not generalize to non-standard antibody-like formats
- Generation of residue-level annotated sequence data (including V/J gene calls, species assignment, per-region sequences, and numbering arrays) that can be used as structured features for downstream computational workflows such as repertoire profiling, developability rule filters, or ML models for antibody design and liability prediction, but is not a substitute for 3D structural modeling or docking analyses

Limitations
-----------

- **Maximum Sequence Length**: Each ``sequence`` must be at most ``2048`` amino acids (``SADIEParams.max_sequence_len``). Longer sequences must be truncated or split before calling ``predictor``.

- **Batch Size**: Each ``predictor`` request must contain between ``1`` and ``8`` items (``SADIEParams.batch_size``). Larger datasets should be processed in multiple requests.

- **Numbering Schemes and Regions**: ``scheme`` is limited to ``imgt``, ``kabat``, or ``chothia`` (``SADIENumbering``). ``region_assign`` is limited to ``imgt``, ``kabat``, ``chothia``, ``abm``, ``contact``, or ``scdr`` (``SADIERegion``); custom schemes or region definitions are not supported.

- **Chain Type Constraints**: ``allowed_chain`` must be a subset of ``["L", "H", "K", "A", "B", "G", "D"]`` and defaults to heavy/light chains (``["H", "K", "L"]``). Other chain types or non-standard receptor formats may fail HMM assignment or produce low-quality annotations.

- **Species and Germline Coverage**: ``hmm_species``, ``v_gene``, and ``j_gene`` in the results are derived from SADIE’s built-in germline database, which is curated mainly for human and common model organisms. Rare species, heavily engineered antibodies, or highly divergent germlines may be mis-assigned or left unannotated.

- **Non-optimal Use Cases**: SADIE is optimized for antibody/BCR/TCR variable-region annotation and numbering. It is not ideal for full-length Ig constructs with long constant regions, non-immunoglobulin proteins, or tasks such as affinity prediction, developability scoring, or structure modeling; other BioLM models are more appropriate for those use cases.

How We Use It
-------------

Sadie Antibody enables standardized numbering, region parsing, and light/heavy chain typing for antibody amino acid sequences, giving downstream design models consistent CDR and framework definitions across large datasets. We integrate Sadie outputs directly into generative design, developability prediction, and clustering workflows, so candidate ranking, epitope-focused mutagenesis, and sequence liability analysis all operate on the same AIRR-compliant region boundaries and V/J gene calls.

- Supports IMGT, Kabat, and Chothia schemes and multiple region definitions (e.g., ABM, contact, SCDR) via a single scalable API, simplifying alignment of internal and partner pipelines.
- Accelerates lead selection by turning raw sequences into structured features (CDRs, FWRs, gene usage, species identity) that can be filtered, embedded, and iteratively optimized across design-make-test cycles.

Related
-------

- ``ImmuneFold Antibody`` – Complements ``Sadie Antibody`` by predicting 3D antibody structures directly from SADIE-annotated variable-region sequences.
- ``ABodyBuilder3 pLDDT`` – Can be used after ``Sadie Antibody`` to build antibody structures and obtain per-residue confidence scores for modeled regions.
- ``IgT5 Paired`` – Uses annotated paired chains to generate and optimize antibody sequences, making it a natural downstream design tool for ``Sadie Antibody`` annotations.
- ``IgBert Paired`` – Provides sequence embeddings for paired antibody chains that can be combined with ``Sadie Antibody`` AIRR annotations for clustering, repertoire analysis, and design.

References
----------

- Walker, L. M., Phogat, S. K., Chan-Hui, P. Y., Wagner, D., Phung, P., Goss, J. L., Wrin, T., Simek, M. D., Fling, S., Mitcham, J. L., Lehrman, J. K., Priddy, F. H., Olsen, O. A., Frey, S. M., Hammond, P. W., Kaminsky, S., Zamb, T., Moyle, M., Koff, W. C., Poignard, P., & Burton, D. R. (2009). `Broad and potent neutralizing antibodies from an African donor reveal a new HIV-1 vaccine target <https://doi.org/10.1126/science.1178746>`_. *Science*, 326(5950), 285–289.

- Deli, A., Kurella, V. B., & Kelsoe, G. (2020). `HuGL mouse models for the study of human antibody repertoires <https://doi.org/10.3389/fimmu.2020.01947>`_. *Frontiers in Immunology*, 11, 1947.

- Dunbar, J., & Deane, C. M. (2016). `ANARCI: antigen receptor numbering and receptor classification <https://doi.org/10.1093/bioinformatics/btv552>`_. *Bioinformatics*, 32(2), 298–300.

- Martin, A. C. R. (2023). `Antibody Numbering and CDR Definitions <http://www.bioinf.org.uk/abs/info.html>`_. *Bioinformatics Group, University College London*.

- Lefranc, M.-P., Giudicelli, V., Duroux, P., Jabado-Michaloud, J., Folch, G., Aouinti, S., Carillon, E., Duvergey, H., Houles, A., Paysan-Lafosse, T., Hadi-Saljoqi, S., Sasorith, S., Lefranc, G., & Kossida, S. (2015). `IMGT®, the international ImMunoGeneTics information system® 25 years on <https://doi.org/10.1093/nar/gku1056>`_. *Nucleic Acids Research*, 43(D1), D413–D422.

- Ye, J., Ma, N., Madden, T. L., & Ostell, J. M. (2013). `IgBLAST: an immunoglobulin variable domain sequence analysis tool <https://doi.org/10.1093/nar/gkt382>`_. *Nucleic Acids Research*, 41(W1), W34–W40.

- Vander Heiden, J. A., Marquez, S., Marthandan, N., Bukhari, S. A. C., Busse, C. E., Corrie, B., Hershberg, U., Kleinstein, S. H., Matsen, F. A., Ralph, D. K., Rosenfeld, A. M., Schramm, C. A., Christley, S., & Laserson, U. (2018). `AIRR Community standardized representations for annotated immune repertoires <https://doi.org/10.3389/fimmu.2018.02206>`_. *Frontiers in Immunology*, 9, 2206.

- Willis, J. R., Sincomb, T., & Kibet, C. K. (2023). `SADIE: Sequencing Analysis and Data Library for Immunoinformatics Exploration <https://sadie.jordanrwillis.com>`_. *GitHub repository*.
