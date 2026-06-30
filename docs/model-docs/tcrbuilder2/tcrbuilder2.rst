TCRBuilder2 API
===============

TCRBuilder2 is a deep learning model for predicting paired T-cell receptor (TCR) α/β variable domain 3D structures directly from amino acid sequences. It is trained on TCR-specific structural data and predicts backbone and side-chain coordinates for all residues, including CDR loops, with mean CDR RMSDs on benchmark sets typically in the 1.8–3.0 Å range. The API returns refined PDB-format structures for batched TCR pairs, supporting high-throughput structural annotation, repertoire analysis, and structure-guided TCR design workflows.

Predict
-------

Predict TCR structure from alpha and beta chain sequences with TCRBuilder2

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="tcrbuilder2",
                action="predict",
                params={},
                items=[
                  {
                    "A": "AQSVTQPSHQVSLGQTVTLSCNYTSSDFQYWYRQNSGTLQLLLKYTAATLTKGINDFAAELKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                    "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                  },
                  {
                    "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVQKNGQKLIFGKGTRLHILP",
                    "B": "ADVTQTPRNLITKTGKRIMLQCSQTQGRDRMYWYRQDPGLGLRLIYYSLDVKDINKGEISDGYSVSRQAQAKFSLSLDSAIPNQTALYFCASSYLGSGNTGQLYYGYTFGSGTRLTVV"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/tcrbuilder2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "A": "AQSVTQPSHQVSLGQTVTLSCNYTSSDFQYWYRQNSGTLQLLLKYTAATLTKGINDFAAELKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                  "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                },
                {
                  "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVQKNGQKLIFGKGTRLHILP",
                  "B": "ADVTQTPRNLITKTGKRIMLQCSQTQGRDRMYWYRQDPGLGLRLIYYSLDVKDINKGEISDGYSVSRQAQAKFSLSLDSAIPNQTALYFCASSYLGSGNTGQLYYGYTFGSGTRLTVV"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/tcrbuilder2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "A": "AQSVTQPSHQVSLGQTVTLSCNYTSSDFQYWYRQNSGTLQLLLKYTAATLTKGINDFAAELKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                      "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                    },
                    {
                      "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVQKNGQKLIFGKGTRLHILP",
                      "B": "ADVTQTPRNLITKTGKRIMLQCSQTQGRDRMYWYRQDPGLGLRLIYYSLDVKDINKGEISDGYSVSRQAQAKFSLSLDSAIPNQTALYFCASSYLGSGNTGQLYYGYTFGSGTRLTVV"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/tcrbuilder2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  A = "AQSVTQPSHQVSLGQTVTLSCNYTSSDFQYWYRQNSGTLQLLLKYTAATLTKGINDFAAELKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
                  B = "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
                ),
                list(
                  A = "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVQKNGQKLIFGKGTRLHILP",
                  B = "ADVTQTPRNLITKTGKRIMLQCSQTQGRDRMYWYRQDPGLGLRLIYYSLDVKDINKGEISDGYSVSRQAQAKFSLSLDSAIPNQTALYFCASSYLGSGNTGQLYYGYTFGSGTRLTVV"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/tcrbuilder2/predict/

   Predict endpoint for TCRBuilder2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 8) — Number of sequences processed per batch (maximum: 8)
        - **max_sequence_len** (*int*, default: 2048) — Maximum allowed length of input sequences (maximum: 2048)

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **H** (*string*, optional, min length: 1, max length: 2048) — Amino acid sequence of antibody heavy chain or nanobody
        - **L** (*string*, optional, min length: 1, max length: 2048) — Amino acid sequence of antibody light chain
        - **A** (*string*, optional, min length: 1, max length: 2048) — Amino acid sequence of T-cell receptor alpha chain
        - **B** (*string*, optional, min length: 1, max length: 2048) — Amino acid sequence of T-cell receptor beta chain

        Valid combinations of fields within each item (exactly one combination required):

          - **H** and **L** provided — Antibody (ABodyBuilder2)
          - **H** provided alone — Nanobody (NanoBodyBuilder2)
          - **A** and **B** provided — T-cell receptor (TCRBuilder2, TCRBuilder2PLUS)

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for NanoBodyBuilder2:

        - **H** (*string*, required, min length: 1, max length: 2048) — Amino acid sequence of nanobody

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for ABodyBuilder2:

        - **H** (*string*, required, min length: 1, max length: 2048) — Amino acid sequence of antibody heavy chain
        - **L** (*string*, required, min length: 1, max length: 2048) — Amino acid sequence of antibody light chain

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for TCRBuilder2:

        - **A** (*string*, required, min length: 1, max length: 2048) — Amino acid sequence of T-cell receptor alpha chain
        - **B** (*string*, required, min length: 1, max length: 2048) — Amino acid sequence of T-cell receptor beta chain

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/tcrbuilder2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "A": "AQSVTQPSHQVSLGQTVTLSCNYTSSDFQYWYRQNSGTLQLLLKYTAATLTKGINDFAAELKKSETSFHLTKPSAHMSDAAEYFCAVSEQDDKIIFGKGTRLHILP",
            "B": "ADVTQTPRNRITKTGKRIMLECSQTKGHDRMYWYRQDPGLGLRLIYYSFDVKDINKGEISDGYSVSRQAQAKFSLSLESAIPNQTALYFCATSDESYGYTFGSGTRLTVV"
          },
          {
            "A": "AQSVTQLGSHVSVSEGALVLLRCNYSSSVPPYLFWYVQYPNQGLQLLLKYTSAATLVKGINGFEAEFKKSETSFHLTKPSAHMSDAAEYFCAVQKNGQKLIFGKGTRLHILP",
            "B": "ADVTQTPRNLITKTGKRIMLQCSQTQGRDRMYWYRQDPGLGLRLIYYSLDVKDINKGEISDGYSVSRQAQAKFSLSLDSAIPNQTALYFCASSYLGSGNTGQLYYGYTFGSGTRLTVV"
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

        - **pdb** (*string*) — Predicted immune protein structure in standard PDB format; includes atomic coordinates for backbone and side-chain atoms; structure generated by ImmuneBuilder deep-learning models (ABodyBuilder2, NanoBodyBuilder2, or TCRBuilder2); accuracy comparable to AlphaFold-Multimer; typical backbone RMSD for CDR loops ranges from approximately 0.4Å to 3.5Å (depending on loop type and model variant); stereochemically refined to remove clashes, cis-peptide bonds, and nonphysical bond lengths; coordinates provided in Angstroms (Å)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "REMARK  TCR STRUCTURE MODELLED USING TCRBUILDER2                                \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-19                          \nATOM      1  N   ALA B   2       1.833... (truncated for documentation)"
          },
          {
            "pdb": "REMARK  TCR STRUCTURE MODELLED USING TCRBUILDER2                                \nREMARK  STRUCTURE REFINED USING OPENMM 8.2, 2025-06-19                          \nATOM      1  N   ALA B   2       0.076... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- CPU-only inference with lightweight memory footprint: TCRBuilder2 runs on 2 vCPUs with 8 GB RAM per worker, with batch processing of up to 8 TCRs per request and up to 2048 amino acids per chain (alpha and beta) as enforced by the API schema.
- Predictive accuracy on TCRs is comparable to AlphaFold-Multimer while outperforming earlier immune-specific models:
  
  - On the ImmuneBuilder benchmark, TCRBuilder2 achieves mean backbone RMSDs of 1.85 Å (CDR-α3) and 1.93 Å (CDR-β3), matching AlphaFold-Multimer (1.84 Å and 1.94 Å) and improving on the original TCRBuilder by ~1 Å on key CDR loops.
  - Framework regions are predicted with mean RMSDs of 0.90 Å (Fw-α) and 0.67 Å (Fw-β), outperforming RepertoireBuilder and TCRBuilder and on par with or better than AlphaFold-Multimer across most regions.
- Compared with other BioLM structure predictors, TCRBuilder2 offers a favorable accuracy–cost trade-off for TCRs:
  
  - Versus general protein models (e.g., AlphaFold2/Multimer-like deployments), TCRBuilder2 reaches similar backbone accuracy for TCR variable domains at a fraction of the computational cost because it does not require MSAs or large sequence databases.
  - Within the ImmuneBuilder family, TCRBuilder2 provides TCR-specific accuracy analogous to ABodyBuilder2 and NanoBodyBuilder2 on antibodies and nanobodies, making it preferable to general-purpose folding models for TCR repertoires.
- Structures are refined for physical plausibility before being returned, with stereochemical quality comparable to crystal structures:
  
  - Benchmarking shows negligible stereochemical violations (no cis-peptide bonds, no D-amino acids, and no peptide-bond length outliers), making TCRBuilder2 outputs suitable for downstream modeling, docking, and design workflows without additional cleanup.

Applications
------------

- Rapid in silico prediction of paired TCR (α/β) variable-domain structures for engineered T-cell therapies, enabling screening and prioritization of TCR candidates for desired antigen recognition and cross-reactivity profiles; useful for biotech companies designing personalized cancer or infectious-disease TCR therapies, but not intended for non-TCR or non-immune protein modeling.
- Structure-guided analysis of TCR–peptide–MHC binding regions by inspecting CDR loop conformations and predicted chemical surfaces in the TCR variable domains, supporting rational mutation design and affinity tuning of therapeutic TCRs; valuable for companies optimizing efficacy and reducing off-target recognition, though detailed complex-level energetics still require docking and experimental follow-up.
- High-throughput structural characterization of TCR repertoires derived from next-generation sequencing by converting many α/β sequence pairs into 3D variable-domain models, enabling clustering by structural similarity and identification of recurrent binding-site geometries linked to disease or treatment response; appropriate for repertoire-scale immune profiling, but does not build full-length membrane-embedded TCR complexes.
- Early-stage assessment of developability risks in therapeutic TCRs by inspecting predicted variable-domain structures for features associated with instability or aggregation (e.g., exposed hydrophobic patches, unusual loop conformations), helping teams filter or re-engineer candidates before costly lab work; not a replacement for biophysical stability or manufacturability assays.
- Generation of multiple TCR structural hypotheses per sequence pair and use of per-residue uncertainty to flag low-confidence regions, allowing researchers to focus experimental characterization on high-confidence models and to treat low-confidence CDR segments as flexible or multiconformational; useful when experimental throughput is limited, though conformational dynamics of highly flexible loops may still be underestimated.

Limitations
-----------

- **Maximum Sequence Length**: Each TCR chain ``A`` and ``B`` must be between ``1`` and ``2048`` amino acids. Longer sequences must be truncated or split before submission.
- **Batch Size**: ``items`` in a ``predictor`` request is limited to ``8`` TCRs (alpha/beta pairs). For larger datasets, split into batches of ``8`` or fewer items.
- **TCR-Specific Model**: TCRBuilder2 is optimized for paired T-cell receptor variable domains (Vα/Vβ). It will not reliably predict antibody or nanobody structures; for those, use ``abodybuilder2`` or ``nanobodybuilder2`` instead.
- **Sequence and repertoire coverage**: Performance is best for typical, IMGT-like TCR variable domains. Non-standard constructs, chimeric designs, or sequences far from known TCR repertoires (e.g. highly engineered loops, unusual domain boundaries) may be modelled with reduced accuracy.
- **Single-conformation output**: For each input TCR pair, the API returns one refined 3D structure in ``pdb`` format. It does not sample multiple conformational states, so it is not suitable when explicit conformational heterogeneity is required.
- **No additional scores or embeddings**: The response includes only the ``pdb`` structure. It does not expose per-residue error estimates, confidence scores, or sequence/structure embeddings; use separate tools if you need these features.

How We Use It
-------------

BioLM uses TCRBuilder2 to rapidly generate 3D structures of paired TCR α/β chains from sequence, enabling structural assessment of candidate receptors at the same scale as repertoire sequencing and generative design. TCRBuilder2 structures are combined with BioLM affinity and developability predictors to prioritize TCRs with favorable CDR loop conformations, epitope accessibility, and biophysical profiles for downstream optimization and experimental validation.

- Accelerates iterative TCR design cycles by turning sequence proposals into structural models suitable for docking, filtering, and ranking.
- Integrates with antibody and nanobody structure prediction (ABodyBuilder2, NanoBodyBuilder2) to support cross-modality immune therapeutic discovery in a single, standardized API workflow.

Related
-------

- ``TCRBuilder2+`` – Updated TCRBuilder2 weights with improved accuracy for TCR α/β structure prediction; available via the same TCRBuilder2 API.
- ``ImmuneFold TCR`` – Alternate TCR-focused structure predictor, useful for cross-checking and benchmarking TCRBuilder2 results.
- ``NanoBodyBuilder2`` – ImmuneBuilder nanobody model; complements TCRBuilder2 when modelling single-chain VHH immune receptors.
- ``ABodyBuilder2`` – ImmuneBuilder antibody model for paired VH/VL structures, enabling comparative studies across antibody and TCR repertoires.

References
----------

- Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). `ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins <https://doi.org/10.1038/s42003-023-04927-7>`_. *Communications Biology*.

