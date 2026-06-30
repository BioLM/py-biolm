DIAMOND API
===========

DIAMOND is a fast, sensitive protein sequence alignment service for large-scale homology search, matching BLASTP-level sensitivity in very-sensitive and ultra-sensitive modes while providing ~80–360× speedups. The API accepts protein sequences (validated amino-acid alphabet) and runs configurable DIAMOND sensitivity modes against selectable protein databases, returning BLAST-like hits with identities, bit scores, *e*-values, and CIGAR strings. It supports high-throughput comparisons for functional annotation, orthology inference, gene age estimation, and metagenomic protein profiling.

Predict
-------

Run DIAMOND protein alignment search against specified databases

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="diamond",
                action="predict",
                params={
                  "e_value": 0.001,
                  "max_hits": 5,
                  "block_size": 6.0,
                  "sensitivity": "default",
                  "databases": [
                    "diamond"
                  ]
                },
                items=[
                  {
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                  },
                  {
                    "sequence": "GHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/diamond/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "e_value": 0.001,
                "max_hits": 5,
                "block_size": 6.0,
                "sensitivity": "default",
                "databases": [
                  "diamond"
                ]
              },
              "items": [
                {
                  "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                },
                {
                  "sequence": "GHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/diamond/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "e_value": 0.001,
                    "max_hits": 5,
                    "block_size": 6.0,
                    "sensitivity": "default",
                    "databases": [
                      "diamond"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
                    },
                    {
                      "sequence": "GHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/diamond/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                e_value = 0.001,
                max_hits = 5,
                block_size = 6.0,
                sensitivity = "default",
                databases = list(
                  "diamond"
                )
              ),
              items = list(
                list(
                  sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"
                ),
                list(
                  sequence = "GHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/diamond/predict/

   Predict endpoint for DIAMOND.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **e_value** (*float*, default: 0.001) — E-value threshold for reported alignments

        - **max_hits** (*int*, range: 1-100, default: 1) — Maximum number of matches returned per query sequence

        - **block_size** (*float*, default: 6.0) — DIAMOND block size setting

        - **sensitivity** (*string*, default: "default") — DIAMOND sensitivity mode identifier

        - **databases** (*array of strings*, default: ["diamond"]) — Names of databases to search


      - **items** (*array of objects*, min: 1, max: 300000) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 2000, required) — Protein sequence using the extended amino acid alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/diamond/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "e_value": 0.001,
          "max_hits": 5,
          "block_size": 6.0,
          "sensitivity": "default",
          "databases": [
            "diamond"
          ]
        },
        "items": [
          {
            "sequence": "MKTAYIAKQRQISFVKSHFSRQDILD"
          },
          {
            "sequence": "GHHHHHHSSGLVPRGSHMASMTGGQQMGRGS"
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

        - **id** (*string*) — Query sequence identifier (SHA256 hash)

        - **sequences** (*array of objects*) — Matched database sequences for this query, length: 0–`max_hits`


        - **sequences[].query_id** (*string*) — Query sequence identifier (SHA256 hash)

        - **sequences[].subject_id** (*string*) — Subject sequence identifier from database

        - **sequences[].subject_seq** (*string*) — Full subject amino acid sequence

        - **sequences[].identity** (*float*, range: 0.0–100.0, unit: percent) — Percentage sequence identity over the alignment

        - **sequences[].alignment_length** (*int*, ≥ 0, unit: residues) — Length of the alignment in residues

        - **sequences[].e_value** (*float*, ≥ 0.0) — E-value of the match

        - **sequences[].score** (*float*) — Bit score of the match

        - **sequences[].cigar** (*string*) — CIGAR string encoding the alignment

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "id": "9b4c85aefb1f4ea71daa1477812280ee58e881454c3c49743719dad740747e2c",
            "sequences": [
              {
                "query_id": "9b4c85aefb1f4ea71daa1477812280ee58e881454c3c49743719dad740747e2c",
                "subject_id": "BL_ORD_ID:834878",
                "subject_seq": "MATATSASLFSTVSSSYSKASSIPHSRLQSVKFNSVPSFTGLKSTSLISGSDSSSLAKTLRGSVTKAQTSDKKPYGFKINAMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDW... (truncated for documentation)",
                "identity": 100.0,
                "alignment_length": 22,
                "e_value": 6.13e-07,
                "score": 46.2,
                "cigar": "22M"
              },
              {
                "query_id": "9b4c85aefb1f4ea71daa1477812280ee58e881454c3c49743719dad740747e2c",
                "subject_id": "BL_ORD_ID:431261",
                "subject_seq": "LLYCRPVLLLNQWQQDAGVIKMKTAYIAKQRQISFVKSHFSRQLEEKLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKTLPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLTPIHSVYVDQWDWERVMGDEERHVGTLKATVEAIYAGIKATELAVSQEFGLTPFLPEQIHFVHSQELLSRYPE... (truncated for documentation)",
                "identity": 100.0,
                "alignment_length": 22,
                "e_value": 6.16e-07,
                "score": 46.2,
                "cigar": "22M"
              }
            ]
          },
          {
            "id": "f7ccf4c1378ba7ef4d204b5df3cc0c4f4c5c3f42a6bcacd0a7ea84cedadc06e2",
            "sequences": [
              {
                "query_id": "f7ccf4c1378ba7ef4d204b5df3cc0c4f4c5c3f42a6bcacd0a7ea84cedadc06e2",
                "subject_id": "BL_ORD_ID:394169",
                "subject_seq": "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGS",
                "identity": 100.0,
                "alignment_length": 30,
                "e_value": 6.01e-10,
                "score": 50.8,
                "cigar": "30M"
              },
              {
                "query_id": "f7ccf4c1378ba7ef4d204b5df3cc0c4f4c5c3f42a6bcacd0a7ea84cedadc06e2",
                "subject_id": "BL_ORD_ID:94946",
                "subject_seq": "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQACGRTRAPPPPPLRSGC",
                "identity": 100.0,
                "alignment_length": 30,
                "e_value": 9.75e-10,
                "score": 50.8,
                "cigar": "30M"
              },
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Algorithm throughput and speed:
  - DIAMOND v2.x on BioLM typically delivers 80–360× higher throughput than BLASTP-like CPU homology search at BLAST-comparable sensitivity, and up to ~8000× faster in the least-sensitive modes on large databases (per Buchfink et al., Nat. Methods 2021)
  - On equivalent datasets, DIAMOND v2 is reported to be ~12–15× faster than MMseqs2 at similar sensitivity; BioLM’s multi-core CPU deployment preserves this gap in API workloads
  - Relative to embedding-based pipelines using BioLM’s large protein language models (e.g., ESM-2 3B, Evo 2 1B plus vector search), DIAMOND is usually 5–20× faster end-to-end for medium-scale annotation tasks (<10^6 queries) because it avoids high-dimensional embedding inference
- Alignment sensitivity and accuracy:
  - In ``--ultrasensitive`` mode DIAMOND matches or slightly exceeds BLASTP v2.10.0 in AUC1 sensitivity at low false-positive rates while retaining ~80× speedup on NCBI nr vs. UniRef50-scale benchmarks
  - ``--very-sensitive`` maintains BLAST-like recall for most protein families with reduced runtime relative to ``--ultrasensitive``, making it well suited for large-scale annotation when extremely remote homology is not the main goal
  - DIAMOND’s tantan-based repeat masking and composition-based score adjustments reduce spurious low-complexity matches compared with plain BLASTP, improving precision in large, repetitive proteomes
- Scaling and resource behavior on BioLM:
  - DIAMOND’s double indexing, spaced seeding, and SIMD-optimized filters exhibit near-linear strong scaling across CPU cores and nodes in published HPC benchmarks (281M × 39M all-vs-all alignment in <18 hours on 20,800 cores); BioLM uses the same distributed-chunking strategy to sustain high throughput
  - Compared with single-GPU sequence models on BioLM (e.g., ESM-2 3B, Evo 2 1B used for embeddings), DIAMOND scales more efficiently with additional CPU cores and disk bandwidth and is typically the most cost-efficient option for bulk homology search (10^5–10^6 queries) with explicit local alignments
  - On-the-fly indexing keeps memory usage below typical protein language models or structure predictors even in ultra-sensitive modes, allowing many queries to be processed in parallel per worker
- Role relative to other BioLM tools:
  - DIAMOND returns local alignments with bit scores and E-values rather than embedding-only similarity, making thresholds more interpretable than scores from models such as ESM-2 or Evo 2
  - In enzyme design, antibody optimization, and variant triage pipelines, DIAMOND is usually used as a high-precision, high-throughput pre-filter before slower generative or structure-based models (e.g., ProGen2, Boltz-2, ProteinMPNN, TEMPRO), where it contributes a small fraction of total runtime while providing most of the search-space reduction

Applications
------------

- High-throughput homology search in enzyme engineering campaigns, using DIAMOND’s very-sensitive or ultra-sensitive modes to align up to hundreds of thousands of designed protein variants per API request against curated reference panels (for example UniRef50 or proprietary enzyme libraries). This enables detection of distant functional homologs and domain architectures that would be impractical with BLASTP at comparable sensitivity and scale. DIAMOND is most effective when you can batch large variant sets; for very small panels, simpler local BLAST workflows may be easier to manage.
- Tree-of-life scale target scouting for biocatalyst discovery, by aligning metagenomic or proprietary protein catalogs against large reference databases to detect remote homologs of industrial enzyme classes (for example transaminases, dehydrogenases, glycosidases). Using the sensitivity parameter and database selection in the API, teams can systematically mine broad phylogenetic diversity and prioritize families that conserve key catalytic motifs. This is particularly valuable when screening tens to hundreds of millions of sequences; for kilobyte-scale datasets, local HMM or BLAST searches may be sufficient.
- Large-scale off-target and liability assessment for therapeutic protein constructs, calling the DIAMOND predictor endpoint to align engineered proteins (for example Fc-fusions, recombinant growth factors, engineered scaffolds) across comprehensive proteome collections from human and relevant preclinical species. This helps flag unexpected homology to proteins associated with toxicity, autoimmunity, or other liabilities before in vivo work. DIAMOND here provides rapid, proteome-wide coverage but does not replace specialized immunogenicity or structural epitope prediction tools and should be one component of a broader safety workflow.
- Functional annotation and clustering of proprietary protein design libraries, by aligning millions of in silico–generated variants through the API against annotated reference sets (for example Swiss-Prot subsets or internal gold-standard enzymes) to infer putative function, EC class, and domain composition, and to group sequences into similarity-based families. DIAMOND’s speed and sensitivity allow frequent re-annotation of evolving design spaces during iterative optimization, with the understanding that definitive functional assignment still requires complementary methods such as structure prediction or experimental assays.
- Cross-species comparability mapping for assay and manufacturability platforms, using DIAMOND to align production or assay protein constructs (for example expression-optimized variants, purification tags) against diverse host, feeder, and common contaminant proteomes to evaluate potential cross-reactivity or analytical interference. This supports selection of host strains, tags, and detection reagents that minimize homology to background proteins. DIAMOND scales efficiently across large databases via its underlying distributed implementation, but remains a sequence-only method and will not capture post-translational modifications or higher-order structural effects without integration with additional tools.

Limitations
-----------

- **Maximum sequence length**: Each ``sequence`` in ``items`` must be an amino-acid string of length ``1``–``2000`` (``DiamondParams.max_sequence_len``). Longer proteins must be truncated or split into windows, which can miss homologous regions outside the queried segment and may not reflect full-length domain architectures.
- **Batch size and throughput**: Each request must contain between ``1`` and ``300000`` query sequences in ``items`` (``DiamondParams.max_sequences_per_request``). Although DIAMOND scales well to large batches, very large requests can result in long-running jobs and very large JSON responses; for interactive or UI-driven workflows, prefer smaller batches and paginate your query set.
- **Search space and result volume**: ``DiamondPredictRequestParams.max_hits`` is limited to ``100`` per query, so this API returns the top-scoring matches rather than an exhaustive all-vs-all alignment. Large ``databases`` lists combined with permissive ``e_value`` thresholds can still create substantial outputs; if you primarily need family-level calls or summary statistics, plan on downstream aggregation instead of storing every ``DiamondMatch``.
- **Sensitivity/speed trade-offs**: ``DiamondPredictRequestParams.sensitivity`` selects DIAMOND’s internal sensitivity mode and directly trades speed for recall. Even in its most sensitive modes, DIAMOND operates at BLASTP-like but not higher-than-BLAST theoretical sensitivity; for extremely remote homology, short low-complexity motifs, or cases where marginal ``e_value`` hits must be exhaustively explored, profile/HMM-based tools (for example, HMMER) or BLASTP itself may be more appropriate.
- **Algorithmic scope and biological context**: This endpoint performs local pairwise protein alignments only. It does not compute structure, embeddings, function annotations, or de novo designs. If you need tasks such as 3D structure ranking, protein design, or embedding-based clustering, use DIAMOND as a homology search component within a broader pipeline that includes structure or representation-learning models.
- **Result interpretation and edge cases**: Each ``DiamondMatch`` (under ``DiamondPredictResponse.results[].sequences``) reports statistics such as ``identity``, ``alignment_length``, ``e_value``, ``score``, and ``cigar`` for a local alignment against ``subject_seq``. Despite internal repeat masking, low-complexity regions, compositional bias, and highly repetitive proteins can still yield artifactual or non-orthologous hits; for such inputs or for error-prone translated data, consider additional filtering or complementary methods before making strong functional or evolutionary inferences.

How We Use It
-------------

BioLM uses DIAMOND as a scalable homology search layer that feeds standardized, high-confidence annotations into downstream protein ML workflows. DIAMOND similarity searches are exposed as APIs that map new protein sequences (up to 2,000 residues and 300,000 queries per request) to curated reference databases, and the resulting alignments are converted into feature sets that integrate with generative models, structure predictors, and sequence-embedding pipelines. This alignment context improves candidate triage in enzyme and antibody programs, supports rapid off-target and immunogenicity risk assessment, and links in silico design with laboratory data by providing a consistent way to relate designed variants to natural diversity at tree-of-life scale.

- DIAMOND outputs are transformed into ML-ready features (for example, homology profiles, domain- and family-level hits, distance and similarity scores) that are consumed by BioLM design, ranking, and optimization APIs.  
- Combined with structure-based and physicochemical property models, DIAMOND-based homology information accelerates multi-round optimization campaigns by prioritizing variants that balance novelty with conserved functional and developability signals.

Related
-------

- ``MMseqs2`` – Alternative large-scale sequence search and clustering engine; useful as a baseline against DIAMOND or for fast pre-clustering before high-sensitivity DIAMOND searches.
- ``ESM-2 650M`` – Protein language model for embeddings; you can use DIAMOND to retrieve homologs and then apply ESM-2 embeddings for function prediction, clustering, or similarity analysis within the DIAMOND hit sets.
- ``ESMFold`` – Structure prediction from sequence; run DIAMOND to identify homologous proteins or families, then submit representative hits or orthologs to ESMFold to obtain 3D structures.
- ``AlphaFold2`` – High-accuracy structure prediction; combine DIAMOND-based homolog and ortholog search with AlphaFold2 to build comparative structural models across protein families returned by DIAMOND.

References
----------

- Buchfink, B., Reuter, K., & Drost, H.-G. (2021). `Sensitive protein alignments at tree-of-life scale using DIAMOND <https://doi.org/10.1038/s41592-021-01101-x>`_. *Nature Methods*, 18, 366–368.

