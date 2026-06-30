ESMFold API
===========

ESMFold is a GPU-accelerated protein structure prediction model that infers atomic-level 3D coordinates directly from amino acid sequences using ESM-2 language model representations, without MSAs or templates. The API supports single chains and multimeric complexes with up to 4 chains encoded as colon-separated sequences, each request batching up to 2 items and sequences up to 768 residues. It returns PDB structures with mean pLDDT and pTM confidence scores, enabling high-throughput workflows in protein engineering, metagenomics, and structural biology.

Predict
-------

Predict 3D structure for one or more protein chains (separated by colons) using ESMFold

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esmfold",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "GAMEDTQVAW"
                  },
                  {
                    "sequence": "MKTIIALSYIFCLVFADYKDDDD:VLLPAGKQ"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esmfold/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "sequence": "GAMEDTQVAW"
                },
                {
                  "sequence": "MKTIIALSYIFCLVFADYKDDDD:VLLPAGKQ"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esmfold/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "sequence": "GAMEDTQVAW"
                    },
                    {
                      "sequence": "MKTIIALSYIFCLVFADYKDDDD:VLLPAGKQ"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esmfold/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  sequence = "GAMEDTQVAW"
                ),
                list(
                  sequence = "MKTIIALSYIFCLVFADYKDDDD:VLLPAGKQ"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esmfold/predict/

   Predict endpoint for ESMFold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 2) — Maximum number of items processed per request
        - **max_sequence_len** (*int*, default: 768) — Maximum number of amino acid residues per chain in a sequence
        - **max_n_multimers** (*int*, default: 4) — Maximum number of chains allowed in a multimer sequence

      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:

        - **sequence** (*string*, min length: 1, max length: 771, required) — Protein sequence using the extended amino acid alphabet with ":" as chain separator, allowing up to 3 non-consecutive ":" characters

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmfold/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "sequence": "GAMEDTQVAW"
          },
          {
            "sequence": "MKTIIALSYIFCLVFADYKDDDD:VLLPAGKQ"
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

        - **pdb** (*string*) — Predicted protein structure in PDB format, including coordinates and standard PDB records

        - **mean_plddt** (*float*) — Mean predicted Local Distance Difference Test (pLDDT) confidence score over all residues

        - **ptm** (*float*) — Predicted Template Modeling (pTM) score estimating global fold accuracy

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "PARENT N/A\nATOM      1  N   GLY A   1       2.704  13.046  14.934  1.00 67.45           N  \nATOM      2  CA  GLY A   1       1.649  12.125  15.325  1.00 67.60           C  \nATOM      3  C   GLY A   1 ... (truncated for documentation)",
            "mean_plddt": 64.428955078125,
            "ptm": 0.012728862464427948
          },
          {
            "pdb": "PARENT N/A\nATOM      1  N   MET A   1       1.328 -16.623 -16.398  1.00 42.52           N  \nATOM      2  CA  MET A   1       2.126 -17.137 -15.289  1.00 44.35           C  \nATOM      3  C   MET A   1 ... (truncated for documentation)",
            "mean_plddt": 47.16063690185547,
            "ptm": 0.10539554059505463
          }
        ]
      }


Performance
-----------

- ESMFold runs inference on NVIDIA A10G GPUs with 4 vCPUs and 16 GB RAM, using mixed-precision computation, chunked attention, and optional CPU offloading to efficiently handle single- or multi-chain inputs up to 768 residues total.
- Inference speed is tuned for rapid turnaround: for a 384-residue monomer, network forward time is around 14 seconds on a single GPU, typically ~6× faster than a single AlphaFold2 model and >60× faster than AlphaFold2 on short sequences (<200 residues) when MSA search is considered.
- Structural accuracy is competitive with other single-sequence and MSA-based predictors: on CASP14 targets, ESMFold reaches mean LDDT ≈ 0.68 (vs. ≈ 0.85 for AlphaFold2 and ≈ 0.81 for RoseTTAFold). On the CAMEO benchmark, mean TM-scores are ≈ 0.90 (easy), 0.79 (medium), and 0.45 (hard), slightly below AlphaFold2 but close to RoseTTAFold on easy and medium targets.
- Confidence estimates (mean pLDDT and pTM) are well calibrated: mean pLDDT strongly correlates with realized accuracy (LDDT/TM-score), enabling downstream pipelines to filter or prioritize BioLM ESMFold predictions by quality when trading off against slower, more accurate models such as AlphaFold2 or against nanobody-specialized models such as NanobodyBuilder.

Applications
------------

- Rapid single- and few-chain structure prediction in protein engineering pipelines, enabling teams to assess structural viability of designed variants in seconds without MSA generation; accelerates iterative design for therapeutics or industrial biocatalysts within the API limits of up to 4 chains and 768 residues total.
- High-throughput structural screening of protein variant or metagenomic libraries by ranking candidates using mean pLDDT and pTM scores from the API response; useful for prioritizing stable folds and well-packed cores for downstream experimental validation; less suitable for detailed modeling of large complexes or interfaces beyond four chains.
- Structural annotation of orphan or low-homology proteins in discovery programs, providing predicted 3D coordinates (PDB output) and confidence metrics directly from sequence; supports target selection and domain boundary assessment when experimental structures or deep MSAs are unavailable; accuracy decreases on very hard, novel folds, so predictions should be combined with additional evidence.
- Computational ranking of protein design candidates, combining ESMFold mean pLDDT and pTM with other design scores to focus wet-lab screening on structurally plausible designs; reduces cost by deprioritizing clearly misfolded or disordered variants; not recommended as the sole criterion for functional optimization, as activity and specificity still require additional modeling or experiments.

Limitations
-----------

- **Maximum Sequence Length**: The ``sequence`` input (a single chain or multiple chains separated by ``:``) must not exceed ``768`` amino acids in total. Longer proteins must be truncated or split into separate API calls.
- **Batch Size**: The ``items`` list in ``ESMFoldPredictRequest`` supports at most ``batch_size`` = ``2`` sequences per request. Larger sets of sequences must be split across multiple requests.
- **Multimeric Input**: Multimer predictions are supported for up to ``4`` chains per ``sequence``, separated by ``:`` characters. Complexes with more than 4 chains must be decomposed into smaller sub-complexes or modeled chain-wise.
- **Protein Complexes**: ESMFold was trained only on single-chain structures. Multimer predictions are therefore out-of-distribution and typically less accurate than specialized multimer models (for example AlphaFold-Multimer), especially for detailed interface geometry.
- **Accuracy vs. Speed**: ESMFold is much faster than AlphaFold2 but usually less accurate on difficult targets (e.g. low-MSA-depth CASP/CAMEO-style benchmarks). For final ranking of a small number of critical designs, higher-accuracy MSA-based models are often preferable.
- **Confidence Estimation (pLDDT / pTM)**: The response fields ``mean_plddt`` and ``ptm`` correlate with model accuracy, but low-confidence predictions (``mean_plddt`` < 70) or highly novel/orphan sequences should be treated cautiously and ideally cross-checked with additional models or experimental data.

How We Use It
-------------

ESMFold enables rapid, high-throughput prediction of protein structures directly from sequence data, accelerating protein design and engineering workflows by removing the need for multiple sequence alignment searches. Within BioLM-driven design cycles, ESMFold predictions feed standardized, API-level structural features (coordinates, pLDDT, pTM) into downstream thermodynamic, biophysical, and sequence-based models, allowing teams to quickly prioritize variants for synthesis and testing in enzyme optimization, antibody maturation, and targeted protein modification campaigns.

- Supports iterative protein engineering by quickly assessing structural impact of sequence changes across single chains and small multimers (up to 4 chains in one request).
- Integrates with BioLM scoring and filtering models to focus experimental resources on candidates with favorable predicted structure, stability, and developability profiles.

Related
-------

- ``AlphaFold2`` – Complementary MSA-based structure predictor; useful for benchmarking accuracy and confidence estimates against ESMFold’s single-sequence predictions.
- ``ESM-2 3B`` – Underlying protein language model whose frozen embeddings ESMFold uses to predict 3D structures directly from sequence.
- ``RoseTTAFold`` – Alternative deep learning-based structure prediction model; useful for cross-validating ESMFold results on challenging targets.
- ``ESM-IF1`` – Inverse folding model that can be paired with ESMFold to design sequences compatible with predicted backbones and to evaluate alternative sequences for a given structure.

References
----------

- Lin, Z., et al. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*, 379(6637), 1123–1130.
