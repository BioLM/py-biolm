ABodyBuilder3 Language API
==========================

ABodyBuilder3 is an antibody-specific model for predicting 3D Fv structures from paired heavy (H) and light (L) chain amino acid sequences. It builds on ImmuneBuilder/ABodyBuilder2-style architectures to generate all-atom PDB structures and can optionally return per-residue pLDDT confidence scores. The API accepts one VH/VL pair per request (batch size 1, up to 2048 residues per chain) and runs on CPU or GPU backends for integration into antibody engineering, repertoire analysis, and therapeutic discovery pipelines.

Predict
-------

Predict the 3D structure for an antibody heavy/light chain pair.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="abodybuilder3-language",
                action="predict",
                params={
                  "plddt": false,
                  "seed": 42
                },
                items=[
                  {
                    "H": "ACDEFGHIKLMNPQRSTVWY",
                    "L": "ACDEFGHIKL"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/abodybuilder3-language/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "H": "ACDEFGHIKLMNPQRSTVWY",
                  "L": "ACDEFGHIKL"
                }
              ],
              "params": {
                "plddt": false,
                "seed": 42
              }
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/abodybuilder3-language/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "H": "ACDEFGHIKLMNPQRSTVWY",
                      "L": "ACDEFGHIKL"
                    }
                  ],
                  "params": {
                    "plddt": false,
                    "seed": 42
                  }
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/abodybuilder3-language/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  H = "ACDEFGHIKLMNPQRSTVWY",
                  L = "ACDEFGHIKL"
                )
              ),
              params = list(
                plddt = FALSE,
                seed = 42
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/abodybuilder3-language/predict/

   Predict endpoint for ABodyBuilder3 Language.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **plddt** (*bool*, default: False) — Whether to include per-residue pLDDT scores in the response

        - **seed** (*int*, optional, default: 42) — Random seed for stochastic components of the prediction

      - **items** (*array of objects*, min items: 1, max items: 1) --- Input antibody chains:

        - **H** (*string*, min length: 1, max length: 2048) — Heavy chain amino acid sequence (extended amino acid alphabet)

        - **L** (*string*, min length: 1, max length: 2048) — Light chain amino acid sequence (extended amino acid alphabet)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/abodybuilder3-language/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "H": "ACDEFGHIKLMNPQRSTVWY",
            "L": "ACDEFGHIKL"
          }
        ],
        "params": {
          "plddt": false,
          "seed": 42
        }
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **pdb** (*string*) — Predicted antibody structure in PDB format

        - **plddt** (*array of arrays of floats*, optional) — Predicted per-residue confidence scores (pLDDT), values range from 0.0 (low confidence) to 100.0 (high confidence); shape: [number_of_chains][number_of_residues]

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "REMARK   1 CREATED WITH OPENMM 7.7, 2025-06-17\nATOM      1  N   ALA H   0      -1.825 -13.109   4.977  1.00  0.00           N  \nATOM      2  H   ALA H   0      -2.288 -14.051   4.344  1.00  0.00      ... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- Runs on NVIDIA T4 GPUs with 4 vCPUs and 12 GB RAM allocated for the ``language`` variant, enabling GPU-accelerated inference for antibody-specific sequence modelling.
- Compared to general protein language models (for example, ESM-1b used for residue masking/completion), the ABodyBuilder3 language model is typically around 7× faster on antibody heavy/light chain inputs of similar length due to its antibody-focused architecture and optimized deployment.
- For antibody-specific residue restoration and masked-sequence recovery tasks, ABodyBuilder3 language achieves up to ~98% residue-level accuracy on benchmarked antibody datasets, outperforming general protein LMs (such as ESM-1b) and traditional IMGT germline-based restoration approaches.
- Within BioLM’s antibody modelling stack, the ABodyBuilder3 language model is optimized for sequence-level tasks (masking, completion, scoring) and is substantially faster and cheaper than full 3D structure predictors (for example, ABodyBuilder2/3 structural or AlphaFold2-like models), while providing accuracy that is more appropriate for antibody sequence design than general-purpose protein LMs.

Applications
------------

- Sequence completion for partially observed antibody heavy/light chains from high-throughput sequencing, by assigning high-likelihood residues to missing or ambiguous positions at the N-terminus or internally, allowing recovery of otherwise unusable sequences in discovery workflows; uses antibody-specific language modeling rather than generic protein models and does not require prior germline calling, though accuracy decreases for long contiguous gaps (>30 residues).
- In silico antibody maturation and optimization by scoring residue-level substitutions in VH/VL with a learned likelihood, enabling rapid prioritization of point mutations or small combinatorial libraries for improved affinity, stability, or developability; well suited for narrowing down design spaces before structural modeling or wet-lab screening, but computational scores must be experimentally confirmed and may not capture all epitope- or liability-related effects.
- Large-scale repertoire analysis, clustering, and candidate ranking using sequence-level embeddings derived from the language model that capture germline usage, CDR patterns, and mutation context, supporting industrial antibody discovery campaigns in narrowing millions of sequences to tractable panels for expression and characterization; effective for similarity search and coarse-grained functional grouping, but not a replacement for detailed 3D structure prediction or epitope mapping.
- Imputation of ambiguous amino acids (e.g., “X”) arising from sequencing or base-calling errors in BCR/antibody datasets by assigning the most probable residue given the full heavy/light context, improving data quality and downstream analytics, particularly for single-cell or repertoire-scale assays; performance is highest for isolated ambiguous positions and degrades when multiple adjacent residues are unknown.
- Estimating and validating antibody sequence boundaries (e.g., N-terminal truncations) by combining ANARCI numbering with language-model likelihoods over candidate N-terminal residues, helping standardize Fv sequence annotation and detect mis-trimmed constructs in therapeutic pipelines; robust to unusual CDR lengths or indels but may require iterative refinement or additional experimental information for edge cases.

Limitations
-----------

- **Maximum Sequence Length**: Antibody heavy (``H``) and light (``L``) chain sequences must each be between ``1`` and ``2048`` amino acids in length.
- **Batch Size**: Only one antibody heavy/light chain pair can be submitted per request (``items`` list must have ``min_items=1`` and ``max_items=1``; effectively ``batch_size = 1``).
- **GPU Type**: The ``language`` variant requires a ``T4`` GPU, which may increase inference cost and latency relative to the ``plddt`` variant (which runs on CPU only).
- ABodyBuilder3 is trained for paired antibody variable domains and is not intended for general protein structure prediction or for non-antibody immune receptors (e.g. TCRs, nanobodies).
- The model is optimised for canonical antibody formats and typical CDR length distributions; accuracy may degrade for highly engineered constructs, non-standard scaffolds, or sequences far outside training distributions.
- For initial screening or ranking of very large antibody libraries (e.g. millions of sequences), faster sequence-based or coarse-grained structure models (such as ``ESMFold``) are usually more practical than running ABodyBuilder3 on every candidate.

How We Use It
-------------

BioLM uses ABodyBuilder3 Language to rapidly generate 3D structures for antibody heavy/light pairs at scale, enabling high-throughput in silico screening, developability assessment, and lead optimization without the overhead of full AlphaFold-style pipelines. The standardized API integrates into multiparameter antibody engineering workflows, where ABodyBuilder3 structures feed downstream stability, liability and epitope/paratope models to prioritize variants for synthesis and iterative optimization.

- Enables fast structure generation for sequence libraries, supporting lab-in-the-loop affinity maturation and specificity tuning.

Related
-------

- ``AbLang-2`` – Updated antibody language model that improves sequence completion and repertoire-aware design; can be paired with ABodyBuilder3 Language to generate and structurally validate antibody variants.
- ``ESM-1v`` – General protein language model that provides broader protein sequence embeddings to study mutational effects and context around antibody regions modeled by ABodyBuilder3 Language.
- ``ImmuneFold Antibody`` – End-to-end antibody structure predictor; use alongside ABodyBuilder3 Language to compare structures from language-driven designs or to obtain alternative structural hypotheses.
- ``IgT5 Paired`` – Paired heavy/light chain antibody language model useful for generating and analyzing realistic paired sequences before 3D structure prediction with ABodyBuilder3 Language.

References
----------

- Abanades, B. et al. (2023). `ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins <https://doi.org/10.1038/s42003-023-04927-7>`_. *Communications Biology*.
- Olsen, T. H., Moal, I. H., & Deane, C. M. (2022). `AbLang: an antibody language model for completing antibody sequences <https://doi.org/10.1093/bioadv/vbac046>`_. *Bioinformatics Advances*.
- Olsen, T. H. et al. (2022). `OAS: a diverse database of cleaned, annotated and translated unpaired and paired antibody sequences <https://doi.org/10.1002/pro.4205>`_. *Protein Science*.
- Kovaltsuk, A. et al. (2018). `Observed antibody space: a resource for data mining next-generation sequencing of antibody repertoires <https://doi.org/10.4049/jimmunol.1800708>`_. *Journal of Immunology*.
