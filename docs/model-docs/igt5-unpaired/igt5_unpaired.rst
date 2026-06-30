IgT5 Unpaired API
=================

IgT5 Unpaired is a GPU-accelerated antibody language model based on the ProtT5 encoder–decoder architecture, further trained on approximately 700 million clustered unpaired variable heavy and light chain sequences from the Observed Antibody Space. The encoder endpoint provides mean and per-residue embeddings for single-chain amino acid sequences up to 512 residues, supporting downstream tasks such as repertoire characterization, sequence clustering, feature extraction for binding or expression models, and design or maturation workflows where antibody-specific representations are required.

Encode
------

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igt5-unpaired",
                action="encode",
                params={
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "heavy": null,
                    "light": null,
                    "sequence": "EVQLVESGGGLVQ"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igt5-unpaired/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "mean"
                ]
              },
              "items": [
                {
                  "heavy": null,
                  "light": null,
                  "sequence": "EVQLVESGGGLVQ"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igt5-unpaired/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "mean"
                    ]
                  },
                  "items": [
                    {
                      "heavy": null,
                      "light": null,
                      "sequence": "EVQLVESGGGLVQ"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igt5-unpaired/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  heavy = None,
                  light = None,
                  sequence = "EVQLVESGGGLVQ"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igt5-unpaired/encode/

   Encode endpoint for IgT5 Unpaired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding types to include in the response; allowed values: "mean", "residue"

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy chain amino acid sequence using extended amino acid codes

        - **light** (*string*, optional, min length: 1, max length: 256) — Light chain amino acid sequence using extended amino acid codes

        - **sequence** (*string*, optional, min length: 1, max length: 512) — Unpaired amino acid sequence using extended amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igt5-unpaired/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "mean"
          ]
        },
        "items": [
          {
            "heavy": null,
            "light": null,
            "sequence": "EVQLVESGGGLVQ"
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

        - **embeddings** (*array of floats*, size: 1024) — Mean embedding vector for the encoded input; numerical values unbounded.

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 1024]) — Per-residue embedding vectors for the encoded input; numerical values unbounded.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              -0.014581063762307167,
              -0.05809000879526138,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- IgT5 Unpaired runs on NVIDIA T4 GPUs with 4 vCPUs and 16 GB RAM allocated per worker, enabling GPU-accelerated embedding generation for large antibody datasets.
- Compared to general protein language models fine-tuned on antibodies (e.g., ProtT5, ProtBert), IgT5 Unpaired achieves substantially higher antibody sequence recovery, especially in hypervariable CDRs (e.g., CDR-H3 recovery ~0.60 vs. ProtT5 ~0.34), while maintaining high framework-region accuracy.
- Within BioLM’s antibody-specific encoders, IgT5 Unpaired matches or slightly exceeds IgBert Unpaired on overall heavy-chain recovery (≈0.912 vs. 0.910) and light-chain recovery (≈0.951 vs. 0.949–0.956), offering improved CDR performance with a larger T5 architecture.
- For downstream regression on antibody-focused tasks, embeddings from IgT5 Unpaired improve binding-affinity prediction over ProtT5 on representative benchmarks (R² ≈ 0.299 vs. 0.290), but general protein models like ProtT5 remain preferable for broad protein properties such as expression (R² ≈ 0.697 vs. 0.567).

Applications
------------

- Antibody affinity maturation and optimization by prioritizing beneficial mutations in complementarity-determining regions (CDRs) using IgT5 embeddings as features in downstream models, enabling rapid in silico ranking of variants for improved antigen binding; valuable for accelerating therapeutic antibody lead optimization, but not sufficient alone to predict broader expression, aggregation, or stability properties.
- Identification and scoring of putative paratope residues from IgT5 residue-level embeddings, allowing efficient prioritization of positions most likely to impact antigen binding; useful for rational engineering tasks such as specificity tuning or epitope focusing, although not a substitute for full structural modeling or epitope mapping.
- High-throughput antibody sequence filtering and ranking by training regression or classification models on IgT5 mean embeddings to approximate antigen binding affinity, enabling rapid down-selection of large antibody libraries in discovery campaigns; effective for sequence-based enrichment but limited in capturing developability attributes such as solubility, viscosity, or immunogenicity.
- Clustering and diversity analysis of antibody repertoires using IgT5 mean or residue-level embeddings, facilitating selection of diverse panels of candidates while maintaining coverage of sequence space; supports library design and repertoire mining, but embeddings alone do not provide detailed functional annotation and still require experimental follow-up.
- Analysis of heavy-light chain pairing signals by comparing IgT5 embeddings of paired versus unpaired sequences, supporting development of custom pairing-compatibility predictors for repertoire and single-cell sequencing data; useful for reconstructing likely native pairs in human-like repertoires, though performance may degrade on highly non-human or unusual antibody formats.

Limitations
-----------

- **Maximum Sequence Length**: For paired inputs (``heavy`` and ``light`` chains), each chain must be ≤ ``256`` amino acids. For unpaired inputs (``sequence``), the maximum length is ``512`` amino acids.
- **Batch Size**: API requests are limited to a maximum of ``8`` sequences per ``items`` list. Larger workloads must be split across multiple requests.
- IgT5 embeddings are designed for antibody variable regions; using arbitrary or non-immunoglobulin proteins for general property prediction (for example, expression across diverse protein families) may underperform compared to general protein language models such as ProtT5.
- Training data is predominantly human antibody sequences from the Observed Antibody Space (OAS); performance may degrade for sequences unlike typical human antibodies (for example, non-human species, synthetic frameworks, or highly engineered CDRs).
- The API exposes IgT5 as an encoder only: it returns sequence embeddings via ``embeddings`` (mean over residues) and optionally ``residue_embeddings``. It is not intended for direct generative use (for example, de novo sequence design) or for antibody–antigen 3D structure prediction.
- Cross-chain context learned from paired heavy–light data is only available when you provide both ``heavy`` and ``light``; using unpaired ``sequence`` inputs will not capture pairing-specific features and may reduce accuracy for tasks that depend on native chain pairing.

How We Use It
-------------

IgT5 Unpaired is used as a backbone encoder in antibody discovery and optimization programs, where its embeddings of unpaired variable regions drive rapid screening, property prediction, and design-space exploration before committing to wet-lab cycles. The model supports scalable, standardized encoding of large repertoires, and its representations feed into downstream models for binding, expression, and developability that we integrate with structure-based tools and lab data.

- IgT5 Unpaired embeddings enable prioritization and clustering of candidates in early antibody campaigns, reducing the number of variants that need to be synthesized and tested.
- Combined with paired IgT5 and other BioLM encoders, IgT5 Unpaired supports end-to-end workflows from repertoire mining through sequence optimization and multi-round affinity maturation.

Related
-------

- ``IgT5 Paired`` – Paired variant of IgT5 trained on heavy–light chain complexes, useful when embeddings must capture cross-chain context beyond what unpaired IgT5 provides.
- ``IgBert Unpaired`` – Antibody-specific BERT encoder trained on unpaired sequences, offering an alternative embedding space for the same types of inputs as IgT5 Unpaired.
- ``ABodyBuilder3 Language`` – Structure-prediction model that consumes sequence-based embeddings, commonly used with IgT5 Unpaired for sequence-to-structure antibody modeling.
- ``AntiFold`` – Inverse folding model for sequence design under structural constraints, complementary to IgT5 Unpaired embeddings in sequence–structure antibody design workflows.

References
----------

- Kenlay, H., Dreyer, F. A., Kovaltsuk, A., Miketa, D., Pires, D., & Deane, C. M. (2024). Large scale paired antibody language models. *PLoS Computational Biology*, 20(12), e1012646. https://doi.org/10.1371/journal.pcbi.1012646
