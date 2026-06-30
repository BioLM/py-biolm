IgT5 Paired API
===============

IgT5 Paired is an antibody-specific encoder–decoder transformer initialised from ProtT5 and fine-tuned on two million natively paired heavy–light variable-region sequences from OAS. The API provides GPU-accelerated residue-level and mean embeddings for either paired (heavy + light) or unpaired antibody sequences, with batch sizes up to 8 and maximum lengths of 256 residues per heavy/light chain or 512 residues for unpaired sequences. IgT5 embeddings support workflows in antibody engineering, affinity maturation, binding prediction, and developability modeling.

Encode
------

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="igt5-paired",
                action="encode",
                params={
                  "include": [
                    "mean"
                  ]
                },
                items=[
                  {
                    "heavy": "EVQLVESGGGLVQ",
                    "light": "DIQMTQ",
                    "sequence": null
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/igt5-paired/encode/ \
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
                  "heavy": "EVQLVESGGGLVQ",
                  "light": "DIQMTQ",
                  "sequence": null
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/igt5-paired/encode/"
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
                      "heavy": "EVQLVESGGGLVQ",
                      "light": "DIQMTQ",
                      "sequence": null
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/igt5-paired/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "mean"
                )
              ),
              items = list(
                list(
                  heavy = "EVQLVESGGGLVQ",
                  light = "DIQMTQ",
                  sequence = None
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/igt5-paired/encode/

   Encode endpoint for IgT5 Paired.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["mean"]) — Embedding outputs to return; allowed values: "mean", "residue"

      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:

        - **heavy** (*string*, optional, min length: 1, max length: 256) — Heavy chain amino acid sequence using the supported extended amino acid alphabet

        - **light** (*string*, optional, min length: 1, max length: 256) — Light chain amino acid sequence using the supported extended amino acid alphabet

        - **sequence** (*string*, optional, min length: 1, max length: 512) — Single unpaired amino acid sequence using the supported extended amino acid alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/igt5-paired/encode/ HTTP/1.1
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
            "heavy": "EVQLVESGGGLVQ",
            "light": "DIQMTQ",
            "sequence": null
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

        - **embeddings** (*array of floats*, size: 1024, optional) — Mean embedding vector for the encoded input item; each float is one embedding dimension

        - **residue_embeddings** (*array of arrays of floats*, shape: [sequence_length, 1024], optional) — Per-residue embedding vectors for the encoded input item; each inner array contains 1024 floats for a single residue

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embeddings": [
              -0.0003371938946656883,
              -0.0146927610039711,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- IgT5 Paired is a ~3B-parameter encoder-decoder model running on NVIDIA T4 GPUs (16 GB VRAM) with GPU-optimized inference. It is more compute-intensive per sequence than BioLM’s IgBert Paired (~420M parameters) but tuned for higher-accuracy embeddings of paired heavy–light variable regions.
- On FLAb antibody binding-affinity benchmarks, IgT5 Paired embeddings yield higher predictive accuracy than general protein language models and ProtT5/ProtBert backbones (e.g., largest dataset: IgT5 Paired :math:`R^2 \approx 0.25` vs. ProtT5 :math:`R^2 \approx 0.21`, ProtBert :math:`R^2 \approx 0.10`), and outperform BioLM’s IgBert Paired on paired binding tasks (e.g., :math:`R^2 \approx 0.297` vs. :math:`0.174`).
- Relative to BioLM’s antibody-specific encoders, IgT5 Paired improves masked-residue recovery in highly variable regions such as CDR-H3 (≈62.0% vs. ≈60.1% for IgBert Paired) while maintaining comparable framework-region recovery, leading to better overall VH/VL variable-region reconstruction on paired sequences.

Applications
------------

- Antibody affinity maturation support by using IgT5 Paired embeddings as features in downstream models to prioritize mutations in complementarity-determining regions (CDRs), helping researchers rank sequence variants with improved predicted binding or expression; useful in lead optimization pipelines, with all in silico suggestions requiring experimental validation to confirm functional impact.
- Detection of sequence-level liabilities via sequence recovery and mutation scoring based on IgT5 Paired representations, enabling early flagging of unusual or poorly supported residues in developability assessments; relevant for biopharmaceutical pipelines aiming to reduce downstream risks, while not directly predicting specific biophysical properties such as aggregation, viscosity, or solubility.
- Generation of paired heavy–light antibody embeddings for supervised prediction of antigen binding affinity or expression using external models, enabling ranking and triage of large candidate panels in discovery and optimization campaigns; valuable for reducing experimental screening burden, though embeddings alone do not capture explicit 3D structure and may need to be combined with structural modeling.
- In silico humanization guidance by comparing IgT5 Paired embeddings or token-level scores for candidate sequences against human antibody repertoires, helping to identify non-human-like residues or regions that might contribute to immunogenicity; suitable for early design filtering, but not a stand‑alone predictor of clinical immunogenicity.
- Cross-chain pairing analysis for variable regions by embedding candidate heavy and light chains (using paired or unpaired encoding as appropriate) and training downstream pairing classifiers on top of IgT5 representations, aiding reconstruction or quality control of inferred heavy–light pairs in repertoire mining and single-cell datasets, with potential reductions in accuracy for highly unusual or sparsely represented sequence families.

Limitations
-----------

- **Maximum Sequence Length**: The ``heavy`` and ``light`` chain inputs each support up to ``256`` amino acids via the ``heavy`` and ``light`` fields. The unpaired ``sequence`` input supports up to ``512`` amino acids. Longer sequences must be truncated or split upstream.
- **Batch Size**: The maximum ``batch_size`` is ``8`` items per ``encoder`` API request in the ``items`` list.
- **Input Type Constraints**: Each entry in ``items`` must be either paired or unpaired. Paired entries require both ``heavy`` and ``light``; unpaired entries require a single ``sequence``. Providing both ``sequence`` and (``heavy``, ``light``) together, or omitting all of them, results in validation errors.
- **Antibody-Specific Training**: IgT5 Paired is trained on antibody variable regions from OAS. Performance on non-antibody proteins or on sequences very different from typical antibody variable regions (for example, long linkers or engineered domains) may be poor.
- **Task-Specific Performance**: IgT5 embeddings are tuned for antibody-focused tasks such as binding- and paratope-related properties. For broader protein properties like generic expression or stability across many protein families, general protein models such as ProtT5 often perform better.
- **Model Choice for Chain Context**: IgT5 Paired embeddings encode heavy–light cross-chain context and are most appropriate when true paired antibody variable regions are available. For single-chain analyses or large repertoires without reliable pairing information, the IgT5 Unpaired model is usually a better fit.

How We Use It
-------------

BioLM uses IgT5 Paired embeddings as a standardized representation of antibody heavy–light variable regions within design and optimization workflows, enabling scalable assessment of candidate repertoires for binding-relevant sequence features and consistent feature inputs to supervised property models. These embeddings are combined with BioLM’s affinity, expression, stability, and developability predictors and with generative design loops, so teams can iteratively propose, filter, and rank heavy–light variants while keeping paired-chain context explicit and accessible through the IgT5 encoder API.

- Integrates with generative antibody design and multi-parameter optimization pipelines by providing a shared embedding space for paired heavy–light variants
- Enables rapid prioritization of heavy–light pairs from large in silico libraries, reducing wet-lab screening load and shortening optimization cycles

Related
-------

- ``IgBert Paired`` – Paired antibody language model trained on the same OAS dataset; provides alternative paired-chain embeddings for comparison or ensembling with IgT5 Paired on affinity and developability prediction tasks.
- ``AntiFold`` – Antibody inverse-folding model for scoring or refining IgT5 Paired–guided sequence designs using sequence-to-structure evaluation workflows.
- ``ImmuneFold Antibody`` – Antibody structure prediction model for converting IgT5 Paired–selected heavy/light variants into 3D structures for downstream stability or interface analysis.
- ``ABodyBuilder3 Language`` – Antibody sequence-to-structure generator informed by language-model features, useful for adding structural context to sequences analysed or generated with IgT5 Paired.

References
----------

- Kenlay, H., Dreyer, F. A., Kovaltsuk, A., Miketa, D., Pires, D., & Deane, C. M. (2024). Large scale paired antibody language models. *PLoS Computational Biology*, 20(12), e1012646. https://doi.org/10.1371/journal.pcbi.1012646
