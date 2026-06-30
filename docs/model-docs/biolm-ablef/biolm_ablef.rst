BioLM-AbLEF API
===============

BioLM-AbLEF is an antibody developability prediction service that uses frozen AbLang embeddings of paired heavy/light Fv sequences to estimate hydrophobic interaction chromatography retention time (HIC-RT) and Fab melting temperature by DSF (Fab Tm). The API provides GPU-accelerated, batched inference for up to 8 antibody pairs (≤160 residues per chain) and can also return 1536-dimensional Fv embeddings for downstream modeling. It is suited for ranking variants, early-stage lead optimization, and sequence-aware screening workflows.

Predict
-------

Predict hydrophobic interaction chromatography retention time (hic_rt, minutes) and Fab melting temperature (fab_tm, °C) for paired antibody Fv sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="biolm-ablef",
                action="predict",
                params={},
                items=[
                  {
                    "heavy": "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEWV",
                    "light": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLI"
                  },
                  {
                    "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYWMHWVRQAPGQGLEWM",
                    "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/biolm-ablef/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "heavy": "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEWV",
                  "light": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLI"
                },
                {
                  "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYWMHWVRQAPGQGLEWM",
                  "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/biolm-ablef/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "heavy": "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEWV",
                      "light": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLI"
                    },
                    {
                      "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYWMHWVRQAPGQGLEWM",
                      "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/biolm-ablef/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  heavy = "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEWV",
                  light = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLI"
                ),
                list(
                  heavy = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYWMHWVRQAPGQGLEWM",
                  light = "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/biolm-ablef/predict/

   Predict endpoint for BioLM-AbLEF.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 8) — Maximum number of sequence pairs accepted in a single request
        - **max_sequence_len** (*int*, default: 160) — Maximum allowed length per heavy or light chain sequence

      - **items** (*array of objects*, min: 1, max: 8, required) --- Input antibody Fv sequences:

        - **heavy** (*string*, min length: 1, max length: 160, required) — Heavy chain amino acid sequence using extended amino acid codes
        - **light** (*string*, min length: 1, max length: 160, required) — Light chain amino acid sequence using extended amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/biolm-ablef/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "heavy": "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEWV",
            "light": "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLNWYQQKPGKAPKLLI"
          },
          {
            "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYWMHWVRQAPGQGLEWM",
            "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
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

        - **hic_rt** (*float*) — Predicted hydrophobic interaction chromatography retention time, units: minutes

        - **fab_tm** (*float*) — Predicted Fab melting temperature by differential scanning fluorimetry (DSF), units: degrees Celsius


      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **embedding** (*array of floats*, size: 1536) — Concatenated Fv embedding (768 heavy-chain dimensions + 768 light-chain dimensions)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "hic_rt": 9.784151077270508,
            "fab_tm": 65.33888244628906
          },
          {
            "hic_rt": 10.333909034729004,
            "fab_tm": 66.85546112060547
          }
        ]
      }


Encode
------

Generate concatenated Fv embeddings (1536-dim) from paired heavy/light chain antibody sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="biolm-ablef",
                action="encode",
                params={},
                items=[
                  {
                    "heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEW",
                    "light": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLI"
                  },
                  {
                    "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWM",
                    "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/biolm-ablef/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEW",
                  "light": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLI"
                },
                {
                  "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWM",
                  "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/biolm-ablef/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEW",
                      "light": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLI"
                    },
                    {
                      "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWM",
                      "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/biolm-ablef/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEW",
                  light = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLI"
                ),
                list(
                  heavy = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWM",
                  light = "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/biolm-ablef/encode/

   Encode endpoint for BioLM-AbLEF.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **batch_size** (*int*, default: 8) — Maximum number of items per request

        - **max_sequence_len** (*int*, default: 160) — Maximum length per heavy or light chain sequence


      - **items** (*array of objects*, min: 1, max: 8, required) --- Input sequence pairs:

        - **heavy** (*string*, min length: 1, max length: 160, required) — Antibody heavy chain amino acid sequence

        - **light** (*string*, min length: 1, max length: 160, required) — Antibody light chain amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/biolm-ablef/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "heavy": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMNWVRQAPGKGLEW",
            "light": "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLI"
          },
          {
            "heavy": "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQGLEWM",
            "light": "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLI"
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

        - **embedding** (*array of floats*, size: 1536) — Concatenated Fv embedding (heavy chain 768-dim + light chain 768-dim), unitless continuous values

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "embedding": [
              -0.12289410829544067,
              0.880843460559845,
              "... (truncated for documentation)"
            ]
          },
          {
            "embedding": [
              -0.4577169418334961,
              0.416167676448822,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Model variant and architecture
  - Implements an AbLang-based, language-only baseline trained on public FLAb data (Jain 2017), not the full sequence+structure AbLEF from Rollins et al. (2024)
  - Uses frozen AbLang Fv embeddings (1536-dim: 768 heavy + 768 light) with shallow regression heads for each property
  - Does not perform any 3D structure generation, ensemble fusion, or Tagg prediction in this hosted version

- Predictive accuracy and calibration
  - On antibody-focused developability tasks, performance is similar to strong antibody-specific language models and typically exceeds generic protein LMs (e.g., ESM-1b / ESM‑2 650M fine-tuned on HIC-like data) at comparable data scales
  - For HIC-like hydrophobicity endpoints, language-only AbLang baselines in Rollins et al. reach R² in the 0.4 range; the hosted model is expected to be in that class of performance but below the best ensemble-fusion AbLEF variants (R² ≈ 0.49 on proprietary HIC-RT)
  - Fab Tm (DSF) predictions are calibrated for IMGT-aligned antibody Fv regions but will not generalize to non-Ig scaffolds or to Tagg-style aggregation temperatures

- Comparison to full AbLEF and other BioLM models
  - Versus full AbLEF (sequence+ensemble distograms): the hosted model sacrifices some accuracy and uncertainty reduction on structure-sensitive endpoints in exchange for simpler deployment and higher throughput
  - Versus BioLM structural or multimodal pipelines (e.g., models that rely on AlphaFold2/ImmuneFold or GNNs over 3D structures): this API is less accurate on strongly structure-dependent properties (e.g., high-concentration viscosity) but competitive for HIC-like and DSF-based thermal metrics in the small-data regime (10²–10³ labeled antibodies)
  - As an encoder, its 1536-dim antibody-specific embeddings are lower dimensional and cheaper to store than many large general-purpose protein LMs offered on BioLM, while often improving downstream antibody-specific tasks relative to generic embeddings

- Robustness, scaling, and typical usage patterns
  - Frozen AbLang backbones and shallow regressors reduce overfitting and training instability compared to deep sequence+structure models when labels are scarce
  - In internal and partner use, AbLEF-style language-only models reliably outperform physicochemical-descriptor baselines for HIC and Fab Tm and often narrow the performance gap to sequence+structure approaches, making this endpoint a practical first-pass filter before invoking more expensive BioLM structural or multimodal models

Applications
------------

- In silico screening of therapeutic IgG1 antibody panels for HIC-RT and Fab melting temperature (Fab Tm) to down-select thousands of variants to a smaller, higher-confidence set before running hydrophobic interaction chromatography and DSF assays, reducing wet-lab load and focusing experimental resources on candidates with predicted acceptable retention time and thermal stability profiles
- Early-stage antibody lead optimization where affinity-matured or humanized IgG1 variants are iteratively proposed and then scored for HIC-RT and Fab Tm, allowing project teams to balance target binding improvements against predicted surface hydrophobicity and Fab thermal stability, and to flag designs likely to show poor chromatographic behavior or low melting temperatures in formulation screens; not optimal for developability properties outside these trained endpoints (for example, viscosity or polyspecificity) unless combined with additional models
- Developability-aware library design for display campaigns and high-throughput antibody generation, where sequence engineers or platform teams virtually pre-filter or bias combinatorial CDR designs using AbLEF predictions so that generated IgG1 libraries are enriched for variants within acceptable HIC-RT and Fab Tm windows, improving the fraction of downstream hits that pass developability gates and lowering the cost per clinical-quality lead
- Portfolio-level risk assessment of existing clinical or late-preclinical IgG1 antibody assets by running Fv sequences through AbLEF to obtain predicted HIC-RT and Fab Tm, providing biopharm and CMC organizations with quantitative, model-based flags for molecules that may require formulation complexity, tighter cold-chain constraints, or additional manufacturability engineering; predictions are most informative when sequences are close to the model’s training space (human-like IgG1 Fv, similar pH and salt conditions)
- Integration into automated antibody design–build–test workflows, where development teams call the BioLM-AbLEF predictor API as a scoring function inside generative or mutational search loops to rank proposed IgG1 Fv variants by predicted hydrophobic interaction behavior and Fab melting temperature, enabling closed-loop optimization of developability alongside potency; currently best suited for sequence-only scenarios without explicit structural ensembles and for conditions similar to the training assays (near-neutral pH, defined salt and temperature ranges)

Limitations
-----------

- **Maximum sequence length and batch size.** Each ``heavy`` and ``light`` chain in an ``AbLEFSequenceItem`` must be at most ``160`` amino acids (per chain), and each ``AbLEFPredictRequest`` or ``AbLEFEncodeRequest`` may contain at most ``8`` ``items``. Longer inputs (e.g., full IgG, Fv/Fab with extra framework residues, or atypical numbering) must be pre-trimmed or mapped to the Fv region before calling the API.
- **Input / output scope.** The API only accepts paired Fv sequences via ``AbLEFSequenceItem`` (``heavy``, ``light``) and exposes two endpoints: ``predictor`` (returns ``hic_rt`` in minutes and ``fab_tm`` in °C via ``AbLEFPredictResponse``) and ``encoder`` (returns a fixed-length numeric ``embedding`` vector via ``AbLEFEncodeResponse``). It does not take structures, antigen sequences, formulation conditions, or assay metadata as inputs, and the ``embedding`` is not a 3D structure or annotation.
- **AbLEF variant differences from the paper.** This hosted model is an AbLang-based baseline: it uses frozen AbLang embeddings (no fine-tuning of LM layers), predicts DSF-based ``fab_tm`` rather than ``Tagg`` (temperature of aggregation), and does not perform 3D conformational ensemble fusion. As a result, behavior and performance may differ from the full AbLEF architecture described in Rollins et al. 2024, especially for properties with strong thermodynamic or structural dependence.
- **Property and data-regime limitations.** The underlying regressors were trained on relatively small monoclonal antibody datasets (``~10^2–10^3`` sequences) targeting developability-style endpoints. They are not calibrated for binding affinity, immunogenicity, polyspecificity, viscosity, or other properties beyond ``hic_rt`` and ``fab_tm``; for those, separate models or bespoke pipelines are typically more appropriate.
- **Sequence and distribution constraints.** The model is antibody-focused and expects canonical Fv-like heavy/light chains similar to therapeutic IgG1s. Predictions may be less reliable for non-IgG formats (nanobodies, scFv, bispecifics, heavily engineered scaffolds), sequences far from the training distribution, or non-human/atypical germlines.
- **Use cases where other methods are preferable.** This API is not optimal if you need (a) explicit structural modeling or conformational ensembles (e.g., AF2/ESMFold-based workflows, docking, epitope mapping), (b) ranking under specific formulation or stress conditions not reflected in ``hic_rt``/``fab_tm``, or (c) uncertainty-quantified, out-of-distribution–robust predictions for regulatory or other high-risk decisions, which usually require tailored model development and experiment-integrated pipelines.

How We Use It
-------------

BioLM-AbLEF enables antibody teams to move from ad hoc, single-metric developability checks to consistent, ensemble-informed scoring that plugs directly into sequence design, ranking, and optimization workflows. Organizations call the AbLEF predictor and encoder APIs alongside BioLM antibody language models and structure-based encoders to drive multi-objective design loops (e.g., binding vs. HIC-RT vs. Fab Tm), using shared developability scores and embeddings to triage large virtual libraries, prioritize variants before synthesis, and iteratively refine candidates as new HIC-RT and thermal-stability data arrive.

- AbLEF predictions integrate with BioLM generative antibody models to down-select in silico variants by hydrophobicity (HIC-RT) and Fab melting temperature risk before synthesis.  
- Sequence-level developability scores and Fv embeddings from AbLEF stream into Bayesian optimization and active-learning pipelines to accelerate multi-round optimization and reduce late-stage developability failures.

Related
-------

- ``AbLang-2`` – Antibody-specific language model for rapid, sequence-only screening of large repertoires before running AbLEF on a smaller, higher-value subset.
- ``ABodyBuilder3 pLDDT`` – High-accuracy antibody Fv structure predictor that can generate starting Fv structures for building or validating the 3D ensembles underpinning AbLEF.
- ``ImmuneFold Antibody`` – End-to-end antibody structure prediction model that can provide alternative Fv conformations and paratope geometries to probe structural variability alongside AbLEF sequence-based predictions.
- ``PROPERMAB`` – Antibody developability predictor offering complementary sequence- and structure-based scores (e.g., stability, aggregation) that can be compared with AbLEF HIC-RT and Fab Tm outputs for multi-metric risk assessment.

References
----------

- Rollins, Z. A., Widatalla, T., Waight, A., Cheng, A. C., & Metwally, E. (2024). `AbLEF: antibody language ensemble fusion for thermodynamically empowered property predictions <https://doi.org/10.1093/bioinformatics/btae268>`_. *Bioinformatics*, 40(6), btae268.
