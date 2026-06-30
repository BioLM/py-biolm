ZymCTRL API
===========

ZymCTRL is a conditional protein language model for controllable enzyme sequence generation, trained as a 738M-parameter GPT2/CTRL-style Transformer decoder on ~36M BRENDA enzymes. It generates amino acid sequences up to 1,024 residues conditioned on EC numbers (full or partial, respecting the EC hierarchy) to target specific catalytic reactions. The API provides GPU-accelerated batch generation (up to 20 samples per EC) with configurable sampling (temperature, top-k, repetition penalty) and perplexity scores, plus an encoder endpoint for EC-aware enzyme embeddings.

Encode
------

Encode enzyme sequences into embeddings, optionally conditioned on EC numbers. Demonstrates per_token pooling and a positive layer index.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="zymctrl",
                action="encode",
                params={
                  "pooling": "per_token",
                  "layer": 12
                },
                items=[
                  {
                    "sequence": "MKKQLFALVSTAAAGVAVAQA",
                    "ec_number": "3.5.5.1"
                  },
                  {
                    "sequence": "MGHHHHHHSSGVDLGTENLYFQSMNNKSTVVVLDGAGKTALTIQLIQNHFV",
                    "ec_number": "1.1.1"
                  },
                  {
                    "sequence": "MADQLTEEQIAEFKEAFSLFDKDGDGCITTRE",
                    "ec_number": null
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/zymctrl/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "pooling": "per_token",
                "layer": 12
              },
              "items": [
                {
                  "sequence": "MKKQLFALVSTAAAGVAVAQA",
                  "ec_number": "3.5.5.1"
                },
                {
                  "sequence": "MGHHHHHHSSGVDLGTENLYFQSMNNKSTVVVLDGAGKTALTIQLIQNHFV",
                  "ec_number": "1.1.1"
                },
                {
                  "sequence": "MADQLTEEQIAEFKEAFSLFDKDGDGCITTRE",
                  "ec_number": null
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/zymctrl/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "pooling": "per_token",
                    "layer": 12
                  },
                  "items": [
                    {
                      "sequence": "MKKQLFALVSTAAAGVAVAQA",
                      "ec_number": "3.5.5.1"
                    },
                    {
                      "sequence": "MGHHHHHHSSGVDLGTENLYFQSMNNKSTVVVLDGAGKTALTIQLIQNHFV",
                      "ec_number": "1.1.1"
                    },
                    {
                      "sequence": "MADQLTEEQIAEFKEAFSLFDKDGDGCITTRE",
                      "ec_number": null
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/zymctrl/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                pooling = "per_token",
                layer = 12
              ),
              items = list(
                list(
                  sequence = "MKKQLFALVSTAAAGVAVAQA",
                  ec_number = "3.5.5.1"
                ),
                list(
                  sequence = "MGHHHHHHSSGVDLGTENLYFQSMNNKSTVVVLDGAGKTALTIQLIQNHFV",
                  ec_number = "1.1.1"
                ),
                list(
                  sequence = "MADQLTEEQIAEFKEAFSLFDKDGDGCITTRE",
                  ec_number = None
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/zymctrl/encode/

   Encode endpoint for ZymCTRL.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Encoder configuration parameters:

        - **pooling** (*string*, allowed: {"mean", "last", "per_token"}, default: "mean") — Embedding pooling mode for encoded outputs

        - **layer** (*int*, range: -36 to 36, default: -1) — Transformer layer index used to compute embeddings


      - **items** (*array of objects*, min: 1, max: 8, required) --- Input enzyme sequences:

        - **sequence** (*string*, min length: 1, max length: 1024, required) — Amino acid sequence using unambiguous standard residue codes

        - **ec_number** (*string*, optional) — EC number tag in the format "X.X", "X.X.X", or "X.X.X.X" with numeric components only (e.g., "3.5.5.1", "3.5.5")

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/zymctrl/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "pooling": "per_token",
          "layer": 12
        },
        "items": [
          {
            "sequence": "MKKQLFALVSTAAAGVAVAQA",
            "ec_number": "3.5.5.1"
          },
          {
            "sequence": "MGHHHHHHSSGVDLGTENLYFQSMNNKSTVVVLDGAGKTALTIQLIQNHFV",
            "ec_number": "1.1.1"
          },
          {
            "sequence": "MADQLTEEQIAEFKEAFSLFDKDGDGCITTRE",
            "ec_number": null
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

        - **sequence_index** (*int*) — Zero-based index of the input sequence within the request items array

        - **embedding** (*array of floats*, optional, size: 1260) — Sequence-level embedding from the specified layer for ``pooling`` values ``"mean"`` or ``"last"``

        - **per_token_embeddings** (*array of arrays of floats*, optional, inner size: 1260) — Token-level embeddings from the specified layer for ``pooling`` value ``"per_token"``; outer array length equals the amino acid sequence length

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "sequence_index": 0,
            "per_token_embeddings": [
              [
                -0.6531957387924194,
                1.674278736114502,
                "... (truncated for documentation)"
              ],
              [
                -0.4254383444786072,
                1.5193023681640625,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 1,
            "per_token_embeddings": [
              [
                -0.06376715004444122,
                -0.649621844291687,
                "... (truncated for documentation)"
              ],
              [
                0.19652745127677917,
                -0.36987558007240295,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          },
          {
            "sequence_index": 2,
            "per_token_embeddings": [
              [
                -0.9010642170906067,
                0.8635432124137878,
                "... (truncated for documentation)"
              ],
              [
                -0.7350242137908936,
                0.8583425879478455,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Generate
--------

Generate dehydrogenase-like enzyme sequences conditioned on EC 1.1.1.1 with custom sampling parameters.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="zymctrl",
                action="generate",
                params={
                  "temperature": 0.9,
                  "top_k": 15,
                  "repetition_penalty": 1.1,
                  "num_samples": 3,
                  "max_length": 180
                },
                items=[
                  {
                    "ec_number": "1.1.1.1"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/zymctrl/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "temperature": 0.9,
                "top_k": 15,
                "repetition_penalty": 1.1,
                "num_samples": 3,
                "max_length": 180
              },
              "items": [
                {
                  "ec_number": "1.1.1.1"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/zymctrl/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "temperature": 0.9,
                    "top_k": 15,
                    "repetition_penalty": 1.1,
                    "num_samples": 3,
                    "max_length": 180
                  },
                  "items": [
                    {
                      "ec_number": "1.1.1.1"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/zymctrl/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                temperature = 0.9,
                top_k = 15,
                repetition_penalty = 1.1,
                num_samples = 3,
                max_length = 180
              ),
              items = list(
                list(
                  ec_number = "1.1.1.1"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/zymctrl/generate/

   Generate endpoint for ZymCTRL.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Generation configuration:

        - **temperature** (*float*, range: 0.0-2.0, default: 0.8) — Sampling temperature

        - **top_k** (*int*, range: 1-50, default: 9) — Number of highest-probability tokens to sample from

        - **repetition_penalty** (*float*, range: 1.0-2.0, default: 1.2) — Penalty factor applied to repeated tokens

        - **num_samples** (*int*, range: 1-20, default: 5) — Number of sequences to generate per input item

        - **max_length** (*int*, range: 50-1024, default: 256) — Maximum length of each generated protein sequence (amino acids)


      - **items** (*array of objects*, min: 1, max: 1) --- Generation requests:

        - **ec_number** (*string*, required) — Enzyme Commission number in the format ``X.X`` to ``X.X.X.X`` (dot-separated positive integers, 2-4 levels)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/zymctrl/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "temperature": 0.9,
          "top_k": 15,
          "repetition_penalty": 1.1,
          "num_samples": 3,
          "max_length": 180
        },
        "items": [
          {
            "ec_number": "1.1.1.1"
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

        - **[i]** (*array of objects*) — One generated sample per ``num_samples`` for input item *i*:

          - **sequence** (*string*) — Generated amino-acid sequence, length: 1–1024 residues, alphabet: unambiguous protein letters (ACDEFGHIKLMNPQRSTVWY)

          - **perplexity** (*float*) — Autoregressive sequence perplexity, unitless, lower values indicate higher model likelihood

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "MKAAVVSKDHHVDVTDKTLRSLKHGEALLKMECCGVCHTDLHVKNGDFGDKTGVILGHEGIGVVAEVGPGVTSLKPGDRASVAWFYEGCGHCEYCNSGNETLCRSVKNAGYSVDGGMAEECIVVADYAVKVPDGLDSAAASSITCAGVTTYKAVKLSKIRPGQWIAIYGLG",
              "perplexity": 1.0562608114055065
            },
            {
              "sequence": "MKAAVVREFGKPLVIEDRPDPQAGPGQILVKLEASGVCHTDLHAATGDWPVKPNPPFIPGHEGVGHIVALGGGVTAVKEGDRVGVPWLYSACGHCEHCLSGWETLCEAQQNTGYSVNGGFAEYALADANYVGTLPDNVGFVDIAPVLCAGVTVYKGLKMTDTRPGQWVVIS",
              "perplexity": 1.4497264813821733
            },
            {
              "sequence": "MRALQFYEPGKLQLRDVAEPAPPDSADVILKVGATGVCRTDWHGWQGHDPDITLPHVPGHELAGTVVEAGSGVDLRRGDRVTVPFIAGCGHCRNCVAGNPQVCENQTQPGFTHWGSFAEYVAIDYADTNLVTLIPEGVGFVEAAPLTCAGLTTYSALRKAGDAQRLGLTGF",
              "perplexity": 2.504095578395442
            }
          ]
        ]
      }


Performance
-----------

- Model architecture and scale
  - 36-layer Transformer decoder with 1,260-dimensional hidden size and 16 attention heads (CTRL/GPT‑style), totaling ~738M parameters
  - Trained autoregressively on ~36M BRENDA enzyme sequences with explicit, tokenized EC-number conditioning
  - Larger and more enzyme-specialized than ESM-2 150M and BioLM’s ProGen2 Medium backends, but smaller than ESM-2 650M / E1 600M class models

- Functional controllability and accuracy (vs BioLM general generators)
  - Under EC conditioning, generated sequences match ProteInfer’s top-level EC class in ~81% of cases, and full four-level EC labels in ~54% of cases where ProteInfer assigns an EC number; natural BRENDA enzymes reach ~62% at four levels under the same evaluation
  - Compared to unconditioned ProGen2-style models served by BioLM, ZymCTRL shows substantially higher enrichment for the requested EC class in zero-shot generation, reducing the need for brute-force generation plus post hoc functional filtering
  - Hierarchical EC tokenization lets ZymCTRL transfer information from well-populated to sparse classes (tens of examples), yielding usable candidates with broader perplexity dispersion but still overlapping the perplexity range of dense classes

- Structural plausibility, novelty, and perplexity
  - IUPred3 globularity predictions are ~97.7% for ZymCTRL-generated sequences vs ~99.3% for natural enzymes; single-sequence structure predictors give mean LDDT/pLDDT ~60–61 for generated vs ~85 for natural, comparable to ProGen2 Medium when matched by length
  - MMseqs2 comparisons to the BRENDA training set show mean sequence identity ~53% ± 23% over ~338-residue alignments, with only ~12.5% of generated sequences exceeding 90% identity to any training enzyme, indicating frequent generation of distant but plausible variants
  - Average perplexity for EC-conditioned samples is ~6.2 ± 2.9, close to natural sequences under the same model; perplexity is higher than for non-conditional BioLM generators (e.g., ProGen2 Large in matched families), reflecting the added EC-control constraint but enabling effective candidate ranking by model confidence

- Embeddings and downstream use within BioLM
  - Encoder embeddings (mean/last pooled or per-token) derive from the same 36-layer, 738M-parameter decoder and are competitive with ESM-2 650M / E1 600M on enzyme function–aware tasks, with clearer clustering by EC class and catalytic mechanism
  - For non-enzymes or structure-centric tasks, BioLM generally recommends ESM-2 650M, E1 600M, or ESM3 Open Small; for EC-aware enzyme design, clustering, or fitness modeling within an EC class, ZymCTRL embeddings provide stronger functional separation than generic encoders of similar size

Applications
------------

- Rapid in silico generation of enzyme variant libraries conditioned on EC number for industrial biocatalyst development, enabling teams to explore thousands of plausible, globular, sequence-diverse candidates for a defined reaction (for example, EC 3.1.1.- esterases for agrochemical intermediate hydrolysis) before committing to DNA synthesis and screening, reducing wet-lab library size while maintaining functional diversity; not optimal if you require control over properties not encoded in the EC class (such as precise temperature optimum or solvent tolerance), which must be layered on with additional predictive filters or directed evolution
- Lead-finding for new enzymatic routes in process development, where process chemists and protein engineers need novel scaffolds for an existing EC class to improve IP position or avoid known liabilities (such as patent-encumbered natural sequences or sequence motifs associated with immunogenicity); ZymCTRL’s EC-conditioned Transformer generation, combined with external models like ProteInfer for orthogonal EC prediction, can propose artificial enzymes that are distant in sequence yet still likely belong to the same EC class, although final activity, stability, and scale-up performance must be confirmed experimentally
- Enzyme portfolio expansion around underrepresented or poorly characterized EC classes, where only tens of natural sequences are known (for example, niche halogenases or C–C bond–forming enzymes) and classical homology mining yields few options; by leveraging its tokenized EC hierarchy, ZymCTRL can extrapolate from related, better-populated subclasses to propose new sequences for those sparse EC labels, giving enzyme suppliers or CDMOs more candidates to test in activity panels, while users should expect higher uncertainty (higher perplexity) and plan for broader screening when training data for an EC class are scarce
- Scaffold diversification for optimization campaigns on an existing industrial enzyme, where a company has a working catalyst in a given EC class but wants multiple, mechanistically similar yet sequence-distant backbones to derisk manufacturability, stability, or regulatory concerns; because ZymCTRL samples from an EC-conditioned sequence distribution rather than doing local mutagenesis, it can propose EC-matched sequences that occupy new regions of sequence space instead of simple point mutants of the natural scaffold, providing alternative starting points for subsequent rounds of rational design or directed evolution, but it is not a drop-in replacement for structure-guided engineering when very fine control of active-site geometry is required
- Front-end generator in automated enzyme design pipelines, where enterprises combine ZymCTRL sampling at a specified EC number with downstream structural prediction (for example, AlphaFold-class models), stability and solubility scoring, and application-specific filters (such as pH profile, secretion signals, or host-compatibility motifs) to build high-throughput design–make–test–analyze loops; in this role, ZymCTRL’s Transformer-based autoregressive generation supplies an EC-aware prior over sequence space, improving the hit rate over random or purely heuristic libraries, while its outputs still require multi-parameter filtering and experimental validation to meet strict industrial specifications (for example, GMP, food-grade, or detergent conditions)

Limitations
-----------

- **Maximum sequence length and batch size.** ZymCTRL only supports amino-acid sequences up to ``1024`` residues (``ZymCTRLParams.max_sequence_len``). Generation requests accept at most ``batch_size = 1`` item (sequential generation) and encode requests at most ``batch_size_encode = 8`` items. Longer sequences must be truncated or split externally, and larger workloads must be batched client-side.
- **Generation is conditional on valid EC numbers only.** The ``ZymCTRLGenerateRequestItem.ec_number`` field must match the EC pattern ``X.X``, ``X.X.X`` or ``X.X.X.X`` (for example, ``"3.5.5.1"`` or ``"3.5"``). Invalid formats are rejected. While partial ECs are allowed, ZymCTRL’s training data are unevenly distributed across classes: sequences from poorly represented or rare ECs can be lower confidence (typically higher perplexity) and may require additional filtering or downstream validation.
- **Model does not guarantee functional or experimentally active enzymes.** ZymCTRL is a sequence-level autoregressive model trained on BRENDA; its ``ZymCTRLGenerateResponseGenerated.sequence`` outputs are statistically plausible but not guaranteed to fold, express, or catalyze the desired reaction. Structural and functional scores reported in the paper are predictive only. For practical enzyme design, users should treat ZymCTRL as a generator for candidate sequences to be ranked by orthogonal models and validated experimentally, not as a standalone design oracle.
- **Embedding behavior and encode-specific constraints.** ``ZymCTRLEncodeRequestItem.sequence`` must be a valid, unambiguous amino-acid string (standard 20-letter alphabet) and ≤ ``1024`` residues. The optional ``ec_number`` in encode requests is format-validated but does not retroactively “fix” or relabel the sequence; it only conditions the internal representation in a way consistent with training. The ``ZymCTRLEncodeParams.pooling`` option controls the shape of the output: ``"mean"`` and ``"last"`` return a single vector in ``embedding``, while ``"per_token"`` returns a list of vectors in ``per_token_embeddings`` and is more memory-intensive. Pooling choice should match your downstream task (e.g., clustering vs. token-level analyses).
- **Not a universal protein model or encoder.** ZymCTRL is enzyme-focused and conditioned on EC numbers. It is not optimal for non-enzymatic proteins (e.g., structural scaffolds, antibodies, receptors) or for general protein embedding tasks where family-level or taxonomy conditioning is required. The model may perform poorly on sequences or conditions far outside BRENDA-like enzyme space, and diffusion or structure-based methods may better capture geometry-driven properties (e.g., active-site shape) than this purely sequence-based language model.
- **Throughput and early-stage screening limitations.** Because generation is limited to ``num_samples <= 20`` per item (``ZymCTRLGenerateParams.num_samples``) and ``batch_size = 1``, ZymCTRL is not ideal for ultra–high-throughput enumeration of millions of candidates as a first-stage filter. It is better used to propose focused sets of candidates per EC class that are then evaluated with faster predictive or structural models; for large library exploration, other models or workflows may be more appropriate.

How We Use It
-------------

ZymCTRL enables EC-conditioned enzyme generation as a standardized component in design–build–test–learn workflows, where teams define EC-targeted design spaces, generate diverse candidate catalysts, and then route these sequences into downstream structure prediction, developability profiling, and experimental testing. Through scalable APIs for both generation (conditioned on EC numbers) and encoding (sequence embeddings with optional EC context), ZymCTRL integrates into automated pipelines for enzyme discovery, route scouting, and multi-round optimization, accelerating iteration from target reaction to ranked, test-ready variant sets that reflect both catalytic intent and process constraints.

- Enzyme discovery and expansion: rapidly proposes EC-specific, sequence-diverse candidates that can seed or extend campaign libraries, followed by similarity/novelty analysis with embedding models and structure-based triage.
- Multi-round optimization: supports iterative campaigns where experimental data feeds into custom ranking models, while ZymCTRL continues to supply EC-consistent variants tuned toward evolving objectives such as activity, selectivity, IP space, or manufacturability.

Related
-------

- ``ProGen2 Large`` – General-purpose protein LM for conditional generation by family or function; use it to explore broader catalytic scaffolds, then refine EC-specific enzyme designs with :mod:`ZymCTRL`.
- ``ProteinMPNN`` – Structure-conditioned sequence design; pair with :mod:`ZymCTRL` by generating EC-conditioned sequences, predicting structures, then using ProteinMPNN to optimize active-site neighborhoods on a fixed backbone.
- ``ESMFold`` – Fast structure prediction from sequence; apply it to :mod:`ZymCTRL`-generated enzymes to assess fold quality, active-site geometry, and suitability for downstream ProteinMPNN design.
- ``ProteInfer`` – Functional annotation from sequence; verify that :mod:`ZymCTRL`-generated sequences match the intended EC class and filter candidates using high-confidence predicted enzymatic functions.

References
----------

- Munsamy, G., Lindner, S., Lorenz, P., & Ferruz, N. (2022). ZymCTRL: a conditional language model for the controllable generation of artificial enzymes. *Machine Learning for Structural Biology Workshop, NeurIPS 2022*. https://openreview.net/forum?id=KoqUQ_gv0v
