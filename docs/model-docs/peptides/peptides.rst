Peptides API
============

Peptides computes a focused set of physicochemical descriptors for antimicrobial peptide (AMP) analysis, including sequence length, amino acid composition, net charge, aliphatic index, molecular weight, isoelectric point, hydrophobicity (GRAVY), instability index, Boman index, and hydrophobic moment, with optional per-residue profiles. The encoder endpoint accepts amino acid sequences up to 2048 residues in batches of up to 10, returning numeric and vector features for AMP classification, virtual screening, and peptide design workflows.

Encode
------

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="peptides",
                action="encode",
                params={
                  "include": [
                    "vector"
                  ]
                },
                items=[
                  {
                    "sequence": "GLPRKILCAI"
                  },
                  {
                    "sequence": "KKKGKCKGPLKLVCKC"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/peptides/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": [
                  "vector"
                ]
              },
              "items": [
                {
                  "sequence": "GLPRKILCAI"
                },
                {
                  "sequence": "KKKGKCKGPLKLVCKC"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/peptides/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": [
                      "vector"
                    ]
                  },
                  "items": [
                    {
                      "sequence": "GLPRKILCAI"
                    },
                    {
                      "sequence": "KKKGKCKGPLKLVCKC"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/peptides/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = list(
                  "vector"
                )
              ),
              items = list(
                list(
                  sequence = "GLPRKILCAI"
                ),
                list(
                  sequence = "KKKGKCKGPLKLVCKC"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/peptides/encode/

   Encode endpoint for Peptides.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, optional, default: []) — Additional feature groups to compute

          - Allowed values:

            - "vector" — Include vector-based feature profiles


      - **items** (*array of objects*, required, min items: 1, max items: 10) --- Input peptide sequences:

        - **sequence** (*string*, required, min length: 1, max length: 2048) — Peptide sequence using extended amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/peptides/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": [
            "vector"
          ]
        },
        "items": [
          {
            "sequence": "GLPRKILCAI"
          },
          {
            "sequence": "KKKGKCKGPLKLVCKC"
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

        - **features** (*object*) — Computed peptide features and descriptor values:

          - **aliphatic_index** (*float*, ≥ 0.0) — Aliphatic index value

          - **boman** (*float*) — Boman (potential protein interaction) index

          - **charge** (*float*) — Net charge at pH 7

          - **descriptors** (*object*, optional) — Collection of peptide structural descriptors (when present):

            - **length** (*int*, ≥ 1) — Number of amino acids in sequence

            - **amino_acid_composition** (*object*) — Amino acid class composition metrics:

              - **Tiny** (*object*) — Class containing amino acids A, C, G, S, T

                - **number** (*int*, ≥ 0) — Count of tiny amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of tiny amino acids

              - **Small** (*object*) — Class containing amino acids A, B, C, D, G, N, P, S, T, V

                - **number** (*int*, ≥ 0) — Count of small amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of small amino acids

              - **Aliphatic** (*object*) — Class containing amino acids A, I, L, V

                - **number** (*int*, ≥ 0) — Count of aliphatic amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of aliphatic amino acids

              - **Aromatic** (*object*) — Class containing amino acids F, H, W, Y

                - **number** (*int*, ≥ 0) — Count of aromatic amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of aromatic amino acids

              - **NonPolar** (*object*) — Class containing amino acids A, C, F, G, I, L, M, P, V, W, Y

                - **number** (*int*, ≥ 0) — Count of non-polar amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of non-polar amino acids

              - **Polar** (*object*) — Class containing amino acids D, E, H, K, N, Q, R, S, T, Z

                - **number** (*int*, ≥ 0) — Count of polar amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of polar amino acids

              - **Charged** (*object*) — Class containing amino acids B, D, E, H, K, R, Z

                - **number** (*int*, ≥ 0) — Count of charged amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of charged amino acids

              - **Basic** (*object*) — Class containing amino acids H, K, R

                - **number** (*int*, ≥ 0) — Count of basic amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of basic amino acids

              - **Acidic** (*object*) — Class containing amino acids B, D, E, Z

                - **number** (*int*, ≥ 0) — Count of acidic amino acids
                - **mole_percent** (*float*, 0.0–100.0) — Mole percentage of acidic amino acids

          - **frequencies** (*object*) — Per-residue frequencies, as fractional counts for each amino acid symbol (e.g. ``A_frequency``, ``R_frequency``)

          - **hydrophobic_moment** (*float*, ≥ 0.0) — Hydrophobic moment value

          - **hydrophobicity** (*float*) — Mean hydrophobicity index

          - **instability_index** (*float*) — Instability index value

          - **isoelectric_point** (*float*) — Isoelectric point (pI) on pH scale

          - **mass_shift** (*float*) — Mass shift value in daltons (Da)

          - **molecular_weight** (*float*) — Molecular weight in daltons (Da)

          - **mz** (*float*) — Mass-to-charge ratio (m/z)

          - **hydrophobic_moment_profile** (*array of floats*, length: sequence length - 10) — Sliding-window hydrophobic moment values (empty if ``vector`` features are not requested)

          - **hydrophobicity_profile** (*array of floats*, length: sequence length - 10) — Sliding-window mean hydrophobicity values (empty if ``vector`` features are not requested)

          - **linker_preference_profile** (*array of floats*, length: sequence length) — Linker preference scores per position (empty if ``vector`` features are not requested)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "features": {
              "aliphatic_index": 166,
              "boman": -0.3240000009536743,
              "charge": 1.9357112646102905,
              "AF1": -0.327700138092041,
              "AF2": -0.07865546643733978,
              "AF3": 0.13956385850906372,
              "AF4": 0.5497181415557861,
              "AF5": 0.4625459909439087,
              "BLOSUM1": -0.28600001335144043,
              "BLOSUM2": -0.6370000243186951,
              "BLOSUM3": -0.07400000095367432,
              "BLOSUM4": -0.10100000351667404,
              "BLOSUM5": 0.12399999797344208,
              "BLOSUM6": -0.014999999664723873,
              "BLOSUM7": 0.2809999883174896,
              "BLOSUM8": 0.1860000044107437,
              "BLOSUM9": -0.05299999937415123,
              "BLOSUM10": -0.0820000022649765,
              "PP1": -0.5479999780654907,
              "PP2": -0.1940000057220459,
              "PP3": 0.2070000022649765,
              "F1": 0.3424000144004822,
              "F2": 0.08630000054836273,
              "F3": -0.08229999989271164,
              "F4": 0.5688999891281128,
              "F5": 0.24549999833106995,
              "F6": 0.21050000190734863,
              "KF1": -0.15800000727176666,
              "KF2": -0.30799999833106995,
              "KF3": 0.23399999737739563,
              "KF4": -0.23999999463558197,
              "KF5": -0.30000001192092896,
              "KF6": -0.4129999876022339,
              "KF7": 0.5270000100135803,
              "KF8": -0.1679999977350235,
              "KF9": 0.23899999260902405,
              "KF10": 0.06300000101327896,
              "MSWHIM1": -0.6159999966621399,
              "MSWHIM2": 0.4359999895095825,
              "MSWHIM3": -0.1459999978542328,
              "E1": -0.05590000003576279,
              "E2": 0.06340000033378601,
              "E3": -0.06650000065565109,
              "E4": -0.11919999867677689,
              "E5": 0.010400000028312206,
              "PD1": -0.4690000116825104,
              "PD2": -0.4410000145435333,
              "PRIN1": 2.281709909439087,
              "PRIN2": -1.503059983253479,
              "PRIN3": 1.5445400476455688,
              "ProtFP1": 1.190000057220459,
              "ProtFP2": -1.402999997138977,
              "ProtFP3": 0.050999999046325684,
              "ProtFP4": 0.8600000143051147,
              "ProtFP5": -0.42899999022483826,
              "ProtFP6": 0.31200000643730164,
              "ProtFP7": 0.21299999952316284,
              "ProtFP8": -0.1340000033378601,
              "SV1": 0.23309999704360962,
              "SV2": 0.14489999413490295,
              "SV3": -0.026900000870227814,
              "SV4": -0.14190000295639038,
              "ST1": -0.9506000280380249,
              "ST2": -0.3418999910354614,
              "ST3": -0.10819999873638153,
              "ST4": 0.11299999803304672,
              "ST5": -0.48919999599456787,
              "ST6": -0.16130000352859497,
              "ST7": 0.21539999544620514,
              "ST8": 0.16130000352859497,
              "SVGER1": -1.7931300401687622,
              "SVGER2": -0.8130599856376648,
              "SVGER3": 0.3334699869155884,
              "SVGER4": -0.838129997253418,
              "SVGER5": 0.7487000226974487,
              "SVGER6": -0.004189999774098396,
              "SVGER7": -3.0354299545288086,
              "SVGER8": 0.7736499905586243,
              "SVGER9": -0.7188799977302551,
              "SVGER10": -2.2506299018859863,
              "SVGER11": 0.6736699938774109,
              "T1": -5.179999828338623,
              "T2": -0.10100000351667404,
              "T3": -0.4480000138282776,
              "T4": 0.7950000166893005,
              "T5": 0.5550000071525574,
              "VHSE1": 0.296999990940094,
              "VHSE2": -0.24699999392032623,
              "VHSE3": -0.1679999977350235,
              "VHSE4": -0.19300000369548798,
              "VHSE5": 0.3709999918937683,
              "VHSE6": -0.6050000190734863,
              "VHSE7": 0.33399999141693115,
              "VHSE8": 0.10100000351667404,
              "VSTPV1": -0.9129999876022339,
              "VSTPV2": 0.32600000500679016,
              "VSTPV3": 0.6340000033378601,
              "VSTPV4": -0.013000000268220901,
              "VSTPV5": -0.006000000052154064,
              "VSTPV6": -0.3070000112056732,
              "Z1": -0.906000018119812,
              "Z2": -1.0449999570846558,
              "Z3": -0.5839999914169312,
              "Z4": 0.02800000086426735,
              "Z5": 0.26100000739097595,
              "A_frequency": 0.1,
              "R_frequency": 0.1,
              "N_frequency": 0,
              "D_frequency": 0,
              "C_frequency": 0.1,
              "Q_frequency": 0,
              "E_frequency": 0,
              "G_frequency": 0.1,
              "H_frequency": 0,
              "I_frequency": 0.2,
              "L_frequency": 0.2,
              "K_frequency": 0.1,
              "M_frequency": 0,
              "F_frequency": 0,
              "P_frequency": 0.1,
              "S_frequency": 0,
              "T_frequency": 0,
              "W_frequency": 0,
              "Y_frequency": 0,
              "V_frequency": 0,
              "O_frequency": 0,
              "U_frequency": 0,
              "B_frequency": 0,
              "Z_frequency": 0,
              "J_frequency": 0,
              "X_frequency": 0,
              "hydrophobic_moment": 0.552782575399268,
              "hydrophobicity": 1.0499999523162842,
              "instability_index": 31.490000000000002,
              "isoelectric_point": 9.833242292515934,
              "mass_shift": 12.040258407592773,
              "molecular_weight": 1083.400634765625,
              "mz": 570.8497314453125,
              "hydrophobic_moment_profile": [],
              "hydrophobicity_profile": [],
              "linker_preference_profile": []
            }
          },
          {
            "features": {
              "aliphatic_index": 66.875,
              "boman": 1.203125,
              "charge": 6.809994697570801,
              "AF1": 0.3024120628833771,
              "AF2": 0.03756478428840637,
              "AF3": -0.08593206107616425,
              "AF4": 0.0802687257528305,
              "AF5": 0.6511217951774597,
              "BLOSUM1": 0.21062499284744263,
              "BLOSUM2": -0.4312500059604645,
              "BLOSUM3": 0.03312499821186066,
              "BLOSUM4": -0.28062498569488525,
              "BLOSUM5": 0.5218750238418579,
              "BLOSUM6": -0.078125,
              "BLOSUM7": 0.171875,
              "BLOSUM8": 0.23499999940395355,
              "BLOSUM9": -0.008124999701976776,
              "BLOSUM10": -0.34375,
              "PP1": -0.17624999582767487,
              "PP2": -0.2175000011920929,
              "PP3": 0.49125000834465027,
              "F1": -0.40418750047683716,
              "F2": 0.02449999935925007,
              "F3": -0.2939375042915344,
              "F4": 0.20343750715255737,
              "F5": -0.07000000029802322,
              "F6": 0.4350624978542328,
              "KF1": 0.008750000037252903,
              "KF2": -0.11812499910593033,
              "KF3": -0.019375000149011612,
              "KF4": 0.3174999952316284,
              "KF5": 0.5706250071525574,
              "KF6": -0.6056249737739563,
              "KF7": 1.024999976158142,
              "KF8": -0.10437499731779099,
              "KF9": -0.13187499344348907,
              "KF10": 0.6656249761581421,
              "MSWHIM1": -0.5674999952316284,
              "MSWHIM2": 0.23375000059604645,
              "MSWHIM3": 0.02437499910593033,
              "E1": 0.06912499666213989,
              "E2": -0.01681249961256981,
              "E3": -0.02850000001490116,
              "E4": -0.09393750131130219,
              "E5": -0.03762499988079071,
              "PD1": -0.5106250047683716,
              "PD2": 0.2731249928474426,
              "PRIN1": -1.6547374725341797,
              "PRIN2": 0.545131266117096,
              "PRIN3": 1.2218937873840332,
              "ProtFP1": -1.2331249713897705,
              "ProtFP2": -0.058125000447034836,
              "ProtFP3": 0.96875,
              "ProtFP4": 1.1349999904632568,
              "ProtFP5": -0.3737500011920929,
              "ProtFP6": 0.7893750071525574,
              "ProtFP7": 0.11812499910593033,
              "ProtFP8": 0.2175000011920929,
              "SV1": 0.22237500548362732,
              "SV2": 0.08187499642372131,
              "SV3": -0.0338749997317791,
              "SV4": -0.14168749749660492,
              "ST1": -0.9298750162124634,
              "ST2": -0.0988750010728836,
              "ST3": 0.11231250315904617,
              "ST4": 0.21168750524520874,
              "ST5": -0.461062490940094,
              "ST6": -0.04574999958276749,
              "ST7": 0.34168750047683716,
              "ST8": 0.30506250262260437,
              "SVGER1": -0.9708062410354614,
              "SVGER2": -0.016462499275803566,
              "SVGER3": -1.2430437803268433,
              "SVGER4": -0.8692125082015991,
              "SVGER5": 1.0701375007629395,
              "SVGER6": 0.2856374979019165,
              "SVGER7": -2.924762487411499,
              "SVGER8": 0.4098062515258789,
              "SVGER9": -0.7742562294006348,
              "SVGER10": -0.6717687249183655,
              "SVGER11": -0.0656687468290329,
              "T1": -5.071249961853027,
              "T2": 0.4662500023841858,
              "T3": -0.8931249976158142,
              "T4": 0.6506249904632568,
              "T5": 0.512499988079071,
              "VHSE1": -0.2718749940395355,
              "VHSE2": -0.2574999928474426,
              "VHSE3": -0.10562500357627869,
              "VHSE4": 0.37937501072883606,
              "VHSE5": 0.6918749809265137,
              "VHSE6": 0.02812499925494194,
              "VHSE7": 0.6456249952316284,
              "VHSE8": -0.0031250000465661287,
              "VSTPV1": -0.8887500166893005,
              "VSTPV2": 0.6181250214576721,
              "VSTPV3": 0.5618749856948853,
              "VSTPV4": 0.20499999821186066,
              "VSTPV5": 0.31687501072883606,
              "VSTPV6": -0.34437501430511475,
              "Z1": 0.6150000095367432,
              "Z2": -0.7418749928474426,
              "Z3": -0.5087500214576721,
              "Z4": 0.48374998569488525,
              "Z5": -0.18000000715255737,
              "A_frequency": 0,
              "R_frequency": 0,
              "N_frequency": 0,
              "D_frequency": 0,
              "C_frequency": 0.1875,
              "Q_frequency": 0,
              "E_frequency": 0,
              "G_frequency": 0.125,
              "H_frequency": 0,
              "I_frequency": 0,
              "L_frequency": 0.125,
              "K_frequency": 0.4375,
              "M_frequency": 0,
              "F_frequency": 0,
              "P_frequency": 0.0625,
              "S_frequency": 0,
              "T_frequency": 0,
              "W_frequency": 0,
              "Y_frequency": 0,
              "V_frequency": 0.0625,
              "O_frequency": 0,
              "U_frequency": 0,
              "B_frequency": 0,
              "Z_frequency": 0,
              "J_frequency": 0,
              "X_frequency": 0,
              "hydrophobic_moment": 0.4868826955054231,
              "hydrophobicity": -0.6499999761581421,
              "instability_index": -17.15625,
              "isoelectric_point": 10.682097629643977,
              "mass_shift": 42.14090347290039,
              "molecular_weight": 1761.322265625,
              "mz": 966.5570068359375,
              "hydrophobic_moment_profile": [
                0.12071507424116135,
                0.2683659791946411,
                "... (truncated for documentation)"
              ],
              "hydrophobicity_profile": [
                -1.7727272510528564,
                -1.0727273225784302,
                "... (truncated for documentation)"
              ],
              "linker_preference_profile": [
                0.03453333303332329,
                0.07180000096559525
              ]
            }
          }
        ]
      }


Performance
-----------

- CPU-only execution with 0.125 vCPU and 1 GB RAM per worker; no GPU is used. The implementation relies on lightweight, analytic descriptors (charge, hydrophobicity, instability index, etc.), so runtime scales approximately linearly with sequence length and batch size.

- Predictive performance for AMP-related tasks is comparable to the original Peptides R package: when these descriptors are used with linear discriminant analysis, reported AMP classification accuracy reaches ~95%, and ~85% with CART-based models on benchmark datasets.

- Compared to structure-focused models in the BioLM ecosystem (e.g., ESMFold or AlphaFold2), Peptides is substantially faster and less compute-intensive because it avoids 3D structure prediction and focuses on closed-form physicochemical calculations. For rapid AMP screening or feature generation, it typically offers higher throughput per CPU core than deep sequence or structure models.

Applications
------------

- Rapid screening and prioritization of antimicrobial peptide (AMP) candidates by computing physicochemical descriptors (e.g., hydrophobic moment, net charge, aliphatic and instability indices, GRAVY, amino-acid composition), enabling biotech teams to rank and filter large in silico peptide libraries for therapeutic or agricultural applications; not optimal for peptides longer than ~50 amino acids where AMP-focused heuristics are less informative.
- Optimization of membrane-active peptide antibiotics by quantifying Boman index, hydrophobicity, and hydrophobic moment profiles, helping researchers tune peptide–membrane selectivity and reduce off-target protein binding in lead series; less informative for peptides whose primary mechanism is via specific intracellular protein targets.
- Stability and manufacturability assessment of AMP hits using aliphatic and instability indices alongside molecular weight and isoelectric point, allowing formulation and CMC teams to flag candidates with predicted poor thermostability, solubility, or proteolytic sensitivity before synthesis; accuracy may be reduced for sequences with noncanonical residues or extensive post-translational modifications, which are not explicitly modeled.
- Classification and pre-filtering of novel peptide sequences by exporting computed descriptors and frequencies as model-ready feature vectors for downstream machine learning (e.g., LDA, CART, other in-house classifiers), enabling companies to build internal AMP vs non-AMP or spectrum-of-activity models; performance may degrade on peptides that are highly dissimilar from the physicochemical space represented in training data.
- Support for peptide–membrane modeling workflows by generating hydrophobicity and hydrophobic moment profiles suitable for integration with external molecular dynamics pipelines (e.g., GROMACS XVG analyses handled outside this API), helping structural biologists rationalize sequence regions that are likely transmembrane, surface-associated, or globular; not suitable for direct parsing or visualization of MD trajectory files within this API.

Limitations
-----------

- **Maximum Sequence Length**: Each ``sequence`` must be at most ``2048`` amino acids long (longer inputs should be truncated or split before submission).
- **Batch Size**: A single request can contain up to ``10`` sequences in ``items`` (larger datasets must be chunked across multiple API calls).
- **Input Scope**: Features are computed from primary amino acid sequences only; no structural (3D), molecular dynamics, or GROMACS ``XVG`` file inputs are supported.
- **Output Scope**: The API returns physicochemical descriptors and profiles in ``features``; it does not directly predict antimicrobial activity, potency, toxicity, or other biological endpoints.
- **Sequence Length Regime**: Descriptors are most relevant for short peptides (typically <50 residues). For long protein-like sequences, interpretations for antimicrobial peptide design or classification may be less meaningful.
- **Embeddings and ML Workflows**: Outputs are engineered physicochemical descriptors (numeric and vector features), not latent sequence embeddings. For unsupervised clustering, visualization, or representation learning, sequence-embedding models are generally more appropriate.

How We Use It
-------------

BioLM uses the Peptides algorithm as a standardized feature generator for antimicrobial peptide R&D, turning raw sequences into rich physicochemical descriptors that feed larger ML workflows for classification, prioritization, and design. These descriptors (charge, hydrophobicity, aliphatic index, instability index, Boman index, hydrophobic moment, membrane-position proxies, and related profiles) are combined with sequence embeddings, AMP classifiers, and generative models to define candidate design spaces, enforce developability constraints, and link in silico optimization with downstream synthesis and testing.

- Enables high-throughput, API-driven computation of peptide properties and profiles for batches of up to 10 sequences (length ≤ 2048 aa).
- Integrates into iterative design loops, where Peptides features guide filtering, ranking, and retraining of AMP and protein engineering models.

Related
-------

- ``ESM-1v`` – Provides sequence-based protein function predictions that can be combined with Peptides descriptors to build richer AMP classification or activity models.
- ``BioLMTox-2`` – Predicts peptide toxicity, complementing Peptides physicochemical profiles when designing safer antimicrobial peptide candidates.
- ``AlphaFold2`` – Predicts 3D structures from sequences, useful for structurally validating AMP designs characterized with Peptides features.
- ``ESMFold`` – Offers fast sequence-to-structure predictions that can help interpret how Peptides-derived properties relate to peptide folding and stability.

References
----------

- Osorio, D., Rondon-Villarreal, P., & Torres, R. (2015). Peptides: A package for data mining of antimicrobial peptides. *The R Journal*, 7(1), 4–14. https://doi.org/10.32614/RJ-2015-001
