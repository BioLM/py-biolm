Sadie Antibody API
==================

Sadie Antibody is an antibody-specific inverse folding model fine-tuned from ESM-IF1, optimized for generating structurally accurate antibody sequences from solved or predicted variable domain structures. It predicts residue-level mutation tolerance, amino acid probabilities, and samples sequences with backbone RMSDs averaging 0.95 Å. GPU-accelerated inference provides efficient, scalable design for antibody engineering, affinity maturation, and stability optimization tasks.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for Sadie Antibody.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="sadie-antibody",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/sadie-antibody/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/sadie-antibody/predictor/"

      headers = {

          "Authorization": "Token YOUR_API_KEY",

          "Content-Type": "application/json"

      }

      data = {}
      
      response = requests.post(url, headers=headers, json=data)

      result = response.json()

      print(result)

    .. tab-item:: R
        :sync: r

        .. code:: r


      library(httr2)

      library(jsonlite)
      
      url <- "https://biolm.ai/api/v3/sadie-antibody/predictor/"

      data <- list()
      

      response <- request(url) %>%

        req_headers(

          Authorization = "Token YOUR_API_KEY",

          `Content-Type` = "application/json"

        ) %>%

        req_body_json(data) %>%

        req_perform()
      

      result <- response %>% resp_body_json()

      print(result)

.. http:post:: /api/v3/sadie-antibody/predict/

   Predict endpoint for Sadie Antibody.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters for the prediction request:
      
        - **heavy_chain** (*string*, optional) — Chain identifier for the heavy chain of the antibody
        - **light_chain** (*string*, optional) — Chain identifier for the light chain of the antibody
        - **nanobody_chain** (*string*, optional) — Chain identifier for the nanobody
        - **antigen_chain** (*string*, optional) — Chain identifier for the antigen
        - **include** (*array of strings*, default: ["mean"]) — Output types to include, options: "mean", "residue", "logits"
        - **num_seq_per_target** (*int*, range: 1-100, default: 1) — Number of sequences to generate per target
        - **sampling_temp** (*float*, range: 0.0-4.0, default: 0.2) — Sampling temperature for sequence diversity
        - **regions** (*array of strings or integers*, default: ["CDR1", "CDR2", "CDR3"]) — Regions to sample sequences for
        - **limit_expected_variation** (*boolean*, default: False) — Whether to limit expected variation in sequences
        - **exclude_heavy** (*boolean*, default: False) — Whether to exclude the heavy chain from sampling
        - **exclude_light** (*boolean*, default: False) — Whether to exclude the light chain from sampling
      
      - **items** (*array of objects*, max: 1) --- Input items for the prediction request:
      
        - **pdb** (*string*, min length: 1, max length: 1024, required) — PDB structure of the antibody variable domain

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/sadie-antibody/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {}

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:
      
        - **sequences** (*array of objects*) — Generated sequences and associated data:
      
          - **global_score** (*float*) — Overall score for the sequence
          - **score** (*float*) — Specific score for the sequence
          - **heavy** (*string*) — Heavy chain sequence
          - **light** (*string*, optional) — Light chain sequence
          - **temperature** (*float*) — Sampling temperature used
          - **mutations** (*int*) — Number of mutations in the sequence
          - **seq_recovery** (*float*) — Sequence recovery percentage
      
        - **logprobs** (*array of arrays of floats*, optional) — Log probabilities for each residue
        - **logits** (*array of arrays of floats*, optional) — Logits for each residue
        - **pdb_posins** (*array of ints*, optional) — PDB position indices
        - **pdb_chain** (*array of strings*, optional) — PDB chain identifiers
        - **pdb_res** (*array of strings*, optional) — PDB residue identifiers
        - **top_res** (*array of strings*, optional) — Top predicted residues
        - **perplexity** (*array of floats*, optional) — Perplexity values for each position
        - **vocab** (*array of strings*, optional) — Vocabulary used for predictions

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
      "score": 184.0,
      "identity_species": "human",
      "v_gene": "IGHV1-3*01",
      "v_identity": 0.98,
      "j_gene": "IGHJ4*01",
      "j_identity": 0.93,
      "Chain": "H",
      "Numbering": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        82,
        82,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        100,
        100,
        100,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113
      ],
      "Insertion": [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "A",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "A",
        "B",
        "C",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "A",
        "B",
        "C",
        "D",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""
      ],
      "scheme": "chothia",
      "region_definition": "kabat",
      "fwr1_aa_gaps": "QVQLVQSGAEVKKPGASVKVSCKASGYTFT",
      "fwr1_aa_no_gaps": "QVQLVQSGAEVKKPGASVKVSCKASGYTFT",
      "cdr1_aa_gaps": "SYAMH",
      "cdr1_aa_no_gaps": "SYAMH",
      "fwr2_aa_gaps": "WVRQAPGQGLEWMG",
      "fwr2_aa_no_gaps": "WVRQAPGQGLEWMG",
      "cdr2_aa_gaps": "WINAGNGNTKYSQKFQG",
      "cdr2_aa_no_gaps": "WINAGNGNTKYSQKFQG",
      "fwr3_aa_gaps": "RVTITRDTSASTAYMELSSLRSEDTAVYYCAK",
      "fwr3_aa_no_gaps": "RVTITRDTSASTAYMELSSLRSEDTAVYYCAK",
      "cdr3_aa_gaps": "VSYLSTASSLDY",
      "cdr3_aa_no_gaps": "VSYLSTASSLDY",
      "fwr4_aa_gaps": "WGQGTLVTVSS",
      "fwr4_aa_no_gaps": "WGQGTLVTVSS",
      "leader": "",
      "follow": "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYNSTYRVVSVLTVLHQDWLNGK..."
    }
  ]
}


Performance
-----------

- AntiFold uses GPU-accelerated inference on NVIDIA A100 GPUs, providing faster inference than CPU-only deployments and comparable speed to other BioLM inverse folding models such as ESM-IF1 and IgBert Paired.

- AntiFold achieves superior predictive accuracy for antibody sequence recovery compared to general inverse folding models available on BioLM, such as ESM-IF1:
  
  - CDRH3 amino acid recovery: AntiFold (60%) significantly outperforms ESM-IF1 (43%) and ProteinMPNN-based models (AbMPNN, 56%).
  
  - Framework region (FR) amino acid recovery: AntiFold achieves 87-94%, surpassing AbMPNN (85-89%).

- AntiFold-designed antibody sequences maintain structural integrity with lower CDR backbone RMSD (0.95 Å) than ESM-IF1 (1.01 Å), ProteinMPNN (1.03 Å), and AbMPNN (0.98 Å), as validated by re-folding with ABodyBuilder2.

- AntiFold demonstrates significantly improved zero-shot prediction of antibody-antigen binding affinity (Spearman correlation = 0.42) compared to ESM-IF1 (0.33), AbMPNN (0.32), and ProteinMPNN (0.30).

- AntiFold's inverse folding probabilities effectively discriminate improved antibody variants (median rank score 80%) from lower-performing variants, outperforming ProteinMPNN (73%), ESM-IF1 (57%), and AbMPNN (55%).

- AntiFold can accurately handle both experimentally solved and computationally predicted antibody structures (ABodyBuilder2, AlphaFold2), with minimal performance degradation (ΔAAR < 0.5%), unlike AbMPNN (ΔAAR -2.7%).

- Input: Antibody variable domain structures in standard PDB format; optionally specify heavy, light, nanobody, and antigen chains.

- Output: Designed antibody sequences, per-residue mutation tolerance (perplexity), residue probabilities, and optional embeddings/logits.


Applications
------------

- Antibody design and optimization for therapeutic applications: AntiFold enables researchers to design antibodies with improved binding affinity and structural integrity by predicting sequences that maintain the backbone structure. This is valuable for developing effective therapeutics targeting specific antigens, such as in cancer or autoimmune disease treatments.

- Predictive modeling for antibody-antigen binding affinity: AntiFold's ability to predict binding affinity in a zero-shot manner helps in identifying high-affinity antibody candidates without extensive experimental validation. This accelerates the drug development process by narrowing down the pool of potential therapeutic antibodies.

- Structural conservation in antibody engineering: By generating sequences that preserve the structural fold, AntiFold allows for the optimization of specific antibody properties, such as stability and manufacturability, without compromising antigen-binding capabilities. This is crucial for producing antibodies that are both effective and manufacturable at scale.

- Antibody sequence diversity generation: AntiFold can generate diverse antibody sequences while maintaining structural similarity, which is useful for exploring a wide range of potential therapeutic antibodies. This diversity is important for overcoming resistance in pathogens or cancer cells.

- Limitations in long CDRH3 loop design: While AntiFold excels in sequence recovery for typical antibody structures, its performance may be less optimal for antibodies with unusually long CDRH3 loops. Researchers should consider additional validation steps when working with such antibodies to ensure desired binding properties.


Limitations
-----------

- **Batch Size**: AntiFold API supports a maximum ``batch_size`` of ``32`` items per prediction request and ``1`` item per generation request.
- **Maximum PDB Length**: Input PDB structures must not exceed ``max_pdb_str_len`` characters; overly large structures may need trimming or preprocessing.
- AntiFold is specifically optimized for antibody variable domain structures (heavy/light chains and nanobodies). It may perform poorly or unpredictably on general protein structures or non-antibody scaffolds.
- Performance decreases significantly with longer CDRH3 loops (≥16 residues), resulting in lower amino acid recovery and potentially less accurate predictions.
- AntiFold is designed for inverse folding tasks (structure-based sequence generation); it does not provide direct predictions of antibody developability properties (e.g., stability, aggregation, immunogenicity). Users should combine AntiFold outputs with complementary predictive tools for comprehensive antibody optimization.
- AntiFold's inverse folding probabilities correlate with antigen binding affinity, but the model primarily identifies structurally disruptive mutations. It does not explicitly optimize for affinity maturation or antigen specificity; dedicated affinity maturation models or experimental validation are recommended for these tasks.


How We Use It
-------------

BioLM integrates Sadie Antibody (AntiFold) into antibody design and optimization pipelines to accelerate antibody engineering workflows, enabling rapid identification of high-affinity antibody variants while preserving critical structure-dependent characteristics. By leveraging AntiFold's antibody-specific inverse folding capabilities, BioLM efficiently prioritizes sequences predicted to maintain structural integrity and antigen binding, significantly reducing downstream experimental validation costs and time-to-market. Integrated with BioLM's predictive modeling tools for immunogenicity, aggregation propensity, and developability assessments, AntiFold systematically filters and ranks antibody candidates, enabling multi-round optimization cycles and improving overall success rates.

- Accelerates antibody affinity maturation and optimization by identifying structurally viable mutations.
- Reduces experimental overhead and improves hit rates through integrated structural and biophysical property prediction.


Related
-------

- ``ABodyBuilder3 pLDDT`` – Accurately predicts antibody variable domain structures, providing reliable inputs for Sadie Antibody's structural analyses.
- ``IgBert Paired`` – Predicts antibody properties from paired heavy-light chain sequences, complementing Sadie Antibody's structural and functional predictions.
- ``ImmuneFold Antibody`` – Generates antibody structural models, useful as inputs for Sadie Antibody's downstream affinity and stability assessments.
- ``AntiFold`` – Performs inverse folding for antibody design, effectively guiding Sadie Antibody's optimization of antibody sequences while preserving structural integrity.


References
----------

- Høie, M. H., Hummer, A., Olsen, T. H., Aguilar-Sanjuan, B., Nielsen, M., & Deane, C. M. (2024). `AntiFold: Improved antibody structure-based design using inverse folding <https://opig.stats.ox.ac.uk/webapps/antifold/>`_. *Bioinformatics*, Oxford University Press.

