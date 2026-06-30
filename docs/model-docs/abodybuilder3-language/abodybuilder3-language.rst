ABodyBuilder3 Language API
==========================

ABodyBuilder3 is an antibody-specific language model designed to accurately restore missing residues in antibody sequences without requiring germline information. The model, trained on the Observed Antibody Space (OAS) database, leverages transformer-based embeddings to predict amino acids at incomplete or ambiguous positions, achieving approximately 97% accuracy and processing 100 sequences in under 7 seconds on standard CPUs. BioLM provides GPU-accelerated API access for antibody repertoire sequencing QC, antibody engineering, and therapeutic discovery applications.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ABodyBuilder3 Language.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="abodybuilder3-language",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/abodybuilder3-language/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/abodybuilder3-language/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/abodybuilder3-language/predictor/"

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

.. http:post:: /api/v3/abodybuilder3-language/predict/

   Predict endpoint for ABodyBuilder3 Language.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **plddt** (*bool*, default: False) — Whether to include pLDDT scores in the output
        - **seed** (*int*, optional, default: 42) — Random seed for reproducibility
      
      - **items** (*array of objects*, max: 1) --- Input sequences:
      
        - **H** (*string*, min length: 1, max length: 2048) — Heavy chain antibody sequence
        - **L** (*string*, min length: 1, max length: 2048) — Light chain antibody sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/abodybuilder3-language/predict/ HTTP/1.1
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
      
        - **pdb** (*string*) — Predicted antibody structure in PDB format
      
        - **plddt** (*array of arrays of floats*, optional) — Predicted per-residue confidence scores (pLDDT), values range from 0.0 (low confidence) to 100.0 (high confidence); shape: [number_of_chains][number_of_residues]

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "pdb": "REMARK   1 CREATED WITH OPENMM 7.7, 2025-06-17\nATOM      1  N   ALA H   0      -1.825 -13.109   4.977  1.00  0.00           N  \nATOM      2  H   ALA H   0      -2.288 -14.051   4.344  1.00  0.00      ..."
    }
  ]
}


Performance
-----------

- **GPU Specifications**: Utilizes NVIDIA T4 GPUs for accelerated processing, ensuring efficient handling of complex computations.
- **Batch Size**: Processes up to 1 sequence per batch, optimizing resource allocation and ensuring consistent performance.
- **Sequence Length**: Supports sequences with a maximum length of 2048 amino acids, accommodating extensive antibody sequences.
- **Processing Speed**: 
  - Typical completion time for a single batch is within seconds, leveraging GPU acceleration for rapid inference.
  - AbodyBuilder3 Language model is optimized for swift execution compared to general protein models like ESM-1b, offering a 7x speed improvement.
- **Predictive Accuracy**: 
  - Outperforms general protein language models such as ESM-1b in restoring missing residues in antibody sequences.
  - Achieves up to 98% accuracy in residue restoration tasks, surpassing traditional methods like IMGT germline predictions.
- **Performance Optimization**:
  - Tailored for antibody-specific tasks, providing more precise predictions than general models.
  - Efficient resource utilization and model scaling enhance throughput and reduce latency in high-demand scenarios.
- **Comparison with Other Models**:
  - AbodyBuilder3 Language model exhibits superior performance in antibody sequence tasks compared to ESM-1b and similar models.
  - While models like AlphaFold2 offer high accuracy in structural predictions, AbodyBuilder3 is specialized for language-based tasks, ensuring optimal performance in antibody-related applications.


Applications
------------

- Restoring incomplete antibody sequences from high-throughput sequencing data by accurately predicting missing residues at the N-terminus or scattered internal positions, enabling biotech companies to recover and utilize otherwise discarded sequences; particularly valuable for antibody discovery workflows where sequencing errors frequently occur; performs significantly better than general protein language models (e.g., ESM-1b) and does not require prior germline identification, although accuracy may decrease slightly when predicting very long missing segments (>30 residues).
- Enhancing antibody maturation and optimization efforts by generating residue-level predictions for potential mutations, enabling researchers to rapidly evaluate amino acid substitutions for improved affinity, stability, or specificity; particularly useful for antibody engineering pipelines aiming to accelerate lead optimization cycles; however, predictions should be experimentally validated, as computational scores alone do not guarantee functional improvements.
- Filtering and ranking antibody candidates by generating sequence-level embeddings (seq-codings) that capture germline origin, mutation patterns, and functional properties, allowing researchers to efficiently cluster and prioritize large antibody repertoires for downstream experimental validation; valuable for therapeutic antibody selection processes where rapid identification of promising candidates is critical; not optimal for detailed structural predictions or epitope mapping tasks, which require complementary computational or experimental methods.
- Imputing ambiguous residues resulting from sequencing errors or ambiguous base calls in antibody repertoire datasets, enabling biotech companies to improve data quality and maximize information extracted from costly sequencing experiments; particularly beneficial for single-cell B-cell receptor sequencing platforms prone to ambiguous base calls; although highly accurate for single-residue imputations, performance may degrade if multiple adjacent residues are ambiguous or missing.
- Determining correct antibody sequence lengths by combining ANARCI numbering with likelihood predictions for the first residue, enabling accurate reconstruction of antibody sequences with unknown N-terminal truncations; valuable for antibody repertoire analysis and therapeutic candidate characterization, where precise sequence annotation is essential; performs robustly even when standard numbering schemes fail due to indels or unusual CDR lengths, but may require iterative refinement for optimal accuracy.


Limitations
-----------

- **Maximum Sequence Length**: Antibody heavy (``H``) and light (``L``) chain sequences must each be between ``1`` and ``2048`` amino acids in length.
- **Batch Size**: Only one antibody sequence pair (``H`` and ``L`` chains) can be processed per API request (``batch_size = 1``).
- **GPU Type**: The ``LANGUAGE`` model variant requires a GPU (``T4``), which may affect inference cost and latency.
- ABodyBuilder3 is optimized specifically for antibody sequences; it is not suitable for general protein structure prediction tasks or non-antibody proteins.
- The model performs best on canonical antibody structures, particularly complementarity-determining regions (CDRs); performance may degrade significantly for atypical antibody formats or heavily engineered antibodies with unusual sequence features.
- For ranking and filtering large antibody libraries or rapid screening of millions of sequences, faster sequence-based models (e.g., ESMFold) may be more appropriate due to computational efficiency.


How We Use It
-------------

BioLM leverages ABodyBuilder3 Language (AbLang) to accelerate antibody design and optimization workflows by rapidly restoring incomplete antibody sequences, significantly improving data quality and downstream predictive accuracy. Integrated seamlessly into BioLM's antibody engineering pipelines, AbLang enables researchers to efficiently recover missing residues and predict amino acid likelihoods without relying on germline identification, facilitating faster and more informed decision-making in therapeutic antibody development.

- Accelerates iterative antibody optimization workflows when combined with predictive modeling and generative antibody design algorithms.
- Integrates smoothly with BioLM's comprehensive suite of antibody engineering tools, enabling streamlined data processing and rapid progression from computational designs to experimental validation.


Related
-------

- ``AbLang-2`` – An updated version of AbLang, this model further enhances antibody sequence completion and can be used in tandem with ABodyBuilder3 Language for improved antibody structure predictions.
- ``ESM-1v`` – A general protein language model that, while not antibody-specific, can complement ABodyBuilder3 Language by providing broader protein sequence insights and enhancing understanding of antibody interactions.
- ``ImmuneFold Antibody`` – Focused on antibody structure prediction, this algorithm can be used alongside ABodyBuilder3 Language to refine structural predictions and improve accuracy in antibody modeling.
- ``IgT5 Paired`` – Specializes in paired antibody sequence analysis, offering a complementary approach to ABodyBuilder3 Language for comprehensive antibody sequence and structure prediction tasks.


References
----------

- Olsen, T. H., Moal, I. H., & Deane, C. M. (2022). `AbLang: an antibody language model for completing antibody sequences <https://doi.org/10.1093/bioadv/vbac046>`_. *Bioinformatics Advances*.
- Olsen, T. H. et al. (2022). `OAS: a diverse database of cleaned, annotated and translated unpaired and paired antibody sequences <https://doi.org/10.1002/pro.4205>`_. *Protein Science*.
- Kovaltsuk, A. et al. (2018). `Observed antibody space: a resource for data mining next-generation sequencing of antibody repertoires <https://doi.org/10.4049/jimmunol.1800708>`_. *Journal of Immunology*.

