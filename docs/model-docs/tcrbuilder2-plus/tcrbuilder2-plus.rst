TCRBuilder2+ API
================

TCRBuilder2+ is a GPU-accelerated deep learning model specialized for rapidly predicting accurate 3D structures of T-cell receptor (TCR) variable domains from amino acid sequences. Optimized specifically for TCR modeling, it predicts backbone conformations with CDR loop RMSDs averaging below 2 Å. Outputs include PDB-format structures and per-residue error estimations. TCRBuilder2+ supports high-throughput workflows for therapeutic TCR discovery, immune repertoire analysis, and structure-guided antigen-binding studies.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for TCRBuilder2+.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="tcrbuilder2-plus",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/tcrbuilder2-plus/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/tcrbuilder2-plus/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/tcrbuilder2-plus/predictor/"

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

.. http:post:: /api/v3/tcrbuilder2-plus/predict/

   Predict endpoint for TCRBuilder2+.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **include** (*array of strings*, default: ["mean"]) — Output types to include:
        
          - Allowed values: "mean", "per_token", "bos", "contacts", "logits", "attentions"
      
      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:
      
        - **H** (*string*, optional, min length: 1, max length: 2048) — Heavy chain amino acid sequence (required for ABodyBuilder2 and NanoBodyBuilder2)
        
        - **L** (*string*, optional, min length: 1, max length: 2048) — Light chain amino acid sequence (required for ABodyBuilder2)
        
        - **A** (*string*, optional, min length: 1, max length: 2048) — Alpha chain amino acid sequence (required for TCRBuilder2)
        
        - **B** (*string*, optional, min length: 1, max length: 2048) — Beta chain amino acid sequence (required for TCRBuilder2)
      
      - **ImmuneBuilderNanoBodyBuilder2PredictRequest** (*object*) --- NanoBodyBuilder2-specific request structure:
      
        - **items** (*array of objects*, min: 1, max: 8) — Input sequences:
        
          - **H** (*string*, required, min length: 1, max length: 2048) — Heavy chain amino acid sequence
      
      - **ImmuneBuilderABodyBuilder2PredictRequest** (*object*) --- ABodyBuilder2-specific request structure:
      
        - **items** (*array of objects*, min: 1, max: 8) — Input sequences:
        
          - **H** (*string*, required, min length: 1, max length: 2048) — Heavy chain amino acid sequence
          
          - **L** (*string*, required, min length: 1, max length: 2048) — Light chain amino acid sequence
      
      - **ImmuneBuilderTCRBuilder2PredictRequest** (*object*) --- TCRBuilder2-specific request structure:
      
        - **items** (*array of objects*, min: 1, max: 8) — Input sequences:
        
          - **A** (*string*, required, min length: 1, max length: 2048) — Alpha chain amino acid sequence
          
          - **B** (*string*, required, min length: 1, max length: 2048) — Beta chain amino acid sequence

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/tcrbuilder2-plus/predict/ HTTP/1.1
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
      
        - **pdb** (*string*) — Predicted immune protein structure in standard PDB format; includes atomic coordinates for all atoms (heavy atoms and hydrogens), residue numbering according to IMGT scheme, and chain identifiers; structure is refined to remove steric clashes, incorrect peptide bond lengths, cis-peptide bonds, and D-amino acids; coordinates are in Angstroms (Å)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "error": true,
  "status_code": 503,
  "message": "{\"error\":\"Uncaught exception\"}"
}


Performance
-----------

- TCRBuilder2+ delivers significantly improved accuracy over the original TCRBuilder2 model, particularly in predicting the challenging CDR loops of T-cell receptor (TCR) structures:
  
  - CDR-α3 loop RMSD: 1.85 Å (TCRBuilder2+) vs. 2.89 Å (original TCRBuilder), a 36% improvement.
  
  - CDR-β3 loop RMSD: 1.93 Å (TCRBuilder2+) vs. 3.12 Å (original TCRBuilder), a 38% improvement.

- Compared to AlphaFold-Multimer, TCRBuilder2+ achieves comparable accuracy in TCR structural predictions, with nearly identical RMSD values for critical loops (CDR-α3: 1.85 Å vs. 1.84 Å; CDR-β3: 1.93 Å vs. 1.94 Å), while being substantially faster and more computationally efficient.

- TCRBuilder2+ is optimized specifically for TCR prediction, leveraging immune receptor-specific training datasets and structural constraints, resulting in significantly faster inference speeds compared to general-purpose structure predictors such as AlphaFold2 and ESMFold.

- Unlike general-purpose models (e.g., AlphaFold2), TCRBuilder2+ does not require extensive multiple sequence alignments or large sequence databases, enabling rapid predictions suitable for high-throughput analysis of large TCR sequence datasets.

- TCRBuilder2+ models are deployed using GPU-accelerated inference on NVIDIA Tesla P100 GPUs, allowing typical structure predictions to complete in approximately 5 seconds per individual TCR prediction, compared to approximately 30 minutes per prediction for AlphaFold-Multimer on similar hardware.

- TCRBuilder2+ generates physically plausible structures with minimal stereochemical errors, comparable to experimentally determined crystal structures, and superior to other specialized methods (e.g., RepertoireBuilder, original TCRBuilder).

- The model outputs structural predictions in standard Protein Data Bank (PDB) format, facilitating immediate downstream structural analysis and visualization.


Applications
------------

- Predicting T-cell receptor (TCR) structural conformations to accelerate therapeutic candidate selection, enabling rapid identification of TCRs with optimal antigen-binding properties; valuable for companies developing TCR-based immunotherapies, though less suitable for predicting highly flexible loop regions beyond typical CDR lengths.
- Structural modeling of TCR-antigen interactions to guide rational design and affinity maturation, providing detailed insights into binding interfaces; beneficial for biotech companies optimizing TCR specificity and affinity, but limited in accuracy for non-canonical or highly unusual TCR sequences.
- High-throughput structural screening of large-scale TCR sequence datasets from next-generation sequencing (NGS), enabling rapid identification of structurally viable receptor candidates; useful for biotech firms performing repertoire analysis and biomarker discovery, though predictions may require experimental validation for clinical applications.
- Computational filtering and prioritization of TCR sequences based on structural stability and predicted conformational variability, helping reduce experimental workload and costs; particularly valuable for companies conducting TCR library generation and screening, although less effective for predicting dynamic structural changes upon antigen binding.
- Generation of reliable TCR structural ensembles to estimate prediction uncertainty, allowing researchers to identify and exclude low-confidence models; advantageous for biotech teams integrating computational predictions into experimental pipelines, but not optimal for modeling large-scale conformational rearrangements or receptor clustering scenarios.


Limitations
-----------

- **Maximum Sequence Length**: The API accepts sequences up to ``2048`` amino acids per chain. Longer sequences must be truncated or split into smaller segments.
- **Batch Size**: Up to ``8`` sequence pairs per request. Larger datasets must be submitted in multiple batches.
- TCRBuilder2+ is specialized for predicting T-cell receptor (TCR) structures. It should not be used for antibody or nanobody structure prediction; for these, use ``ABodyBuilder2`` or ``NanoBodyBuilder2`` respectively.
- While TCRBuilder2+ provides accuracy comparable to AlphaFold-Multimer for TCR structures, it may be less accurate for highly unusual or novel TCR sequences, particularly in the highly variable CDR3 regions.
- TCRBuilder2+ predicts a single representative structure per sequence pair. It does not provide alternative conformations or ensembles, limiting its utility for modeling structural flexibility or conformational diversity.
- TCRBuilder2+ does not provide sequence embeddings or encodings. If downstream clustering, visualization, or embedding-based analyses are required, consider using embedding-focused models instead.


How We Use It
-------------

TCRBuilder2+ enables BioLM users to rapidly generate accurate structural models of T-cell receptors (TCRs) from sequence data, directly integrating into our protein engineering workflows to accelerate design cycles. By providing consistent predictions of TCR complementarity-determining regions (CDRs), the algorithm supports informed selection and optimization of therapeutic candidates, significantly reducing experimental trial and error. Within BioLM pipelines, TCRBuilder2+ integrates seamlessly with downstream property prediction tools, embedding generation, and candidate ranking algorithms, facilitating rapid prioritization of biologically viable and therapeutically promising TCR sequences.

- Accelerates TCR-based therapeutic discovery by providing accurate structural context for sequence-based datasets.
- Integrates effectively with BioLM predictive modeling and candidate ranking workflows, enhancing selection efficiency and experimental success rates.


Related
-------

- ``TCRBuilder2`` – Provides rapid, accurate structural predictions for T-cell receptors, serving as a faster alternative when ensemble predictions from ``TCRBuilder2+`` are not required.
- ``ImmuneFold TCR`` – Complements ``TCRBuilder2+`` by leveraging a different deep-learning architecture, useful for cross-validation and ensemble modeling of TCR structures.
- ``NanoBodyBuilder2`` – Offers specialized deep-learning models for nanobody structure prediction, analogous to ``TCRBuilder2+`` but focused on single-domain antibodies.
- ``ABodyBuilder3 pLDDT`` – Predicts antibody structures with residue-level confidence scores, complementing ``TCRBuilder2+`` by providing similar uncertainty estimation capabilities for antibodies.


References
----------

- Abanades, B., Wong, W. K., Boyles, F., Georges, G., Bujotzek, A., & Deane, C. M. (2023). `ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins <https://doi.org/10.1038/s42003-023-04927-7>`_. *Communications Biology*.

