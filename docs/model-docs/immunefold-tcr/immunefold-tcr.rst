ImmuneFold TCR API
==================

ImmuneFold TCR is a GPU-accelerated structure prediction model for T-cell receptor (TCR) proteins, utilizing parameter-efficient transfer learning (LoRA) based on the ESMFold architecture. It accurately predicts atomic-resolution 3D structures directly from TCR sequence data without relying on homologous sequences or MSAs, achieving RMSD values of 1.31 Å for the challenging CDR3β region in approximately 3 seconds per structure on Nvidia A100 GPUs. ImmuneFold TCR supports prediction of unbound TCRs and TCR-epitope complexes, enabling high-throughput workflows in immunotherapy design, neoantigen screening, and epitope specificity analysis.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ImmuneFold TCR.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="immunefold-tcr",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/immunefold-tcr/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/immunefold-tcr/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/immunefold-tcr/predictor/"

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

.. http:post:: /api/v3/immunefold-tcr/predict/

   Predict endpoint for ImmuneFold TCR.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **contact_idx** (*int*, optional) — Index for contact prediction
      
      - **items** (*array of objects*, max: 32) --- Input sequences for prediction:
      
        - **H** (*string*, min length: 1, max length: 256, optional) — Heavy chain sequence for antibodies
        - **L** (*string*, min length: 1, max length: 256, optional) — Light chain sequence for antibodies
        - **B** (*string*, min length: 1, max length: 256, optional) — Beta chain sequence for TCRs
        - **A** (*string*, min length: 1, max length: 256, optional) — Alpha chain sequence for TCRs
        - **P** (*string*, min length: 1, max length: 256, optional) — Peptide sequence for TCRs
        - **M** (*string*, min length: 1, max length: 256, optional) — MHC sequence for TCRs
        - **pdb** (*string*, min length: 1, max length: 1000000, optional) — PDB formatted structure data

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/immunefold-tcr/predict/ HTTP/1.1
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
      
        - **ptm** (*float*, range: 0.0–1.0) — Predicted TM-score (pTM), global structure prediction confidence score
      
        - **full_plddt** (*float*, range: 0.0–100.0) — Average predicted Local Distance Difference Test (pLDDT) score over all residues
      
        - **plddt** (*array of arrays of floats*, shape: [num_chains, num_residues], range: 0.0–100.0) — Predicted per-residue pLDDT confidence scores for each chain
      
        - **pdb** (*string*) — Predicted protein structure in PDB format

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "error": true,
  "status_code": 500,
  "message": "{\"error\":\"Uncaught exception: \",\"status_code\":500}"
}


Performance
-----------

- ImmuneFold TCR API provides GPU-accelerated inference on Nvidia T4 GPUs, optimized specifically for immune protein structure prediction tasks.

- ImmuneFold TCR significantly outperforms general protein structure prediction models such as ESMFold and AlphaFold-Multimer for T-cell receptor (TCR) modeling:

  - Achieves a root mean squared deviation (RMSD) of 1.31 Å for the challenging CDR3β region, compared to 1.57 Å for AlphaFold-Multimer and 3.67 Å for ESMFold.

  - Demonstrates superior accuracy across all TCR complementarity-determining regions (CDRs) and framework regions relative to TCR-specific models like TCRmodel2 and ImmuneBuilder.

  - Provides more accurate modeling of inter-chain orientations between TCR α and β chains, achieving an interface RMSD (iRMS) of 1.43 Å and DockQ score of 0.84, surpassing both TCRmodel2 and AlphaFold-Multimer.

- Predictive confidence scores (pLDDT) generated by ImmuneFold TCR correlate strongly with actual RMSD values, demonstrating Pearson correlation coefficients of 0.87 at the protein level and 0.63 at the residue level, providing users with reliable quality estimates.

- ImmuneFold TCR predictions can be directly integrated with Rosetta energy calculations to enable zero-shot TCR-epitope binding affinity prediction, outperforming supervised methods such as PanPep and TEIM, achieving an AUROC of 0.69 compared to 0.55 and 0.57, respectively.

- Inference speed is significantly faster than comparable MSA-based methods (AlphaFold-Multimer, TCRmodel2):

  - ImmuneFold TCR: approximately 3 seconds per structure prediction on Nvidia T4 GPUs.

  - AlphaFold-Multimer and TCRmodel2: typically 5–20 minutes per prediction.

- BioLM's optimized deployment leverages low-rank adaptation (LoRA) fine-tuning, resulting in no additional computational overhead during inference compared to base ESMFold, while providing substantially improved accuracy for immune protein structures.


Applications
------------

- Accurate TCR structure prediction for immunotherapy development  
  Predicting the structure of T-cell receptors (TCRs) is crucial for designing targeted immunotherapies that can effectively recognize and bind to specific antigens. This application is valuable because it allows researchers and biotech companies to engineer TCRs with enhanced specificity and binding affinity, potentially leading to more effective cancer treatments. ImmuneFold's ability to accurately predict TCR structures, particularly in the hypervariable CDR3β region, addresses the challenge of limited experimental data and enhances the design of personalized TCR-based therapies.

- Zero-shot TCR-epitope binding prediction for neoantigen identification  
  ImmuneFold's integration with Rosetta software enables zero-shot prediction of TCR-epitope binding affinities, which is essential for identifying neoantigens that can trigger immune responses. This application is particularly valuable for developing personalized cancer vaccines and immunotherapies, as it allows for the identification of novel targets without the need for extensive experimental binding data. By accurately predicting binding interactions, researchers can prioritize candidate neoantigens for further development, streamlining the immunotherapy design process.

- Antibody and nanobody structure prediction for therapeutic development  
  Predicting the structures of antibodies and nanobodies is critical for designing new therapeutic agents with high specificity and affinity for their targets. ImmuneFold's ability to accurately model the highly variable CDR regions, especially CDR H3, provides researchers with the structural insights needed to optimize antibody and nanobody candidates. This application is valuable for biotech companies focused on developing monoclonal antibodies for various diseases, as it accelerates the lead optimization phase and reduces reliance on costly experimental methods like X-ray crystallography.

- Structure prediction in the presence of antigens for virtual screening  
  ImmuneFold supports the prediction of antibody and nanobody structures in the presence of specific antigens, enhancing virtual screening efforts. This capability is crucial for identifying potential therapeutic candidates that maintain binding efficacy in complex biological environments. By providing accurate structural models that account for antigen context, researchers can better assess the binding potential and stability of antibody-antigen interactions, thereby improving the efficiency of drug discovery pipelines.

- Efficient training and inference for academic research  
  The use of Low-Rank Adaptation (LoRA) in ImmuneFold allows for parameter-efficient fine-tuning, making it accessible to academic research groups with limited computational resources. This application is valuable for researchers aiming to explore immune protein structures without the need for extensive computational infrastructure. By reducing memory usage and training time, ImmuneFold democratizes access to advanced protein modeling techniques, fostering innovation and collaboration in the scientific community.


Limitations
-----------

- **Maximum Sequence Length**: Input sequences for each chain (`B`, `A`, `P`, `M`) must not exceed ``256`` amino acids. Longer sequences are not supported and must be truncated or split accordingly.
- **Batch Size**: The maximum number of items per request is ``32``. Larger batches must be divided into multiple API calls.
- ImmuneFold is specifically fine-tuned for T-cell receptor (TCR) sequences; predictions for other immune proteins (e.g., antibodies, nanobodies) require a separate model type (`antibody`).
- ImmuneFold does not explicitly model conformational flexibility or dynamics. Predictions represent static structures and may not accurately reflect conformational changes upon antigen binding.
- ImmuneFold predictions are optimized for TCR-peptide-MHC complexes. For general protein-protein docking tasks or large antigen complexes, alternative methods such as AlphaFold-Multimer may yield better accuracy.
- The model provides predicted confidence scores (`plddt`, `ptm`) which correlate with accuracy. However, highly unique or unusual sequences without close homologs may result in lower confidence and less accurate predictions.


How We Use It
-------------

ImmuneFold TCR enables accurate structural modeling and zero-shot binding affinity prediction for T-cell receptors, directly facilitating the design and optimization of TCR-based immunotherapies. Integrated into BioLM's protein engineering workflows, ImmuneFold TCR accelerates candidate screening and prioritization by providing reliable structural insights, especially for TCR hypervariable CDR3 regions, where traditional homology-based methods fall short. By combining ImmuneFold's structure predictions with downstream tools such as Rosetta energy calculations, researchers rapidly identify promising TCR-epitope interactions, reducing experimental cycles and driving faster therapeutic development.

- Accelerates identification and ranking of TCR candidates through accurate structure-based affinity predictions.
- Integrates seamlessly with downstream computational tools, improving efficiency and reducing experimental validation costs.


Related
-------

- ``ImmuneFold Antibody`` – Uses the same LoRA-based fine-tuning of ESMFold specifically adapted for antibody and nanobody structure prediction tasks.
- ``TCRBuilder2`` – Provides an alternative deep-learning approach to TCR structure prediction, allowing comparative analysis and validation of ImmuneFold TCR results.
- ``AlphaFold2`` – General-purpose protein structure prediction tool that complements ImmuneFold TCR by enabling baseline comparisons and predictions for non-immune proteins.
- ``ESMFold`` – The foundational model fine-tuned by ImmuneFold TCR, useful for benchmarking performance improvements gained through specialized immune protein training.


References
----------

- Tian Zhu, Milong Ren, Zaikai He, Siyuan Tao, Ming Li, Dongbo Bu, & Haicang Zhang. (2024). Accurate structure prediction of immune proteins using parameter-efficient transfer learning. *Journal of Computational Biology*.

