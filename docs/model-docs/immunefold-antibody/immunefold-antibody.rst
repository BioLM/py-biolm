ImmuneFold Antibody API
=======================

ImmuneFold Antibody is a GPU-accelerated algorithm providing antibody and nanobody structure prediction via parameter-efficient fine-tuning of ESMFold's Evoformer module, using low-rank adaptation (LoRA). It supports inference of unbound antibody structures or bound complexes when antigen context is available, achieving a CDR H3 RMSD of 2.65 Å, with typical inference times around 3 seconds per structure. ImmuneFold Antibody enables rapid, accurate modeling for antibody design, epitope mapping, and structural immunology research.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ImmuneFold Antibody.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="immunefold-antibody",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/immunefold-antibody/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/immunefold-antibody/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/immunefold-antibody/predictor/"

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

.. http:post:: /api/v3/immunefold-antibody/predict/

   Predict endpoint for ImmuneFold Antibody.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **contact_idx** (*int*, optional) — Index specifying residue position for contact analysis
      
      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences and structures:
      
        - **H** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of antibody heavy chain (single-domain VHH antibodies use only this field)
      
        - **L** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of antibody light chain (requires presence of heavy chain sequence)
      
        - **B** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of T-cell receptor β chain
      
        - **A** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of T-cell receptor α chain
      
        - **P** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of peptide antigen
      
        - **M** (*string*, optional, min length: 1, max length: 256) — Amino acid sequence of MHC molecule
      
        - **pdb** (*string*, optional, min length: 1, max length: 1000000) — Structure of antigen provided in PDB format
      
          - Cannot be provided simultaneously with TCR inputs (`B`, `A`, `P`, `M`)
      
          - Requires antibody inputs (`H` and optionally `L`) if provided

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/immunefold-antibody/predict/ HTTP/1.1
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
      
        - **ptm** (*float*, range: 0.0–1.0) — Predicted TM-score (pTM) confidence metric for the predicted structure.
      
        - **full_plddt** (*float*, range: 0.0–100.0) — Average predicted Local Distance Difference Test (pLDDT) score for the entire predicted structure.
      
        - **plddt** (*array of arrays of floats*, shape: [num_chains, num_residues], range: 0.0–100.0) — Per-residue predicted Local Distance Difference Test (pLDDT) scores for each chain in the predicted structure.
      
        - **pdb** (*string*) — Predicted 3D atomic coordinates of the immune protein structure in standard PDB format.

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

- ImmuneFold Antibody API runs inference on NVIDIA T4 GPUs, optimized for efficient and rapid antibody and nanobody structure predictions.

- Typical inference time per antibody or nanobody structure is approximately 3 seconds, significantly faster than AlphaFold2 and AlphaFold-Multimer, which typically require 5-20 minutes per prediction.

- ImmuneFold Antibody achieves superior predictive accuracy compared to general protein structure prediction models such as ESMFold, AlphaFold2, and AlphaFold-Multimer, particularly in the highly variable CDR H3 region:
  
  - CDR H3 RMSD: ImmuneFold Antibody (2.65 Å) vs. AlphaFold2 (3.07 Å), AlphaFold-Multimer (3.07 Å), and ESMFold (3.07 Å).
  
  - Framework RMSD: ImmuneFold Antibody (0.72 Å) vs. AlphaFold2 (0.78 Å), AlphaFold-Multimer (0.78 Å), and ESMFold (0.92 Å).

- Compared to antibody-specific models such as IgFold, BALMFold, and tFold-Ab, ImmuneFold Antibody consistently provides more accurate predictions across all complementarity-determining regions (CDRs) and framework regions:
  
  - CDR H3 RMSD: ImmuneFold Antibody (2.65 Å) vs. tFold-Ab (2.99 Å), IgFold (3.31 Å), BALMFold (3.07 Å), and ImmuneBuilder (3.42 Å).

- ImmuneFold Antibody effectively predicts the relative orientation between heavy and light chains, achieving an Orientational Coordinate Distance (OCD) of 2.66, comparable to the best-performing antibody-specific model tFold-Ab (2.66) and outperforming general models such as AlphaFold-Multimer (3.06) and ESMFold (3.28).

- Predictive confidence scores (pLDDT) generated by ImmuneFold Antibody strongly correlate with actual structural accuracy, demonstrating Pearson correlation coefficients of 0.57 for antibodies and 0.55 for nanobodies, providing reliable estimates of prediction quality.

- ImmuneFold Antibody is fine-tuned using Low-Rank Adaptation (LoRA), a parameter-efficient method that reduces GPU memory usage by nearly 50% compared to full-parameter fine-tuning, enabling faster deployment and scaling without compromising predictive accuracy.

- ImmuneFold Antibody supports optional antigen context input (antigen PDB structure), significantly improving prediction accuracy for bound antibody and nanobody structures:
  
  - Antibody CDR H3 RMSD improves from 2.91 Å (without antigen) to 2.57 Å (with antigen context).
  
  - Nanobody CDR3 RMSD improves from 3.00 Å (without antigen) to 2.46 Å (with antigen context).

- Input Types:
  
  - Antibody: Heavy chain (H) and optional Light chain (L) amino acid sequences; optional antigen structure (PDB format).
  
  - Nanobody: Single-domain antibody sequence provided as Heavy chain (H); optional antigen structure (PDB format).

- Output Types:
  
  - Predicted 3D structure (PDB format), predicted template modeling score (pTM), overall predicted Local Distance Difference Test (full pLDDT), and per-residue pLDDT confidence scores.


Applications
------------

- Rapid antibody structure prediction for therapeutic antibody optimization, enabling biotech companies to efficiently screen and rank antibody variants based on predicted structural stability and antigen-binding affinity; particularly valuable when experimental determination (e.g., crystallography or cryo-EM) is slow, costly, or limited by protein stability issues; however, predictions may be less accurate for antibodies with unusually long or highly disordered CDR loops.
- Nanobody engineering for targeted therapeutics, allowing researchers to quickly predict and optimize nanobody binding interactions with antigens, significantly accelerating the design of highly specific, stable, and manufacturable biologics; especially useful for companies developing therapeutics against challenging targets with limited structural data; less optimal for predicting large conformational changes upon antigen binding.
- Computational antibody-antigen docking and binding affinity estimation, enabling biotech companies to rapidly screen candidate antibodies against known antigen targets, prioritize leads, and reduce the number of costly wet-lab experiments; particularly valuable for early-stage antibody discovery programs with limited experimental data; less suitable for large antigen complexes or cases where the antigen epitope is unknown or structurally ambiguous.
- Antibody humanization and maturation workflows, providing a computational method to predict structural effects of sequence modifications, enabling biotech companies to efficiently design humanized antibodies with reduced immunogenicity and improved manufacturability; especially useful for therapeutic antibodies derived from non-human sources; however, predictions may not fully capture subtle immunogenicity risks or stability issues that require experimental validation.
- High-throughput screening of antibody libraries for antigen specificity, enabling biotech companies to rapidly identify and prioritize antibody candidates with desired binding profiles, significantly reducing experimental screening costs and timelines; particularly valuable for companies with large antibody libraries targeting multiple antigens; less optimal for antibodies requiring precise kinetic or thermodynamic binding characterization, which still requires experimental assays.


Limitations
-----------

- **Maximum Sequence Length**: ImmuneFold supports a maximum sequence length of ``256`` amino acids per chain for antibody (``H``, ``L``) and TCR (``A``, ``B``, ``P``, ``M``) inputs. Single-domain antibodies (VHH) can use the ``H`` chain alone. For antigen structures provided as input (``pdb``), ensure the total PDB string length does not exceed the defined limit.
- **Batch Size**: The maximum number of sequences allowed per API request is ``32``. Larger datasets must be split into multiple requests.
- ImmuneFold is specifically optimized for antibody, nanobody, and T-cell receptor (TCR) structure prediction. It is not suitable for general protein structure prediction or modeling large protein complexes beyond immune protein-antigen interactions.
- ImmuneFold leverages a parameter-efficient fine-tuning of ESMFold, which relies on single-chain protein language models. Therefore, it may not accurately capture inter-chain evolutionary constraints for large antibody-antigen complexes, particularly when the antigen structure is large or the epitope is unknown.
- While ImmuneFold significantly improves prediction accuracy for hypervariable regions (such as CDR H3 in antibodies and CDR3 in TCRs), predictions for extremely long or unusually structured CDR loops may be less accurate due to limited training data.
- ImmuneFold predicts static structures and does not model conformational flexibility or dynamics of immune proteins. For applications requiring conformational sampling or dynamic modeling, alternative methods or additional molecular dynamics simulations should be considered.


How We Use It
-------------

ImmuneFold Antibody enables rapid and accurate antibody and nanobody structure prediction, directly integrating into BioLM's predictive and generative protein engineering workflows. By leveraging low-rank adaptation (LoRA) fine-tuning on ESMFold, this model significantly improves the prediction of challenging hypervariable regions (such as CDR H3), enabling more precise antibody optimization and maturation strategies. ImmuneFold Antibody integrates seamlessly with downstream computational tools such as Rosetta, facilitating virtual screening, affinity maturation, and thermostability optimization projects for antibodies and nanobodies, thus accelerating experimental validation and reducing time-to-market.

- Supports both antigen-bound and unbound antibody structure prediction, enabling flexible design scenarios.
- Efficient inference enables rapid iteration cycles, accelerating antibody engineering pipelines.


Related
-------

- ``ESMFold`` – Serves as the foundational model for ImmuneFold, providing the initial protein structure prediction capabilities that are fine-tuned for immune proteins.
- ``AlphaFold2`` – Known for its high accuracy in protein structure prediction, it complements ImmuneFold by offering a benchmark for evaluating structural prediction accuracy.
- ``TCRBuilder2`` – Focuses on T-cell receptor structure prediction, complementing ImmuneFold's capabilities in predicting TCR structures with specialized models.
- ``IgFold`` – Specializes in antibody structure prediction using language models, offering complementary insights and methods for predicting antibody structures alongside ImmuneFold.


References
----------

- Zhu, T., Ren, M., He, Z., Tao, S., Li, M., Bu, D., & Zhang, H. (2024). Accurate structure prediction of immune proteins using parameter-efficient transfer learning. *Journal Name*.

