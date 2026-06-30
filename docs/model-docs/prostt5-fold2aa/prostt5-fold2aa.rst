ProstT5 Fold2AA API
===================

ProstT5 Fold2AA is a GPU-accelerated bilingual protein language model that translates structural protein representations (Foldseek 3Di tokens) into corresponding amino acid sequences. Utilizing a transformer-based encoder-decoder architecture trained on AlphaFold-derived protein structures, it supports inverse folding tasks through conditional sequence generation. Typical use cases include generating protein sequence variants, structure-guided mutagenesis, and exploration of remote homologs at scale.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Encode
------

This endpoint encodes for ProstT5 Fold2AA.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.encoder(

          model="prostt5-fold2aa",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/prostt5-fold2aa/encoder/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/prostt5-fold2aa/encoder/"

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
      
      url <- "https://biolm.ai/api/v3/prostt5-fold2aa/encoder/"

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

.. http:post:: /api/v3/prostt5-fold2aa/encode/

   Encode endpoint for ProstT5 Fold2AA.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature for sequence generation
      
        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Nucleus sampling cumulative probability threshold
      
        - **top_k** (*int*, range: 1-20, default: 3) — Number of highest probability tokens to consider at each step
      
        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor for repeated tokens
      
        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate per input sequence
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using standard amino acid codes (AA2fold) or 3Di tokens (fold2AA)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-fold2aa/encode/ HTTP/1.1
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
      
        - **mean_representation** (*array of floats*, size: 1024) — Mean embedding vector representing the input protein sequence or structure, derived from the ProstT5 encoder. Each embedding dimension is a float without fixed range constraints.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "error": true,
  "status_code": 500,
  "message": "{\"error\":\"Uncaught exception: missing a required argument: 'direction'\",\"status_code\":500}"
}


Generate
--------

This endpoint gensats for ProstT5 Fold2AA.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.generator(

          model="prostt5-fold2aa",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/prostt5-fold2aa/generator/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/prostt5-fold2aa/generator/"

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
      
      url <- "https://biolm.ai/api/v3/prostt5-fold2aa/generator/"

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

.. http:post:: /api/v3/prostt5-fold2aa/generate/

   Generate endpoint for ProstT5 Fold2AA.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **temperature** (*float*, range: 0.0-8.0, default: 1.2) — Sampling temperature
        - **top_p** (*float*, range: 0.0-1.0, default: 0.95) — Nucleus sampling cumulative probability threshold
        - **top_k** (*int*, range: 1-20, default: 6) — Number of highest probability tokens considered at each step
        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor applied to repeated tokens
        - **num_samples** (*int*, range: 1-3, default: 1) — Number of generated sequences per input
        - **num_beams** (*int*, range: 1-3, default: 3) — Beam search width for decoding
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence using standard amino acid codes
      
      - **params** (*object*, optional) --- Configuration parameters:
      
        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature
        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Nucleus sampling cumulative probability threshold
        - **top_k** (*int*, range: 1-20, default: 3) — Number of highest probability tokens considered at each step
        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor applied to repeated tokens
        - **num_samples** (*int*, range: 1-3, default: 1) — Number of generated sequences per input
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 512, required) — 3Di sequence using characters "acdefghiklmnpqrstvwy" representing protein structure tokens
      
      - **items** (*array of objects*, min: 1, max: 16) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 1000, required) — Protein sequence using standard amino acid codes
      
      - **items** (*array of objects*, min: 1, max: 16) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 1000, required) — 3Di sequence using characters "acdefghiklmnpqrstvwy" representing protein structure tokens

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-fold2aa/generate/ HTTP/1.1
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
      
        - **sequence** (*string*) — Generated sequence (AA or 3Di tokens); length: 1–512 tokens (AA or 3Di alphabet)

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "error": true,
  "status_code": 500,
  "message": "{\"error\":\"Uncaught exception: missing a required argument: 'direction'\",\"status_code\":500}"
}


Performance
-----------

- **Hardware Specifications:** Inference runs on NVIDIA L4 GPUs, leveraging half-precision (fp16) for optimized performance.

- **Batch Size and Sequence Length Constraints:** ProstT5 Fold2AA supports a batch size of up to 2 sequences per request, with each sequence limited to a maximum length of 512 tokens.

- **Prediction Accuracy:** ProstT5 Fold2AA achieves high-quality inverse folding (structure-to-sequence translation) performance, producing novel amino acid sequences that adopt the desired structural templates. When benchmarked against ProteinMPNN, a state-of-the-art inverse folding model, ProstT5 Fold2AA-generated sequences achieve similar structural fidelity (average lDDT of 72 vs. ProteinMPNN's 77) but with greater sequence diversity (average pairwise identity ~22% vs. ~30% for ProteinMPNN).

- **Structural Fidelity Metrics:** Generated sequences evaluated by structure prediction (ESMFold) achieve:
  
  - Average lDDT: 72 (compared to ProteinMPNN's 77, and native/ESMFold upper bound of 78)
  
  - Average RMSD: 2.90 Å (ProteinMPNN: 2.61 Å)
  
  - Average TM-score: 0.58 (ProteinMPNN: 0.61)

- **Sequence Diversity:** ProstT5 Fold2AA generates sequences with lower sequence identity to native sequences (~22% identity), indicating higher sequence diversity compared to ProteinMPNN (~30% identity).

- **Optimization and Scaling:** ProstT5 Fold2AA employs optimized inference settings, including beam search (num_beams=3), nucleus sampling (top_p=0.85), and top-k sampling (top_k=3), to balance structural accuracy and sequence diversity.

- **Inference Speed:** ProstT5 Fold2AA inference involves autoregressive decoding, resulting in typical completion times of approximately 0.6–2.5 seconds per protein sequence (depending on sequence length), making it significantly faster than structure-based inverse folding methods that require explicit 3D structure prediction (e.g., AlphaFold2).

- **Comparison to Related Models:** ProstT5 Fold2AA is faster than AlphaFold2-based inverse folding workflows, as it directly translates structural tokens (3Di) into amino acid sequences without explicit 3D structure prediction. However, it is slower than encoder-only embedding extraction models (such as ProtT5 or ESM-2), due to its autoregressive decoding step.

- **Input and Output Types:**
  
  - **Input:** One-dimensional strings representing protein structures encoded as 3Di tokens (20-letter alphabet: "acdefghiklmnpqrstvwy").
  
  - **Output:** Amino acid sequences (standard 20-letter alphabet) predicted to adopt the provided structural template.


Applications
------------

- Rapid screening of metagenomic sequence databases for remote homolog detection by converting amino acid sequences directly into 3Di structural tokens, enabling structure-level sensitivity at sequence-level speed; valuable for companies mining large-scale metagenomic datasets to discover novel proteins or functional homologs, significantly reducing computational time compared to traditional structure prediction methods; not optimal for precise atomic-level modeling or detailed structural refinement tasks.
- High-throughput inverse folding for protein design by generating novel amino acid sequences conditioned on a desired 3Di structural representation; enables biotech companies to rapidly explore diverse sequence space for protein scaffolds with specific structural motifs, useful in developing novel protein therapeutics or industrial enzymes; limited by the accuracy of the underlying 3Di representation and may require additional structural validation.
- Structural embedding extraction for protein fold classification and annotation transfer, leveraging ProstT5 embeddings to classify proteins into structural superfamilies (e.g., CATH, SCOPe); valuable for bioinformatics pipelines in biotech companies performing large-scale protein annotation or functional inference, particularly when sequence similarity alone is insufficient; less suitable for functional annotations unrelated to structural similarity.
- Generation of structurally diverse protein sequence variants for stability and solubility optimization by translating between amino acid sequences and corresponding 3Di structural tokens; beneficial for protein engineering workflows aimed at improving biophysical properties of therapeutic proteins or industrial enzymes, enabling targeted exploration of structurally constrained sequence diversity; not optimal for fine-tuning protein function or catalytic activity without additional experimental validation.
- Accelerated structural clustering and redundancy reduction of protein libraries by encoding protein structures into compact 3Di token sequences, facilitating rapid clustering and filtering of large protein datasets; useful in biotech companies managing extensive protein libraries for directed evolution or combinatorial protein engineering, significantly reducing computational overhead compared to full 3D structural comparisons; not ideal for detailed structural analyses or precise structural alignment tasks.


Limitations
-----------

- **Maximum Sequence Length**: ProstT5 accepts sequences up to ``max_sequence_len = 1000`` residues for encoding tasks and up to ``max_sequence_len = 512`` residues for generation tasks. Longer sequences must be truncated or split.
- **Batch Size**: The maximum supported ``batch_size`` is ``16`` for encoding and ``2`` for generation requests. Larger datasets must be processed in multiple batches.
- ProstT5 is optimized for structured proteins; it may perform poorly on intrinsically disordered proteins or highly repetitive sequences due to biases introduced during training data filtering.
- ProstT5-generated 3Di sequences are highly effective for rapid fold-based remote homology detection but may not reach the sensitivity of full 3D structure comparison methods (e.g., AlphaFold2 predictions) for very subtle structural differences.
- ProstT5 embeddings primarily capture structural information; tasks heavily dependent on functional annotations, subcellular localization signals, or ligand-binding predictions may see reduced performance compared to general-purpose protein language models.
- The inverse folding capability (generating amino acid sequences from 3Di structures) can produce diverse sequences, but specialized graph-based methods like ProteinMPNN may yield higher-quality sequences for precise protein design tasks.


How We Use It
-------------

The ProstT5 Fold2AA model enables rapid translation of protein structural information (3Di sequences) into amino acid sequences, accelerating workflows in protein engineering, antibody optimization, and enzyme design. By integrating ProstT5 Fold2AA into protein design pipelines, researchers can efficiently generate diverse protein sequence variants predicted to adopt specified structural features, significantly reducing experimental screening cycles and resource allocation. ProstT5 Fold2AA integrates smoothly with downstream structural modeling tools such as ESMFold and Foldseek, providing a complete, scalable solution for structure-guided protein optimization.

- Accelerates structure-guided protein variant generation, reducing experimental cycles.
- Integrates directly with downstream BioLM workflows for structural modeling and protein optimization.


Related
-------

- ``ProstT5 AA2Fold`` – Complementary to ProstT5 Fold2AA, translates amino acid sequences into 3Di structural representations, enabling bidirectional structure-sequence modeling.
- ``Foldseek`` – Uses 3Di sequences for rapid structural similarity searches; integrates seamlessly with ProstT5 Fold2AA-generated sequences for enhanced remote homology detection.
- ``ESMFold`` – Predicts atomic-level 3D structures from amino acid sequences; complements ProstT5 Fold2AA by providing detailed structural models for generated sequences.
- ``AlphaFold2`` – Provides highly accurate protein structure predictions used to train ProstT5 Fold2AA, serving as the foundational data resource for its structure-to-sequence translation capabilities.


References
----------

- Heinzinger, M., Weissenow, K., Gomez Sanchez, J., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). `Bilingual language model for protein sequence and structure <https://doi.org/10.1093/nargab/lqae150>`_. *NAR Genomics and Bioinformatics*.

