ProstT5 AA2Fold API
===================

ProstT5 AA2Fold is a GPU-accelerated encoder-decoder model that translates amino acid (AA) sequences into 3Di structure tokens, enabling rapid protein structure inference. Derived by fine-tuning ProtT5 on 17M AlphaFold2 predictions encoded via Foldseek's 3Di alphabet, ProstT5 AA2Fold delivers near structure-level sensitivity for remote homology detection at ~3600-fold faster runtimes than conventional structure prediction, suitable for proteome-scale annotation, structural classification, and high-throughput dataset analyses.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Encode
------

This endpoint encodes for ProstT5 AA2Fold.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.encoder(

          model="prostt5-aa2fold",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/prostt5-aa2fold/encoder/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/prostt5-aa2fold/encoder/"

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
      
      url <- "https://biolm.ai/api/v3/prostt5-aa2fold/encoder/"

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

.. http:post:: /api/v3/prostt5-aa2fold/encode/

   Encode endpoint for ProstT5 AA2Fold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature for sequence generation
      
        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Nucleus sampling cumulative probability threshold
      
        - **top_k** (*int*, range: 1-20, default: 3) — Number of highest probability tokens to consider for sampling
      
        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty factor applied to repeated tokens
      
        - **num_samples** (*int*, range: 1-3, default: 1) — Number of generated sequences per input item
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:
      
        - **sequence** (*string*, required, length: 1-512) — Protein sequence using standard amino acid codes (20 standard amino acids plus extended codes B, J, O, U, X, Z)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-aa2fold/encode/ HTTP/1.1
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
      
        - **mean_representation** (*array of floats*, size: 1024) — Mean embedding vector representing the input sequence; each float typically ranges from approximately -10.0 to +10.0

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

This endpoint gensats for ProstT5 AA2Fold.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.generator(

          model="prostt5-aa2fold",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/prostt5-aa2fold/generator/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/prostt5-aa2fold/generator/"

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
      
      url <- "https://biolm.ai/api/v3/prostt5-aa2fold/generator/"

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

.. http:post:: /api/v3/prostt5-aa2fold/generate/

   Generate endpoint for ProstT5 AA2Fold.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **temperature** (*float*, range: 0.0-8.0, default: 1.2) — Sampling temperature for sequence generation
        - **top_p** (*float*, range: 0.0-1.0, default: 0.95) — Cumulative probability threshold for token selection
        - **top_k** (*int*, range: 1-20, default: 6) — Number of top tokens considered for next prediction
        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty for repeated tokens in the output
        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate
        - **num_beams** (*int*, range: 1-3, default: 3) — Number of beams for beam search in sequence generation
      
      - **items** (*array of objects*, max: 2) --- Input sequences for generation:
      
        - **sequence** (*string*, max length: 512, required) — Protein sequence using standard amino acid codes for AA2Fold generation
      
      - **params** (*object*, optional) --- Configuration parameters:
      
        - **temperature** (*float*, range: 0.0-8.0, default: 1.0) — Sampling temperature for sequence generation
        - **top_p** (*float*, range: 0.0-1.0, default: 0.85) — Cumulative probability threshold for token selection
        - **top_k** (*int*, range: 1-20, default: 3) — Number of top tokens considered for next prediction
        - **repetition_penalty** (*float*, range: 0.0-3.0, default: 1.2) — Penalty for repeated tokens in the output
        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate
      
      - **items** (*array of objects*, max: 2) --- Input sequences for generation:
      
        - **sequence** (*string*, max length: 512, required) — 3Di sequence using designated characters for fold2AA generation

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/prostt5-aa2fold/generate/ HTTP/1.1
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
      
        - **mean_representation** (*array of floats*) — Mean representation of the protein sequence, with dimensions corresponding to the embedding size
      
        - **token** (*int*) — Token identifier for the sequence element
      
        - **token_str** (*string*) — String representation of the token
      
        - **score** (*float*) — Score associated with the token, indicating its relevance or confidence
      
        - **sequence** (*string*) — The sequence of amino acids or 3Di tokens as processed by the model

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

- ProstT5 AA2Fold accepts a maximum batch size of 2 sequences, with each sequence limited to 512 amino acids.

- ProstT5 AA2Fold inference runs on Nvidia L4 GPUs, leveraging half-precision (fp16) computation for optimized throughput.

- Typical inference completion is approximately 0.6 to 2.5 seconds per protein sequence, depending on sequence length (shorter sequences closer to 0.6 seconds, longer sequences approaching 2.5 seconds).

- ProstT5 AA2Fold achieves a three orders-of-magnitude speedup for structural remote homology detection compared to traditional methods that rely on full 3D structure prediction (e.g., AlphaFold2, ESMFold).

  - ProstT5 AA2Fold combined with Foldseek reaches near-experimental sensitivity for remote homology detection (ROC-AUC of 0.45 at the superfamily level), significantly outperforming traditional sequence alignment methods such as MMseqs2 (ROC-AUC of 0.06).

- ProstT5 AA2Fold predictive accuracy for structural classification tasks (e.g., CATH fold classification) consistently surpasses ProtT5 and ESM-1b embeddings.

  - For CATH superfamily classification, ProstT5 AA2Fold embeddings achieve an accuracy of 73%, compared to ProtT5's 64% and ESM-1b's 57%.

- ProstT5 AA2Fold inverse folding capability generates novel amino acid sequences that adopt desired structures with an average structural similarity (lDDT) of 72, slightly below ProteinMPNN (77), but with sequences exhibiting greater diversity (average pairwise sequence identity of 21.9% compared to ProteinMPNN's 29.6%).

- ProstT5 AA2Fold inference is optimized by replacing autoregressive decoding with a two-layer CNN classifier on encoder embeddings, maintaining high remote homology detection sensitivity while significantly reducing inference latency.


Applications
------------

- Rapid structural annotation of novel protein sequences by predicting 3Di structure tokens directly from amino acid sequences, enabling scalable remote homology detection without computationally expensive 3D modeling; valuable for biotech companies screening metagenomic datasets for novel protein scaffolds or enzyme candidates, although less suitable for detailed atomic-level analyses.
- Accelerated inverse folding tasks by generating diverse amino acid sequences conditioned on a target 3Di structural representation, enabling protein engineers to efficiently explore sequence space for optimized stability or novel functionality; useful in protein therapeutic design and scaffold engineering, though not optimal for high-precision sequence recovery tasks requiring exact residue-level constraints.
- High-throughput fold classification of large protein libraries by leveraging ProstT5 embeddings for unsupervised structural similarity detection, allowing rapid prioritization of protein variants or engineered constructs based on structural novelty or functional potential; beneficial in directed evolution workflows or protein library screening, but limited in cases requiring explicit functional annotation or biochemical activity prediction.
- Structure-guided mutation effect prediction by comparing ProstT5-generated 3Di sequences for wild-type and mutant proteins, enabling rapid identification of structurally disruptive mutations; useful for protein stability engineering or affinity maturation campaigns, although predictions may not capture subtle functional or dynamic changes.
- Embedding-based clustering and filtering of protein variants in combinatorial libraries by using ProstT5 embeddings to identify structurally redundant sequences, significantly reducing the experimental screening burden; valuable for protein engineering companies performing high-throughput screening, though less effective for distinguishing variants with subtle functional differences unrelated to structure.


Limitations
-----------

- **Maximum Sequence Length**: Input sequences must not exceed ``1000`` residues for encoding (``encode``) and ``512`` residues for generation (``generate``). Longer sequences must be truncated or split into smaller segments.
- **Batch Size**: The maximum number of sequences per request is ``16`` for encoding and ``2`` for generation. Larger datasets must be processed in multiple API calls.
- **GPU Type**: ProstT5 inference runs on ``L4`` GPUs, which may limit throughput for very large-scale inference tasks or extremely low-latency requirements.
- ProstT5 is optimized for structure-related tasks (e.g., remote homology detection, inverse folding). It is not optimal for purely functional predictions (e.g., subcellular localization, binding residues), where other embedding-based models (e.g., ProtT5, ESM-2, Ankh) may perform better.
- The underlying 3Di representation inherently biases predictions toward structured, well-folded proteins. ProstT5 may perform poorly on intrinsically disordered proteins or proteins with highly repetitive or low-complexity regions.
- ProstT5-generated 3Di sequences are highly useful for rapid structural similarity searches, but they do not provide explicit atomic coordinates. For applications requiring precise 3D structural details (e.g., detailed molecular docking or high-resolution structural modeling), slower but more accurate methods (e.g., AlphaFold2, ESMFold) should be used instead.


How We Use It
-------------

BioLM integrates ProstT5 AA2Fold into protein engineering workflows to accelerate structural annotation, remote homology detection, and protein optimization tasks. By translating amino acid sequences directly into 3Di structural representations, ProstT5 AA2Fold enables rapid, large-scale structural similarity screening and filtering of candidate sequences without the computational overhead of full 3D structure prediction. Integrated with downstream filtering models such as Foldseek, this algorithm supports iterative design cycles, improving the efficiency of antibody maturation, enzyme optimization, and targeted protein design.

- Streamlines candidate selection and optimization cycles through rapid structural annotation.
- Enables large-scale structural similarity searches for accelerated protein engineering and discovery.


Related
-------

- ``ProstT5 Fold2AA`` – Complementary to ``ProstT5 AA2Fold``, translates structure (3Di) sequences back into amino acid sequences, enabling inverse folding workflows.
- ``Foldseek`` – Directly complementary, uses 3Di sequences generated by ``ProstT5 AA2Fold`` to rapidly search for structurally related proteins at sequence-search speed.
- ``ESMFold`` – Complementary structure prediction algorithm; can validate 3Di sequences predicted by ``ProstT5 AA2Fold`` by generating full 3D coordinates.
- ``AlphaFold2`` – Provides accurate 3D structure predictions used to train ``ProstT5 AA2Fold``, essential for generating high-quality datasets and benchmarking results.


References
----------

- Heinzinger, M., Weissenow, K., Gomez Sanchez, J., Henkel, A., Mirdita, M., Steinegger, M., & Rost, B. (2024). `Bilingual language model for protein sequence and structure <https://doi.org/10.1093/nargab/lqae150>`_. *NAR Genomics and Bioinformatics*.

