Evo 1.5 8k Base API
===================

Evo 1.5 8k Base is a genomic foundation model trained on prokaryotic genomes, enabling predictive modeling and generative design of DNA, RNA, and protein sequences at single-nucleotide resolution. Based on the StripedHyena architecture with GPU-accelerated inference, Evo operates at an 8k context length, performing zero-shot function prediction and generative sequence design for CRISPR-Cas complexes, transposable elements, and regulatory DNA. Evo is suitable for protein engineering, genome annotation, and synthetic biology research.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for Evo 1.5 8k Base.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="evo-15-8k-base",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/evo-15-8k-base/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/evo-15-8k-base/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/evo-15-8k-base/predictor/"

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

.. http:post:: /api/v3/evo-15-8k-base/predict/

   Predict endpoint for Evo 1.5 8k Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters for sequence generation:
      
        - **max_new_tokens** (*int*, range: 1-4096, default: 100) — Maximum number of tokens to generate
        - **temperature** (*float*, range: ≥0.0, default: 0.0) — Sampling temperature
        - **top_k** (*int*, range: ≥1, default: 1) — Top-k sampling parameter
        - **top_p** (*float*, range: 0.0-1.0, default: 1.0) — Top-p (nucleus) sampling parameter
        - **prepend_bos** (*bool*, default: False) — Whether to prepend beginning-of-sequence token
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences for generation:
      
        - **prompt** (*string*, min length: 1, max length: 4096, required) — DNA sequence consisting of unambiguous nucleotide characters (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo-15-8k-base/predict/ HTTP/1.1
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
      
        - **log_prob** (*float*, range: negative infinity to 0.0) — Logarithmic probability (natural log scale) of the provided DNA sequence, computed by Evo model.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "log_prob": -30.015625
    }
  ]
}


Generate
--------

This endpoint gensats for Evo 1.5 8k Base.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.generator(

          model="evo-15-8k-base",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/evo-15-8k-base/generator/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/evo-15-8k-base/generator/"

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
      
      url <- "https://biolm.ai/api/v3/evo-15-8k-base/generator/"

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

.. http:post:: /api/v3/evo-15-8k-base/generate/

   Generate endpoint for Evo 1.5 8k Base.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Generation parameters:
      
        - **max_new_tokens** (*int*, range: 1-4096, default: 100) — Maximum number of tokens to generate
        - **temperature** (*float*, range: ≥0.0, default: 0.0) — Sampling temperature
        - **top_k** (*int*, range: ≥1, default: 1) — Top-k sampling cutoff
        - **top_p** (*float*, range: 0.0-1.0, default: 1.0) — Top-p (nucleus) sampling cutoff
        - **prepend_bos** (*bool*, default: False) — Whether to prepend beginning-of-sequence token
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:
      
        - **prompt** (*string*, min length: 1, max length: 4096, required) — DNA sequence prompt consisting of unambiguous nucleotide characters (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/evo-15-8k-base/generate/ HTTP/1.1
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
      
        - **generated** (*string*, length: 1 to 4096 nucleotides) — Generated DNA sequence at single-nucleotide resolution.
      
        - **score** (*float*) — Log-likelihood score of the generated sequence.
      
      
      - **results** (*array of objects*) --- One result per input item, in the order requested:
      
        - **log_prob** (*float*) — Log probability of the input DNA sequence under the Evo model.

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "generated": "ATCGATACCGGTGCTGCGATCCCGGGCAGCAGGCCGGCAGGCGGTGCCAGCTTGCGGTCGCCGCGGCCCGCCCGCACACCGATCTCGCCGGCGATCAGCCAGGCATCGCGCGATTCGTCG",
      "score": -1.4464020729064941
    }
  ]
}


Performance
-----------

- Evo 1.5 8k Base utilizes GPU-accelerated inference on NVIDIA L4 GPUs, optimized specifically for efficient processing of biological sequences at single-nucleotide resolution.

- Evo 1.5 8k Base employs the StripedHyena architecture, a hybrid model combining data-controlled convolutional ("hyena") layers with rotary self-attention layers. This architecture significantly improves computational efficiency and accuracy compared to traditional Transformer-based DNA language models (e.g., DNABERT-2 and Omni-DNA 1B).
  
  - StripedHyena achieves superior scaling efficiency relative to dense Transformer architectures, with sub-quadratic computational complexity, enabling faster inference speeds and lower memory consumption at long sequence lengths.
  
  - Compared to DNABERT-2, Evo 1.5 8k Base demonstrates lower perplexity scores on DNA sequence prediction tasks, indicating superior predictive accuracy at single-nucleotide resolution.

- Evo 1.5 8k Base significantly outperforms other nucleotide language models (e.g., DNABERT-2, Omni-DNA 1B) in zero-shot functional prediction tasks involving bacterial proteins, non-coding RNAs, and regulatory DNA elements, as measured by Spearman correlation to experimental fitness data.

- Evo 1.5 8k Base achieves predictive accuracy competitive with specialized protein language models (e.g., ESM-2 650M, ESM-1v) on bacterial protein mutational effect prediction, despite being trained on genomic DNA sequences without explicit protein annotations.

- Evo 1.5 8k Base demonstrates superior accuracy compared to RNA-specific language models (e.g., nanoBERT) in predicting mutational effects on non-coding RNA function.

- Evo 1.5 8k Base is optimized for rapid inference; typical generation tasks (e.g., generating sequences of 100-1,000 nucleotides) complete within seconds per batch, significantly faster than comparable nucleotide language models (e.g., DNABERT-2).

- Evo 1.5 8k Base has been extensively optimized by BioLM for efficient GPU utilization, ensuring stable inference performance and minimal latency during high-throughput sequence analysis tasks.


Applications
------------

- Zero-shot prediction of mutational effects on bacterial protein and RNA function, enabling rapid identification of beneficial mutations in industrial enzymes, antibodies, or RNA therapeutics; valuable for biotech companies performing protein or RNA engineering to enhance stability, activity, or specificity, without requiring extensive experimental screening; however, not optimal for eukaryotic proteins or human therapeutic antibodies due to prokaryotic-only training data.
- Generative design of novel CRISPR-Cas protein-RNA complexes, enabling biotech companies to rapidly generate diverse Cas9, Cas12, or Cas13 variants for genome editing applications; valuable for developing CRISPR-based diagnostics, therapeutics, or agricultural gene editing tools, by providing sequences that diverge significantly from known natural variants; however, experimental validation is required to confirm functional activity of generated complexes.
- In silico identification of essential bacterial genes at whole-genome scale, enabling rapid prioritization of antibiotic targets by predicting gene essentiality directly from genomic sequence; valuable for pharmaceutical and biotech companies developing new antibiotics or antimicrobial agents, by reducing the time and cost associated with experimental gene knockout screens; however, predictions are limited to prokaryotic organisms and may not generalize to complex eukaryotic genomes.
- Generative design of synthetic transposable elements (IS200/IS605 systems), enabling biotech companies to engineer novel mobile genetic elements for controlled genomic insertion and gene delivery applications; valuable for developing gene therapy vectors, microbial strain engineering, or synthetic biology chassis with predictable insertion behavior; however, functional validation and optimization are necessary to ensure efficient and safe transposition activity in desired host organisms.
- Generation of genome-scale DNA sequences with realistic coding density and operon-like structure, enabling synthetic biology companies to rapidly prototype large synthetic genomes or metabolic pathways; valuable for designing bacterial strains optimized for biomanufacturing, metabolic engineering, or bioremediation applications, by providing plausible genomic architectures that facilitate downstream experimental refinement; however, generated sequences currently lack critical conserved elements such as complete tRNA repertoires, requiring manual curation and further optimization for viability.


Limitations
-----------

- **Maximum Sequence Length**: Input sequences are limited to ``4096`` nucleotides. Longer sequences must be truncated or split into smaller segments before submission.
- **Batch Size**: The maximum batch size is ``2`` sequences per request. For larger-scale inference, multiple requests must be submitted sequentially.
- Evo 1.5 8k Base was pretrained exclusively on prokaryotic genomes. As a result, it is not suitable for tasks involving eukaryotic sequences, such as human genomic or mammalian protein variant analysis.
- Evo's generative capabilities may produce sequences with plausible genomic organization, but generated sequences often lack key functional elements (e.g., complete tRNA repertoires, rRNA genes). Thus, Evo is not optimal for generating fully functional synthetic genomes without additional downstream refinement.
- While Evo demonstrates strong zero-shot performance for predicting mutational effects on prokaryotic proteins, ncRNAs, and regulatory DNA, it has limited predictive accuracy for highly novel or synthetic sequences that significantly diverge from natural genomic distributions.
- Evo employs a hybrid architecture (StripedHyena) optimized for long-context modeling, but it may not be computationally efficient for very short-sequence tasks (e.g., single-gene or short regulatory element analysis), where smaller, specialized models could be more appropriate.


How We Use It
-------------

Evo 1.5 8k Base enables accelerated protein and nucleic acid sequence optimization workflows by providing scalable, standardized APIs for zero-shot prediction and generative design tasks. Integrated into BioLM's broader protein engineering pipelines, Evo supports enzyme and antibody design projects by rapidly evaluating functional impacts of sequence variants and generating candidate sequences informed by genome-scale evolutionary context. This algorithm seamlessly integrates with predictive models for biophysical property assessment and downstream filtering tools, streamlining iterative cycles of molecule design, ranking, and refinement.

- Facilitates iterative optimization of protein sequences informed by genomic evolutionary patterns.
- Integrates smoothly with downstream prediction workflows for biophysical and structural properties.


Related
-------

- ``Evo 2 1B Base`` – Offers similar genomic-scale modeling and design capabilities as Evo 1.5 8k Base, but scaled to larger model size and context length for more complex tasks.
- ``DNABERT-2`` – Complements Evo by providing transformer-based nucleotide modeling, useful for shorter sequence contexts or specialized genomic tasks requiring different tokenization strategies.
- ``ESMFold`` – Works well alongside Evo by predicting protein structures from sequences, enabling structural validation of Evo-generated protein coding sequences.
- ``AlphaFold2`` – Provides complementary high-accuracy protein structure prediction, useful for validating and refining protein complexes and systems designed by Evo.


References
----------

- Nguyen, E., Poli, M., Durrant, M. G., Thomas, A. W., Kang, B., Sullivan, J., Ng, M. Y., Lewis, A., Patel, A., Lou, A., Ermon, S., Baccus, S. A., Hernandez-Boussard, T., Ré, C., Hsu, P. D., & Hie, B. L. (2024). `Sequence modeling and design from molecular to genome scale with Evo <https://github.com/evo-design/evo>`_. *bioRxiv*.

