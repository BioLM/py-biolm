DNA Chisel API
==============

DNA Chisel is a Python-based DNA sequence optimization algorithm designed to solve multi-objective sequence design challenges in synthetic biology. It enables users to define constraints and objectives via Python scripts or annotated Genbank files, supporting over 15 built-in specification classes including codon optimization, GC content control, and removal of unwanted motifs. DNA Chisel provides detailed optimization reports and supports user-defined custom specifications, facilitating efficient DNA sequence engineering workflows.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for DNA Chisel.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="dna-chisel",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/dna-chisel/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/dna-chisel/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/dna-chisel/predictor/"

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

.. http:post:: /api/v3/dna-chisel/predict/

   Predict endpoint for DNA Chisel.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **include** (*array of strings*, default: ["gc_content", "cai", "hairpin_score", "melting_temperature", "restriction_site_count", "codon_usage_entropy", "rare_codon_frequency", "homopolymer_run_length", "dinucleotide_frequencies", "sequence_length", "tata_box_count", "non_unique_6mer_count", "in_frame_stop_codon_count", "methionine_frequency", "at_skew", "gc_skew", "nucleotide_entropy", "tandem_repeat_count", "gc_content_std_dev", "kozak_sequence_strength"]) — Features to include in response
      
        - **species** (*string*, default: "e_coli") — Species for codon-related features; options: "e_coli", "s_cerevisiae", "h_sapiens", "c_elegans", "b_subtilis", "d_melanogaster"
      
        - **restriction_enzymes** (*array of strings*, default: ["EcoRI", "BsaI"], optional) — Restriction enzymes for site-count feature; set empty or null to disable; valid enzyme names from Biopython restriction enzyme database
      
      
      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, required) — DNA sequence containing only unambiguous nucleotide characters (A, T, C, G)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dna-chisel/predict/ HTTP/1.1
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
      
        - **gc_content** (*float*, range: 0.0–1.0) — Fractional GC content of the sequence
      
        - **cai** (*float*, range: 0.0–1.0) — Codon Adaptation Index (CAI) relative to selected species
      
        - **hairpin_score** (*float*, ≥ 0.0) — Predicted hairpin formation score
      
        - **melting_temperature** (*float*, °C) — Melting temperature of the DNA sequence
      
        - **restriction_site_count** (*object*) — Counts of specified restriction enzyme sites:
        
          - **{enzyme_name}** (*int*, ≥ 0) — Number of occurrences of enzyme recognition site
      
        - **codon_usage_entropy** (*float*, ≥ 0.0) — Shannon entropy of codon usage distribution
      
        - **rare_codon_frequency** (*float*, range: 0.0–1.0) — Frequency of rare codons relative to selected species
      
        - **homopolymer_run_length** (*int*, ≥ 1) — Length of the longest homopolymer run
      
        - **dinucleotide_frequencies** (*object*) — Frequencies of all dinucleotide combinations (AA, AC, ..., TT):
        
          - **{dinucleotide}** (*float*, range: 0.0–1.0) — Fractional frequency of dinucleotide
      
        - **sequence_length** (*int*, ≥ 1) — Total length of the DNA sequence in nucleotides
      
        - **tata_box_count** (*int*, ≥ 0) — Number of TATA box motifs identified
      
        - **non_unique_6mer_count** (*int*, ≥ 0) — Count of non-unique 6-mer sequences
      
        - **in_frame_stop_codon_count** (*int*, ≥ 0) — Number of in-frame stop codons detected
      
        - **methionine_frequency** (*float*, range: 0.0–1.0) — Frequency of methionine codons (ATG) in sequence
      
        - **at_skew** (*float*, range: -1.0–1.0) — AT skew calculated as (A–T)/(A+T)
      
        - **gc_skew** (*float*, range: -1.0–1.0) — GC skew calculated as (G–C)/(G+C)
      
        - **nucleotide_entropy** (*float*, ≥ 0.0) — Shannon entropy of nucleotide distribution
      
        - **tandem_repeat_count** (*int*, ≥ 0) — Number of tandem repeat sequences identified
      
        - **gc_content_std_dev** (*float*, ≥ 0.0) — Standard deviation of GC content across sliding windows
      
        - **kozak_sequence_strength** (*float*, range: 0.0–1.0) — Strength score of Kozak consensus sequence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "gc_content": 0.5,
      "cai": 1.0,
      "hairpin_score": 0.0,
      "melting_temperature": 14.254361957460812,
      "restriction_site_count": {
        "EcoRI": 0,
        "BsaI": 0
      },
      "codon_usage_entropy": 1.0,
      "rare_codon_frequency": 0.0,
      "homopolymer_run_length": 1,
      "dinucleotide_frequencies": {
        "AA": 0.0,
        "AC": 0.0,
        "AG": 0.0,
        "AT": 0.2857142857142857,
        "CA": 0.14285714285714285,
        "CC": 0.0,
        "CG": 0.0,
        "CT": 0.0,
        "GA": 0.0,
        "GC": 0.2857142857142857,
        "GG": 0.0,
        "GT": 0.0,
        "TA": 0.0,
        "TC": 0.0,
        "TG": 0.2857142857142857,
        "TT": 0.0
      },
      "sequence_length": 8,
      "tata_box_count": 0,
      "non_unique_6mer_count": 0,
      "at_skew": 0.0,
      "gc_skew": 0.0,
      "nucleotide_entropy": 2.0,
      "tandem_repeat_count": 0,
      "gc_content_std_dev": 0.0,
      "kozak_sequence_strength": 0.0
    }
  ]
}


Performance
-----------

- DNA Chisel runs exclusively on CPU resources (0.25 CPU cores, 1 GB RAM); GPU acceleration is not utilized or required.

- Processes sequences individually (batch size = 1); each request handles exactly one input sequence.

- Typical completion time is on the order of seconds per sequence, depending on sequence length and complexity of optimization constraints.

- DNA Chisel employs efficient local sequence optimization heuristics, significantly outperforming more general-purpose genetic algorithm frameworks in speed and convergence for DNA sequence optimization tasks.

- Compared to traditional sequence optimization tools (e.g., D-Tailor), DNA Chisel achieves faster constraint resolution and objective maximization due to its two-step optimization approach (constraint resolution followed by objective maximization).

- The algorithm uses local problem simplification and targeted stochastic or exhaustive search methods, enabling rapid resolution of complex multi-objective optimization scenarios.

- User-defined custom specifications can further enhance performance by implementing specialized local resolution methods, reducing the computational overhead compared to generic optimization approaches.

- Performance remains consistent across supported species (e.g., E. coli, S. cerevisiae, H. sapiens), as codon usage and restriction site data are preloaded from optimized internal databases.

- DNA Chisel's optimization speed and scalability make it particularly suitable for iterative design workflows requiring rapid turnaround of optimized DNA sequences.


Applications
------------

- Codon optimization for heterologous protein expression in industrial host strains, enabling improved protein yield and solubility by adjusting DNA sequences to match host-specific codon usage; useful for biopharma and industrial enzyme manufacturers producing recombinant proteins in E. coli, yeast, or CHO cells; not optimal for addressing post-translational modifications or protein folding issues.
- Removal of problematic restriction enzyme sites and repetitive DNA motifs from synthetic gene constructs, preventing synthesis failures or cloning issues caused by unwanted restriction digestion or recombination; valuable for gene synthesis providers and synthetic biology companies designing large or complex DNA constructs; not intended for genome-scale editing or CRISPR guide RNA design.
- Multi-constraint DNA sequence optimization for gene therapy vector design, balancing GC content, CpG dinucleotide frequency, and regulatory sequence compatibility to enhance vector stability, manufacturability, and transgene expression; beneficial for biotech companies developing viral vectors (AAV, lentivirus) for therapeutic applications; less suitable for optimizing RNA-based therapeutics or non-viral delivery systems.
- Generation of biologically neutral DNA sequences for genetic circuit components, minimizing unintended host interactions or metabolic burden by removing cryptic promoters, terminators, or regulatory elements; essential for synthetic biology startups and metabolic engineering companies constructing predictable genetic circuits; not designed for functional RNA sequence optimization or RNA secondary structure prediction.
- Programmatic optimization of large-scale DNA libraries for protein variant screening, systematically encoding combinatorial diversity while maintaining synthesis feasibility and experimental constraints; critical for protein engineering and enzyme evolution companies performing high-throughput screening campaigns; not suitable for antibody affinity maturation workflows or structure-guided antibody design.


Limitations
-----------

- **Batch Size**: The API processes exactly one sequence per request (``batch_size = 1``); batch processing of multiple sequences is not supported.
- **Sequence Length**: Input sequences must be at least 1 nucleotide in length; extremely long sequences may significantly increase processing time or fail to optimize due to computational complexity.
- DNA Chisel resolves constraints first and then objectives; conflicting or overly strict constraints can result in optimization failures or suboptimal sequences.
- The algorithm uses local heuristics and stochastic searches; thus, it does not guarantee globally optimal solutions, especially for complex multi-objective scenarios.
- DNA Chisel is not suitable for tasks requiring protein-level structural predictions or embedding-based clustering; it focuses solely on nucleotide-level sequence optimization.
- While DNA Chisel can optimize codon usage and remove undesired sequence patterns, it does not model biological functions such as protein folding, activity, or stability; consider specialized predictive models if these properties are critical.


How We Use It
-------------

DNA Chisel enables BioLM to rapidly optimize DNA sequences as part of integrated protein engineering workflows, ensuring sequences meet practical biological requirements and manufacturability criteria. By standardizing constraints such as codon optimization, GC content balancing, and removal of synthesis-blocking motifs, BioLM integrates DNA Chisel outputs seamlessly with downstream predictive modeling, embedding generation, and ML-driven design pipelines. This integration accelerates research cycles, reduces synthesis failure rates, and improves experimental outcomes.

- Integration with predictive sequence models and structure-based property prediction algorithms.
- Accelerates multi-round protein optimization by providing standardized, synthesis-ready sequences.


Related
-------

- ``DNABERT-2`` – Complements DNA Chisel by providing robust DNA sequence embeddings useful for identifying biologically significant motifs prior to optimization.
- ``Omni-DNA 1B`` – Enhances DNA Chisel workflows by predicting functional DNA elements, aiding targeted sequence optimization.
- ``Evo 2 1B Base`` – Works alongside DNA Chisel to generate optimized sequences informed by evolutionary constraints and biological fitness criteria.
- ``ESM-2 650M`` – Provides protein-level insights and constraints, useful when DNA Chisel is optimizing coding sequences for protein expression.


References
----------

- Zulkower, V., & Rosser, S. (2020). `DNA Chisel, a versatile sequence optimizer <https://doi.org/10.1093/bioinformatics/btaa558>`_. *Bioinformatics*.

