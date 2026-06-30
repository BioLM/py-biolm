DNA Chisel API
==============

DNA Chisel is a Python-based DNA sequence analysis and optimization framework for multi-objective synthetic biology design problems. In this API, it is used as a fast, CPU-only predictor of sequence-level design features, including GC content and GC-content variance, codon adaptation index (CAI), codon usage entropy, rare-codon and methionine frequencies, melting temperature, hairpin score, homopolymer runs, dinucleotide frequencies, motif counts (restriction sites, TATA boxes, tandem repeats, non-unique 6-mers, in-frame stops), nucleotide skew/entropy, and Kozak sequence strength for host-specific expression and manufacturability assessment.

Predict
-------

Predict DNA design features for a single sequence

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="dna-chisel",
                action="predict",
                params={},
                items=[
                  {
                    "sequence": "ATGCATGC"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dna-chisel/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {},
              "items": [
                {
                  "sequence": "ATGCATGC"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dna-chisel/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {},
                  "items": [
                    {
                      "sequence": "ATGCATGC"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dna-chisel/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(),
              items = list(
                list(
                  sequence = "ATGCATGC"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dna-chisel/predict/

   Predict endpoint for DNA Chisel.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*array of strings*, default: ["gc_content", "cai", "hairpin_score", "melting_temperature", "restriction_site_count", "codon_usage_entropy", "rare_codon_frequency", "homopolymer_run_length", "dinucleotide_frequencies", "sequence_length", "tata_box_count", "non_unique_6mer_count", "in_frame_stop_codon_count", "methionine_frequency", "at_skew", "gc_skew", "nucleotide_entropy", "tandem_repeat_count", "gc_content_std_dev", "kozak_sequence_strength"]) — Feature keys to include in the response; each value must be one of the supported feature option strings

        - **species** (*string*, default: "e_coli") — Species identifier for codon-related features; one of: "e_coli", "s_cerevisiae", "h_sapiens", "c_elegans", "b_subtilis", "d_melanogaster"

        - **restriction_enzymes** (*array of strings*, default: ["EcoRI", "BsaI"], optional) — Restriction enzyme names for site-count features; set to an empty array or null to disable; each value must be a valid enzyme name from the Biopython restriction enzyme database


      - **items** (*array of objects*, min: 1, max: 1) --- Input sequences:

        - **sequence** (*string*, min length: 1, required) — DNA sequence containing only unambiguous nucleotides (A, C, G, T)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dna-chisel/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {},
        "items": [
          {
            "sequence": "ATGCATGC"
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

        - **gc_content** (*float*, optional, range: 0.0–1.0) — Fractional GC content of the sequence

        - **cai** (*float*, optional, range: 0.0–1.0) — Codon Adaptation Index relative to the selected species

        - **hairpin_score** (*float*, optional, ≥ 0.0) — Predicted hairpin formation score

        - **melting_temperature** (*float*, optional, units: °C) — Melting temperature of the sequence

        - **restriction_site_count** (*object*, optional) — Counts of restriction enzyme recognition sites:

          - **{enzyme_name}** (*int*, ≥ 0) — Count for a single restriction enzyme

        - **codon_usage_entropy** (*float*, optional, ≥ 0.0) — Shannon entropy of codon usage distribution

        - **rare_codon_frequency** (*float*, optional, range: 0.0–1.0) — Fractional frequency of rare codons relative to the selected species

        - **homopolymer_run_length** (*int*, optional, ≥ 1) — Length of the longest homopolymer run

        - **dinucleotide_frequencies** (*object*, optional) — Frequencies of all 16 dinucleotide combinations:

          - **{dinucleotide}** (*float*, range: 0.0–1.0) — Fractional frequency of a single dinucleotide

        - **sequence_length** (*int*, optional, ≥ 1) — Sequence length in nucleotides

        - **tata_box_count** (*int*, optional, ≥ 0) — Count of TATA box motifs

        - **non_unique_6mer_count** (*int*, optional, ≥ 0) — Count of 6-mer subsequences occurring more than once

        - **in_frame_stop_codon_count** (*int*, optional, ≥ 0) — Count of in-frame stop codons

        - **methionine_frequency** (*float*, optional, range: 0.0–1.0) — Fractional frequency of methionine codons (ATG)

        - **at_skew** (*float*, optional, range: -1.0–1.0) — AT skew (A−T)/(A+T)

        - **gc_skew** (*float*, optional, range: -1.0–1.0) — GC skew (G−C)/(G+C)

        - **nucleotide_entropy** (*float*, optional, ≥ 0.0) — Shannon entropy of nucleotide distribution

        - **tandem_repeat_count** (*int*, optional, ≥ 0) — Count of tandem repeat sequences

        - **gc_content_std_dev** (*float*, optional, ≥ 0.0) — Standard deviation of GC content across internal sliding windows

        - **kozak_sequence_strength** (*float*, optional, range: 0.0–1.0) — Strength score of the Kozak consensus sequence

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

- Runs on CPU-only resources (0.25 vCPU, 1 GB RAM); no GPU is used or required because all features are computed with lightweight, local sequence operations.
- Uses DNA Chisel’s constraint-first, objective-second local optimization heuristics, which converge faster on comparable DNA design problems than generic genetic algorithms and earlier multi-objective tools such as D-Tailor.
- For codon- and restriction-related features, performance is largely species-independent (e.g., *E. coli*, *S. cerevisiae*, *H. sapiens*), as codon usage tables and restriction enzyme definitions are cached in memory instead of recomputed per request.
- Within BioLM’s DNA tooling, DNA Chisel is optimized for rapid local DNA feature evaluation and constraint handling, making it substantially faster and cheaper for DNA design analytics than heavier protein-structure or genome-scale models that require GPU acceleration.

Applications
------------

- Codon optimization assessment for heterologous protein expression using CAI, rare-codon frequency, codon-usage entropy and GC content features to quantify how well a coding sequence matches host-specific usage in supported species such as *E. coli*, yeast or human; useful for biopharma and industrial protein production teams when triaging candidate designs prior to wet-lab testing; not a full de novo sequence optimizer or predictor of expression level or folding.
- Detection of problematic motifs in synthetic constructs via restriction site counts, homopolymer run length, tandem repeat count and non-unique 6-mer metrics, helping gene synthesis providers and synthetic biology companies identify sequences likely to cause synthesis failures or undesired recombination; not intended for designing cloning strategies or genome-scale editing plans.
- Evaluation of DNA manufacturability and basic stability constraints for viral vectors and plasmids by combining GC content, GC/AT skew, GC content standard deviation and melting temperature features, enabling gene therapy and vector engineering groups to flag designs that may be hard to synthesize or amplify; less suitable for modeling in vivo vector performance or immunogenicity.
- Screening of genetic circuit components and “neutral” sequence spacers using hairpin score, TATA box count, Kozak sequence strength and in-frame stop codon count to reduce unintended transcriptional starts, translation initiation or premature stops; valuable for metabolic engineering and synthetic biology teams designing predictable multi-gene constructs; not designed for detailed RNA secondary structure prediction.
- Programmatic quality control of large DNA libraries and variant panels by computing sequence length, dinucleotide frequencies, nucleotide entropy and methionine frequency across many constructs, ensuring diversity while staying within experimental constraints for high-throughput screening; not a replacement for specialized antibody or protein structure-based design tools.

Limitations
-----------

- **Batch Size**: The predictor endpoint processes exactly one sequence per request (``DnaChiselParams.batch_size = 1``); sending multiple ``items`` in a single ``DnaChiselPredictRequest`` is not supported.
- **Minimum Sequence Length**: Each ``items[].sequence`` must contain at least 1 unambiguous DNA nucleotide (A, C, G, T); ambiguous bases are rejected by ``validate_dna_unambiguous``. Very long sequences are accepted but will increase runtime and may be impractical to analyze at scale.
- **Feature Scope Only**: This API reports predefined sequence descriptors in ``params.include`` (for example ``gc_content``, ``cai``, ``melting_temperature``) and never alters ``items[].sequence``. It does not expose DNA Chisel's sequence optimization or design capabilities described in the original publication.
- **Codon and Species Limitations**: Codon-related outputs such as ``cai``, ``codon_usage_entropy`` and ``rare_codon_frequency`` are computed only for ``params.species`` values in ``SupportedSpecies`` (``e_coli``, ``s_cerevisiae``, ``h_sapiens``, ``c_elegans``, ``b_subtilis``, ``d_melanogaster``). Using other organisms requires mapping to one of these codon tables and may not reflect true in vivo behavior.
- **Enzyme and Motif Coverage**: ``restriction_site_count`` is limited to enzymes returned by ``list_supported_restriction_enzymes``; any unsupported name in ``params.restriction_enzymes`` raises a validation error. Motif-derived counts such as ``tata_box_count`` and ``tandem_repeat_count`` rely on simple pattern-based heuristics and are not substitutes for full promoter, regulatory-element, or repeat-annotation pipelines.
- **No Higher-Level Functional Predictions**: Outputs like ``gc_content``, ``hairpin_score``, ``kozak_sequence_strength``, or ``tandem_repeat_count`` are low-level sequence metrics only. The API does not predict protein structure, expression levels, folding, activity, stability, or other cellular or organismal phenotypes, and is not suitable as a standalone design or validation tool for complex genetic constructs.

How We Use It
-------------

DNA Chisel enables us to score DNA sequences for manufacturability and basic design risks before synthesis, providing standardized sequence-quality metrics that feed into protein engineering and lab-in-the-loop optimization. By turning GC content, codon usage statistics, restriction site counts, sequence complexity, and related features into structured outputs, we link DNA Chisel analyses with sequence-level predictors, structure-based property models, and generative design loops to de-risk synthesis and focus wet-lab effort on candidates with robust DNA-level properties.

- Integrates with predictive models and embeddings to rank or filter designs using DNA designability metrics alongside protein-level scores.
- Supports multi-round optimization by supplying consistent, synthesis-aware sequence features for each candidate across design cycles.

Related
-------

- ``DNABERT-2`` – Provides DNA sequence embeddings that can help flag motifs and regions of interest before running DNA Chisel feature analysis.
- ``Omni-DNA 1B`` – Predicts functional and regulatory DNA elements to guide which regions to preserve or avoid editing when interpreting DNA Chisel feature outputs.
- ``Evo 2 1B Base`` – Generates candidate coding sequences under evolutionary and fitness-related constraints that can then be scored with DNA Chisel design metrics.
- ``ESM-2 650M`` – Supplies protein-level context for coding regions whose nucleotide-level properties (e.g., CAI, GC content, Kozak strength) are quantified by DNA Chisel.

References
----------

- Zulkower, V., & Rosser, S. (2020). DNA Chisel, a versatile sequence optimizer. *Bioinformatics*, 36(16), 4508–4509. https://doi.org/10.1093/bioinformatics/btaa558
