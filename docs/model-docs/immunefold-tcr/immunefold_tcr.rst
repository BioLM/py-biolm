ImmuneFold TCR API
==================

ImmuneFold TCR is a GPU-accelerated structure prediction model for T-cell receptor proteins, fine-tuned from ESMFold using parameter-efficient LoRA adaptation. It predicts atomic-resolution 3D structures for TCR–pMHC complexes from sequence alone, without MSAs or templates, and returns PDB coordinates with per-residue pLDDT and pTM scores. Typical runtimes are on the order of seconds per complex. This service supports high-throughput TCR structure modeling for applications in immunotherapy design, neoantigen screening, and epitope specificity analysis.

Predict
-------

Predict 3D structures for TCR proteins

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="immunefold-tcr",
                action="predict",
                params={
                  "contact_idx": null
                },
                items=[
                  {
                    "H": null,
                    "L": null,
                    "B": "VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL",
                    "A": "VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS",
                    "P": "YLQPRTFLL",
                    "M": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ",
                    "pdb": null
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/immunefold-tcr/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "contact_idx": null
              },
              "items": [
                {
                  "H": null,
                  "L": null,
                  "B": "VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL",
                  "A": "VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS",
                  "P": "YLQPRTFLL",
                  "M": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ",
                  "pdb": null
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/immunefold-tcr/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "contact_idx": null
                  },
                  "items": [
                    {
                      "H": null,
                      "L": null,
                      "B": "VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL",
                      "A": "VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS",
                      "P": "YLQPRTFLL",
                      "M": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ",
                      "pdb": null
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/immunefold-tcr/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                contact_idx = None
              ),
              items = list(
                list(
                  H = None,
                  L = None,
                  B = "VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL",
                  A = "VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS",
                  P = "YLQPRTFLL",
                  M = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ",
                  pdb = None
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/immunefold-tcr/predict/

   Predict endpoint for ImmuneFold TCR.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **contact_idx** (*int*, optional) — Index for residue-residue contact analysis in the predicted structure


      - **items** (*array of objects*, min: 1, max: 32) --- Input sequences for prediction:

        - **H** (*string*, min length: 1, max length: 256, optional) — Antibody heavy or single-chain variable domain amino acid sequence

        - **L** (*string*, min length: 1, max length: 256, optional) — Antibody light chain variable domain amino acid sequence

        - **B** (*string*, min length: 1, max length: 256, optional) — TCR beta chain amino acid sequence

        - **A** (*string*, min length: 1, max length: 256, optional) — TCR alpha chain amino acid sequence

        - **P** (*string*, min length: 1, max length: 256, optional) — Peptide (epitope) amino acid sequence

        - **M** (*string*, min length: 1, max length: 256, optional) — MHC amino acid sequence

        - **pdb** (*string*, min length: 1, max length: 1000000, optional) — Antigen structure in PDB text format

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/immunefold-tcr/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "contact_idx": null
        },
        "items": [
          {
            "H": null,
            "L": null,
            "B": "VSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSVSRDRNTGELFFGEGSRLTVL",
            "A": "VEQDPGPFNVPEGATVAFNCTYSNSASQSFFWYRQDCRKEPKLLMSVYSSGNEDGRFTAQLNRASQYISLLIRDSKLSDSATYLCVVNEEDALIFGKGTTLSVSS",
            "P": "YLQPRTFLL",
            "M": "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ",
            "pdb": null
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

        - **ptm** (*float*, range: 0.0–1.0) — Predicted TM-score-like (pTM) global structure confidence

        - **full_plddt** (*float*, range: 0.0–100.0) — Mean per-residue pLDDT confidence over all residues and chains

        - **plddt** (*array of arrays of floats*, shape: [num_chains, num_residues_per_chain], range: 0.0–100.0) — Per-residue pLDDT confidence scores for each chain

        - **pdb** (*string*) — Predicted multi-chain protein structure in PDB format

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "pdb": "ATOM      1  N   VAL B   1      12.257   4.414  11.069  1.00 94.96           N  \nATOM      2  CA  VAL B   1      11.290   5.443  11.439  1.00 94.96           C  \nATOM      3  C   VAL B   1      11.439... (truncated for documentation)"
          }
        ]
      }


Performance
-----------

- Runs on Nvidia T4 GPUs with no additional inference overhead relative to base ESMFold, due to LoRA parameters being merged into the original weights after fine-tuning.
- For TCR structure prediction, ImmuneFold TCR improves accuracy over general and TCR-specific models:
  
  - CDR3β backbone RMSD 1.31 Å vs 1.57 Å (AlphaFold-Multimer) and 3.67 Å (ESMFold).
  - Lower RMSD across all CDRs and framework regions than TCRmodel2 and ImmuneBuilder.
  - Better α/β inter-chain geometry with iRMS 1.43 Å and DockQ 0.84, exceeding TCRmodel2 and AlphaFold-Multimer.

- Confidence scores correlate strongly with true accuracy, providing usable quality estimates: protein-level Pearson correlation between pLDDT and RMSD 0.87, residue-level 0.63.
- When combined offline with Rosetta (not part of the API), ImmuneFold-based structures support zero-shot TCR–epitope binding ranking with AUROC 0.69 vs 0.55 (PanPep) and 0.57 (TEIM).

Applications
------------

- TCR variable domain and CDR3β structure prediction for TCR-engineering workflows  
  ImmuneFold enables atomic-level modeling of paired α/β TCRs, including the hypervariable CDR3β loops that are poorly covered by homology templates. This is useful for companies designing TCR-based cell therapies or soluble TCRs, for example comparing alternative CDR3β designs for a given target peptide–MHC and filtering out candidates with clearly suboptimal loop geometries. Predictions are most reliable for TCR lengths within the 256-residue per-chain API limit and should be treated as structural hypotheses, not a substitute for experimental validation.

- Zero-shot TCR–epitope ranking using structure-based binding scores  
  By combining ImmuneFold TCR–pMHC complex models (using the B/A/P/M chains in the API) with external docking or Rosetta energy calculations, teams can rank peptide variants for a given TCR without training a supervised binding model. This supports neoantigen prioritization in personalized oncology (e.g. scoring 10–20 candidate HLA-A*02:01 peptides for one TCR) and early in silico feasibility checks when binding data are sparse. Energy-based rankings depend on the downstream scoring protocol and should be interpreted comparatively within a peptide or TCR panel rather than as absolute affinities.

- Antibody and nanobody variable region structure prediction for lead optimization  
  For antibodies, the API accepts heavy (H) and optional light (L) chain sequences (single VHHs via H only) and predicts Fv structures with improved accuracy in CDR H3 and chain orientation compared to general-purpose folders. This helps therapeutic antibody teams visualize paratope geometry, identify clashes introduced by affinity-maturation mutations, and select a small number of variants for expression based on structural plausibility. Results are most informative when Fv lengths respect the 256-residue per-chain limit; ImmuneFold does not predict developability metrics such as aggregation or immunogenicity directly.

- Antibody/nanobody modeling in the context of a known antigen structure  
  When an antigen PDB is supplied with an antibody sequence (H/L plus pdb in the API), ImmuneFold refines the paratope conformation in the presence of that fixed antigen, which can modestly improve CDR H3/CDR3 placement at the interface. This is useful for structure-based virtual screening or re-scoring of docked poses when a high-confidence antigen model is already available, for example ranking 50–100 humanized variants against a single antigen structure. The API does not perform full antibody–antigen docking or epitope discovery; users must provide the antigen structure and interpret predicted complexes in combination with external docking and scoring tools.

- High-throughput immune repertoire modeling under sequence-length constraints  
  The parameter-efficient ImmuneFold architecture allows batched prediction of up to 32 immune proteins per API call, with each chain up to 256 residues. This supports industrial-scale analyses such as building structural catalogs from TCR repertoires or antibody discovery campaigns and feeding those structures into downstream pipelines (e.g. Rosetta design, developability screening, or ML models on 3D features). Very long or highly glycosylated constructs, or complexes beyond the supported chain layout, are not currently handled and may require alternative tools or custom engineering support.

Limitations
-----------

- **Maximum Sequence Length**: Each chain field (`H`, `L`, `B`, `A`, `P`, `M`) must be ``<= 256`` amino acids (validated as protein sequences). Longer sequences are rejected and must be truncated or split client-side.
- **Batch Size**: The ``items`` array in ``ImmuneFoldPredictRequest`` supports at most ``32`` entries per call (``batch_size = 32``). Larger workloads must be split across multiple requests.
- **Input Combinations**: Each item must be either an antibody or a TCR, not both. Antibodies require ``H`` (with optional ``L``); TCR–pMHC complexes require all of ``B``, ``A``, ``P``, and ``M``. Providing invalid combinations (for example, ``L`` without ``H`` or mixing antibody and TCR fields) will raise validation errors.
- **Antigen Context**: Antigen structures via ``pdb`` are only supported for antibodies and must not be combined with TCR inputs (``B``, ``A``, ``P``, ``M``). TCR predictions always use sequence-only inputs; generic protein–protein docking is not supported through this API.
- **Static Structures Only**: Predictions are single static 3D structures (returned in ``pdb``) and do not capture conformational dynamics, induced fit, or full ensembles upon antigen binding.
- **Confidence and Out-of-Distribution Sequences**: Confidence scores (``plddt``, ``ptm``) correlate with accuracy but may be unreliable for highly unusual or out-of-distribution immune sequences, which can lead to lower-quality structures.

How We Use It
-------------

ImmuneFold TCR enables accurate structural modeling and zero-shot binding-oriented scoring of T-cell receptors in complex with peptide–MHC, supporting design and prioritization of TCRs for cell therapies and epitope-focused vaccine programs. Within broader BioLM workflows, ImmuneFold TCR provides high-confidence 3D structures and per-residue confidence (pLDDT), which can be combined with external physics-based tools (for example, Rosetta run by the user) and BioLM sequence-level models to iteratively propose, filter, and refine TCR variants focused on CDR3-driven recognition.

- Accelerates identification and ranking of TCR candidates by supplying reliable complex structures for downstream affinity and developability assessments.
- Integrates with sequence engineering and external docking/energy workflows, reducing low-value experimental screening and enabling scalable, standardized in silico triage of TCR designs.

Related
-------

- ``ImmuneFold Antibody`` – LoRA-fine-tuned ESMFold model specialized for antibody and nanobody structure prediction; use alongside ImmuneFold TCR for comprehensive immune repertoire modeling.
- ``TCRBuilder2`` – Alternative TCR structure prediction method for benchmarking and cross-validating ImmuneFold TCR results on the same input sequences.
- ``AlphaFold2`` – General protein structure predictor useful as a baseline for TCR chains and for modeling non-immune proteins in related studies.
- ``ESMFold`` – Base monomeric structure model from which ImmuneFold TCR is adapted; useful for comparing the impact of immune-specific fine-tuning.

References
----------

- Zhu, T., Ren, M., He, Z., Tao, S., Li, M., Bu, D., & Zhang, H. (2024). Accurate structure prediction of immune proteins using parameter-efficient transfer learning. *Journal of Computational Biology*.
