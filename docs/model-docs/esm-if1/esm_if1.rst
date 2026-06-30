ESM-IF1 API
===========

ESM-IF1 is an inverse folding model that designs amino acid sequences conditioned on protein backbone atom coordinates (N, Cα, C), using an autoregressive transformer with invariant geometric layers trained on 12M AlphaFold2-predicted structures plus CATH. It achieves ~51% native sequence recovery and ~72% recovery for buried residues on structurally held-out backbones. The API generates up to 3 sequence samples per backbone (single- or multi-chain), with configurable temperature, for structure-conditioned protein design and sequence optimization tasks.

Generate
--------

Generate protein sequences from a provided protein backbone using the ESM-IF1 inverse folding model.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="esm-if1",
                action="generate",
                params={
                  "chain": "A",
                  "num_samples": 1,
                  "temperature": 0.6,
                  "multichain_backbone": false
                },
                items=[
                  {
                    "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND\n"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/esm-if1/generate/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "chain": "A",
                "num_samples": 1,
                "temperature": 0.6,
                "multichain_backbone": false
              },
              "items": [
                {
                  "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND\n"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/esm-if1/generate/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "chain": "A",
                    "num_samples": 1,
                    "temperature": 0.6,
                    "multichain_backbone": false
                  },
                  "items": [
                    {
                      "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND\n"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/esm-if1/generate/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                chain = "A",
                num_samples = 1,
                temperature = 0.6,
                multichain_backbone = FALSE
              ),
              items = list(
                list(
                  pdb = "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  
            ATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  
            ATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  
            ATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  
            ATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  
            TER       6      ALA A   1                                                      
            END
            "
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/esm-if1/generate/

   Generate endpoint for ESM-IF1.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, required) --- Configuration parameters:

        - **chain** (*string*, max length: 1, default: "A") — Chain identifier in the input PDB to use for sequence generation

        - **num_samples** (*int*, range: 1-3, default: 1) — Number of sequences to generate for each input item

        - **temperature** (*float*, range: 0.0-8.0, default: 0.6) — Sampling temperature applied during sequence generation

        - **multichain_backbone** (*bool*, default: False) — Indicates whether the input PDB contains multiple chains in the backbone



      - **items** (*array of objects*, min: 1, max: 1) --- Input data items:

        - **pdb** (*string*, min length: 1, max length: 100000, required) — Protein backbone structure as a PDB-format text block

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm-if1/generate/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "chain": "A",
          "num_samples": 1,
          "temperature": 0.6,
          "multichain_backbone": false
        },
        "items": [
          {
            "pdb": "ATOM      1  N   ALA A   1      11.104  13.207  11.947  1.00 20.00           N  \nATOM      2  CA  ALA A   1      12.560  13.051  11.824  1.00 20.00           C  \nATOM      3  C   ALA A   1      13.069  11.615  12.062  1.00 20.00           C  \nATOM      4  O   ALA A   1      12.436  10.671  11.586  1.00 20.00           O  \nATOM      5  CB  ALA A   1      13.255  13.861  10.726  1.00 20.00           C  \nTER       6      ALA A   1                                                      \nEND\n"
          }
        ]
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of arrays of objects*) --- One result per input item, in the order requested:

        - (*array of objects*, length: num_samples) — Generated sequence samples for the input PDB item

          - **sequence** (*string*) — Generated amino acid sequence (single-letter amino acid codes)

          - **recovery** (*float*, range: 0.0 - 1.0) — Fraction of residues matching the reference sequence used for recovery calculation

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          [
            {
              "sequence": "G",
              "recovery": 0.0
            }
          ]
        ]
      }


Performance
-----------

- ESM-IF1 is optimized for fixed-backbone inverse folding, recovering on average ~51% of native residues on structurally held‑out CATH backbones and ~72% for buried core residues, substantially improving over earlier GVP-GNN inverse folding models trained only on experimental structures.
- Compared to structure-prediction models such as ESMFold and AlphaFold2, ESM-IF1 is specialized for sequence recovery and zero-shot mutational scoring conditioned on a known backbone, offering substantially higher sequence recovery and better perplexity for this task, while not providing full 3D structure prediction.
- Within the ESM family, ESM-IF1 (a 142M‑parameter GVP‑Transformer) achieves lower perplexity and higher sequence recovery than GVP-GNN and GVP-GNN-large variants, and generalizes well to complexes, partially masked backbones, and multi-state design when provided appropriate backbone coordinates.

Applications
------------

- Fixed-backbone protein sequence design from 3D coordinates, enabling rapid exploration of sequence variants for a given structure. Useful for engineering more stable or functional protein scaffolds, binding domains, or industrial proteins. Performance is strongest for well-resolved, globular backbones; surface-exposed and highly flexible regions are less reliably designed.
- Optimization of protein–protein interfaces by proposing sequences at a specified backbone interface, supporting design of tighter or more specific binders, receptor–ligand pairs, or protein-based biosensors. Requires reasonably accurate backbone coordinates for all interacting chains; predictions degrade when interface geometry is poorly defined.
- Zero-shot scoring of sequence variants on a fixed backbone to prioritize mutations that are more compatible with a desired structure, aiding stability or affinity engineering campaigns. Works best for point mutations or small local changes; not well suited for variants that require large backbone rearrangements, long insertions, or disordered segments.
- Multi-state sequence design by evaluating sequences against multiple backbone conformations (submitted in separate API calls), allowing users to search for sequences compatible with several functional states. Useful for designing conformational switches or allosteric proteins; effectiveness is reduced when conformations differ by large-scale unfolding or when structural data are low confidence.
- Completion and redesign of structured regions surrounding missing or low-confidence segments in a backbone by conditioning on the available coordinates, supporting loop redesign or local patching in structural biology workflows. Accuracy drops for long unresolved spans (e.g., >30 residues) or intrinsically disordered loops where the backbone conformation is uncertain.

Limitations
-----------

- **Batch Size**: The maximum number of input items per request is ``1`` (``ESMIF1Params.batch_size``); to process multiple structures, submit separate requests.
- **Maximum Input Length**: The ``pdb`` string for each item must not exceed ``max_pdb_str_len`` characters; longer PDBs must be truncated or split before submission.
- **Chain Selection**: The model designs sequences for a single chain indicated by the ``chain`` parameter (default ``"A"``); enabling ``multichain_backbone=True`` allows multi-chain backbones but may reduce accuracy.
- **Sampling Limits**: Each request can generate up to ``3`` sequences per backbone via ``num_samples`` (range ``1``–``3``); higher diversity requires multiple requests or temperature tuning via ``temperature`` (range ``0.0``–``8.0``).
- **Backbone Length and Quality**: ESM-IF1 was trained primarily on backbones up to 500 residues from CATH and AlphaFold2-predicted structures; very long chains, highly flexible/disordered regions, or novel folds may yield lower sequence recovery.
- **Task Scope**: ESM-IF1 performs fixed-backbone inverse folding only. It does not model backbone flexibility, side-chain packing, or large insertions/deletions, and is not a general structure prediction or stability/ΔΔG prediction tool.

How We Use It
-------------

ESM-IF1 enables rapid, structure-guided protein sequence design by proposing amino acid sequences conditioned on user-provided backbone coordinates. In BioLM workflows, it is used to generate alternative sequences for fixed backbones (including antibody domains, enzymes, and designed scaffolds), and the resulting variants are then filtered and ranked with sequence language models, embedding-based similarity search, and structure- or property-prediction models. Through iterative design–evaluate–redesign cycles built on standardized API calls, teams can focus synthesis on candidates predicted to preserve fold and interface geometry while improving stability, activity, or binding.

- Integrates with structure predictors (e.g., ESMFold or AlphaFold-derived backbones) and sequence-based models for closed-loop protein engineering.
- Prioritizes sequences for synthesis by combining ESM-IF1 sequence proposals with downstream predictors of stability, binding affinity, and other biophysical properties.

Related
-------

- ``AlphaFold2`` – Predicts protein structures from amino acid sequences, which you can feed into ``ESM-IF1`` to generate compatible sequences for fixed-backbone design.
- ``ESMFold`` – Fast single-sequence structure prediction; use its models as backbones for sequence design with ``ESM-IF1`` in iterative design–evaluate workflows.
- ``ESM-2 650M`` – Protein language model capturing evolutionary constraints; useful for sequence-only evaluation or refinement of candidates designed with ``ESM-IF1``.
- ``ProstT5 AA2Fold`` – Sequence-to-structure model that provides alternative backbone predictions, enabling comparison of ``ESM-IF1`` designs across different structural hypotheses.

References
----------

- Hsu, C., Verkuil, R., Liu, J., Lin, Z., Hie, B., Sercu, T., Lerer, A., & Rives, A. (2022). `Learning inverse folding from millions of predicted structures <https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2>`_. *bioRxiv*.
