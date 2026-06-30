AbLang-2 API
============

AbLang-2 is an antibody-specific language model optimized to reduce germline bias and better model non-germline residues relevant for binding and developability. Trained on 35.6M unpaired and 1.26M paired VH/VL sequences from OAS using a modified masked language modeling objective with focal loss, it supports paired heavy/light-chain inputs up to 1024 residues per chain and batched processing of up to 32 antibodies per request. The API exposes sequence- and residue-level encodings, per-position likelihoods, and sequence restoration for masked or missing residues.

Predict
-------

Predict likelihood for these input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="ablang2",
                action="predict",
                params={
                  "include": "likelihood"
                },
                items=[
                  {
                    "heavy": "QVQLVQSGAEVKKPGASVKVSCK",
                    "light": "DIQMTQSPASLSASVGDRVTITC"
                  },
                  {
                    "heavy": "EVQLVESGGGLVKPGGSLKLSCA",
                    "light": "KVVMTQSPDSLSASLGDRVTITC"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/ablang2/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": "likelihood"
              },
              "items": [
                {
                  "heavy": "QVQLVQSGAEVKKPGASVKVSCK",
                  "light": "DIQMTQSPASLSASVGDRVTITC"
                },
                {
                  "heavy": "EVQLVESGGGLVKPGGSLKLSCA",
                  "light": "KVVMTQSPDSLSASLGDRVTITC"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/ablang2/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": "likelihood"
                  },
                  "items": [
                    {
                      "heavy": "QVQLVQSGAEVKKPGASVKVSCK",
                      "light": "DIQMTQSPASLSASVGDRVTITC"
                    },
                    {
                      "heavy": "EVQLVESGGGLVKPGGSLKLSCA",
                      "light": "KVVMTQSPDSLSASLGDRVTITC"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/ablang2/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = "likelihood"
              ),
              items = list(
                list(
                  heavy = "QVQLVQSGAEVKKPGASVKVSCK",
                  light = "DIQMTQSPASLSASVGDRVTITC"
                ),
                list(
                  heavy = "EVQLVESGGGLVKPGGSLKLSCA",
                  light = "KVVMTQSPDSLSASLGDRVTITC"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/ablang2/predict/

   Predict endpoint for AbLang-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*string*, default: "likelihood") — Output type to include; must be "likelihood"

      - **items** (*array of objects*, min: 1, max: 32) --- Input antibody sequence pairs:

        - **heavy** (*string*, min length: 1, max length: 1024, required) — Heavy chain amino acid sequence using extended amino acid alphabet

        - **light** (*string*, min length: 1, max length: 1024, required) — Light chain amino acid sequence using extended amino acid alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/ablang2/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": "likelihood"
        },
        "items": [
          {
            "heavy": "QVQLVQSGAEVKKPGASVKVSCK",
            "light": "DIQMTQSPASLSASVGDRVTITC"
          },
          {
            "heavy": "EVQLVESGGGLVKPGGSLKLSCA",
            "light": "KVVMTQSPDSLSASLGDRVTITC"
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

        - **likelihood** (*array of arrays of floats*, shape: [num_sequence_tokens, vocab_size]) — Per-token logit scores over the model vocabulary

        - **sequence_tokens** (*array of strings*, length: num_sequence_tokens) — Tokenized heavy and light input sequences, including special and separator tokens

        - **vocab_tokens** (*array of strings*, length: vocab_size) — Vocabulary tokens corresponding to columns in ``likelihood``

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "likelihood": [
              [
                -4.1002655029296875,
                -3.792412519454956,
                "... (truncated for documentation)"
              ],
              [
                -4.9303059577941895,
                -2.364351749420166,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "<",
              "Q",
              "... (truncated for documentation)"
            ],
            "vocab_tokens": [
              "M",
              "R",
              "... (truncated for documentation)"
            ]
          },
          {
            "likelihood": [
              [
                -3.3265843391418457,
                -5.960440158843994,
                "... (truncated for documentation)"
              ],
              [
                -3.3216309547424316,
                -3.2128522396087646,
                "... (truncated for documentation)"
              ],
              "... (truncated for documentation)"
            ],
            "sequence_tokens": [
              "<",
              "E",
              "... (truncated for documentation)"
            ],
            "vocab_tokens": [
              "M",
              "R",
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Encode
------

Generate embeddings for input sequences

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="ablang2",
                action="encode",
                params={
                  "include": "seqcoding",
                  "align": false
                },
                items=[
                  {
                    "heavy": "QVQLVQSGAEVKKQ",
                    "light": "DVVMTQTPLSLPVTP"
                  },
                  {
                    "heavy": "QVQLVESGGGSVQPGRSLR",
                    "light": "EIVLTQSPGTLSLSPGERA"
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/ablang2/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "params": {
                "include": "seqcoding",
                "align": false
              },
              "items": [
                {
                  "heavy": "QVQLVQSGAEVKKQ",
                  "light": "DVVMTQTPLSLPVTP"
                },
                {
                  "heavy": "QVQLVESGGGSVQPGRSLR",
                  "light": "EIVLTQSPGTLSLSPGERA"
                }
              ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/ablang2/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "params": {
                    "include": "seqcoding",
                    "align": false
                  },
                  "items": [
                    {
                      "heavy": "QVQLVQSGAEVKKQ",
                      "light": "DVVMTQTPLSLPVTP"
                    },
                    {
                      "heavy": "QVQLVESGGGSVQPGRSLR",
                      "light": "EIVLTQSPGTLSLSPGERA"
                    }
                  ]
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/ablang2/encode/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              params = list(
                include = "seqcoding",
                align = FALSE
              ),
              items = list(
                list(
                  heavy = "QVQLVQSGAEVKKQ",
                  light = "DVVMTQTPLSLPVTP"
                ),
                list(
                  heavy = "QVQLVESGGGSVQPGRSLR",
                  light = "EIVLTQSPGTLSLSPGERA"
                )
              )
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/ablang2/encode/

   Encode endpoint for AbLang-2.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

        - **include** (*string*, optional, default: "seqcoding", enum: ["seqcoding", "rescoding"]) — Encoding type to compute for each item

        - **align** (*boolean*, optional, default: false) — Alignment flag used when include is "rescoding"

      - **items** (*array of objects*, min: 1, max: 32) --- Input antibody sequence pairs:

        - **heavy** (*string*, required, min length: 1, max length: 1024) — Heavy chain amino acid sequence using the extended amino acid alphabet

        - **light** (*string*, required, min length: 1, max length: 1024) — Light chain amino acid sequence using the extended amino acid alphabet

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/ablang2/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "params": {
          "include": "seqcoding",
          "align": false
        },
        "items": [
          {
            "heavy": "QVQLVQSGAEVKKQ",
            "light": "DVVMTQTPLSLPVTP"
          },
          {
            "heavy": "QVQLVESGGGSVQPGRSLR",
            "light": "EIVLTQSPGTLSLSPGERA"
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

        - **seqcoding** (*array of floats*, size: 480) — Fixed-size embedding for the concatenated heavy and light chains

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "seqcoding": [
              -0.21477425201138592,
              -0.12190765886129264,
              "... (truncated for documentation)"
            ]
          },
          {
            "seqcoding": [
              -0.23653832053072577,
              -0.12340051473963053,
              "... (truncated for documentation)"
            ]
          }
        ]
      }


Performance
-----------

- Runs on 2 vCPUs and 4 GB RAM with CPU‑only inference; AbLang‑2 deployments do not use GPUs, making them significantly cheaper to operate than BioLM’s larger transformer antibody models (e.g. IgT5 Paired, IgBert Paired)
- AbLang‑2 uses a compact ESM‑2–derived architecture (12 layers, 480‑dimensional embeddings) and an optimized inference pipeline, giving lower latency for seqcoding and rescoding than larger antibody encoders at BioLM (e.g. IgT5, IgBert)
- Likelihood estimation and sequence restoration complete in milliseconds per VH/VL pair under typical loads, providing substantially lower end‑to‑end latency for design loops than heavier antibody language models that require GPU‑backed inference
- Relative to other antibody‑focused transformers at BioLM, AbLang‑2 delivers markedly better computational efficiency while maintaining competitive embedding quality and superior non‑germline mutation modeling accuracy for antibody design and optimization tasks

Applications
------------

- Antibody sequence optimization to prioritize non‑germline mutations that are frequently observed in affinity‑matured and therapeutic antibodies, using per‑residue likelihoods from the likelihood endpoint to highlight residues where mutations away from germline remain probable; useful for proposing sequence changes beyond simple germline reversion, but not a substitute for explicit binding or developability prediction models.
- In silico ranking of mutated antibody panels by comparing per‑residue likelihood profiles across VH/VL variants that differ at a limited number of positions (typically outside CDR3), enabling triage of alternative lead designs before experimental affinity maturation; less informative for variants involving major insertions/deletions, large structural rearrangements, or sequences far outside natural antibody space.
- Computational assessment of somatic hypermutation patterns within clonotypes by embedding paired VH/VL sequences with the encoder (seqcoding/rescoding) endpoints and comparing non‑germline‑like positions, supporting analysis of maturation trajectories and mutational hotspots in discovery or patient repertoires; not intended as a standalone phylogenetic or lineage reconstruction tool.
- Completion and restoration of partially known antibody variable regions by supplying “*” placeholders in VH/VL sequences to the restore/generate endpoint, enabling reconstruction of paired sequences from repertoire or display data with internal gaps but preserved framework context; performance degrades for highly fragmented inputs, unusual numbering, or non‑canonical antibodies.
- Rapid repertoire‑scale embedding of paired antibodies for clustering and diversity analyses in discovery campaigns, using encoder outputs (seqcoding for sequence‑level encodings or rescoding for residue‑level encodings) to group similar VH/VL pairs and quantify diversification from germline; suitable for selecting representative leads from large BCR‑seq datasets, but not as a standalone predictor of aggregation, immunogenicity, or manufacturability.

Limitations
-----------

- **Maximum Sequence Length**: Each ``heavy`` and ``light`` chain must be between 1 and ``1024`` amino acids. Longer chains must be truncated or split before submission.
- **Batch Size**: Each request can include between 1 and ``32`` antibody pairs in ``items``. Larger datasets should be split across multiple API calls.
- **Antibody-Specific Embeddings**: The ``seqcoding`` (per-sequence embedding) and ``rescoding`` (per-residue embedding, optionally aligned when ``align=True``) outputs are trained on antibody VH/VL sequences and may not generalize to non-antibody proteins or unrelated biological sequences.
- **Restore Requirements**: The ``restore`` / ``generate`` functionality only accepts sequences where at least one position in ``heavy`` or ``light`` is marked as ``*`` and all other characters are valid amino acids; sequences without ``*`` or with other invalid tokens cannot use this feature.
- **No Structural Predictions**: AbLang-2 does not output 3D structural information (e.g. full antibody structures or CDR loop conformations). Use dedicated structure prediction tools for such tasks.
- **No Binding or Epitope Predictions**: Likelihood scores (from ``likelihood`` / ``predict``) reflect sequence plausibility under the model and are not direct measures of antigen binding affinity, specificity, developability, or epitope/paratope location; these tasks require additional models or experimental validation.

How We Use It
-------------

BioLM uses AbLang-2 as a component in antibody design cycles to score VH–VL sequence pairs with reduced germline bias, highlight non-germline positions that are still plausible, and provide alternative residue suggestions that are more likely to yield productive affinity or liability shifts without destabilizing the scaffold. AbLang-2 sequence- and residue-level encodings, along with per-position likelihoods exposed via standardized, scalable APIs, are combined with structural modeling, developability screens, and experimental data to focus campaigns on non-germline modifications that preserve overall antibody integrity.

- Supports in silico affinity maturation by ranking mutations outside CDR3 using sequence likelihoods
- Integrates with structure- and biophysics-based tools for multi-parameter antibody optimization across large variant libraries

Related
-------

- ``IgT5 Paired`` – Sequence-to-sequence modeling for paired heavy/light chains; can be used after ``AbLang-2`` to generate or refine paired variants guided by AbLang-2 likelihoods.
- ``IgBert Paired`` – Produces transformer embeddings for paired chains; combine with ``AbLang-2`` likelihoods for joint representation, mutation scoring, and downstream tasks.
- ``ABodyBuilder3 pLDDT`` – Antibody structure prediction with per-residue confidence; apply to sequences evaluated, restored, or designed with ``AbLang-2`` to assess structural feasibility.
- ``ImmuneFold Antibody`` – Antibody structure prediction from sequence; use to structurally evaluate paired sequences for which ``AbLang-2`` has suggested or scored mutations.

References
----------

- Olsen, T. H., Moal, I. H., & Deane, C. M. (2024). `Addressing the antibody germline bias and its effect on language models for improved antibody design <https://doi.org/10.1101/2024.02.02.578678>`_. *bioRxiv*.
- Olsen, T. H., Moal, I. H., & Deane, C. M. (2022). `AbLang: an antibody language model for completing antibody sequences <https://doi.org/10.1093/bioadv/vbac046>`_. *Bioinformatics Advances*.
- Olsen, T. H., Boyles, F., & Deane, C. M. (2022). `Observed Antibody Space: A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequences <https://doi.org/10.1002/pro.4205>`_. *Protein Science*.
- Raybould, M. I. J., Marks, C., Lewis, A. P., Shi, J., Bujotzek, A., Taddese, B., & Deane, C. M. (2020). `Thera-SAbDab: the Therapeutic Structural Antibody Database <https://doi.org/10.1093/nar/gkz827>`_. *Nucleic Acids Research*.
- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*.
- Ruffolo, J. A., Gray, J. J., & Sulam, J. (2021). `Deciphering antibody affinity maturation with language models and weakly supervised learning <https://doi.org/10.48550/arXiv.2112.07782>`_. *arXiv preprint*.
- Prihoda, D., Maamary, J., Waight, A., Juan, V., Fayadat-Dilman, L., Svozil, D., & Bitton, D. A. (2022). `BioPhi: A platform for antibody design, humanization, and humanness evaluation based on natural antibody repertoires and deep learning <https://doi.org/10.1080/19420862.2021.2020203>`_. *mAbs*.
- Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., Ott, M., Zitnick, C. L., Ma, J., & Fergus, R. (2021). `Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences <https://doi.org/10.1073/pnas.2016239118>`_. *Proceedings of the National Academy of Sciences*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://doi.org/10.48550/arXiv.1810.04805>`_. *arXiv preprint*.
