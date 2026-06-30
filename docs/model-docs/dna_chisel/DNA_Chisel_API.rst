DNA-Chisel API
==============

DNA-Chisel is a lightweight analytics engine that extracts >20 sequence-level
features from raw DNA.  It supports GC-content, codon-adaptation index (CAI),
hairpin detection, melting temperature and many more statistics in a single call.

.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#dna-chisel

Encode (Analytics)
------------------

The sole endpoint returns a JSON object for every sequence, populated with the
requested features.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM

            seq = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"

            resp = BioLM(
                entity="dna-chisel",
                action="encode",
                params={
                    "include": ["gc_content", "cai"],
                    "species": "e_coli",
                },
                items=[{"sequence": seq}],
            )
            print(resp)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/dna-chisel/encode/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d @payload.json

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/dna-chisel/encode/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json",
            }
            payload = {
                "params": {
                    "include": ["gc_content", "cai"],
                    "species": "e_coli",
                },
                "items": [{"sequence": "ATGGCCATT..."}],
            }

            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/dna-chisel/encode/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type"  = "application/json"
            )
            payload <- list(
              params = list(include = list("gc_content", "cai"), species = "e_coli"),
              items  = list(list(sequence = "ATGGCCATT..."))
            )

            res <- POST(url, add_headers(.headers = headers), body = payload, encode = "json")
            print(content(res))

.. http:post:: /api/v3/dna-chisel/encode/

   Returns selected analytics for each DNA sequence.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   **Request JSON Object**

   - **params** (*object*)  
     - **include** (*list*) — subset of analytics to compute  
     - **species** (*string*) — species codon table (for CAI)  
     - **restriction_enzymes** (*list[str], optional*)
   - **items** (*array*, max 1*)  
     - **sequence** (*string*) — DNA (IUPAC unambiguous) ≤ 4 096 bp

   :statuscode 200: Successful. ``results`` array mirrors input order.
   :statuscode 400: Validation error (non-DNA chars, unknown enzyme …)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/dna-chisel/encode/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {
          "include": ["gc_content", "cai"],
          "species": "e_coli"
        },
        "items": [
          {"sequence": "ATGGCCATT..."}
        ]
      }

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "results": [
          {
            "gc_content": 0.54,
            "cai": 0.88
          }
        ]
      }

Performance
-----------

- CPU-only (0.25 vCPU, 1 GB RAM); median latency < 0.5 s.  
- No external calls – fully deterministic & cache-friendly.

Applications
------------

- **Codon-usage checks** before synthesis.  
- **Promoter / terminator QC** in synthetic circuits.  
- **Large-scale feature extraction** feeding sequence-to-vector ML pipelines.

Limitations
-----------

- Single sequence per request (batching planned).  
- Focused on unambiguous DNA (IUPAC).  
- No optimisation / design – analytics only.

How BioLM Uses DNA-Chisel
-------------------------

BioLM pipelines call DNA-Chisel prior to LLM-based design to surface GC skew,
rare-codon stretches or hairpin hotspots that might impair expression.

Related
-------

- ``DNABERT-2`` – language-model embeddings for DNA  
- ``OmniDNA`` – large-context DNA transformer

References
----------

- Zulkower V. & Rosser S. DNA-Chisel, a versatile sequence optimizer.  
  *Nucleic Acids Res.* **48**, W70–W76 (2020). https://doi.org/10.1093/nar/gkaa369

Schema reference: ``algorithms.schemas_v3.static.dna_chisel``. 