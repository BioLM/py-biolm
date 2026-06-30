AntiFold API
============

AntiFold is an antibody-specific inverse-folding model fine-tuned from ESM-IF1.  
It scores, embeds and designs antibody variable domains given a PDB structure,  
keeping the backbone fixed while exploring sequence space that preserves the fold.

.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#antifold

Predict (Score)
--------------

This endpoint returns a ``global_score`` —the (negative) log-probability of the
input structure according to AntiFold *(lower = more plausible)*.  Chain IDs let
you specify VL/VH or nanobody context.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM

            pdb_str = open("6W41.pdb").read()

            resp = BioLM(
                entity="antifold",
                action="predict",
                params={"heavy_chain": "H", "light_chain": "L"},
                items=[{"pdb": pdb_str}],
            )
            print(resp)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/antifold/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d @payload.json

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests, json

            pdb_str = open("6W41.pdb").read()
            url = "https://biolm.ai/api/v3/antifold/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json",
            }
            payload = {
                "params": {"heavy_chain": "H", "light_chain": "L"},
                "items": [{"pdb": pdb_str}],
            }

            resp = requests.post(url, headers=headers, json=payload)
            print(resp.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            pdb_str <- readChar("6W41.pdb", file.info("6W41.pdb")$size)
            url <- "https://biolm.ai/api/v3/antifold/predict/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type"  = "application/json"
            )
            payload <- list(
              params = list(heavy_chain = "H", light_chain = "L"),
              items  = list(list(pdb = pdb_str))
            )

            res <- POST(url, add_headers(.headers = headers), body = payload, encode = "json")
            print(content(res))

.. http:post:: /api/v3/antifold/predict/

   Structure-aware scoring – returns ``global_score`` per item.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   **Request JSON Object**

   - **params** (*object*)  
     - **heavy_chain**, **light_chain** (*char*) — chain IDs of VL/VH  
       *(exclusive with* ``nanobody_chain``*)*
     - **nanobody_chain** (*char*) — chain ID for single-domain antibody
     - **antigen_chain** (*char, optional*) — include antigen context
   - **items** (*array*, max 32*) — list of structures  
     - **pdb** (*string*) — raw PDB text (≤ 2 MB)

   :statuscode 200: Successful. Returns ``results[].global_score``
   :statuscode 400: Validation error (missing chains, bad PDB …)

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/antifold/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {
          "heavy_chain": "H",
          "light_chain": "L"
        },
        "items": [
          {"pdb": "HEADER\nATOM ..."}
        ]
      }

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
        "results": [
          {"global_score": -57.3}
        ]
      }

Encode / Generate
-----------------

``POST /api/v3/antifold/encode/`` – embeddings (``mean`` / ``residue`` / ``logits``)  
``POST /api/v3/antifold/generate/`` – mutate selected regions with sampling temperature.

Performance
-----------

- CPU-only (1 vCPU, 2 GB RAM); typical latency < 3 s for single antibody.  
- Batch scoring up to 32 structures per request.

Applications
------------

- **In-silico affinity maturation** – prioritise mutations that keep structure intact.  
- **Liability assessment** – highlight disruptive CDR substitutions via perplexity.  
- **Sequence design** – generate diverse, structure-compatible CDR variants.

Limitations
-----------

- Requires reasonably accurate input structure (experimental or AlphaFold-predicted).  
- Not suited for multi-chain antibodies beyond heavy/light (+antigen) context.  
- Generation currently limited to ≤ 100 sequences per target.

How BioLM Uses AntiFold
-----------------------

BioLM plugs AntiFold into LLM-driven antibody design workflows, combining structure
perplexity with language-model priors to triage variants before costly wet-lab work.

Related
-------

- ``ESM-IF1`` – general inverse folding  
- ``ProteinMPNN`` – backbone-conditioned design (non-antibody)

References
----------

- Høie M H *et al.* AntiFold: Improved antibody structure-based design using inverse
  folding. *Bioinformatics* (2024). https://doi.org/10.1093/bioinformatics/xxxx

Schema reference: ``algorithms.schemas_v3.static.antifold``. 