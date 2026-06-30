.. _model_api:

ModelName API
=============

Short description of the model and its purpose.  
Mention key features, supported actions, and typical applications.

ActionName
----------

This endpoint does XYZ for ModelName.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code-block:: python

            from biolmai import BioLM
            response = BioLM(
                entity="modelname",
                action="actionname",
                params={
                    # params here
                },
                items=[
                    # items here
                ]
            )
            print(response)

    .. tab-item:: Python Requests
        :sync: python

        .. code-block:: python

            import requests

            url = "https://biolm.ai/api/v3/modelname/actionname/"
            headers = {"Authorization": "Token YOUR_API_KEY", "Content-Type": "application/json"}
            payload = {
                "params": {
                    # params here
                },
                "items": [
                    # items here
                ]
            }
            resp = requests.post(url, json=payload, headers=headers)
            print(resp.json())

    .. tab-item:: cURL
        :sync: curl

        .. code-block:: bash

            curl -X POST https://biolm.ai/api/v3/modelname/actionname/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
                "params": {
                  # params here
                },
                "items": [
                  # items here
                ]
              }'

    .. tab-item:: R
        :sync: r

        .. code-block:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/modelname/actionname/"
            headers <- c(
              "Authorization" = "Token YOUR_API_KEY",
              "Content-Type" = "application/json"
            )
            body <- list(
              params = list(
                # params here
              ),
              items = list(
                # items here
              )
            )
            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/modelname/actionname/

   Short description of what this endpoint does.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON

   .. container:: field-definition

      - **params** (*object*) --- Description of params:

        - **param1** (*type*) — description
        - **param2** (*type*) — description

      - **items** (*array of objects*, max. N) --- Description of items:

        - **field1** (*type*) — description
        - **field2** (*type*) — description

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/modelname/actionname/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {
        "params": {
          # params here
        },
        "items": [
          # items here
        ]
      }

   :statuscode 200: Description of success.
   :statuscode 400: Description of client error.
   :statuscode 401: Unauthorized.
   :statuscode 500: Internal server error.

   .. container:: field-heading

      Response JSON

   .. container:: field-definition

      - **results** (*array of objects*) — Description of results:

        - **field1** (*type*) — description
        - **field2** (*type*) — description

   **Example response**:

   .. code-block:: json

      {
        "results": [
          {
            "field1": "value",
            "field2": "value"
          }
        ]
      }

Performance
-----------
- Bullet points about performance, batch size, latency, etc.

Applications
------------
- Bullet points about technical use cases.

Limitations
-----------
- Bullet points about known limitations.

How BioLM Uses ModelName
------------------------
- Bullet points or short paragraph about integration into scientific/synbio protein/DNA/enzyme/antibody workflows.

Related
-------
- :ref:`other_model_api` — Short description

References
----------
- Author(s), Paper Title, Journal, Year. DOI/URL