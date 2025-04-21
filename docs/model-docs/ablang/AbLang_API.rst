================================
AbLang API
================================

.. article-info::
    :avatar: ../img/book_icon.png
    :date: Feb, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*On this page, we will show and explain the use of AbLang for Antibody Sequence Completion. As well as document the BioLM API, and demonstrate no-code  and code interfaces antibody design*

---------------------------
Endpoints
---------------------------

There are 2 BioLM endpoints corresponding to the AbLang Heavy and Light Chain models.


* AbLang Heavy Chain endpoint at `https://biolm.ai/api/v2/ablang-heavy/<model_action>/ <https://api.biolm.ai/#1b94f7dc-5dd7-48c6-9944-d933d85bc601>`_
* AbLang Light Chain endpoint at `https://biolm.ai/api/v2/ablang-light/<model_action>/ <https://api.biolm.ai/#867e99f1-7049-434b-a9a2-d6ff5da0986c>`_

Each of these endpoints has the encode, predict and generate actions

encode generates embeddings, predict returns the likelihoods for the sequence and generate restores the specified positions of a sequence.

----------------------------------------
Encode API Usage
----------------------------------------
Using AbLang Heavy Chain as an example, the encode endpoint is at
API endpoint for `https://biolm.ai/api/v2/ablang-heavy/encode/ <https://api.biolm.ai/#1b94f7dc-5dd7-48c6-9944-d933d85bc601>`_.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell
            curl --location 'https://biolm.ai/api/v2/ablang-heavy/encode/' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{
            "items": [
               {
                  "sequence": "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS"
               }
            ],
            "params": {
               "include": "seqcoding"
            }
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v2/ablang-heavy/encode/"

            payload = {
                "items": [
                    {
                        "sequence": "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS"
                    }
                ],
                "params": {
                    "include": "seqcoding"
                }
            }
            headers = {
            'Authorization': 'Token {}'.format(os.environ["BIOLMAI_TOKEN"]),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: biolm SDK
        :sync: sdk

        .. code:: sdk

                import biolm
                seqs = ["EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS"]

                cls = biolm.AbLangHeavy()
                resp = cls.encode(seqs, params={
                        "include": [
                            "seqcoding"
                        ]
                    })

    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN')),
            "Content-Type" = "application/json"
            )
            payload = "
                \"items\": [
                    {
                        \"sequence\": \"EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS\"
                    }
                ],
                \"params\": {
                    \"include\": \"seqcoding\"
                }
            }"
            res <- postForm("https://biolm.ai/api/v2/ablang-heavy/encode/", .opts=list(postfields = payload, followlocation = TRUE), style = "httppost")
            cat(res)



^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

         {
         "results": [
            {
               "seqcoding": [
               -0.6615958659340097,
               0.13918796144733744,
               -0.9715563959080326,
               -0.24384153723208743,
               0.0955913498129865,
               0.6615201387831495,
               -0.3109214511846215,
               0.4820148539248361,


.. note::
  The above response is only a small snippet of the full JSON response. However, all the relevant response keys are included.


^^^^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^^^^

params:
   Additional parameters for the request.

include:
   Specifies additional data to be included in the response. "seqcoding" indicates that sequence embeddings should be included.

payload:
   A string variable containing the JSON payload to be sent in the POST request. It consists of items and sequence.

sequence:
   The amino acid sequence of the antibody heavy chain for which you want to generate embeddings.

items:
   A list of dictionaries, each representing an item to be processed by the ABLang model. Each dictionary has a key.

^^^^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^^^^


results:
   A list containing the results of the ABLang model's encoding process. Each element in this list is a dictionary representing the results for one input item (in this case, one antibody heavy chain sequence).


seqcoding:
   A key within each result dictionary that corresponds to the sequence embeddings generated by the ABLang model for the input antibody heavy chain sequence. The value is a list of floating-point numbers, each representing a dimension in the embedding space. These embeddings capture the characteristics of the input sequence and can be used for various downstream tasks, such as similarity comparisons, clustering, or as input features for machine learning models.


----------
Related
----------

:doc:`/model-docs/ablang/AbLang_Additional`


.. _Status Page: https://status.biolm.ai






