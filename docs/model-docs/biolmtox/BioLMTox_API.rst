================
BioLMTox API
================

.. article-info::
    :avatar: ../img/book_icon.png
    :date: Dec 26, 2023
    :read-time: 6 min read
    :author: Chance Challacombe
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*This page explains applications of BioLMTox and documents
it's usage for classification and embedding extraction on BioLM.*

---------------
Endpoints
---------------

The BioLM endpoint for BioLMTox is `https://biolm.ai/api/v1/models/biolmtox_v1/<model_action>/ <https://api.biolm.ai/#8616fff6-33c4-416b-9557-429da180ef92>`_.



---------------------------
Embedding API Usage
---------------------------

The endpoint for BioLMTox embedding extraction is `https://biolm.ai/api/v1/models/biolmtox_v1/transform/ <https://api.biolm.ai/#723bb851-3fa0-40fa-b4eb-f56b16d954f5>`_.

^^^^^^^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/biolmtox_v1/transform/' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{
            "instances": [{
               "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
            }]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/biolmtox_v1/transform/"

            payload = json.dumps({
            "instances": [
               {
                  "data": {
                  "text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"
                  }
               }
            ]
            })
            headers = {
            'Authorization': 'Token {}'.format(os.environ['BIOLMAI_TOKEN']),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            seqs = [""MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"]

            cls = biolmai.BioLMToxv1()
            resp = cls.transform(seqs)

    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN')),
            "Content-Type" = "application/json"
            )
            params = "{
            \"instances\": [
               {
                  \"data\": {
                  \"text\": \"MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARLGWQDIKVADNADNDALLRALQ"
                  }
               }
            ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/biolmtox_v1/transform/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)

^^^^^^^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

        {"predictions": [
        [
            0.05734514817595482,
            -0.38758233189582825,
            0.14011333882808685,
            0.1311631053686142,
            0.6449017524719238,
            0.042671725153923035,
            0.04185352101922035,

.. note::
  The above response is only a small snippet of the full JSON response. However, all the relevant response keys are included.

^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^

data:
   Inside each instance, there's a key named "data" that holds another
   dictionary. This dictionary contains the actual input data for the
   endpoint action.

text:
   Inside the "data" dictionary, there's a key named "text". The value
   associated with "text" should be a string containing the amino acid sequence
   that the user wants to submit for toxin classification or embedding extraction.

^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^

predictions:
   This is the main key in the JSON object that contains an array of embedding extraction results with one embedding array per sequence in the request


---------------------------
Prediction API Usage
---------------------------
The endpoint for BioLMTox toxin classification is `https://biolm.ai/api/v1/models/biolmtox_v1/predict/ <https://api.biolm.ai/#8616fff6-33c4-416b-9557-429da180ef92>`_.

^^^^^^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/biolmtox_v1/predict/' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{
            "instances": [{
               "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
            }]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/biolmtox_v1/predict/"

            payload = json.dumps({
            "instances": [
               {
                  "data": {
                  "text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"
                  }
               }
            ]
            })
            headers = {
            'Authorization': 'Token {}'.format(os.environ["BIOLMAI_TOKEN"]),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            seqs = [""MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"]

            cls = biolmai.BioLMToxv1()
            resp = cls.predict(seqs)

    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN')),
            "Content-Type" = "application/json"
            )
            params = "{
            \"instances\": [
               {
                  \"data\": {
                  \"text\": \"MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ\"
                  }
               }
            ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/biolmtox_v1/predict/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)


^^^^^^^^^^^^^^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

        {"predictions": [
            {
            "label":"not toxin",
            "score":0.9998562335968018
            }
        ]
        }


^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^

data:
   Inside each instance, there's a key named "data" that holds another
   dictionary. This dictionary contains the actual input data for the
   endpoint action.

text:
   Inside the "data" dictionary, there's a key named "text". The value
   associated with "text" should be a string containing the amino acid sequence
   that the user wants to submit for toxin classification or embedding extraction.

^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^

predictions:
   This is the main key in the JSON object that contains an array of prediction results. Each element in the array represents a set of predictions for one input instance.

label:
   This key holds the predicted classification label for the input instance, it will be either toxin or not toxin

score:
   The model score for predicted class label, the closer the score is to 1 the more confident the model is in the prediction.

-------
Related
-------

:doc:`/model-docs/esm2/index`

.. _biolmtox2_api:

BioLMTox2 API
=============

Model Overview
--------------
BioLMTox2 is a protein toxin classifier and embedding extractor. It supports batch processing of up to 16 protein sequences (1–2048 AAs, unambiguous) for both embedding extraction and toxin prediction.

Embedding Extraction (Encode)
-----------------------------

.. http:post:: /api/v3/biolmtox2/encode/

   Extracts mean embeddings for input protein sequences.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   **Request JSON Object**

   .. code-block:: json

      {
        "items": [
          {"sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
        ]
      }

   - **items** (*array of objects*, max. 16):
     - **sequence** (*string*, 1–2048 AAs, unambiguous)

   **Response JSON Object**

   .. code-block:: json

      {
        "results": [
          {"mean_representation": [0.1, 0.2, ...]}
        ]
      }

   - **results** (*array of objects*):
     - **mean_representation** (*array of float*): Embedding vector for each input sequence.

Toxin Prediction (Predict)
-------------------------

.. http:post:: /api/v3/biolmtox2/predict/

   Predicts whether each input sequence is a toxin.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   **Request JSON Object**

   .. code-block:: json

      {
        "items": [
          {"sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
        ]
      }

   - **items** (*array of objects*, max. 16):
     - **sequence** (*string*, 1–2048 AAs, unambiguous)

   **Response JSON Object**

   .. code-block:: json

      {
        "results": [
          {"label": "not-toxin", "score": 0.9998}
        ]
      }

   - **results** (*array of objects*):
     - **label** (*string*): "toxin" or "not-toxin"
     - **score** (*float*): Confidence score for the predicted label

Constraints
-----------
- **Batch size**: up to 16 sequences per request
- **Sequence length**: 1–2048 amino acids
- **Amino acids**: Only unambiguous AAs allowed

Example Usage
-------------
.. code-block:: python

   from biolmai import BioLM
   # Embedding extraction
   response = BioLM(entity="biolmtox2", action="encode", items=[{"sequence": "MSILV..."}])
   print(response)

   # Toxin prediction
   response = BioLM(entity="biolmtox2", action="predict", items=[{"sequence": "MSILV..."}])
   print(response)

References
----------
- [BioLM API documentation](https://docs.biolm.ai/)
