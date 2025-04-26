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

The BioLM endpoint for BioLMTox is ```https://biolm.ai/api/v1/models/biolmtox_v1/<model_action>/ <https://api.biolm.ai/#8616fff6-33c4-416b-9557-429da180ef92>```_.



---------------------------
Embedding API Usage
---------------------------

The endpoint for BioLMTox embedding extraction is ```https://biolm.ai/api/v1/models/biolmtox_v1/transform/ <https://api.biolm.ai/#723bb851-3fa0-40fa-b4eb-f56b16d954f5>```_.

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
The endpoint for BioLMTox toxin classification is ```https://biolm.ai/api/v1/models/biolmtox_v1/predict/ <https://api.biolm.ai/#8616fff6-33c4-416b-9557-429da180ef92>```_.

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
