================
BioLMTox
================

.. article-info::
    :avatar: img/book_icon.png
    :date: Dec 26, 2023
    :read-time: 6 min read
    :author: Chance Challacombe
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*This page explains applications of BioLMTox and documents
it's usage for classification and embedding extraction on BioLM.*

-----------
Description
-----------

Toxin classification
has important applications in both industry and research settings and has been a
concern for some time with respect to biosecurity and in the fields of protein, DNA
and drug design. BioLMTox is an application of the pre-train fine-tune paradigm,
honing the ESM-2 Pre-Trained Protein Language Model for general toxin classification.



--------
Benefits
--------

* Always-on, auto-scaling GPU-backed APIs; highly-scalable parallelization.
* Save money on infrastructure, GPU costs, and development time.
* Quickly integrate multiple embeddings into your workflows.
* Interact with the endpoint using natural language and our Chat Agents.
* Rapidly screen for biosecurity risks
* Get ahead of potential biosecurity regulation and laws

--------------------
Classifier API Usage
--------------------

--------------------
Embeddings API Usage
--------------------

The endpoint for BioLMTox toxin classification is `https://biolm.ai/api/v1/models/biolmtox_v1/predict/ <https://api.biolm.ai/#8616fff6-33c4-416b-9557-429da180ef92>`_.

The endpoint for BioLMTox embeddings is `https://biolm.ai/api/v1/models/biolmtox_v1/transform/ <https://api.biolm.ai/#723bb851-3fa0-40fa-b4eb-f56b16d954f5>`_.

^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

++++++++++++++++++++++
Classification Request
++++++++++++++++++++++

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/biolmtox_v1/predict/' \
            --header "Authorization: Token ed3fa24ec0432c5ba812a66d7b8931914c73a208d287af387b97bb3ee4cf907e" \
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
            'Authorization': 'Token {}'.format(os.environ['ed3fa24ec0432c5ba812a66d7b8931914c73a208d287af387b97bb3ee4cf907e']),
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

++++++++++++++++++
Embedding Requests
++++++++++++++++++

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/biolmtox_v1/transform/' \
            --header "Authorization: Token ed3fa24ec0432c5ba812a66d7b8931914c73a208d287af387b97bb3ee4cf907e" \
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
            'Authorization': 'Token {}'.format(os.environ['ed3fa24ec0432c5ba812a66d7b8931914c73a208d287af387b97bb3ee4cf907e']),
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

+++++++++++
Definitions
+++++++++++

data:
   A dictionary containing the input data for the request.

text:
   Inside the "data" dictionary, there is a key named "text". The value
   associated with "text" should be a string containing the amino acid sequence
   that the user wants to submit for toxin classification or embedding extraction.


^^^^^^^^^^^^^^
JSON Responses
^^^^^^^^^^^^^^

+++++++++++++++++++++++
Classification Response
+++++++++++++++++++++++

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

+++++++++++++++++++
Embeddings Response
+++++++++++++++++++

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

  The above response is only a small snippet of the full output. However, all the relevant response keys are included.

+++++++++++
Definitions
+++++++++++

predictions:
   Root-level key whose value is a list of predictions, one for each submitted sequence.

label:
   The predicted classification label for an input instance, either 'toxin' or 'not toxin'.

score:
   The model score for the given label, between 0 and 1. Closer to one indicates increasing confidence.


----------------
Model Background
----------------

BioLMTox is a protein language model fine-tuned for general toxin classification across multiple
domains of life and varying sequence lengths. The model was trained on a selection of sequences
from UniProt and UniRef50, as well as published datasets from previous works in this area.

------------------------
Applications of BioLMTox
------------------------

BioLMTox classification predictions and embeddings can be used to:

* augment biosecurity screening, prior to wet-lab use or synthesis, and alongside other computational screening software.

* discriminate between toxin and not-toxin homolologs that may bypass standard sequence-similarity or k-mer based screening methods.

* incorporate into public facing APIs, web apps and chat agents to reduce dual-use risks.










