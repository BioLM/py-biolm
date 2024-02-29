
===================
ProGen2 API
===================

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Zeeshan Siddiqui
    :date: October 19th, 2023
    :read-time: 8 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of the ProGen2 OAS. As well as document the BioLM API for prediction, and demonstrate no-code and code interfaces for predictions.*

-------------
Description:
-------------

ProGen2 represents one of the largest protein language models, leveraging self-supervised pretraining on extensive protein sequence data to generate useful representations applicable to diverse protein structure and function prediction and design applications. As an attention-based model trained on protein sequences, ProGen2 employs a mechanism to selectively focus on informative regions of input data, learning intricate patterns and relationships among amino acids within protein sequences. Specifically, ProGen2 is trained via masked language modeling to predict amino acids from surrounding sequence context. As a protein language model, ProGen2 shows considerable promise for generating synthetic libraries of functional proteins to empower discovery and iterative optimization.

The BioLM API offers access to ProGen2 OAS, Medium, Large, and BDF90.


--------
Benefits
--------

* The BioLM API allows scientists to programmatically interact with ProGen2, making it easier to integrate the models into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.


---------
API Usage
---------

There are 4 BioLM endpoints corresponding to 4 different ProGen2 models.
These models are:

* ProGen2 OAS with an endpoint at `https://biolm.ai/api/v2/progen2-oas/generate/ <https://api.biolm.ai/#2424fa3a-144c-4d8e-8391-9e69b9df15d0>`_
* ProGen2 Medium with an endpoint at `https://biolm.ai/api/v2/progen2-medium/generate/ <https://api.biolm.ai/#06aed4bf-34ac-46da-99d7-791a271c2c53>`_
* ProGen2 Large with an endpoint at `https://biolm.ai/api/v2/progen2-large/generate/ <https://api.biolm.ai/#4526f1c0-6811-4b4c-9406-8fb774139bc4>`_
* ProGen2 BFD90 with an endpoint at `https://biolm.ai/api/v2/progen2-bfd90/generate/ <https://api.biolm.ai/#769693ef-1022-4850-8124-06de134d5413>`_


Each of these endpoints has the generate action. The generate action generates a sequence from a given sequence context.
. For example, using the OAS endpoint the url would be
`https://biolm.ai/api/v2/progen2-oas/generate/ <https://api.biolm.ai/#2424fa3a-144c-4d8e-8391-9e69b9df15d0>`_.


^^^^^^^^^^^^^^^^^^^^^^^^
Making Generate Requests
^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v2/progen2-oas/generate/' \
            --header 'Content-Type: application/json' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --data '{
                "params": {
                    "temperature": 0.7,
                    "top_p": 0.6,
                    "num_samples": 2,
                    "max_length": 105
                },
                "items": [
                    {
                        "context": "EVQ"
                    }
                ]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v2/progen2-oas/generate/"

            payload = json.dumps({
                "params": {
                    "temperature": 0.7,
                    "top_p": 0.6,
                    "num_samples": 2,
                    "max_length": 105
                },
                "items": [
                    {
                        "context": "EVQ"
                    }
                ]
            })
            headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Token {}'.format(os.environ['BIOLMAI_TOKEN']),
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)


    .. tab-item:: biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            ctxts = ["EVQ"]

            cls = biolmai.ProGen2Oas()
            resp = cls.generate(ctxts, params={
                    "temperature": 0.7,
                    "top_p": 0.6,
                    "num_samples": 2,
                    "max_length": 105
                })

    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            "Content-Type" = "application/json",
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN')),
            )
            payload = "{
                \"params\": {
                    \"temperature\": 0.7,
                    \"top_p\": 0.6,
                    \"num_samples\": 2,
                    \"max_length\": 105
                },
                \"items\": [
                    {
                        \"context\": \"EVQ\"
                    }
                ]
            }"
            res <- postForm("https://biolm.ai/api/v2/progen2-oas/generate/", .opts=list(postfields = payload, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
JSON Generate Response
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

        {
            "results": [
                [
                    {
                        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDGGGYS",
                        "ll_sum": -22.08656406402588,
                        "ll_mean": -0.21237081289291382
                    },
                    {
                        "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDRYSSS",
                        "ll_sum": -20.830227851867676,
                        "ll_mean": -0.20029065012931824
                    }
                ]
            ]
        }

^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^

items:
   Inside items are a list of dictionaries with each dictionary corresponding to one model input.
    context:
        The input context the model uses for generation
temperature:
    Represents the temperature parameter for the generation process. The temperature affects the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic

top_p:
    Represent a nucleus sampling parameter, which is a method to control the randomness of the generation by only considering a subset of the most probable tokens for sampling at each step.  Lower nucleus sampling probability, which usually makes sequence generation more conservative, results in sequences more closely matching the training dataset

max_length:
    The maximum length of the generated sequence. The model will stop generating once this length is reached.

num_samples:
    The number of independent sequences the user wants the model to generate for the given prompt. For example, if this value is set to 2, you will get two different generated sequences for the prompt.


^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^

results:
   This is the main key in the JSON object that contains an array of model results. Each element in the array represents a set of predictions for one input instance.

sequence:
    The generated sequence output of the model based on the provided context and parameters

ll_sum:
    Represents the sum of log-likelihoods for each token in the generated sequence. The log-likelihood gives an indication of how probable or confident the model was in generating each token. A higher log-likelihood indicates higher confidence.

ll_mean:
    This represents the average log-likelihood per token for the generated sequence. It's calculated by taking the mean of the log-likelihoods of all the tokens in the sequence. It provides an indication of the model's confidence in the generation.



----------
Related
----------

:doc:`/model-docs/progen2/ProGen2_Additional`

