================
ESM-1v API
================

.. article-info::
    :avatar: ../img/book_icon.png
    :date: Oct 18, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

On this page, we will show and explain the use of ESM-1v. As well as document the BioLM API for folding, demonstrate no-code and code interfaces to folding.


---------
API Usage
---------


There are 6 BioLM endpoints corresponding to 5 different sized ESM-1v model endpoints and 1 endpoint combining all 5 models.
These endpoints are:

* ESM-1v model 1 with an endpoint at ```https://biolm.ai/api/v2/esm1v-n1/predict/ <https://api.biolm.ai/#ce3145ea-d930-44a8-a468-4d39710381f7>```_
* ESM-1v model 2 with an endpoint at ```https://biolm.ai/api/v2/esm1v-n2/predict/ <https://api.biolm.ai/#abd06778-9836-431a-937b-cfa0479b2632>```_
* ESM-1v model 3 with an endpoint at ```https://biolm.ai/api/v2/esm1v-n3/predict/ <https://api.biolm.ai/#e5e4e266-a42e-455e-8813-3184c170735c>```_
* ESM-1v model 4 with an endpoint at ```https://biolm.ai/api/v2/esm1v-n4/predict/ <https://api.biolm.ai/#2708787e-a1f3-4d3c-aa07-311079c947cc>```_
* ESM-1v model 5 with an endpoint at ```https://biolm.ai/api/v2/esm1v-n5/predict/ <https://api.biolm.ai/#9191c028-bab4-4e7e-b986-0f80d546f6f0>```_
* ESM-1v all models with an endpoint at ```https://biolm.ai/api/v2/esm1v-all/predict/ <https://api.biolm.ai/#67f77d96-c4d8-4a1f-953a-d26330c27315>```_


The BioLM API ESM-1v predict endpoints have been customized to return the likelihoods for every AA unmasked at any <mask> position, so you can easily see how the likelihood of the sequence being functional with the wild-type residue compares to a single-AA mutation at that position.
The way to get a straight, “what is the likelihood of function of this sequence” out of this model, is to mask one AA, then get the WT probability for the WT AA, returned by the API.
Furthermore, the BioLM API has 5 distinct endpoints, as there are five models trained randomly on the same data. Hence, the likelihoods coming out of each one for the same input are slightly different.
The best results are achieved by averaging the likelihoods given by all 5 models for a given AA at a given position corresponding to the 6th BioLM API endpoint.

For example, using the ESM-1v model 1, the predict API endpoint is
```https://biolm.ai/api/v2/esm1v-n1/predict/ <https://api.biolm.ai/#ce3145ea-d930-44a8-a468-4d39710381f7>```_.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v2/esm1v-n1/predict/' \
               --header "Authorization: Token $BIOLMAI_TOKEN" \
               --header 'Content-Type: application/json' \
               --data '{
                "items": [
                    {
                        "sequence": "QERLEUTGR<mask>SLYNIVAT"
                    }
                ]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v2/esm1v-n1/predict/"

            payload = json.dumps({
                "items": [
                    {
                        "sequence": "QERLEUTGR<mask>SLYNIVAT"
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
            seqs = ["QERLEUTGR<mask>SLYNIVAT"]

            cls = biolmai.ESM1v1()
            resp = cls.Predict(seqs)

    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN')),
            "Content-Type" = "application/json"
            )
            payload = "{
                \"items\": [
                    {
                        \"sequence\": \"QERLEUTGR<mask>SLYNIVAT\"
                    }
                ]
            }"
            res <- postForm("https://biolm.ai/api/v2/esm1v-n1/predict/", .opts=list(postfields = payload, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)


^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

         {
            "results": [
                [
                    {
                        "token": 4,
                        "token_str": "L",
                        "score": 0.10017549991607666,
                        "sequence": "Q E R L E U T G R L S L Y N I V A T"
                    },
                    {
                        "token": 8,
                        "token_str": "S",
                        "score": 0.07921414822340012,
                        "sequence": "Q E R L E U T G R S S L Y N I V A T"
                    },
                    {
                        "token": 10,
                        "token_str": "R",
                        "score": 0.0782080590724945,
                        "sequence": "Q E R L E U T G R R S L Y N I V A T"
                    },


.. note::
  The above response is only a small snippet of the full JSON response. Each of these dictionaries corresponds
to one of the acceptable amino acids

^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^

items:
   Inside items are a list of dictionaries with each dictionary corresponding to one model input.
    sequence:
        The input sequence for the model

^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^

results:
   This is the main key in the JSON object that contains an array of model results. Each element in the array represents a set of predictions for one input instance.

score:
   This represents the confidence or probability of the model's prediction for the masked token. A higher score indicates higher confidence.

token:
   The predicted token's identifier as per the model's tokenization scheme. It's an integer that corresponds to a particular token (in this case, a particular amino acid) in the model's vocabulary.

token_str:
   Represents the predicted token as a string. That is, the amino acid that was predicted to fill in the masked position in the sequence.

sequence:
   Represents the complete sequence with the masked position filled in by the predicted token.


-------
Related
-------
:doc:`/model-docs/esm1v/ESM-1v_Additional`

:doc:`/model-docs/esmif/index`

:doc:`/model-docs/esmfold/index`

:doc:`/model-docs/esm2/index`

