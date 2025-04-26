..
   Copyright (c) 2021 Pradyun Gedam
   Licensed under Creative Commons Attribution-ShareAlike 4.0 International License
   SPDX-License-Identifier: CC-BY-SA-4.0


===============
ProteInfer API
===============

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Article Information
    :date: Oct 24, 2023
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of ProteInfer for enzyme function prediction. As well as document the BioLM API, and demonstrate no-code  and code interfaces to enzyme function prediction.*

---------
API Usage
---------
here are 2 BioLM endpoints corresponding to ProteInfer GO and EC.
These endpoints are

* ProteInfer EC with an endpoint at ```https://biolm.ai/api/v2/proteinfer-ec/ <https://api.biolm.ai/#9bcdd520-f163-4624-bd55-ff73103526a5>```_
* ProteInfer GO with an endpoint at ```https://biolm.ai/api/v2/proteinfer-go/ <https://api.biolm.ai/#ab42dafe-1c8d-4b35-9186-25abec5d9615>```_

Each of these endpoints can be accessed with the predict action by appending predict to the urls above.
For example, the BioLM ProteInfer EC prediction endpoint is ```https://biolm.ai/api/v2/proteinfer-ec/predict/ <https://api.biolm.ai/#9bcdd520-f163-4624-bd55-ff73103526a5>```_.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v2/proteinfer-ec/predict/' \
            --header 'Content-Type: application/json' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --data '{
                "items": [
                    {
                        "sequence": "MMQTVLAKIVADKAIWVEARKQQQPLASFQNEVQPSTRHFYDALQGARTAFILECKKASPSKGVIRDDFDPARIAAIYKHYASAISVLTDEKYFQGSFNFLPIVSQIAPQPILCKDFIIDPYQIYLARYYQADACLLMLSVLDDDQYRQLAAVAHSLEMGVLTEVSNEEEQERAIALGAKVVGINNRDLRDLSIDLNRTRELAPKLGHNVTVISESGINTYAQVRELSHFANGFLIGSALMAHDDLHAAVRRVLLGENKVCGLTRGQDAKAAYDAGAIYGGLIFVATSPRCVNVEQAQEVMAAAPLQYVGVFRNHDIADVVDKAKVLSLAAVQLHGNEEQLYIDTLREALPAHVAIWKALSVGETLPAREFQHVDKYVLDNGQGGSGQRFDWSLLNGQSLGNVLLAGGLGADNCVEAAQTGCAGLDFNSAVESQPGIKDARLLASVFQTLRAY"
                    }
                ]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v2/proteinfer-ec/predict/"

            payload = json.dumps({
                "items": [
                    {
                        "sequence": "MMQTVLAKIVADKAIWVEARKQQQPLASFQNEVQPSTRHFYDALQGARTAFILECKKASPSKGVIRDDFDPARIAAIYKHYASAISVLTDEKYFQGSFNFLPIVSQIAPQPILCKDFIIDPYQIYLARYYQADACLLMLSVLDDDQYRQLAAVAHSLEMGVLTEVSNEEEQERAIALGAKVVGINNRDLRDLSIDLNRTRELAPKLGHNVTVISESGINTYAQVRELSHFANGFLIGSALMAHDDLHAAVRRVLLGENKVCGLTRGQDAKAAYDAGAIYGGLIFVATSPRCVNVEQAQEVMAAAPLQYVGVFRNHDIADVVDKAKVLSLAAVQLHGNEEQLYIDTLREALPAHVAIWKALSVGETLPAREFQHVDKYVLDNGQGGSGQRFDWSLLNGQSLGNVLLAGGLGADNCVEAAQTGCAGLDFNSAVESQPGIKDARLLASVFQTLRAY"
                    }
                ]
            })
            headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Token {}'.format(os.environ['BIOLMAI_TOKEN'])
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: biolmai SDK
            :sync: sdk

            .. code:: sdk

                import biolmai
                seqs = ["MMQTVLAKIVADKAIWVEARKQQQPLASFQNEVQPSTRHFYDALQGARTAFILECKKASPSKGVIRDDFDPARIAAIYKHYASAISVLTDEKYFQGSFNFLPIVSQIAPQPILCKDFIIDPYQIYLARYYQADACLLMLSVLDDDQYRQLAAVAHSLEMGVLTEVSNEEEQERAIALGAKVVGINNRDLRDLSIDLNRTRELAPKLGHNVTVISESGINTYAQVRELSHFANGFLIGSALMAHDDLHAAVRRVLLGENKVCGLTRGQDAKAAYDAGAIYGGLIFVATSPRCVNVEQAQEVMAAAPLQYVGVFRNHDIADVVDKAKVLSLAAVQLHGNEEQLYIDTLREALPAHVAIWKALSVGETLPAREFQHVDKYVLDNGQGGSGQRFDWSLLNGQSLGNVLLAGGLGADNCVEAAQTGCAGLDFNSAVESQPGIKDARLLASVFQTLRAY"]

                cls = biolmai.ProteInferEC()
                resp = cls.generate(seqs)


    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            "Content-Type" = "application/json",
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN'))
            )
            payload = "{
                \"items\": [
                    {
                        \"sequence\": \"MMQTVLAKIVADKAIWVEARKQQQPLASFQNEVQPSTRHFYDALQGARTAFILECKKASPSKGVIRDDFDPARIAAIYKHYASAISVLTDEKYFQGSFNFLPIVSQIAPQPILCKDFIIDPYQIYLARYYQADACLLMLSVLDDDQYRQLAAVAHSLEMGVLTEVSNEEEQERAIALGAKVVGINNRDLRDLSIDLNRTRELAPKLGHNVTVISESGINTYAQVRELSHFANGFLIGSALMAHDDLHAAVRRVLLGENKVCGLTRGQDAKAAYDAGAIYGGLIFVATSPRCVNVEQAQEVMAAAPLQYVGVFRNHDIADVVDKAKVLSLAAVQLHGNEEQLYIDTLREALPAHVAIWKALSVGETLPAREFQHVDKYVLDNGQGGSGQRFDWSLLNGQSLGNVLLAGGLGADNCVEAAQTGCAGLDFNSAVESQPGIKDARLLASVFQTLRAY\"
                    }
                ]
            }"
            res <- postForm("https://biolm.ai/api/v2/proteinfer-ec/predict/", .opts=list(postfields = payload, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)



^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

        {
            "results": [
                {
                    "sequence_id": "0",
                    "predictions": [
                        {
                            "label": "EC:4.-.-.-",
                            "confidence": 1.0,
                            "description": "Lyases."
                        },
                        {
                            "label": "EC:4.1.-.-",
                            "confidence": 1.0,
                            "description": "Carbon-carbon lyases."
                        },
                        {
                            "label": "EC:4.1.1.-",
                            "confidence": 1.0,
                            "description": "Carboxy-lyases."
                        }

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

predictions:
    This key holds a list of dictionaries, each containing a prediction result. Each item in the list represents a predicted Enzyme Commission (EC) number or Gene Ontology (GO) terms along with additional information related to the prediction.

sequence_id:
    Identifier for the input protein sequence for which the EC numbers are being predicted.

label:
    Represents the predicted EC number or GO term. EC numbers are used to classify enzymes and includes four levels of classification, each separated by a dot. ( "EC:3.-.-.-" and "EC:3.2.1.-" are examples of predicted EC numbers). "GO:0008150" and "GO:0003674" are examples of predicted GO term IDs.

confidence:
    This is a measure of the model's certainty or confidence in the predicted EC number, ranging from 0 to 1, with higher values indicating higher confidence.

description:
    This provides a textual description or annotation related to the predicted EC number or GO term. For EC number this gives some  information about the type of reaction the enzyme catalyzes. FOr GO term Textual description or name of the predicted GO term descriptions like "biological_process" and "molecular_function" provide a brief understanding of what each GO term represents in biological terminology.

----------
Related
----------

:doc:`/model-docs/proteinfer/ProteInfer_Additional`

