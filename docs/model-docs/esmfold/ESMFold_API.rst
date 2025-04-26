=================
ESMFold API
=================

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Article Information
    :date: Oct 24, 2023
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*This page explains the use of ESMFold, as well as documents
its usage on BioLM for protein structure prediction.*

---------
API Usage
---------
here are 2 BioLM endpoints to ESMFold single and multi chain.

* ESMFold single chain is at ```https://biolm.ai/api/v2/esmfold-singlechain/predict/ <https://api.biolm.ai/#f835034e-f0cf-46c4-b74e-8283993063f9>```_.
* ESMFold multi chain is at ```https://biolm.ai/api/v2/esmfold-multichain/predict/ <https://api.biolm.ai/#756b174f-e306-4ff7-ba6d-973fef3f6714>```_.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

Using ESMFold single chain as an example, folding requests to the endpoints can be made with
in several ways including 'python requests' or the biolmai SDK

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v2/esmfold-singlechain/predict/' \
              --header 'Cookie: access=MY_ACCESS_TOKEN;refresh=MY_REFRESH_TOKEN' \
              --header 'Content-Type: application/json' \
              --data '{
                    "items": [
                        {
                            "sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"
                        }
                    ]
                }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v2/esmfold-singlechain/predict/"

            payload = json.dumps({
                    "items": [
                        {
                            "sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"
                        }
                    ]
                })
            headers = {
              'Cookie': 'access=MY_ACCESS_TOKEN;refresh=MY_REFRESH_TOKEN',
              'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            seqs = ["MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"]

            cls = biolmai.ESMFoldSingleChain()
            resp = cls.predict(seqs)

    .. tab-item:: R
        :sync: r

        .. code:: shell

            library(RCurl)
            headers = c(
              "Cookie" = "access=MY_ACCESS_TOKEN;refresh=MY_REFRESH_TOKEN",
              "Content-Type" = "application/json"
            )
            payload = "{
              \"items\": [
                {
                  \"sequence\" \"MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ\"
                }
              ]
            }"
            res <- postForm("https://biolm.ai/api/v2/esmfold-singlechain/predict/", .opts=list(postfields = payload, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)

^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

    {
        "results": [
            {
                "pdb": "PARENT N/A\nATOM      1  N   MET A   1      -3.717 -20.294 -18.979  1.00 87.61           N  \nATOM      2  CA  MET A   1
                "mean_plddt": 94.2749252319336,
                "ptm": 0.9202359914779663
                }
              ]
    }

^^^^^^^^^^^^^^^^^^^^
 Request Definitions
^^^^^^^^^^^^^^^^^^^^

items:
   Inside items is a list of dictionaries with each dictionary corresponding to one model input.
    sequence:
        The input sequence for the model


^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^

results:
   This is the main key in the JSON object that contains an array of model results. Each element in the array represents a set of predictions for one input instance.

pdb:
  Contains a string representing the 3D structure of the protein predicted by the model in PDB (Protein Data Bank) format.

mean_plddt:
  Contains a string representing the mean pLDDT score of the predicted structure. The pLDDT (predicted Local Distance Difference Test) score is a measure of the accuracy of the predicted structure, with values ranging from 0 to 100. Higher scores indicate higher confidence in the prediction.




.. note::
   This graph will be available soon.

   The duration for folding predominantly depends on sequence length. A sequence of length 60 might fold in 6 seconds, however a sequence of
   length 500 might fold in 400 seconds.

-------
Related
-------
:doc:`/model-docs/esmfold/ESMFold_Additional`

:doc:`/model-docs/esmif/index`

:doc:`/model-docs/esm2/index`

:doc:`/model-docs/esm1v/index`

