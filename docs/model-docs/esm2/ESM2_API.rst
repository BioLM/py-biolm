================
ESM-2 API
================

.. article-info::
    :avatar: ../img/book_icon.png
    :date: Oct 18, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*This page explains the use of ESM-2 for generating embeddings, contacts, attentions, and predicting logits and how
to access these capabilities with the the BioLM API.*

---------------------------
Endpoints
---------------------------

There are 5 BioLM endpoints corresponding to 5 different sized ESM-2 models.
These model sizes, with M representing millions of parameters and B representing billions of
parameters, are:

* esm2_t6_8M_UR50D, 6 layers and 8M parameters, endpoint at ```https://biolm.ai/api/v2/esm2-8m/<model_action>/ <https://api.biolm.ai/#571be64b-f9f4-4303-8ff8-6abc67abb80c>```_
* esm2_t12_35M_UR50D, 12 layers and 35M parameters, endpoint at ```https://biolm.ai/api/v2/esm2-35m/<model_action>/ <https://api.biolm.ai/#a4355a60-93d4-43b5-a2d6-83519065b225>```_
* esm2_t30_150M_UR50D, 30 layers and 150M parameters, endpoint at  ```https://biolm.ai/api/v2/esm2-150m/<model_action>/ <https://api.biolm.ai/#7afcd793-f9a9-4dab-b931-340648531130>```_
* esm2_t33_650M_UR50D,  33 layers and 650M parameters, endpoint at ```https://biolm.ai/api/v2/esm2-650m/<model_action>/ <https://api.biolm.ai/#48318bdc-ff48-47ac-a464-f67fdec2e20b>```_
* esm2_t36_3B_UR50D, 36 layers and 3B parameters, endpoint at ```https://biolm.ai/api/v2/esm2-3B/<model_action>/ <https://api.biolm.ai/#31743328-cbef-49eb-8650-26fd9d9bc43f>```_



---------------------------
Embedding API Usage
---------------------------

The encode action produces embeddings, contacts, attention maps and logits.
Appending 'encode/' to model endpoints above gives access to these outputs.

Using the 650M model as an example, the encode endpoint is ```https://biolm.ai/api/v2/esm2-650m/encode/ <https://api.biolm.ai/#daa50ec7-0da2-4bff-ab4c-3ead7f377154>```_

^^^^^^^^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v2/esm2-650m/encode/' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{
                "params": {
                    "include": [
                        "mean",
                        "contacts",
                        "logits",
                        "attentions"
                    ]
                },
                "items": [
                    {
                        "sequence": "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCRYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG"
                    }
                ]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v2/esm2-650m/encode/"

            payload = json.dumps({
                "params": {
                    "include": [
                        "mean",
                        "contacts",
                        "logits",
                        "attentions"
                    ]
                },
                "items": [
                    {
                        "sequence": "MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCRYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG"
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
            seqs = ["MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCRYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG"]

            cls = biolmai.ESM2_650M()
            resp = cls.encode(seqs, params={
                    "include": [
                        "mean",
                        "contacts",
                        "logits",
                        "attentions"
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
            payload = "{
                \"params\": {
                    \"include\": [
                        \"mean\",
                        \"contacts\",
                        \"logits\",
                        \"attentions\"
                    ]
                },
                \"items\": [
                    {
                        \"sequence\": \"MAETAVINHKKRKNSPRIVQSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCRYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG\"
                    }
                ]
            }"
            res <- postForm("https://biolm.ai/api/v2/esm2-650m/encode/", .opts=list(postfields = payload, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)


^^^^^^^^^^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

         {
            "results": [
                {
                    "sequence_index": 0,
                    "mean_representations": {
                        "33": [
                            0.005844539031386375,
                            -0.00489774439483881,
                            -0.007498568389564753,
                    "contacts": [
                                    [
                                        0.004600186832249165,
                                        0.5025275349617004,
                                        0.023159209638834,
                    "logits": [
                                    [
                                        -0.8352559208869934,
                                        -0.3333878219127655,
                                        -1.3698017597198486,
                    "attentions": [
                                    [
                                        0.00449674716219306,
                                        0.003284697188064456,
                                        0.003496115328744054,




.. note::
  The above response is only a small snippet of the full JSON response. For every item in include there is a corresponding field for each dictionary in results. Each of these dictionaries corresponds to one of the items submitted

^^^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^^^

items:
   Inside items are a list of dictionaries with each dictionary corresponding to one model input.
sequence:
    The input sequence for the model
params:
    These are additional parameters for the endpoint that are used with every input in items. By
    default the ESM-2 encode endpoints will only return the extracted mean ESM-2 embeddings for the last layer of the model,
    modifying params allows other outputs such as contacts to be returned or different representative layers for the embeddings to be selected.
repr_layers:
    This parameter specifies the representative layer of the ESM-2 model that embeddings are extracted from.
    If unspecified it defaults to [-1] and returns embeddings/representations for that layer (-1 indexes the last layer, -2 the second to last).
include:
    For the encode endpoint, the include param in params specifies what outputs to include in the response.
    These could be any of 'logits', 'attentions', 'contacts', 'per_token', 'bos', or 'mean'.
    'per_token', 'bos', and 'mean' are types of embeddings. 'per_token' returns the entire model hidden states for each token at the representative layer(s).
    (this can be specified with repr_layers).
    These full representations can be used for additional kinds of pooling such as min or max pooling.
    'bos' returns the hidden states for the 'bos' (beginning of sequence) token at the representative layer(s)
    'mean' is the average of the 'per_token' representations at the representative layer(s). 'mean' is the default option if include is unspecified.

^^^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^^^

results:
   This is the main key in the JSON object that contains an array of model results. Each element in the array represents a set of predictions for one input instance.

mean_representations:
   This key holds the embeddings generated by the ESM-2 model for the corresponding input sequence. These embeddings represent average values computed over certain dimensions of the model's output.

representations:
   This key holds the entire per token hidden states generated by the ESM-2 model for the corresponding input sequence.

bos_representations:
   This key holds the embeddings for the 'bos' (beginning of sequence) tokens generated by the ESM-2 model for the corresponding input sequence.

33:
   The layer numbers corresponding to the selected representative layers in the request are sub keys under the different representations.
    These keys hold the corresponding embeddings for that specific layer. This is different for each model size, ESM-2 8M only has 6 layers while ESM-2 650M has 33.
    If using the ESM-2 8M endpoint, this subkey would never exceed 6.

logits:
    This key contains the model logits for each token in the input sequence. The returned values are of size Length of Sequence X 20 (the number of natural amino acids)

attentions:
    This key corresponds to the computed attentions over each layer of the model corresponding to the input sequence. These attentions are of size Number of Layers X Sequence Length

contacts:
    This key contains the predicted contacts (residues that are close together in structural space) for the input sequence. These contacts are of shape Length of Sequence X Length of Sequence

---------------------------
Prediction API Usage
---------------------------

The predict action returns model computed logits from masked sequences (one
or more amino acids are masked and unknown to the model)
Appending 'predict/' to model endpoints above gives access to these outputs.

Using the 650M model as an example, the predict endpoint is ```https://biolm.ai/api/v2/esm2-650m/predict/ <https://api.biolm.ai/#f26afcaa-1745-486b-99eb-89e752f7dba1>```_.

^^^^^^^^^^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v2/esm2-650m/predict/' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --header 'Content-Type: application/json' \
            --data '{
                "items": [
                    {
                        "sequence": "MAETAVINHKKRKNSPRI<mask>QSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCQYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG"
                    }
                ]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v2/esm2-650m/predict/"

            payload = json.dumps({
                "params": {
                    "include": [
                        "mean",
                        "logits",
                        "attentions"
                    ]
                },
                "items": [
                    {
                        "sequence": "MAETAVINHKKRKNSPRI<mask>QSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCQYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG"
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
            seqs = ["MAETAVINHKKRKNSPRI<mask>QSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCQYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG"]

            cls = biolmai.ESM2_650M()
            resp = cls.predict(seqs)

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
                        \"sequence\": \"MAETAVINHKKRKNSPRI<mask>QSNDLTEAAYSLSRDQKRMLYLFVDQIRKSDGTLQEHDGICEIHVAKYAEIFGLTSAEASKDIRQALKSFAGKEVVFYRPEEDAGDEKGYESFPWFIKRAHSPSRGLYSVHINPYLIPFFIGLQNRFTQFRLSETKEITNPYAMRLYESLCQYRKPDGSGIVSLKIDWIIERYQLPQSYQRMPDFRRRFLQVCVNEINSRTPMRLSYIEKKKGRQTTHIVFSFRDITSMTTG\"
                    }
                ]
            }"
            res <- postForm("https://biolm.ai/api/v2/esm2-650m/predict/", .opts=list(postfields = payload, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)

^^^^^^^^^^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

         {
            "results": [
                {
                    "logits": [
                        [
                            -0.8320964574813843,
                            -0.3259419798851013,
                            -1.3772594928741455,
                    "sequence_tokens": [
                                    "M",
                                    "A",
                                    "E",
                                    "T",
                                    "A",
                                    "V",
                                    "I",
                                    "N",
                                    "H",
                                    "K",
                                    "K",
                                    "R",
                                    "K",
                                    "N",
                                    "S",
                                    "P",
                                    "R",
                                    "I",
                                    "<mask>",
                                    "Q",

                    "alphabet_tokens": [
                                    "L",
                                    "A",
                                    "G",
                                    "V",
                                    "S",
                                    "E",
                                    "R",
                                    "T",
                                    "I",
                                    "D",
                                    "P",
                                    "K",
                                    "Q",
                                    "N",
                                    "F",
                                    "Y",
                                    "M",
                                    "H",
                                    "W",
                                    "C"]



.. note::
  The above response is only small snippets of the full JSON response. Each of these dictionaries corresponds to one of the items submitted

^^^^^^^^^^^^^^^^^^^^^^
Request Definitions
^^^^^^^^^^^^^^^^^^^^^^

items:
   Inside items are a list of dictionaries with each dictionary corresponding to one model input.
sequence:
    The input sequence for the model

^^^^^^^^^^^^^^^^^^^^^^
Response Definitions
^^^^^^^^^^^^^^^^^^^^^^

results:
   This is the main key in the JSON object that contains an array of model results. Each element in the array represents a set of predictions for one input instance.

logits:
   This key contains the models output logits for each position in the input sequence. There are 20 logits for each position corresponding to the 20 natural amino acids. These logits can be mapped to the models confidence in which of the 20 natural amino acids should be at that specif position. In the case of the mask token, these logits give the models prediction for which token most likely occupies the masked position. The logits are of size Length of Sequence X 20
sequence_tokens:
    Contains the tokens of the input sequence. Size Length of Sequence
alphabet_tokens:
    the 20 amino acids corresponding to the 20 output logits for each position in the sequence

-------
Related
-------
:doc:`/model-docs/esm2/ESM2_Additional`

:doc:`/model-docs/esmif/index`

:doc:`/model-docs/esmfold/index`

:doc:`/model-docs/esm1v/index`

.. _Status Page: https://status.biolm.ai






