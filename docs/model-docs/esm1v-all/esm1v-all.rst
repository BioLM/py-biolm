ESM-1v API
==========

ESM-1v is a GPU-accelerated protein language model for zero-shot inference of mutational effects on protein function, predicting variant impact directly from amino acid sequences without task-specific training or MSAs. It returns log-odds variant scores reflecting functional constraints learned from 98 million evolutionary sequences. ESM-1v supports high-throughput variant evaluation, protein engineering, antibody optimization, and enzyme design workflows, providing predictions correlated with deep mutational scanning experiments.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ESM-1v.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="esm1v-all",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esm1v-all/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esm1v-all/predictor/"

      headers = {

          "Authorization": "Token YOUR_API_KEY",

          "Content-Type": "application/json"

      }

      data = {}
      
      response = requests.post(url, headers=headers, json=data)

      result = response.json()

      print(result)

    .. tab-item:: R
        :sync: r

        .. code:: r


      library(httr2)

      library(jsonlite)
      
      url <- "https://biolm.ai/api/v3/esm1v-all/predictor/"

      data <- list()
      

      response <- request(url) %>%

        req_headers(

          Authorization = "Token YOUR_API_KEY",

          `Content-Type` = "application/json"

        ) %>%

        req_body_json(data) %>%

        req_perform()
      

      result <- response %>% resp_body_json()

      print(result)

.. http:post:: /api/v3/esm1v-all/predict/

   Predict endpoint for ESM-1v.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **model_number** (*string*, enum: ["n1", "n2", "n3", "n4", "n5", "all"], default: "all") — Identifier specifying which ESM1v model variant to use for prediction
      
      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 512, required) — Protein sequence containing exactly one "<mask>" token; valid characters include standard amino acids, extended amino acids, and the special token "<mask>"

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm1v-all/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

      {}

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response JSON Object

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:
      
        - **esm1v-n[1-5]** (*array of objects*) — Predictions from individual ESM-1v model variants (n1 through n5):
      
          - **token** (*int*) — Amino acid token index (0-32, inclusive)
      
          - **token_str** (*string*) — Amino acid single-letter code (20 standard amino acids, plus special tokens)
      
          - **score** (*float*, range: 0.0-1.0) — Predicted probability score for the amino acid at the masked position
      
          - **sequence** (*string*, length: 1-512 amino acids) — Protein sequence with the predicted amino acid replacing the "<mask>" token
      
        - **esm1v-n[1-5]** (*array length: up to 33*) — One prediction per possible amino acid substitution at the masked position, sorted by descending probability score
      
        - **esm1v-n[1-5]** (*object keys*) — Exactly one key per model variant ("esm1v-n1", "esm1v-n2", "esm1v-n3", "esm1v-n4", "esm1v-n5")
      
        - **esm1v-n[1-5]** (*array dimensions*) — [num_predictions, 4], where num_predictions ≤ 33 (20 standard amino acids + special tokens)
      
        - **esm1v-n[1-5]** (*score range*) — Probability scores normalized across all possible amino acids at masked position (sum to 1.0)
      
        - **esm1v-n[1-5]** (*sequence length*) — Matches input sequence length (1-512 amino acids), with "<mask>" replaced by predicted amino acid
      
        - **esm1v-n[1-5]** (*token indices*) — Integer encoding of amino acids: 0-19 standard amino acids, 20-32 special tokens (including gap, mask, padding)
      
        - **esm1v-n[1-5]** (*token_str values*) — Single-letter amino acid codes ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y") and special tokens ("<mask>", "<pad>", "<gap>", etc.)
      
        - **esm1v-n[1-5]** (*sequence constraints*) — Exactly one "<mask>" token replaced per sequence
      
        - **esm1v-n[1-5]** (*score precision*) — Floating-point precision (float32)
      
        - **esm1v-n[1-5]** (*token_str constraints*) — Single-character strings for standard amino acids, multi-character strings for special tokens ("<mask>", "<pad>", "<gap>", etc.)
      
        - **esm1v-n[1-5]** (*token constraints*) — Integer values correspond exactly to ESM-1v vocabulary indices (0-32 inclusive)
      
        - **esm1v-n[1-5]** (*sequence format*) — Protein sequence represented as a single continuous string of amino acid characters (no whitespace or delimiters)
      
        - **esm1v-n[1-5]** (*sequence length constraints*) — Matches input sequence length exactly (1-512 amino acids)
      
        - **esm1v-n[1-5]** (*prediction ordering*) — Predictions ordered by descending probability score (highest confidence first)
      
        - **esm1v-n[1-5]** (*probability distribution*) — Scores represent normalized probability distribution at masked position (sum exactly to 1.0)
      
        - **esm1v-n[1-5]** (*special tokens*) — Special tokens ("<mask>", "<pad>", "<gap>", etc.) included in predictions with associated probability scores
      
        - **esm1v-n[1-5]** (*model variant*) — Predictions provided separately for each model variant (n1, n2, n3, n4, n5)
      
        - **esm1v-n[1-5]** (*ensemble predictions*) — When "all" variant selected, predictions from all five model variants provided separately under respective keys ("esm1v-n1" through "esm1v-n5")

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "esm1v-n1": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.09930282086133957,
          "sequence": "A C D G L H I K L M N"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.07655816525220871,
          "sequence": "A C D G I H I K L M N"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.07052073627710342,
          "sequence": "A C D G S H I K L M N"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.06859572976827621,
          "sequence": "A C D G K H I K L M N"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.0646892637014389,
          "sequence": "A C D G V H I K L M N"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.06145920976996422,
          "sequence": "A C D G F H I K L M N"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.05439462140202522,
          "sequence": "A C D G N H I K L M N"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.05385798588395119,
          "sequence": "A C D G G H I K L M N"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.05338464304804802,
          "sequence": "A C D G R H I K L M N"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.04514595866203308,
          "sequence": "A C D G T H I K L M N"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.04505712911486626,
          "sequence": "A C D G C H I K L M N"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.04452083259820938,
          "sequence": "A C D G E H I K L M N"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.043652020394802094,
          "sequence": "A C D G A H I K L M N"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.041521020233631134,
          "sequence": "A C D G D H I K L M N"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.04042752832174301,
          "sequence": "A C D G Y H I K L M N"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.036572158336639404,
          "sequence": "A C D G Q H I K L M N"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.03288909047842026,
          "sequence": "A C D G H H I K L M N"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.025613322854042053,
          "sequence": "A C D G M H I K L M N"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.024750934913754463,
          "sequence": "A C D G P H I K L M N"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.0163724422454834,
          "sequence": "A C D G W H I K L M N"
        }
      ],
      "esm1v-n2": [
        {
          "token": 8,
          "token_str": "S",
          "score": 0.0869770348072052,
          "sequence": "A C D G S H I K L M N"
        },
        {
          "token": 4,
          "token_str": "L",
          "score": 0.08455677330493927,
          "sequence": "A C D G L H I K L M N"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.06859893351793289,
          "sequence": "A C D G K H I K L M N"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.06776954978704453,
          "sequence": "A C D G V H I K L M N"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.0648026317358017,
          "sequence": "A C D G T H I K L M N"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.06313733756542206,
          "sequence": "A C D G I H I K L M N"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.05539366602897644,
          "sequence": "A C D G R H I K L M N"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.05390600487589836,
          "sequence": "A C D G G H I K L M N"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.0532502681016922,
          "sequence": "A C D G E H I K L M N"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.05202822759747505,
          "sequence": "A C D G A H I K L M N"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.05071841925382614,
          "sequence": "A C D G D H I K L M N"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.04738819599151611,
          "sequence": "A C D G N H I K L M N"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.046372637152671814,
          "sequence": "A C D G F H I K L M N"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.037274036556482315,
          "sequence": "A C D G C H I K L M N"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.03686069697141647,
          "sequence": "A C D G Y H I K L M N"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.032883867621421814,
          "sequence": "A C D G Q H I K L M N"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.0328129343688488,
          "sequence": "A C D G H H I K L M N"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.03094547800719738,
          "sequence": "A C D G P H I K L M N"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.019779501482844353,
          "sequence": "A C D G M H I K L M N"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.013496983796358109,
          "sequence": "A C D G W H I K L M N"
        }
      ],
      "esm1v-n4": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.09174387156963348,
          "sequence": "A C D G L H I K L M N"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.07290017604827881,
          "sequence": "A C D G S H I K L M N"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.06795021146535873,
          "sequence": "A C D G I H I K L M N"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.06504393368959427,
          "sequence": "A C D G K H I K L M N"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.06306225061416626,
          "sequence": "A C D G V H I K L M N"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.0549919493496418,
          "sequence": "A C D G F H I K L M N"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.054266273975372314,
          "sequence": "A C D G G H I K L M N"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.05355978384613991,
          "sequence": "A C D G N H I K L M N"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.051960933953523636,
          "sequence": "A C D G C H I K L M N"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.05111481994390488,
          "sequence": "A C D G A H I K L M N"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.05029285326600075,
          "sequence": "A C D G R H I K L M N"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.0497501865029335,
          "sequence": "A C D G E H I K L M N"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.04764137044548988,
          "sequence": "A C D G T H I K L M N"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.04649464413523674,
          "sequence": "A C D G D H I K L M N"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.040316879749298096,
          "sequence": "A C D G H H I K L M N"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.03982219099998474,
          "sequence": "A C D G Y H I K L M N"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.0342683270573616,
          "sequence": "A C D G Q H I K L M N"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.027182502672076225,
          "sequence": "A C D G P H I K L M N"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.02308276854455471,
          "sequence": "A C D G M H I K L M N"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.013834582641720772,
          "sequence": "A C D G W H I K L M N"
        }
      ],
      "esm1v-n5": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.09874340891838074,
          "sequence": "A C D G L H I K L M N"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.07558094710111618,
          "sequence": "A C D G S H I K L M N"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.07126651704311371,
          "sequence": "A C D G I H I K L M N"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.06954080611467361,
          "sequence": "A C D G K H I K L M N"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.06663636118173599,
          "sequence": "A C D G V H I K L M N"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.055239368230104446,
          "sequence": "A C D G F H I K L M N"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.05484839528799057,
          "sequence": "A C D G R H I K L M N"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.053184155374765396,
          "sequence": "A C D G G H I K L M N"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.05232826620340347,
          "sequence": "A C D G T H I K L M N"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.05214473605155945,
          "sequence": "A C D G N H I K L M N"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.04786435887217522,
          "sequence": "A C D G A H I K L M N"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.04770158231258392,
          "sequence": "A C D G E H I K L M N"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.0428353026509285,
          "sequence": "A C D G D H I K L M N"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.03945346176624298,
          "sequence": "A C D G Q H I K L M N"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.036879561841487885,
          "sequence": "A C D G Y H I K L M N"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.03288180008530617,
          "sequence": "A C D G C H I K L M N"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.03153132274746895,
          "sequence": "A C D G H H I K L M N"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.029899941757321358,
          "sequence": "A C D G P H I K L M N"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.024092454463243484,
          "sequence": "A C D G M H I K L M N"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.0159302968531847,
          "sequence": "A C D G W H I K L M N"
        }
      ],
      "esm1v-n3": [
        {
          "token": 6,
          "token_str": "G",
          "score": 0.10201912373304367,
          "sequence": "A C D G G H I K L M N"
        },
        {
          "token": 4,
          "token_str": "L",
          "score": 0.08753731101751328,
          "sequence": "A C D G L H I K L M N"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.07383615523576736,
          "sequence": "A C D G K H I K L M N"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.06900347024202347,
          "sequence": "A C D G S H I K L M N"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.06557872891426086,
          "sequence": "A C D G I H I K L M N"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.05982591584324837,
          "sequence": "A C D G V H I K L M N"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.053227655589580536,
          "sequence": "A C D G N H I K L M N"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.053097281605005264,
          "sequence": "A C D G E H I K L M N"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.04920322448015213,
          "sequence": "A C D G F H I K L M N"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.047672368586063385,
          "sequence": "A C D G R H I K L M N"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.04733869060873985,
          "sequence": "A C D G D H I K L M N"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.04422014579176903,
          "sequence": "A C D G T H I K L M N"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.043792881071567535,
          "sequence": "A C D G A H I K L M N"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.0381644181907177,
          "sequence": "A C D G C H I K L M N"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.037753913551568985,
          "sequence": "A C D G Y H I K L M N"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.03307907655835152,
          "sequence": "A C D G Q H I K L M N"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.03011476993560791,
          "sequence": "A C D G H H I K L M N"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.024792706593871117,
          "sequence": "A C D G P H I K L M N"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.022936400026082993,
          "sequence": "A C D G M H I K L M N"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.01500384695827961,
          "sequence": "A C D G W H I K L M N"
        }
      ]
    },
    {
      "esm1v-n1": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.10145029425621033,
          "sequence": "X P Q R S L F G T"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.09379883855581284,
          "sequence": "X P Q R S S F G T"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.07906772196292877,
          "sequence": "X P Q R S R F G T"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.07735856622457504,
          "sequence": "X P Q R S G F G T"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.0653463751077652,
          "sequence": "X P Q R S A F G T"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.0629093199968338,
          "sequence": "X P Q R S P F G T"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.06283947825431824,
          "sequence": "X P Q R S V F G T"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.056243106722831726,
          "sequence": "X P Q R S T F G T"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.04761470854282379,
          "sequence": "X P Q R S I F G T"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.04633283242583275,
          "sequence": "X P Q R S F F G T"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.038117945194244385,
          "sequence": "X P Q R S K F G T"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.035761695355176926,
          "sequence": "X P Q R S E F G T"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.03405534848570824,
          "sequence": "X P Q R S C F G T"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.03339006006717682,
          "sequence": "X P Q R S D F G T"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.032739341259002686,
          "sequence": "X P Q R S Q F G T"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.031960997730493546,
          "sequence": "X P Q R S N F G T"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.02955177240073681,
          "sequence": "X P Q R S H F G T"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.026163289323449135,
          "sequence": "X P Q R S Y F G T"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.019282586872577667,
          "sequence": "X P Q R S W F G T"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.016610007733106613,
          "sequence": "X P Q R S M F G T"
        }
      ],
      "esm1v-n2": [
        {
          "token": 8,
          "token_str": "S",
          "score": 0.0989096611738205,
          "sequence": "X P Q R S S F G T"
        },
        {
          "token": 4,
          "token_str": "L",
          "score": 0.09306105226278305,
          "sequence": "X P Q R S L F G T"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.08760379999876022,
          "sequence": "X P Q R S R F G T"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.07870138436555862,
          "sequence": "X P Q R S G F G T"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.06757384538650513,
          "sequence": "X P Q R S P F G T"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.06717503815889359,
          "sequence": "X P Q R S A F G T"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.06286092102527618,
          "sequence": "X P Q R S T F G T"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.05829164758324623,
          "sequence": "X P Q R S V F G T"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.04884663224220276,
          "sequence": "X P Q R S I F G T"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.04315071180462837,
          "sequence": "X P Q R S F F G T"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.03545709326863289,
          "sequence": "X P Q R S D F G T"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.034110330045223236,
          "sequence": "X P Q R S K F G T"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.03255288302898407,
          "sequence": "X P Q R S N F G T"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.0321805477142334,
          "sequence": "X P Q R S C F G T"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.031518179923295975,
          "sequence": "X P Q R S Q F G T"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.03113524802029133,
          "sequence": "X P Q R S H F G T"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.031019393354654312,
          "sequence": "X P Q R S E F G T"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.027974633499979973,
          "sequence": "X P Q R S Y F G T"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.01858220063149929,
          "sequence": "X P Q R S W F G T"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.014232822693884373,
          "sequence": "X P Q R S M F G T"
        }
      ],
      "esm1v-n4": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.3064996600151062,
          "sequence": "X P Q R S L F G T"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.11773010343313217,
          "sequence": "X P Q R S I F G T"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.05838964506983757,
          "sequence": "X P Q R S V F G T"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.055545445531606674,
          "sequence": "X P Q R S G F G T"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.053945016115903854,
          "sequence": "X P Q R S S F G T"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.04092787578701973,
          "sequence": "X P Q R S P F G T"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.03917316347360611,
          "sequence": "X P Q R S F F G T"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.0378505140542984,
          "sequence": "X P Q R S R F G T"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.036431681364774704,
          "sequence": "X P Q R S T F G T"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.03578076511621475,
          "sequence": "X P Q R S N F G T"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.03526788577437401,
          "sequence": "X P Q R S A F G T"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.026970142498612404,
          "sequence": "X P Q R S Q F G T"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.025123395025730133,
          "sequence": "X P Q R S K F G T"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.023874927312135696,
          "sequence": "X P Q R S M F G T"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.02182905003428459,
          "sequence": "X P Q R S D F G T"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.018361248075962067,
          "sequence": "X P Q R S H F G T"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.01753968931734562,
          "sequence": "X P Q R S E F G T"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.01618822105228901,
          "sequence": "X P Q R S C F G T"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.015631720423698425,
          "sequence": "X P Q R S Y F G T"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.006293799262493849,
          "sequence": "X P Q R S W F G T"
        }
      ],
      "esm1v-n5": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.6074760556221008,
          "sequence": "X P Q R S L F G T"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.05280425772070885,
          "sequence": "X P Q R S F F G T"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.044951871037483215,
          "sequence": "X P Q R S I F G T"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.040224019438028336,
          "sequence": "X P Q R S P F G T"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.027877938002347946,
          "sequence": "X P Q R S V F G T"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.027868865057826042,
          "sequence": "X P Q R S S F G T"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.025413621217012405,
          "sequence": "X P Q R S T F G T"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.02375607192516327,
          "sequence": "X P Q R S R F G T"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.01721598580479622,
          "sequence": "X P Q R S M F G T"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.01636522077023983,
          "sequence": "X P Q R S G F G T"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.014775858260691166,
          "sequence": "X P Q R S A F G T"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.011473353952169418,
          "sequence": "X P Q R S C F G T"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.008070611394941807,
          "sequence": "X P Q R S W F G T"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.007745381910353899,
          "sequence": "X P Q R S Y F G T"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.0070585887879133224,
          "sequence": "X P Q R S Q F G T"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.006492833141237497,
          "sequence": "X P Q R S H F G T"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.006362810265272856,
          "sequence": "X P Q R S K F G T"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.005260193254798651,
          "sequence": "X P Q R S N F G T"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.004203920252621174,
          "sequence": "X P Q R S E F G T"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.0038706620689481497,
          "sequence": "X P Q R S D F G T"
        }
      ],
      "esm1v-n3": [
        {
          "token": 4,
          "token_str": "L",
          "score": 0.10762723535299301,
          "sequence": "X P Q R S L F G T"
        },
        {
          "token": 6,
          "token_str": "G",
          "score": 0.10102758556604385,
          "sequence": "X P Q R S G F G T"
        },
        {
          "token": 8,
          "token_str": "S",
          "score": 0.09847157448530197,
          "sequence": "X P Q R S S F G T"
        },
        {
          "token": 10,
          "token_str": "R",
          "score": 0.07277119159698486,
          "sequence": "X P Q R S R F G T"
        },
        {
          "token": 14,
          "token_str": "P",
          "score": 0.07179819792509079,
          "sequence": "X P Q R S P F G T"
        },
        {
          "token": 5,
          "token_str": "A",
          "score": 0.06924095749855042,
          "sequence": "X P Q R S A F G T"
        },
        {
          "token": 7,
          "token_str": "V",
          "score": 0.059523701667785645,
          "sequence": "X P Q R S V F G T"
        },
        {
          "token": 11,
          "token_str": "T",
          "score": 0.05258268117904663,
          "sequence": "X P Q R S T F G T"
        },
        {
          "token": 18,
          "token_str": "F",
          "score": 0.04812784492969513,
          "sequence": "X P Q R S F F G T"
        },
        {
          "token": 12,
          "token_str": "I",
          "score": 0.04757078364491463,
          "sequence": "X P Q R S I F G T"
        },
        {
          "token": 13,
          "token_str": "D",
          "score": 0.03384622186422348,
          "sequence": "X P Q R S D F G T"
        },
        {
          "token": 9,
          "token_str": "E",
          "score": 0.03186787664890289,
          "sequence": "X P Q R S E F G T"
        },
        {
          "token": 16,
          "token_str": "Q",
          "score": 0.03024449571967125,
          "sequence": "X P Q R S Q F G T"
        },
        {
          "token": 15,
          "token_str": "K",
          "score": 0.03006196767091751,
          "sequence": "X P Q R S K F G T"
        },
        {
          "token": 17,
          "token_str": "N",
          "score": 0.02946416847407818,
          "sequence": "X P Q R S N F G T"
        },
        {
          "token": 21,
          "token_str": "H",
          "score": 0.026086673140525818,
          "sequence": "X P Q R S H F G T"
        },
        {
          "token": 23,
          "token_str": "C",
          "score": 0.02601403370499611,
          "sequence": "X P Q R S C F G T"
        },
        {
          "token": 19,
          "token_str": "Y",
          "score": 0.024845046922564507,
          "sequence": "X P Q R S Y F G T"
        },
        {
          "token": 22,
          "token_str": "W",
          "score": 0.017783671617507935,
          "sequence": "X P Q R S W F G T"
        },
        {
          "token": 20,
          "token_str": "M",
          "score": 0.015471836552023888,
          "sequence": "X P Q R S M F G T"
        }
      ]
    }
  ]
}


Performance
-----------

- GPU-accelerated inference using NVIDIA T4 GPUs, enabling rapid, scalable predictions.
- Typical inference latency for single-sequence mutation effect prediction: approximately 2-4 seconds per sequence.
- Zero-shot mutation effect prediction accuracy (Spearman ρ = 0.51) comparable to state-of-the-art unsupervised methods (e.g., DeepSequence ρ = 0.52, EVMutation ρ = 0.50), without requiring task-specific model training or multiple sequence alignments (MSAs).
- Fine-tuning ESM-1v on protein-specific MSAs boosts predictive accuracy (Spearman ρ = 0.54), surpassing EVMutation and approaching DeepSequence, while still offering faster inference.
- ESM-1v significantly outperforms earlier protein language models provided by BioLM (e.g., ESM-1b Spearman ρ = 0.46, ProtBERT Spearman ρ = 0.43) in zero-shot mutational effect prediction tasks.
- Predictive performance is robust across diverse protein families, with ESM-1v outperforming DeepSequence on 17 out of 41 benchmarked deep mutational scanning datasets.
- Inference efficiency significantly exceeds models requiring explicit multiple sequence alignments (e.g., MSA Transformer), as ESM-1v requires only a single forward pass per sequence.
- BioLM's optimized deployment enables efficient parallel processing and GPU utilization, ensuring consistent inference speed even at scale.


Applications
------------

- Predicting functional impacts of protein mutations to accelerate protein engineering workflows, enabling rapid filtering and prioritization of candidate variants without costly experimental screening; valuable for biotech companies engineering industrial enzymes, therapeutic proteins, or agricultural proteins; limited accuracy for highly novel mutations lacking evolutionary context.
- Rapid computational screening of protein variants to identify stabilizing mutations or improve thermostability, enabling biotech firms to efficiently engineer proteins that function under harsh industrial conditions such as high temperature or extreme pH; less suitable for proteins with minimal evolutionary sequence data available.
- Identifying critical binding-site residues to guide targeted mutagenesis experiments, enabling protein engineers to focus experimental resources on residues most likely to impact ligand binding affinity or specificity; valuable for companies optimizing therapeutic proteins or biosensors; may not accurately predict impacts of mutations distant from conserved functional sites.
- Computational assessment of variant pathogenicity for clinical diagnostics, enabling genetic testing companies to rapidly classify novel protein-coding variants as benign or deleterious based on evolutionary constraints; useful for interpreting patient genomic data but less reliable for variants occurring in poorly conserved regions or proteins with limited evolutionary data.
- Predicting protein stability changes due to mutations to guide formulation and manufacturing decisions, enabling biopharma companies to proactively identify sequence variants that may negatively impact protein yield, shelf-life, or aggregation propensity; beneficial for biologics manufacturing but not optimal for assessing mutations affecting post-translational modifications or glycosylation patterns.


Limitations
-----------

- **Maximum Sequence Length**: Input sequences must be no longer than ``512`` amino acids. Longer sequences must be truncated or split into smaller segments for processing.
- **Batch Size**: Requests are limited to a maximum of ``5`` sequences per batch, which may impact throughput for large-scale analyses.
- **GPU Type**: GPU acceleration (``T4``) is only available when using the ensemble model (``all``). Individual models (``n1`` to ``n5``) run exclusively on CPU, potentially resulting in slower inference times.
- ESM-1v is optimized for zero-shot prediction of mutational effects on protein function. It does not directly predict protein structure or provide embeddings for clustering or visualization tasks.
- While ESM-1v performs well for initial screening and ranking of large libraries of protein variants, it may not be optimal for precise structural validation or late-stage ranking tasks, where specialized structural models (e.g., AlphaFold2, NanobodyBuilder) are more appropriate.
- Predictions are based solely on sequence context learned from evolutionary data. Novel sequences with limited evolutionary representation may yield less accurate predictions.


How We Use It
-------------

BioLM integrates ESM-1v into protein engineering pipelines to enable rapid assessment of mutational effects on protein functionality, accelerating research cycles and reducing experimental overhead. Through zero-shot predictions and optional fine-tuning with protein family-specific datasets, ESM-1v efficiently informs antibody maturation, enzyme optimization, and general protein design tasks. This integration complements other BioLM workflows, such as embedding-based clustering, predictive modeling, and 3D structure-informed ranking, allowing comprehensive and scalable analysis of protein sequence designs.

- Enables rapid prioritization and filtering of candidate protein sequences prior to lab synthesis and testing
- Integrates seamlessly with embedding-based models and 3D structure analysis pipelines to inform iterative design cycles


Related
-------

- ``ESM-2 650M`` – Provides an alternative protein language model with similar scale, useful for comparative mutation effect analysis.
- ``ESMFold`` – Complements ESM-1v by predicting protein structures, linking mutational effects to structural changes.
- ``AlphaFold2`` – Generates highly accurate protein structures, enabling structural validation of ESM-1v predicted mutation effects.
- ``ESM-IF1`` – Predicts functional impacts of mutations, offering complementary insights into protein variant effects.


References
----------

- Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021). `Language models enable zero-shot prediction of the effects of mutations on protein function <https://doi.org/10.1101/2021.07.09.450648>`_. *bioRxiv*.

