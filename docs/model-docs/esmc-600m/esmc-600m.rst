ESM C 600M API
==============

ESM C 600M is a GPU-accelerated protein language model trained on evolutionary-scale protein sequence data (UniRef, MGnify, JGI), using a masked language modeling objective. With 600 million parameters (36 layers, 1152 hidden units, 18 attention heads), it generates biologically meaningful embeddings for protein representation learning. The API supports high-throughput embedding generation to facilitate protein design, functional annotation, and bioinformatics analysis workflows.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ESM C 600M.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="esmc-600m",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esmc-600m/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esmc-600m/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/esmc-600m/predictor/"

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

.. http:post:: /api/v3/esmc-600m/predict/

   Predict endpoint for ESM C 600M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **repr_layers** (*array of integers*, default: [-1]) — Indices of representation layers to return embeddings from
        - **include** (*array of strings*, default: ["mean"]) — Types of embeddings or logits to include; allowed values: "mean", "per_token", "logits"
      
      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid codes plus "-" character; must contain at least one character
      
      
      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for prediction:
      
        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid codes plus "<mask>" token; must contain at least one "<mask>" token
      
      
      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences for log probability calculation:
      
        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using unambiguous amino acid codes only

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmc-600m/predict/ HTTP/1.1
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
      
        - **embeddings** (*array of objects*, optional) — Mean embeddings per requested layer:
        
          - **layer** (*int*) — Layer index as specified in request
          - **embedding** (*array of floats*, size: 960 for 300m, 1152 for 600m) — Mean embedding vector for the sequence
        
        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer:
        
          - **layer** (*int*) — Layer index as specified in request
          - **embeddings** (*array of arrays of floats*, shape: [sequence_length, embedding_size], embedding_size: 960 for 300m, 1152 for 600m) — Embedding vectors for each token in the sequence
        
        - **logits** (*array of arrays of floats*, optional, shape: [sequence_length, vocab_size]) — Raw logits for each token position; vocab_size: 33 (20 standard amino acids + additional tokens)
        
        - **vocab_tokens** (*array of strings*, optional, size: 33) — Vocabulary tokens corresponding to logits indices
      
      
      - **results** (*array of objects*) --- One result per input item, in the order requested:
      
        - **logits** (*array of arrays of floats*, shape: [num_masked_positions, vocab_size]) — Raw logits for masked positions; vocab_size: 33 (20 standard amino acids + additional tokens)
        
        - **sequence_tokens** (*array of strings*, size: sequence_length) — Tokenized input sequence including predicted tokens
        
        - **vocab_tokens** (*array of strings*, size: 33) — Vocabulary tokens corresponding to logits indices

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "logits": [
        [
          24.125,
          23.625,
          24.25,
          24.25,
          24.625,
          24.25,
          23.625,
          24.875,
          24.375,
          25.125,
          26.5,
          24.0,
          23.875,
          23.625,
          24.125,
          24.5,
          24.25,
          24.875,
          22.875,
          24.25
        ],
        [
          25.375,
          25.125,
          25.125,
          25.375,
          25.25,
          25.25,
          24.375,
          25.625,
          25.5,
          26.0,
          24.5,
          25.0,
          24.875,
          24.625,
          25.25,
          25.5,
          25.375,
          25.875,
          23.75,
          25.0
        ],
        [
          26.875,
          26.25,
          28.125,
          27.125,
          26.875,
          27.0,
          26.375,
          27.125,
          27.0,
          27.625,
          26.25,
          27.125,
          26.75,
          26.75,
          26.875,
          27.25,
          26.875,
          27.125,
          25.625,
          26.625
        ],
        [
          27.25,
          26.75,
          27.5,
          28.0,
          27.375,
          27.375,
          26.625,
          27.625,
          27.375,
          27.75,
          26.375,
          27.125,
          27.0,
          26.875,
          27.25,
          27.75,
          27.125,
          27.375,
          25.75,
          27.0
        ],
        [
          24.375,
          24.0,
          24.75,
          24.625,
          25.375,
          24.5,
          24.5,
          24.875,
          24.875,
          25.125,
          23.75,
          24.875,
          23.875,
          24.125,
          24.5,
          24.75,
          24.375,
          24.625,
          23.125,
          25.0
        ],
        [
          27.75,
          27.625,
          28.0,
          28.0,
          28.125,
          29.125,
          27.625,
          28.125,
          28.0,
          28.75,
          27.25,
          28.0,
          27.625,
          27.625,
          27.875,
          28.375,
          27.75,
          28.125,
          26.75,
          27.75
        ],
        [
          25.625,
          25.125,
          25.875,
          26.0,
          25.625,
          25.875,
          25.625,
          25.875,
          26.25,
          26.0,
          24.875,
          26.5,
          25.125,
          25.375,
          26.125,
          26.25,
          25.75,
          25.75,
          24.5,
          25.375
        ],
        [
          27.375,
          27.0,
          26.875,
          27.5,
          27.5,
          27.5,
          26.5,
          28.0,
          27.5,
          28.375,
          27.0,
          26.875,
          27.375,
          27.0,
          27.375,
          27.625,
          27.375,
          28.0,
          26.375,
          26.75
        ],
        [
          28.0,
          27.625,
          28.0,
          28.25,
          28.125,
          28.125,
          27.75,
          28.5,
          28.625,
          28.875,
          27.375,
          28.0,
          28.0,
          28.0,
          28.375,
          28.625,
          28.25,
          28.5,
          27.0,
          28.0
        ],
        [
          26.875,
          26.5,
          27.0,
          27.0,
          26.875,
          27.125,
          26.5,
          27.125,
          27.25,
          27.5,
          26.125,
          26.875,
          27.0,
          26.625,
          27.25,
          27.375,
          27.0,
          27.125,
          25.75,
          26.75
        ],
        [
          25.5,
          25.125,
          25.25,
          25.5,
          25.125,
          25.375,
          24.875,
          25.625,
          25.75,
          26.125,
          24.625,
          25.25,
          25.25,
          25.375,
          25.375,
          25.75,
          25.375,
          25.625,
          24.125,
          24.875
        ]
      ],
      "sequence_tokens": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "_",
        "L"
      ],
      "vocab_tokens": [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y"
      ]
    }
  ]
}


Encode
------

This endpoint encodes for ESM C 600M.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.encoder(

          model="esmc-600m",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esmc-600m/encoder/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esmc-600m/encoder/"

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
      
      url <- "https://biolm.ai/api/v3/esmc-600m/encoder/"

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

.. http:post:: /api/v3/esmc-600m/encode/

   Encode endpoint for ESM C 600M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **repr_layers** (*array of integers*, default: [-1]) — Layers from which embeddings are extracted
        - **include** (*array of strings*, default: ["mean"]) — Types of embeddings or logits to include; allowed values: "mean", "per_token", "logits"
      
      - **items** (*array of objects*, min: 1, max: 8) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using extended amino acid alphabet plus "-" character

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esmc-600m/encode/ HTTP/1.1
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
      
        - **embeddings** (*array of objects*, optional) — Mean embeddings per requested layer:
      
          - **layer** (*int*) — Layer index as requested (e.g. -1 for last layer)
          - **embedding** (*array of floats*, size: 1280 for 300m model, 1536 for 600m model) — Mean embedding vector for the sequence
      
        - **per_token_embeddings** (*array of objects*, optional) — Per-token embeddings per requested layer:
      
          - **layer** (*int*) — Layer index as requested (e.g. -1 for last layer)
          - **embeddings** (*array of arrays of floats*, shape: [sequence_length, embedding_size]) — Embedding vectors for each token position in the sequence; embedding_size is 1280 for 300m model, 1536 for 600m model
      
        - **logits** (*array of arrays of floats*, optional, shape: [sequence_length, vocab_size]) — Logit scores for each token position; vocab_size = 33 (20 standard amino acids + additional tokens); scores are unbounded floats
      
        - **vocab_tokens** (*array of strings*, optional, size: 33) — Vocabulary tokens corresponding to logits indices; included only when logits are requested

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "embeddings": [
        {
          "layer": 35,
          "embedding": [
            6.875,
            87.5,
            -65.0,
            -52.25,
            0.380859375,
            -47.25,
            22.25,
            39.75,
            -47.75,
            41.25,
            10.75,
            11.125,
            10.5,
            -24.25,
            77.5,
            -17.5,
            19.375,
            16.875,
            0.1875,
            99.5,
            5.15625,
            -22.25,
            31.75,
            -86.0,
            -5.75,
            115.5,
            58.75,
            80.0,
            -18.625,
            70.5,
            -79.0,
            57.75,
            -29.25,
            2.078125,
            88.5,
            -78.0,
            71.5,
            -83.5,
            -12.8125,
            68.0,
            -22.0,
            42.5,
            18.875,
            86.5,
            10.5625,
            82.5,
            46.25,
            -15.4375,
            89.0,
            -29.5,
            39.25,
            21.125,
            -38.5,
            -8.4375,
            43.0,
            -40.75,
            -87.0,
            -35.75,
            -44.0,
            -2.53125,
            61.25,
            -149.0,
            74.5,
            -22.125,
            22.25,
            -98.5,
            67.5,
            22.375,
            103.5,
            -26.625,
            12.8125,
            15.5625,
            21.875,
            34.25,
            -20.375,
            65.5,
            -50.25,
            -24.0,
            26.75,
            41.75,
            -5.5,
            -52.25,
            -2.609375,
            -59.75,
            -29.5,
            -13.625,
            29.5,
            12.9375,
            -101.0,
            10.25,
            -0.9140625,
            -71.0,
            14.3125,
            -65.0,
            24.75,
            87.0,
            -54.5,
            -11.875,
            -33.75,
            -6.34375,
            -58.25,
            -7.375,
            44.0,
            7.53125,
            25.25,
            56.5,
            -0.455078125,
            -39.5,
            -54.25,
            22.5,
            26.75,
            24.5,
            71.0,
            -20.625,
            37.75,
            23.875,
            -17.875,
            72.5,
            -22.75,
            30.75,
            29.125,
            -52.25,
            42.5,
            31.75,
            25.875,
            10.6875,
            -79.5,
            -30.5,
            105.0,
            -63.75,
            65.0,
            -36.5,
            -19.0,
            0.69921875,
            -44.5,
            62.0,
            18.375,
            -4.75,
            83.0,
            -42.75,
            22.75,
            52.75,
            61.25,
            10.125,
            29.5,
            59.25,
            94.5,
            6.09375,
            92.0,
            50.25,
            63.0,
            -1.46875,
            55.5,
            -44.75,
            -57.0,
            -47.75,
            -35.5,
            27.625,
            4.625,
            -56.25,
            16.375,
            1.8359375,
            -123.5,
            -48.75,
            37.25,
            48.5,
            1.15625,
            -17.875,
            74.0,
            71.0,
            -23.75,
            -7.9375,
            -53.75,
            35.75,
            99.5,
            67.0,
            -34.75,
            27.375,
            34.75,
            13.9375,
            -15.625,
            21.0,
            29.375,
            -30.875,
            -111.5,
            95.0,
            -33.0,
            49.0,
            63.25,
            -47.0,
            94.0,
            -13.375,
            88.5,
            72.0,
            -37.5,
            -30.5,
            13.0625,
            86.5,
            120.5,
            21.625,
            15.5625,
            5.34375,
            -31.25,
            -33.5,
            -29.25,
            33.5,
            -19.875,
            78.0,
            12.0625,
            -19.25,
            -123.0,
            -68.0,
            -16.5,
            -5.09375,
            -75.5,
            63.5,
            -24.25,
            -70.0,
            11.1875,
            34.25,
            -60.0,
            -2.203125,
            -85.5,
            81.0,
            -17.5,
            -14.3125,
            -3.796875,
            -85.0,
            17.75,
            46.75,
            -9.8125,
            70.5,
            52.75,
            -82.5,
            13.375,
            -16.625,
            10.75,
            10.3125,
            -2.1875,
            21.0,
            1.28125,
            -106.0,
            53.75,
            -13.0625,
            34.5,
            63.25,
            9.375,
            65.0,
            -0.92578125,
            -13.9375,
            21.375,
            -64.5,
            -44.5,
            136.0,
            51.25,
            -118.5,
            -25.0,
            30.875,
            -63.0,
            46.25,
            -55.25,
            27.125,
            -12.375,
            -10.9375,
            -25.0,
            10.5,
            50.5,
            17.25,
            6.125,
            290.0,
            15.8125,
            42.0,
            48.5,
            -26.25,
            -73.0,
            37.0,
            -58.0,
            22.0,
            -119.0,
            -7.78125,
            75.5,
            -28.5,
            124.0,
            24.5,
            -16.375,
            6.375,
            -41.75,
            -40.25,
            101.0,
            76.0,
            8.625,
            37.25,
            -6.28125,
            29.375,
            2.328125,
            -66.0,
            -62.75,
            -55.0,
            -32.25,
            -3.578125,
            22.75,
            93.5,
            5.375,
            -2.375,
            -19.125,
            44.0,
            -40.75,
            55.5,
            -15.6875,
            -38.75,
            31.625,
            -14.0,
            90.0,
            -92.0,
            -44.0,
            -20.875,
            1.3828125,
            58.5,
            20.375,
            -11.0625,
            46.75,
            31.125,
            -14.8125,
            -1.1015625,
            -17.375,
            -36.5,
            11.3125,
            -61.75,
            80.5,
            -17.375,
            58.75,
            20.125,
            -24.5,
            77.0,
            -160.0,
            65.0,
            38.25,
            48.0,
            -43.25,
            -81.0,
            91.0,
            -13.625,
            16.875,
            -52.0,
            38.75,
            -20.875,
            -6.65625,
            -12.4375,
            -55.75,
            3.703125,
            26.375,
            26.125,
            -64.5,
            77.5,
            39.25,
            76.5,
            -20.0,
            -63.75,
            -28.375,
            -13.125,
            33.25,
            -6.5625,
            49.5,
            15.625,
            -47.75,
            -38.0,
            -7.3125,
            -86.5,
            -27.875,
            10.8125,
            109.5,
            -41.75,
            8.0625,
            9.0625,
            -15.0625,
            112.0,
            66.0,
            -20.875,
            -37.25,
            59.0,
            -31.5,
            -63.75,
            -58.75,
            -11.4375,
            -6.25,
            110.0,
            98.5,
            5.25,
            37.25,
            -6.28125,
            -5.34375,
            120.5,
            4.78125,
            -3.78125,
            80.0,
            16.125,
            69.0,
            56.5,
            -86.0,
            19.375,
            8.875,
            -37.75,
            -20.75,
            -15.75,
            27.625,
            13.0,
            -0.337890625,
            43.0,
            93.0,
            -27.5,
            9.375,
            18.0,
            -122.5,
            30.5,
            38.75,
            14.0625,
            57.0,
            41.75,
            -61.75,
            -54.0,
            -1.171875,
            100.5,
            -5.03125,
            -109.5,
            -7.15625,
            -44.75,
            64.5,
            34.5,
            -34.75,
            -48.5,
            21.875,
            -24.0,
            6.71875,
            -0.63671875,
            13.0,
            48.25,
            -3.078125,
            17.625,
            95.5,
            16.5,
            27.625,
            -37.0,
            47.5,
            31.125,
            94.5,
            103.0,
            33.0,
            28.875,
            40.25,
            96.5,
            10.5625,
            -23.125,
            35.25,
            25.5,
            13.4375,
            -16.25,
            4.625,
            -0.8125,
            -13.3125,
            20.625,
            -24.0,
            13.5625,
            13.5,
            44.25,
            31.375,
            -28.75,
            17.25,
            -59.75,
            -9.0,
            24.5,
            48.25,
            30.75,
            9.5625,
            87.5,
            68.0,
            9.5625,
            -7.03125,
            -44.0,
            -30.625,
            -25.25,
            -34.75,
            132.0,
            -7.71875,
            -36.75,
            -12.9375,
            -38.5,
            -67.5,
            -55.5,
            1.765625,
            89.0,
            -6.71875,
            -7.0625,
            10.0625,
            -28.625,
            16.375,
            60.75,
            43.5,
            -69.0,
            65.0,
            3.125,
            -177.0,
            22.5,
            -37.5,
            23.125,
            77.0,
            -77.5,
            108.0,
            29.625,
            -7.90625,
            68.5,
            -33.25,
            -84.5,
            35.75,
            -7.6875,
            72.5,
            21.75,
            -10.3125,
            23.25,
            4.375,
            -20.25,
            86.5,
            68.5,
            436.0,
            -7.34375,
            78.0,
            -48.5,
            -66.0,
            70.0,
            28.25,
            -12.625,
            -128.0,
            29.75,
            -20.25,
            -43.25,
            24.25,
            -49.0,
            -22.125,
            96.5,
            -24.5,
            47.75,
            65.0,
            -21.25,
            52.5,
            -35.0,
            35.75,
            9.875,
            2.984375,
            224.0,
            74.0,
            -70.5,
            -20.25,
            -49.25,
            -29.375,
            14.75,
            42.75,
            -18.25,
            -39.25,
            24.125,
            -26.75,
            -35.25,
            45.25,
            -83.0,
            -37.0,
            -14.625,
            -33.75,
            121.5,
            -5.15625,
            -29.0,
            97.0,
            52.25,
            -3.21875,
            -19.0,
            6.25,
            110.0,
            -3.28125,
            -45.5,
            -7.5,
            183.0,
            -48.75,
            -56.0,
            16.5,
            28.875,
            8.8125,
            -55.25,
            -67.5,
            -22.75,
            -64.5,
            9.9375,
            39.0,
            13.9375,
            -68.5,
            -18.75,
            6.96875,
            15.125,
            82.0,
            98.0,
            -63.25,
            24.25,
            0.5625,
            98.0,
            30.875,
            -10.6875,
            98.0,
            -86.0,
            -18.75,
            -39.25,
            -18.75,
            -45.0,
            -60.75,
            -46.5,
            -12.8125,
            7.15625,
            -12.875,
            89.0,
            -49.5,
            51.25,
            72.5,
            22.875,
            -30.25,
            108.0,
            193.0,
            -98.0,
            -2.9375,
            -73.0,
            -53.0,
            45.5,
            52.25,
            152.0,
            -17.75,
            98.0,
            -24.625,
            18.625,
            29.625,
            -60.0,
            -16.125,
            34.0,
            22.375,
            74.5,
            -3.984375,
            20.25,
            9.0,
            -32.5,
            23.375,
            -7.78125,
            -5.875,
            13.75,
            117.0,
            6.25,
            -16.0,
            32.75,
            59.25,
            -6.03125,
            99.5,
            53.0,
            44.0,
            -40.0,
            42.75,
            57.5,
            52.0,
            -31.375,
            -29.0,
            85.0,
            8.9375,
            115.5,
            19.875,
            -56.75,
            16.75,
            5.46875,
            -75.5,
            -20.5,
            76.0,
            51.25,
            -3.359375,
            -30.75,
            -52.0,
            -6.71875,
            -100.0,
            -416.0,
            70.5,
            39.75,
            59.75,
            22.875,
            5.9375,
            -30.0,
            112.5,
            -54.75,
            23.125,
            139.0,
            -41.0,
            -45.75,
            14.5,
            -86.0,
            17.875,
            -36.5,
            -51.5,
            30.625,
            -7.65625,
            24.125,
            -80.5,
            -12.0625,
            -472.0,
            -94.5,
            25.375,
            -98.5,
            -17.0,
            -34.5,
            -54.5,
            8.8125,
            -3.59375,
            -41.25,
            22.125,
            24.125,
            -3.625,
            -86.0,
            -122.5,
            -37.25,
            31.125,
            91.5,
            -22.875,
            -0.5859375,
            55.25,
            -49.75,
            -20.875,
            23.875,
            34.5,
            82.5,
            34.0,
            22.25,
            -8.8125,
            6.84375,
            -2.578125,
            34.5,
            30.0,
            -73.5,
            -56.5,
            -29.25,
            86.5,
            -3.0,
            56.5,
            -18.625,
            55.5,
            -13.25,
            -38.25,
            -60.25,
            -29.125,
            -38.0,
            15.125,
            41.0,
            68.5,
            66.5,
            49.75,
            -55.5,
            11.0,
            12.1875,
            3.625,
            60.0,
            53.5,
            -84.0,
            64.5,
            46.75,
            -75.0,
            27.0,
            11.0,
            92.0,
            -9.6875,
            5.75,
            47.25,
            -52.5,
            21.75,
            -13.125,
            -99.5,
            45.5,
            39.5,
            -28.0,
            50.75,
            106.0,
            88.5,
            13.625,
            39.75,
            -1.9609375,
            0.412109375,
            -37.0,
            -1.2734375,
            -35.5,
            36.0,
            24.375,
            -26.375,
            13.875,
            5.15625,
            71.0,
            69.5,
            -31.75,
            46.0,
            -3.984375,
            -49.25,
            -82.5,
            82.0,
            16.625,
            -90.5,
            14.0,
            -31.75,
            -8.3125,
            22.0,
            -11.75,
            0.67578125,
            23.5,
            -38.75,
            -7.4375,
            82.5,
            28.625,
            -14.5625,
            111.0,
            84.0,
            -15.0,
            -87.0,
            24.125,
            -20.875,
            21.625,
            19.375,
            69.5,
            -122.0,
            -35.5,
            28.375,
            51.25,
            -97.5,
            -33.75,
            6.9375,
            35.5,
            48.25,
            -28.625,
            7.9375,
            25.875,
            33.25,
            -24.75,
            16.5,
            14.4375,
            -25.75,
            -8.8125,
            -57.0,
            7.9375,
            -64.5,
            -51.0,
            19.125,
            42.75,
            -17.25,
            21.125,
            -1.875,
            -40.0,
            -55.25,
            22.625,
            35.25,
            9.0625,
            -22.0,
            -91.5,
            30.125,
            12.9375,
            -38.0,
            14.0625,
            75.5,
            -19.0,
            84.5,
            14.4375,
            -12.5,
            48.75,
            -21.875,
            34.0,
            -35.25,
            59.0,
            48.0,
            22.0,
            -37.75,
            76.5,
            -65.5,
            90.0,
            -3.375,
            90.5,
            -28.875,
            -57.5,
            11.5625,
            4.3125,
            -118.0,
            -8.375,
            55.5,
            26.0,
            -49.75,
            6.53125,
            56.75,
            12.125,
            4.375,
            -134.0,
            -16.125,
            -13.0,
            3.84375,
            -21.5,
            -6.25,
            62.75,
            -54.5,
            1.6640625,
            -34.25,
            -30.375,
            59.0,
            25.625,
            -70.5,
            85.5,
            35.75,
            -3.25,
            49.75,
            5.09375,
            14.75,
            -50.0,
            46.25,
            38.25,
            21.125,
            60.0,
            -58.5,
            -4.8125,
            -4.375,
            43.25,
            8.25,
            10.375,
            -85.5,
            2.65625,
            6.59375,
            54.0,
            15.25,
            -1.65625,
            19.875,
            -18.25,
            68.0,
            22.125,
            -110.0,
            -46.75,
            -112.5,
            96.0,
            112.0,
            -55.25,
            27.25,
            -67.5,
            75.5,
            -39.25,
            5.3125,
            -78.5,
            -1.1484375,
            19.75,
            -16.875,
            -44.5,
            -17.0,
            26.5,
            51.25,
            -63.75,
            30.5,
            -12.3125,
            -120.5,
            48.5,
            -13.75,
            20.75,
            5.375,
            -15.125,
            -14.8125,
            79.0,
            -10.5625,
            6.84375,
            -56.75,
            4.4375,
            -4.1875,
            7.03125,
            20.125,
            9.3125,
            -79.0,
            -18.125,
            -31.125,
            26.25,
            7.21875,
            34.5,
            -17.0,
            17.25,
            -4.71875,
            34.75,
            -3.4375,
            -94.5,
            -77.0,
            -15.125,
            34.75,
            -27.375,
            35.25,
            3.9375,
            98.0,
            16.75,
            0.1875,
            -29.0,
            57.0,
            76.0,
            112.0,
            45.0,
            -35.25,
            148.0,
            114.5,
            50.75,
            -10.8125,
            40.75,
            -66.0,
            27.75,
            -51.75,
            -32.75,
            -26.125,
            15.75,
            173.0,
            -12.25,
            92.0,
            -118.0,
            7.9375,
            -3.453125,
            53.0,
            3.25,
            31.75,
            49.5,
            58.75,
            13.875,
            -51.25,
            85.5,
            46.25,
            81.5,
            75.0,
            61.5,
            62.0,
            -44.0,
            44.25,
            76.5,
            -53.25,
            -9.75,
            54.75,
            -13.75,
            45.0,
            3.328125,
            44.5,
            -174.0,
            15.5,
            -32.25,
            -10.6875,
            -30.125,
            -21.75,
            101.0,
            128.0,
            50.5,
            -14.5625,
            45.5,
            -14.3125,
            8.25,
            -5.78125,
            -35.75,
            52.75,
            -81.5,
            -5.65625,
            -60.5,
            4.71875,
            -21.25,
            -73.0,
            -46.25,
            47.0,
            -28.875,
            59.0,
            -36.5,
            -23.375,
            34.0,
            -34.5,
            -96.0,
            -85.5,
            108.0,
            -93.0,
            41.75,
            -0.9296875,
            44.0,
            -6.75,
            -21.0,
            12.4375,
            40.5,
            47.5,
            -29.5,
            38.5,
            37.75,
            14.375,
            71.5,
            -125.0,
            -59.75,
            34.75,
            -42.0,
            -40.25,
            -16.375,
            113.0,
            -46.75,
            42.5,
            -3168.0,
            -66.5,
            -31.125,
            -1.921875,
            -4.15625,
            -42.25,
            22.5,
            -20.75,
            13.75,
            39.25,
            -33.25,
            35.25,
            -24.875,
            -5.4375,
            -38.25,
            -32.25,
            -9.5,
            -48.75,
            19.75,
            24.25,
            21.375,
            22.5,
            92.0,
            3.671875,
            13.25,
            19.25,
            21.25,
            -95.5,
            -87.0,
            127.0,
            9.1875,
            10.25,
            12.5,
            -13.8125,
            50.75,
            42.5,
            -27.625,
            38.0,
            99.0,
            5.0,
            -108.0,
            -11.0,
            50.5,
            46.5,
            87.0,
            65.5,
            -40.75,
            -25.5,
            79.0,
            -16.75,
            -30.375,
            -17.625,
            -54.75,
            70.5,
            71.5,
            54.75,
            23.0,
            -18.125,
            16.875,
            0.212890625,
            5.65625
          ]
        }
      ]
    }
  ]
}


Performance
-----------

- ESM C 600M provides significantly improved inference speed and memory efficiency compared to previous-generation ESM-2 models:
  
  - Matches predictive accuracy of ESM-2 3B (3 billion parameters) model, with approximately 5x fewer parameters.
  
  - Approaches the predictive performance of the substantially larger ESM-2 15B model, enabling high-quality predictions at substantially reduced computational cost.

- GPU-accelerated inference on NVIDIA T4 GPUs ensures reliable, consistent performance for protein embedding and masked sequence prediction tasks.

- Optimized transformer architecture (Pre-LN, rotary embeddings, SwiGLU activations, bias-free linear layers and layer norms) significantly reduces memory footprint and improves inference throughput relative to previous ESM architectures.

- Ideal for high-throughput protein representation learning, embedding generation, and masked residue prediction tasks, providing state-of-the-art accuracy with significantly lower computational overhead compared to larger ESM-2 models.

- ESM C 600M offers superior scalability and throughput relative to BioLM's ESM-2 650M model, delivering substantially higher predictive accuracy at comparable inference speeds.

- Recommended for applications where near state-of-the-art predictive performance is required, but computational resources or inference latency constraints preclude the use of larger models such as ESM-2 3B or 15B.

Input and Output Types:

- **Encoding Tasks**:
  
  - *Input*: Amino acid sequences (single-letter code, up to 2048 residues per sequence).
  
  - *Output*: Embeddings (mean or per-token representations), logits over amino acid vocabulary.

- **Masked Sequence Prediction Tasks**:
  
  - *Input*: Amino acid sequences containing one or more masked positions ("<mask>" tokens).
  
  - *Output*: Predicted logits for masked positions, sequence tokens, and vocabulary tokens.


Applications
------------

- Protein embedding generation to support downstream predictive modeling tasks, enabling researchers to rapidly screen protein libraries for desirable functional properties, such as thermostability or catalytic efficiency; useful in enzyme engineering pipelines but not optimal for direct antibody affinity maturation.
- Unsupervised clustering of protein sequence space to identify novel functional families, facilitating discovery of previously unknown enzyme classes or protein scaffolds; valuable for biotech companies exploring novel biocatalysts but less effective for high-resolution structural predictions.
- Representation learning for protein variant effect prediction, enabling accurate identification of beneficial mutations that enhance protein stability or activity; beneficial for protein engineering teams optimizing industrial enzymes, though not suitable for antibody specificity optimization.
- Protein similarity search and retrieval based on learned embeddings, allowing rapid identification of homologous sequences with desired properties across large databases; practical for enzyme discovery and protein design workflows, but not recommended for precise antibody-antigen binding predictions.
- Generating protein sequence embeddings for downstream machine learning models aimed at predicting protein-protein interactions, helping biotech companies accelerate the design of novel protein therapeutics or scaffolds; effective for general protein engineering applications but limited in predicting highly specific antibody-antigen interactions.


Limitations
-----------

- **Maximum Sequence Length**: Input sequences are limited to ``2048`` amino acids; longer sequences must be truncated or processed in segments.
- **Batch Size**: The maximum allowed batch size per request is ``8`` sequences; larger datasets must be split into multiple requests.
- **GPU Type**: Inference is performed on ``T4`` GPUs; performance may vary depending on computational complexity and batch size.
- ESM C 600M is optimized for representation learning and embeddings; it does not perform generative tasks or structure prediction directly.
- The model is trained exclusively on protein sequences; it may not accurately represent non-natural amino acids or sequences containing ambiguous residues.
- For antibody-specific structural predictions (e.g., CDR3 loops), specialized models like NanobodyBuilder or ABodybuilder typically yield better accuracy than general-purpose models such as ESM C 600M.


How We Use It
-------------

BioLM's integration of the ESM C 600M model into its workflows significantly enhances protein engineering and research acceleration by enabling advanced protein sequence analysis and representation learning. This model allows for the effective design and optimization of proteins by leveraging its ability to predict and analyze the biological properties embedded within protein sequences. By integrating ESM C 600M with BioLM's existing tools and pipelines, researchers can efficiently filter and rank candidate proteins, thereby streamlining the protein design process. The model's capabilities are further amplified when used in conjunction with other BioLM services, such as sequence embeddings and predictive models, which collectively provide a robust framework for innovative scientific solutions. 

- ESM C 600M facilitates scalable and standardized API access, enabling rapid deployment in diverse bioengineering tasks.
- The model's integration within BioLM workflows supports multi-round protein optimization, enhancing experimental accuracy and reducing time to market for new bioproducts.


Related
-------

- ``ESM C 300M`` – Smaller variant of the ESM Cambrian family, ideal for rapid prototyping and lightweight inference tasks before scaling to ESM C 600M.
- ``ESMFold`` – Predicts 3D protein structures from sequences; complements ESM C 600M's representations by providing structural interpretation.
- ``ESM-IF1`` – Inverse folding model that generates sequences from protein structures; pairs effectively with ESM C 600M for protein design workflows.
- ``ESM3 Open Small`` – Generative model for protein sequence, structure, and function; extends ESM C 600M's representations into multimodal protein generation tasks.


References
----------

- ESM Team (2024). `ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning <https://evolutionaryscale.ai/blog/esm-cambrian>`_. *EvolutionaryScale Website*. December 4, 2024.

