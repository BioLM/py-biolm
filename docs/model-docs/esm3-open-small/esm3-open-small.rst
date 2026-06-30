ESM3 Open Small API
===================

ESM3 Open Small is a GPU-accelerated protein language model API designed for encoding protein sequences into numerical embeddings and predicting residue-level contacts without MSAs or structural templates. Trained on UniRef50 sequences using rotary position embeddings (RoPE), ESM3 Open Small provides embeddings suitable for downstream machine learning tasks, such as clustering, annotation, and protein function prediction, supporting scalable workflows in bioinformatics and protein engineering.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ESM3 Open Small.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="esm3-open-small",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esm3-open-small/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esm3-open-small/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/esm3-open-small/predictor/"

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

.. http:post:: /api/v3/esm3-open-small/predict/

   Predict endpoint for ESM3 Open Small.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **batch_size** (*int*, default: 2) — Number of sequences processed in a single batch
        - **max_sequence_len** (*int*, default: 768) — Maximum length of a single sequence
        - **max_n_multimers** (*int*, default: 4) — Maximum number of chains in a sequence
      
      - **items** (*array of objects*, max: 2) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 771, required) — Protein sequence using extended amino acid codes, allowing up to 3 non-consecutive ":" characters to separate chains

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm3-open-small/predict/ HTTP/1.1
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
      
        - **pdb** (*string*) — Predicted protein structure in standard PDB format
      
        - **mean_plddt** (*float*, range: 0.0-1.0) — Mean predicted Local Distance Difference Test (pLDDT) confidence score across all residues
      
        - **ptm** (*float*, range: 0.0-1.0) — Predicted Template Modeling (pTM) score indicating global structural accuracy

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "pdb": "ATOM      1  N   ALA A   1      -1.303  -8.207   7.997  1.00  0.73           N  \nATOM      2  CA  ALA A   1      -2.337  -7.286   7.537  1.00  0.73           C  \nATOM      3  C   ALA A   1      -2.004...",
      "mean_plddt": 0.7593482136726379
    }
  ]
}


Encode
------

This endpoint encodes for ESM3 Open Small.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.encoder(

          model="esm3-open-small",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esm3-open-small/encoder/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esm3-open-small/encoder/"

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
      
      url <- "https://biolm.ai/api/v3/esm3-open-small/encoder/"

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

.. http:post:: /api/v3/esm3-open-small/encode/

   Encode endpoint for ESM3 Open Small.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
      
      - **items** (*array of objects*, min: 1, max: 2) --- Input sequences:
      
        - **sequence** (*string*, required, min length: 1, max length: 771) — Protein sequence composed of standard amino acid codes and ":" separators for multimers (up to 3 occurrences).

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm3-open-small/encode/ HTTP/1.1
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
      
        - **pdb** (*string*) — Predicted protein structure in standard PDB format
      
        - **mean_plddt** (*float*, range: 0.0-1.0) — Mean predicted Local Distance Difference Test (pLDDT) score indicating prediction confidence
      
        - **ptm** (*float*, range: 0.0-1.0) — Predicted Template Modeling (pTM) score evaluating global structural accuracy

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "embeddings": [
        -7.0,
        55.875,
        26.75,
        -43.1875,
        -3.25,
        -57.5,
        11.3125,
        98.875,
        44.40625,
        74.5,
        123.9375,
        117.0,
        17.6875,
        84.25,
        -76.90625,
        4.9921875,
        29.6875,
        182.625,
        -290.25,
        -62.875,
        -63.25,
        166.375,
        -193.5,
        53.625,
        -3.8125,
        35.03125,
        29.796875,
        23.5,
        -90.59375,
        40.8125,
        152.609375,
        51.5,
        -35.6875,
        25.53125,
        -97.21875,
        -35.0,
        45.125,
        -79.78125,
        -50.0625,
        -31.625,
        -169.125,
        104.5,
        70.28125,
        48.25,
        -52.78125,
        -119.875,
        -4.53125,
        -43.25,
        192.0,
        160.875,
        -43.5,
        140.875,
        -66.203125,
        88.875,
        -71.4375,
        -133.25,
        -26.4375,
        -34.0625,
        0.65625,
        -286.5,
        -33.75,
        143.875,
        -68.75,
        -43.0,
        -72.625,
        -60.8125,
        116.9375,
        67.125,
        107.859375,
        -49.25,
        -148.75,
        -16.46875,
        68.828125,
        -40.21875,
        200.25,
        95.3125,
        10.5,
        -66.125,
        -57.875,
        -12.375,
        -99.375,
        -76.0,
        3.03125,
        -117.4375,
        -44.59375,
        -143.25,
        245.25,
        147.75,
        -25.3125,
        7.78125,
        81.375,
        -46.0,
        -71.8125,
        76.1875,
        80.375,
        252.375,
        -321.75,
        141.875,
        -51.4375,
        108.1875,
        18.25,
        -72.5625,
        84.25,
        -133.0,
        70.625,
        40.46875,
        -16.46875,
        -177.25,
        119.25,
        -68.3125,
        -61.5,
        13.625,
        211.25,
        123.625,
        247.75,
        83.34375,
        81.484375,
        114.9375,
        175.25,
        -67.3125,
        23.75,
        -69.1875,
        85.0625,
        150.375,
        76.125,
        115.28125,
        -30.6875,
        24.4375,
        -123.125,
        -55.84375,
        -16.46875,
        -28.8125,
        8.09375,
        -80.875,
        53.03125,
        -190.625,
        -62.75,
        -119.375,
        1.3125,
        184.0,
        -1.3125,
        27.4375,
        16.6875,
        48.625,
        -71.75,
        -97.125,
        -37.03125,
        6.21875,
        -50.0,
        1.4375,
        -93.75,
        154.75,
        -49.25,
        -44.9375,
        -58.8125,
        -63.46875,
        23.5625,
        21.875,
        -113.625,
        -22.3125,
        98.125,
        110.5,
        18.125,
        196.0,
        10.5625,
        35.15625,
        -64.34375,
        104.5,
        64.6875,
        -34.859375,
        60.5625,
        57.0625,
        83.5625,
        87.125,
        277.5,
        207.25,
        44.125,
        -51.125,
        67.875,
        -66.9375,
        -26.625,
        24.6875,
        144.5,
        -73.125,
        44.109375,
        -47.375,
        -75.15625,
        109.25,
        47.59375,
        1.25,
        -67.0,
        -48.5625,
        -32.5,
        108.5,
        -22.5625,
        59.0,
        69.5,
        191.0,
        -42.5,
        166.625,
        35.3125,
        -43.125,
        111.75,
        45.03125,
        -136.375,
        -99.5625,
        7.875,
        -71.71875,
        -27.9375,
        23.90625,
        147.4375,
        50.375,
        187.5,
        -32.5625,
        156.25,
        -115.375,
        182.125,
        42.0,
        -110.4375,
        -102.375,
        64.875,
        63.75,
        -27.125,
        167.0,
        -14.0625,
        100.21875,
        39.796875,
        23.5625,
        34.375,
        -16.8125,
        46.84375,
        152.875,
        103.90625,
        60.3359375,
        25.875,
        100.0625,
        -64.125,
        -87.75,
        -236.0,
        36.78125,
        -34.375,
        -94.0625,
        19.0,
        -111.25,
        51.75,
        37.0,
        102.8125,
        74.125,
        37.6875,
        91.375,
        -69.96875,
        172.25,
        -43.703125,
        148.25,
        -58.9375,
        84.28125,
        0.4375,
        17.03125,
        -16.4375,
        -59.4375,
        -120.5625,
        87.8125,
        -171.0,
        -138.5,
        -249.5,
        -129.75,
        143.125,
        -71.9375,
        -17.25,
        -122.25,
        126.5,
        103.5,
        -138.75,
        -193.75,
        -24.6875,
        19.390625,
        23.4375,
        -24.75,
        32.1796875,
        -24.75,
        140.875,
        84.75,
        202.25,
        92.25,
        -110.375,
        -98.375,
        -19.9375,
        62.625,
        123.4375,
        110.625,
        -61.5,
        177.125,
        -169.5,
        142.75,
        93.375,
        75.4375,
        -151.125,
        -68.5,
        121.5625,
        -17.375,
        -104.75,
        1.625,
        98.3125,
        157.375,
        -29.15625,
        -101.46875,
        23.375,
        -57.375,
        -110.375,
        -84.125,
        156.5,
        236.0,
        63.625,
        89.25,
        7.28125,
        -176.25,
        44.265625,
        -0.046875,
        85.875,
        112.625,
        69.5390625,
        177.875,
        -56.0,
        -92.375,
        145.875,
        -24.875,
        97.0,
        72.0,
        71.125,
        -107.25,
        -35.375,
        544.0,
        -17.1875,
        131.5,
        -61.21875,
        -74.875,
        74.9375,
        43.625,
        -47.375,
        -32.40625,
        42.625,
        15.3125,
        -134.5,
        122.0,
        -51.0625,
        78.0,
        109.625,
        -200.0,
        -6.6875,
        30.125,
        -71.625,
        -47.125,
        -53.3125,
        236.25,
        -65.125,
        -2.375,
        -37.125,
        20.625,
        259.875,
        -22.5625,
        -15.5625,
        150.0,
        60.4375,
        81.875,
        -128.65625,
        18.25,
        22.25,
        -156.125,
        19.88671875,
        68.25,
        123.25,
        -127.5,
        -50.25,
        -227.25,
        -546.5,
        92.4375,
        -183.375,
        -99.125,
        -267.75,
        -103.875,
        17.0,
        -154.875,
        79.1875,
        132.25,
        217.4375,
        188.75,
        0.0,
        -28.4375,
        71.625,
        -176.5,
        -37.984375,
        -3.78125,
        -190.125,
        27.0625,
        61.0390625,
        -13.5625,
        -7.25,
        -50.875,
        -28.375,
        -34.8125,
        141.25,
        -110.25,
        -228.0,
        178.5,
        22.125,
        1.625,
        -37.75,
        119.0,
        -10.0625,
        -226.0,
        95.6875,
        262.25,
        33.25,
        13.125,
        39.40625,
        104.1875,
        -47.125,
        -89.625,
        -108.625,
        -119.625,
        -112.375,
        -59.625,
        -12.0625,
        13.5,
        35.25,
        -81.46875,
        -24.1875,
        103.53125,
        -133.84375,
        -128.0,
        -54.0,
        -110.4375,
        -70.15625,
        -84.9375,
        -107.0,
        -141.875,
        80.375,
        -30.71875,
        -9.375,
        -4.125,
        30.0625,
        -58.375,
        55.21875,
        -45.125,
        20.78125,
        14.625,
        -217.75,
        -42.0,
        -39.5,
        238.75,
        -26.015625,
        -68.828125,
        -3.28125,
        2.6875,
        -192.75,
        -3.125,
        45.6875,
        17.125,
        43.640625,
        113.25,
        -144.0,
        -67.1875,
        -108.0625,
        191.875,
        158.21875,
        164.375,
        -3.796875,
        55.375,
        -107.625,
        12.59375,
        -21.875,
        128.125,
        66.875,
        36.5,
        -163.125,
        112.875,
        -34.5625,
        114.6875,
        -117.5,
        274.625,
        -48.6875,
        -45.625,
        29.9921875,
        0.96875,
        -95.125,
        61.9375,
        -57.59375,
        103.5625,
        -10.484375,
        118.5,
        -9.625,
        133.875,
        -73.0,
        -8.25,
        98.375,
        -115.25,
        60.5,
        133.875,
        -88.125,
        109.875,
        -32.75,
        -53.375,
        -48.875,
        -10.3125,
        210.25,
        7.09375,
        -196.375,
        80.75,
        85.1875,
        -179.625,
        -6.0,
        115.0,
        126.75,
        -137.25,
        -49.0625,
        113.6875,
        70.0625,
        161.5,
        19.0625,
        -160.125,
        153.8125,
        -134.9375,
        101.375,
        45.90625,
        18.0625,
        92.0625,
        37.0,
        18.625,
        45.75,
        59.70703125,
        202.75,
        37.3125,
        -181.0,
        225.875,
        -89.03125,
        -117.625,
        -236.0,
        -86.71875,
        -289.5,
        -150.625,
        -69.125,
        52.9375,
        -15.875,
        57.25,
        -192.625,
        17.125,
        104.9375,
        -93.59375,
        277.75,
        -33.75,
        243.25,
        152.0,
        55.0,
        76.96875,
        -104.65625,
        71.0,
        121.25,
        82.375,
        0.6875,
        -83.4375,
        22.9921875,
        242.0,
        106.125,
        -126.625,
        -229.5,
        -73.0,
        -81.03125,
        -68.0625,
        -51.5625,
        37.625,
        -41.5625,
        103.75,
        -226.0,
        -72.625,
        -17.859375,
        -162.875,
        90.8125,
        46.25,
        -121.25,
        -0.8125,
        97.875,
        -155.0,
        -38.53125,
        -14.875,
        176.0,
        61.6875,
        -142.125,
        103.3125,
        -34.3125,
        18.0,
        120.375,
        119.75,
        95.9375,
        26.25,
        -10.90625,
        -21.1875,
        218.5,
        36.25,
        59.28125,
        69.0625,
        38.5625,
        1.90625,
        -86.5,
        60.5625,
        -50.9375,
        169.75,
        -14.0,
        6.5625,
        144.5,
        103.25,
        71.375,
        -79.1875,
        -50.4375,
        5.0625,
        -21.46875,
        -1.53125,
        87.0,
        -173.5,
        18.25,
        -150.5,
        2.75,
        -83.6875,
        -52.375,
        -110.25,
        -19.5,
        -139.875,
        -167.0,
        -177.0,
        -11.4296875,
        93.8125,
        -7.546875,
        230.625,
        -164.375,
        7.65625,
        47.9375,
        60.71875,
        223.25,
        59.8125,
        121.96875,
        72.75,
        -20.078125,
        -12.6875,
        62.75,
        169.4375,
        45.625,
        80.1875,
        97.125,
        196.375,
        181.0,
        151.875,
        -102.5,
        52.0,
        -19.8125,
        127.0,
        -47.21875,
        72.8125,
        -53.46875,
        -20.0,
        -14.125,
        5.375,
        132.34375,
        -146.375,
        82.125,
        144.265625,
        -44.28125,
        -101.875,
        -173.5,
        8.0,
        -72.625,
        77.0,
        241.25,
        201.125,
        163.8125,
        -15.1875,
        -298.75,
        1.0625,
        -27.515625,
        -145.125,
        126.875,
        103.328125,
        -0.46875,
        61.1171875,
        100.375,
        -29.125,
        -142.625,
        -80.125,
        193.875,
        132.0,
        91.5625,
        -156.875,
        113.125,
        -147.0,
        -73.875,
        -4.5,
        148.625,
        85.140625,
        48.125,
        13.75,
        -112.875,
        -161.25,
        -146.1875,
        87.875,
        -150.8125,
        -48.0,
        -109.1875,
        16.25,
        -31.875,
        -84.5625,
        59.6875,
        -34.71875,
        162.0,
        8.75,
        33.0,
        81.375,
        87.875,
        42.25,
        48.75,
        120.828125,
        86.3125,
        -65.46875,
        -331.0,
        97.0,
        123.125,
        63.75,
        76.9375,
        -23.0,
        -45.0625,
        164.75,
        16.0625,
        45.9375,
        -4.625,
        55.4375,
        25.4375,
        -138.0625,
        -56.28125,
        0.90625,
        -52.5625,
        -47.40625,
        -183.0,
        -69.5,
        45.25,
        192.75,
        118.0,
        46.1875,
        33.0625,
        -22.875,
        -72.6875,
        -128.125,
        152.625,
        11.34375,
        -220.875,
        -0.5,
        -31.625,
        136.25,
        28.25,
        101.3125,
        -8.875,
        19.0,
        222.0,
        44.875,
        197.0,
        -49.5625,
        -16.78125,
        95.5,
        32.96875,
        -81.59375,
        -151.625,
        69.78125,
        9.4375,
        -20.75,
        74.4375,
        -92.078125,
        -0.625,
        159.0,
        -42.75,
        13.875,
        -98.5,
        35.75,
        48.3125,
        -168.0,
        -15.5625,
        1.53125,
        6.125,
        -46.28125,
        -220.25,
        -49.53125,
        -72.5,
        79.5,
        -115.0,
        172.25,
        -26.359375,
        9.34375,
        32.5,
        16.625,
        99.875,
        -83.1875,
        196.375,
        -64.375,
        42.0625,
        -111.75,
        12.8984375,
        -6.5625,
        143.1875,
        43.0,
        11.75,
        13.96875,
        107.9375,
        28.59375,
        86.8125,
        39.125,
        -45.1875,
        -63.3125,
        60.125,
        -9.9375,
        81.125,
        -44.0,
        -62.25,
        136.375,
        -54.75,
        84.34375,
        82.34375,
        -4.625,
        -17.25,
        113.25,
        -72.875,
        35.3125,
        98.46875,
        17.3125,
        31.3125,
        138.625,
        -79.1875,
        224.5,
        172.75,
        -2.375,
        -109.625,
        -152.625,
        118.0,
        72.84375,
        178.125,
        -2.8125,
        52.0,
        -64.375,
        -140.5,
        68.71875,
        -14.8125,
        -25.875,
        262.5,
        20.3125,
        127.75,
        8.78125,
        -38.359375,
        58.25,
        -126.25,
        -125.5,
        23.125,
        -42.34375,
        -105.125,
        -94.3125,
        -45.0625,
        66.125,
        -1.375,
        -26.265625,
        26.59375,
        123.0,
        101.875,
        282.0,
        -14.0,
        -185.5,
        -159.75,
        -11.96875,
        50.96875,
        -65.5625,
        189.5,
        -35.5,
        -10.6875,
        -132.0,
        -52.5,
        69.5,
        -65.1875,
        -112.6875,
        96.875,
        -37.0,
        -134.875,
        80.9375,
        12.1875,
        -56.75,
        38.0625,
        -10.8125,
        -35.375,
        45.875,
        21.0625,
        161.875,
        -162.125,
        34.34375,
        30.3125,
        -67.0,
        -15.625,
        42.25,
        -19.984375,
        186.4375,
        -6.703125,
        116.90625,
        -5.578125,
        18.8125,
        -40.75,
        194.5,
        -92.46875,
        -66.375,
        -152.5,
        -100.4375,
        183.75,
        -108.25,
        -32.375,
        -347.5,
        98.125,
        125.0,
        64.25,
        -106.5,
        1.25,
        34.375,
        28.1875,
        75.375,
        192.5,
        -42.1875,
        36.875,
        -114.875,
        -156.0,
        148.125,
        15.0,
        -16.4375,
        132.25,
        -47.84375,
        -174.75,
        116.0,
        182.375,
        137.625,
        -84.5,
        -76.1875,
        48.375,
        -171.25,
        -29.25,
        -23.6875,
        -68.375,
        19.625,
        -156.875,
        -63.625,
        62.5625,
        41.375,
        11.125,
        -196.25,
        93.5625,
        15.09375,
        162.6875,
        -116.1875,
        16.40625,
        -87.1875,
        -13.375,
        83.53125,
        -48.0625,
        -74.8125,
        131.0,
        56.5,
        -144.625,
        -146.0,
        77.4375,
        102.75,
        48.75,
        -143.5,
        -50.15625,
        -154.25,
        -38.984375,
        155.375,
        47.5625,
        30.3125,
        -4.625,
        98.5,
        -68.25,
        -28.6875,
        127.625,
        111.0625,
        23.625,
        52.5,
        29.046875,
        22.125,
        35.0625,
        20.15625,
        -60.28125,
        166.75,
        -44.5,
        177.5,
        -80.75,
        51.1875,
        27.625,
        106.3515625,
        -120.25,
        205.875,
        -18.5,
        -35.875,
        -79.625,
        -103.5625,
        -137.6875,
        22.6875,
        -173.5625,
        21.5,
        140.25,
        -26.625,
        5.0625,
        -101.96875,
        36.15625,
        0.09375,
        108.875,
        -5.25,
        -43.25,
        66.75,
        -51.53125,
        133.0625,
        -55.78125,
        106.125,
        163.375,
        -81.875,
        41.625,
        19.875,
        -99.0,
        2.421875,
        58.75,
        -32.390625,
        -31.6875,
        40.125,
        -69.390625,
        -127.75,
        49.5625,
        41.125,
        -174.0625,
        -107.25,
        -15.65625,
        68.5,
        53.1875,
        -171.5625,
        -40.75,
        -194.0,
        -31.75,
        -13.328125,
        -198.625,
        128.75,
        -346.25,
        106.875,
        109.8125,
        37.1875,
        -107.625,
        37.34375,
        -92.5,
        -36.0625,
        57.1875,
        56.5,
        135.875,
        -38.0625,
        -73.75,
        -89.3125,
        -59.1875,
        86.5,
        161.875,
        302.0,
        -58.375,
        59.375,
        -102.125,
        114.5625,
        -44.875,
        92.25,
        201.1875,
        114.5,
        177.0,
        -68.8125,
        51.3125,
        -58.875,
        44.875,
        -64.0,
        82.71875,
        -223.0,
        -144.25,
        254.5,
        -1.34375,
        -39.9453125,
        -39.91796875,
        18.4375,
        45.6875,
        99.75,
        206.5,
        -86.4375,
        107.125,
        30.625,
        32.5625,
        -80.6875,
        7.25,
        -49.5625,
        24.1875,
        10.09375,
        -75.75,
        -16.625,
        -64.5,
        -22.375,
        -60.875,
        98.125,
        -11.125,
        -23.625,
        39.375,
        81.59375,
        4.125,
        -24.9375,
        -60.34375,
        204.75,
        93.59375,
        63.5,
        -60.0,
        24.8125,
        -5.875,
        -38.0,
        -86.375,
        -8.28125,
        58.1875,
        194.125,
        153.75,
        25.14453125,
        -91.875,
        75.8125,
        48.875,
        -71.75,
        -11.0625,
        56.5625,
        82.0859375,
        -16.546875,
        137.125,
        43.34375,
        -50.375,
        60.3125,
        -58.625,
        46.0625,
        61.875,
        -42.375,
        24.875,
        119.125,
        -86.4375,
        -94.875,
        -60.046875,
        -19.0,
        123.625,
        39.7734375,
        79.0,
        -60.375,
        -144.5,
        203.75,
        116.375,
        138.6875,
        61.75,
        -127.25,
        23.0,
        -3.625,
        25.4375,
        193.3125,
        52.75,
        37.4140625,
        112.875,
        -64.0,
        0.5625,
        189.25,
        -22.125,
        49.8125,
        67.46875,
        -33.4375,
        11.375,
        -29.8125,
        115.25,
        -0.5625,
        -30.671875,
        22.3125,
        -132.34375,
        -11.9375,
        149.25,
        -48.1875,
        -85.375,
        133.75,
        -30.125,
        -104.375,
        -174.5,
        22.375,
        205.875,
        60.375,
        -141.75,
        107.375,
        13.5859375,
        277.0,
        39.1875,
        -98.1875,
        26.75,
        -41.375,
        27.625,
        -55.4375,
        -58.015625,
        -28.78125,
        2.875,
        180.125,
        18.84375,
        2.5625,
        102.25,
        46.375,
        135.125,
        -48.5,
        57.40625,
        200.25,
        57.1875,
        93.875,
        -21.1875,
        51.34375,
        -76.0,
        -113.25,
        58.546875,
        213.0,
        72.125,
        46.296875,
        -74.375,
        -57.0,
        -132.875,
        101.59375,
        -158.875,
        -45.46875,
        -77.8125,
        45.90625,
        -25.25,
        56.375,
        109.75,
        -154.25,
        -6.9375,
        -133.5,
        -123.25,
        -104.25,
        -50.375,
        -115.375,
        -12.6875,
        92.375,
        -90.125,
        86.625,
        161.875,
        -63.25,
        102.9375,
        -6.3125,
        -35.8125,
        -117.5,
        -1.0,
        -137.6875,
        30.3125,
        106.375,
        -153.125,
        -71.46875,
        -71.5,
        -0.96875,
        -39.4375,
        -161.5,
        43.125,
        -98.0625,
        11.359375,
        -165.125,
        -109.5,
        150.25,
        -20.5,
        -54.375,
        190.5,
        -73.5,
        -17.75,
        -24.5,
        111.125,
        -28.15625,
        -166.75,
        86.375,
        -5.28125,
        -117.8125,
        -48.3125,
        81.9375,
        31.5,
        38.78125,
        158.0,
        78.0,
        9.75,
        10.90625,
        -50.8125,
        -51.125,
        258.5,
        326.5,
        64.0625,
        -83.625,
        37.4375,
        83.125,
        116.5625,
        -40.9375,
        135.5,
        -10.25,
        -119.625,
        -6.5,
        -30.1796875,
        92.0,
        -79.125,
        14.875,
        76.28125,
        -38.9375,
        55.6875,
        160.0,
        -105.625,
        -49.875,
        24.875,
        186.875,
        -127.515625,
        -32.3125,
        71.0625,
        145.5,
        99.5,
        91.5,
        -154.75,
        -23.890625,
        36.4375,
        -58.25,
        113.0625,
        82.75,
        22.875,
        152.65625,
        15.125,
        -54.625,
        60.890625,
        -15.625,
        4.234375,
        4.75,
        155.125,
        -63.75,
        -23.0625,
        -70.5,
        -57.78125,
        60.75,
        -146.78125,
        -125.625,
        94.6875,
        28.078125,
        173.0,
        236.25,
        -55.96875,
        136.9375,
        -8.8125,
        35.25,
        -55.46875,
        102.5625,
        -53.71875,
        99.625,
        -98.375,
        -91.5625,
        -187.625,
        52.4375,
        -39.6875,
        80.5,
        -166.375,
        14.8125,
        235.25,
        177.5,
        90.875,
        3.59375,
        -132.25,
        -7088.0,
        112.8125,
        15.9375,
        -252.25,
        24.53125,
        -201.25,
        126.25,
        46.3125,
        53.375,
        96.5625,
        25.375,
        11.125,
        -65.5625,
        111.8125,
        23.0625,
        -57.0,
        99.875,
        -148.25,
        210.0,
        33.25,
        51.984375,
        57.6875,
        46.96875,
        -60.421875,
        -179.125,
        -62.875,
        106.4375,
        -95.8125,
        -148.25,
        -50.75,
        486.0,
        -113.8125,
        -171.625,
        15.0625,
        137.8125,
        -92.125,
        -73.5625,
        38.25,
        4.625,
        179.0625,
        -25.78125,
        97.5,
        -165.625,
        -38.125,
        20.9375,
        -24.25,
        -143.1875,
        19.46875,
        165.4375,
        -2.75,
        11.3515625,
        -147.125,
        131.25,
        189.0,
        253.0,
        30.1875,
        30.875,
        89.625,
        148.5,
        58.125,
        39.625,
        29.0625,
        -124.75,
        85.625,
        106.53125,
        -158.25,
        130.75,
        -101.125,
        113.125,
        -135.875,
        91.375,
        -25.0625,
        -78.9375,
        141.6875,
        -11.828125,
        -3.75,
        -5.625,
        110.875,
        -32.25,
        -96.203125,
        -132.5,
        127.375,
        -179.5,
        44.3125,
        -13.6875,
        36.453125,
        33.9375,
        81.5,
        -111.625,
        113.8125,
        149.375,
        -130.0625,
        72.625,
        79.546875,
        0.875,
        -26.25,
        108.59375,
        56.0,
        76.265625,
        -2.4375,
        66.09375,
        75.3125,
        37.5,
        77.7578125,
        4.5625,
        -21.3125,
        37.375,
        72.0,
        -142.28125,
        13.125,
        -251.0,
        186.25,
        -97.875,
        35.625,
        -122.875,
        -4.9375,
        -42.5625,
        -28.25,
        124.875,
        26.75,
        34.6875,
        122.5,
        49.25,
        7.0,
        220.75,
        20.90625,
        169.25,
        -201.125,
        -5148.0,
        147.5,
        -57.0,
        12.375,
        -9.75,
        -271.65625,
        -122.0,
        -36.6875,
        90.875,
        -127.75,
        19.8125,
        -39.25,
        -49.0,
        -131.625,
        -40.0,
        84.21875,
        -78.78125,
        -7.875,
        -115.1875,
        84.75,
        -23.0,
        -8.15625,
        -36.0,
        -28.4375,
        -31.75,
        2.59375,
        74.6875,
        -132.625,
        93.4375,
        -10.375,
        20.75,
        -42.5,
        28.875,
        86.125,
        206.125,
        -216.625,
        176.25,
        -75.5,
        -17.28125,
        87.125,
        22.6875,
        24.625,
        -48.6875,
        -58.5,
        -49.0625,
        150.25,
        0.0625
      ]
    }
  ]
}


Performance
-----------

- GPU-accelerated inference using NVIDIA A10G GPUs, optimized specifically for rapid structure prediction tasks
- ESM3 Open Small achieves significantly faster inference compared to AlphaFold2, typically completing predictions in seconds per sequence, versus minutes for AlphaFold2
- Predictive accuracy (measured by lDDT and TM-score) is slightly lower than AlphaFold2, but significantly higher than simpler embedding-based models such as ESM-2 150M or ESM-2 650M
- ESM3 Open Small demonstrates robust performance on orphan proteins (proteins with limited evolutionary data), outperforming traditional MSA-dependent methods like AlphaFold2 and RoseTTAFold on sequences with fewer than 100 homologous sequences
- For short to medium-length proteins (up to ~384 residues), ESM3 Open Small provides approximately 6x faster inference than AlphaFold2, primarily due to the removal of computationally expensive MSA searches
- Performance scaling is efficient up to the maximum supported sequence length (768 residues), with inference times increasing approximately cubically with sequence length due to pairwise representation computations
- BioLM has optimized deployment by employing mixed precision inference, chunked attention, and CPU offloading techniques, enabling efficient predictions even for longer sequences within the supported range
- Unlike AlphaFold2, ESM3 Open Small does not require MSA or template searches, significantly reducing computational overhead and improving throughput for large-scale batch predictions
- Predictive confidence metrics (mean pLDDT and predicted TM-score) correlate strongly with experimentally validated accuracy, allowing users to reliably filter high-confidence predictions for downstream tasks
- While AlphaFold2 remains the gold standard for maximum accuracy, ESM3 Open Small offers a practical trade-off between speed and accuracy, suitable for high-throughput screening, rapid iterative design cycles, or exploratory analyses where computational efficiency is critical


Applications
------------

- Protein structure prediction for novel drug development
  - ESM3 enables accurate atomic-level protein structure predictions, which are crucial for identifying binding sites and understanding protein-ligand interactions in drug development.
  - Pharmaceutical companies can leverage these predictions to design more effective and targeted drugs, reducing the time and cost associated with experimental structure determination.
  - Not optimal for proteins with limited sequence homology data, as prediction accuracy may decrease.

- Antibody design and optimization for therapeutic applications
  - ESM3's capabilities in predicting protein structures facilitate the design and optimization of antibodies, ensuring higher affinity and specificity for target antigens.
  - Biotech companies can use these predictions to enhance the development of monoclonal antibodies, improving therapeutic efficacy and reducing off-target effects.
  - Limitations may arise in predicting complex antibody-antigen interactions without additional experimental data.

- Enzyme engineering for industrial biocatalysis
  - By predicting enzyme structures, ESM3 aids in the engineering of enzymes with improved catalytic properties and stability, essential for industrial applications such as biofuel production and waste treatment.
  - Companies can utilize these predictions to modify enzyme active sites, enhancing reaction rates and substrate specificity.
  - Not ideal for enzymes with highly flexible or disordered regions, where structural predictions might be less reliable.

- DNA sequence analysis for synthetic biology
  - ESM3 supports the analysis of DNA sequences, enabling the design of synthetic genes and pathways for applications in synthetic biology, such as metabolic engineering and biosensor development.
  - Researchers can use these insights to optimize genetic constructs for increased expression and functionality in host organisms.
  - May require complementary experimental validation to confirm predicted sequence-function relationships.


Limitations
-----------

- **Maximum Sequence Length**: The API accepts sequences up to ``768`` amino acids in length. Longer sequences must be truncated or split into smaller segments.
- **Batch Size**: A maximum of ``2`` sequences per request is supported. For larger-scale predictions, multiple requests must be submitted sequentially.
- **Multimer Input**: The API supports predictions for multimers containing up to ``4`` chains. Chains must be separated by the colon character (``:``).
- **Prediction Accuracy**: ESMFold predictions are generally accurate for proteins with abundant evolutionary information. However, accuracy decreases significantly for orphan proteins or sequences with limited evolutionary homologs.
- **Complexes and Multimers**: While ESMFold can predict multimer structures, it was trained exclusively on single-chain proteins. For optimal accuracy in multimeric protein complexes, specialized multimer-focused models (e.g., AlphaFold-Multimer) are recommended.
- **Confidence Scores (mean_plddt, ptm)**: The returned confidence metrics (``mean_plddt``, ``ptm``) correlate well with prediction accuracy, but low-confidence predictions (``mean_plddt`` < 0.7) should be interpreted cautiously, especially for downstream tasks such as protein engineering or experimental validation.


How We Use It
-------------

BioLM integrates the ESM3 Open Small model to accelerate protein design and engineering by providing accurate, computationally efficient predictions of protein structure and function from sequence data. By leveraging ESM3 Open Small's ability to quickly generate high-quality sequence embeddings and predictive scores, BioLM enables researchers to rapidly screen, rank, and prioritize candidate sequences prior to laboratory synthesis and testing. This integration directly complements BioLM's broader workflows, including masked language modeling-based protein optimization, 3D structure-informed filtering, and biophysical property predictions, facilitating iterative protein optimization cycles and significantly reducing experimental timelines.

- Enables rapid sequence prioritization to reduce experimental costs and timelines
- Integrates seamlessly with existing BioLM predictive and generative modeling pipelines for iterative protein engineering


Related
-------

- ``ESMFold`` – Integrates ESM language model embeddings to rapidly predict protein structures, complementary for structure prediction tasks.
- ``ESM-2 150M`` – A smaller-scale language model providing efficient embeddings for faster inference or resource-constrained environments.
- ``AlphaFold2`` – Provides highly accurate protein structure predictions using MSAs, complementary for benchmarking and comparative analysis.
- ``ESM-IF1`` – Inverse folding model leveraging ESM embeddings, complementary for protein design and structure-based sequence optimization tasks.


References
----------

- Lin, Z., et al. (2023). `Evolutionary-scale prediction of atomic-level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*, 379, 1123.

