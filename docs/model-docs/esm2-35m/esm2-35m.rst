ESM-2 35M API
=============

ESM-2 35M is a GPU-accelerated protein language model trained on evolutionary-scale sequence data, enabling the generation and scoring of novel protein sequences beyond known natural families. It supports structure-conditioned design (fixed-backbone), unconstrained de novo sequence generation, and sequence likelihood evaluation through pseudo-perplexity metrics. The API provides efficient inference for high-throughput protein engineering, sequence design, and exploration of structural motifs and sequence spaces not observed in nature.


.. seealso::
   :class: important

   Postman Collection: https://api.biolm.ai/#80869d4d-5ebd-436a-b78f-c281b47342fd

Predict
-------

This endpoint predicts for ESM-2 35M.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.predictor(

          model="esm2-35m",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esm2-35m/predictor/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esm2-35m/predictor/"

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
      
      url <- "https://biolm.ai/api/v3/esm2-35m/predictor/"

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

.. http:post:: /api/v3/esm2-35m/predict/

   Predict endpoint for ESM-2 35M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **include** (*array of strings*, default: ["mean"]) — Output types to include in response embeddings; allowed values: "mean", "per_token", "logits"
      
      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences for prediction:
      
        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard unambiguous amino acid codes

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-35m/predict/ HTTP/1.1
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
      
        - **pdb** (*string*) — Predicted protein structure in PDB format
        - **mean_plddt** (*float*, range: 0.0-100.0) — Mean predicted Local Distance Difference Test (pLDDT) score indicating prediction confidence

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "logits": [
        [
          0.8847527503967285,
          -0.13512253761291504,
          0.19969357550144196,
          1.151706337928772,
          -0.1531011462211609,
          -0.0028736740350723267,
          -0.2561047673225403,
          -0.5353550314903259,
          -0.19937662780284882,
          -0.7912870645523071,
          -0.6497725248336792,
          -0.1751900017261505,
          -0.7316774129867554,
          -1.3250795602798462,
          -0.7398577332496643,
          -1.6736376285552979,
          7.480727672576904,
          -1.6944754123687744,
          -1.379725694656372,
          -1.0519800186157227
        ],
        [
          0.3143197000026703,
          0.1602669060230255,
          -0.007087230682373047,
          3.6206164360046387,
          0.05813796818256378,
          -0.06776130199432373,
          -0.14420102536678314,
          -0.5095621347427368,
          -0.533858060836792,
          -0.5948295593261719,
          -0.568466067314148,
          -0.13696156442165375,
          -0.6174976825714111,
          -0.7373144030570984,
          -0.8859179615974426,
          -1.4303297996520996,
          -0.8957475423812866,
          -1.1306742429733276,
          -0.9877399802207947,
          -1.1650208234786987
        ],
        [
          3.653970956802368,
          -0.11865468323230743,
          0.20023618638515472,
          0.08971625566482544,
          0.3580380976200104,
          -0.36090484261512756,
          -0.20173552632331848,
          -0.4205380380153656,
          -0.5280295610427856,
          -0.715300440788269,
          -0.6383615732192993,
          -0.5129550695419312,
          -0.9578472375869751,
          -0.8396717309951782,
          -0.603166937828064,
          -1.2277979850769043,
          -1.1701500415802002,
          -1.3492285013198853,
          -1.110398769378662,
          -0.7579864859580994
        ],
        [
          0.35314008593559265,
          -0.4303331971168518,
          0.0325181782245636,
          0.1884450614452362,
          3.2223711013793945,
          -0.27374857664108276,
          -0.332050085067749,
          -0.8182462453842163,
          -0.6600256562232971,
          -0.7808043956756592,
          -0.8952537775039673,
          -0.4260849356651306,
          -0.7504400014877319,
          -0.96523118019104,
          -0.7315552830696106,
          -1.4334217309951782,
          -1.0923724174499512,
          -1.2500863075256348,
          -1.2082922458648682,
          -1.1873130798339844
        ],
        [
          0.9409108757972717,
          0.2935412526130676,
          0.8284621834754944,
          0.5983229875564575,
          0.7555992603302002,
          0.315284788608551,
          0.4872781038284302,
          0.1520216166973114,
          0.13138282299041748,
          -0.018999792635440826,
          -0.12769004702568054,
          0.2094501256942749,
          -0.18194586038589478,
          -0.0920477956533432,
          0.019332580268383026,
          -0.659536600112915,
          -0.5000015497207642,
          -0.550428569316864,
          -0.6473647356033325,
          -0.3100384771823883
        ],
        [
          0.36226579546928406,
          -0.5923343896865845,
          3.315760612487793,
          -0.1949518769979477,
          -0.1079130470752716,
          -0.27730488777160645,
          -0.1722545474767685,
          -0.5977306365966797,
          -0.6863598823547363,
          -0.9366760849952698,
          -0.8254699110984802,
          -0.29091909527778625,
          -0.7834360003471375,
          -0.9434289932250977,
          -0.7796578407287598,
          -1.4095619916915894,
          -0.8925543427467346,
          -1.4347946643829346,
          -0.9552619457244873,
          -1.1954394578933716
        ],
        [
          0.1513819694519043,
          -0.7164874076843262,
          0.007183492183685303,
          -0.4266624450683594,
          -0.05242812633514404,
          2.735940933227539,
          -0.14146265387535095,
          -0.7489705681800842,
          -0.6288569569587708,
          -1.0576789379119873,
          -0.964796245098114,
          -0.43710875511169434,
          -0.8276097774505615,
          -0.8629176020622253,
          -0.7900262475013733,
          -1.2362877130508423,
          -1.365944266319275,
          -1.3762744665145874,
          -0.9539467692375183,
          -0.856998085975647
        ],
        [
          0.6746570467948914,
          0.08225008845329285,
          0.5708328485488892,
          0.381375789642334,
          0.320120632648468,
          0.5735371112823486,
          0.5163217782974243,
          -0.09506973624229431,
          0.059787318110466,
          -0.28048115968704224,
          -0.5900707244873047,
          0.5409629940986633,
          0.046062588691711426,
          -0.09730616211891174,
          -0.5446329712867737,
          -0.770743727684021,
          -0.5062323808670044,
          -0.7860196232795715,
          2.6042208671569824,
          -0.7237383127212524
        ],
        [
          0.6265190839767456,
          -0.3451893925666809,
          0.20685246586799622,
          0.004094183444976807,
          0.3284705877304077,
          -0.056085020303726196,
          0.5286262035369873,
          -0.3374505043029785,
          -0.19932124018669128,
          -0.7495543360710144,
          -0.41693639755249023,
          0.052674151957035065,
          3.1295182704925537,
          -0.41657018661499023,
          -0.6250114440917969,
          -0.9812415838241577,
          -0.7406615614891052,
          -0.8289814591407776,
          -0.4939512610435486,
          -0.49352240562438965
        ]
      ],
      "sequence_tokens": [
        "M",
        "V",
        "L",
        "S",
        "<mask>",
        "G",
        "E",
        "W",
        "Q"
      ],
      "vocab_tokens": [
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
        "C"
      ]
    }
  ]
}


Encode
------

This endpoint encodes for ESM-2 35M.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python


      from biolmai import Client
      
      client = Client("YOUR_API_KEY")
      
      result = client.encoder(

          model="esm2-35m",

          # Add your parameters here

      )

      print(result)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

      curl -X POST https://biolm.ai/api/v3/esm2-35m/encoder/ \

        -H 'Authorization: Token YOUR_API_KEY' \

        -H 'Content-Type: application/json' \

        -d '{}'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python


      import requests
      
      url = "https://biolm.ai/api/v3/esm2-35m/encoder/"

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
      
      url <- "https://biolm.ai/api/v3/esm2-35m/encoder/"

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

.. http:post:: /api/v3/esm2-35m/encode/

   Encode endpoint for ESM-2 35M.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request JSON Object

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:
      
        - **include** (*array of strings*, default: ["mean"]) — Types of embeddings or logits to return; allowed values: "mean", "per_token", "logits"
      
      - **items** (*array of objects*, min: 1, max: 5) --- Input sequences:
      
        - **sequence** (*string*, min length: 1, max length: 2048, required) — Protein sequence using standard unambiguous amino acid codes; ambiguous amino acids not allowed

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/esm2-35m/encode/ HTTP/1.1
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
      
        - **pdb** (*string*) — Predicted protein structure in standard PDB file format.
      
        - **mean_plddt** (*float*, range: 0.0 - 1.0) — Mean predicted Local Distance Difference Test (pLDDT) confidence score for the predicted structure, indicating prediction accuracy (0.0 = low confidence, 1.0 = high confidence).

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

      {
  "results": [
    {
      "embeddings": [
        {
          "layer": 12,
          "embedding": [
            -0.04749707132577896,
            0.0048291562125086784,
            0.18371263146400452,
            0.024757906794548035,
            -0.07571955025196075,
            -0.061270572245121,
            -0.14540418982505798,
            0.004863913170993328,
            0.24599158763885498,
            0.06768592447042465,
            0.15966983139514923,
            -0.019701583310961723,
            0.065043143928051,
            -0.24901528656482697,
            -0.08538667857646942,
            -0.11100925505161285,
            0.07086732238531113,
            0.15286721289157867,
            0.061651647090911865,
            -0.25318285822868347,
            0.018937435001134872,
            0.10253562778234482,
            -0.06557448208332062,
            0.06486032158136368,
            -0.06569051742553711,
            -0.002738773822784424,
            0.10550546646118164,
            0.0808994248509407,
            -0.23315301537513733,
            0.013505654409527779,
            -0.3013409972190857,
            0.031698327511548996,
            -0.018290486186742783,
            -0.03061847761273384,
            -0.042040083557367325,
            -0.33903104066848755,
            -0.008044108748435974,
            0.09811365604400635,
            0.09238491207361221,
            0.038183681666851044,
            0.05439556762576103,
            0.23295184969902039,
            -0.2585785388946533,
            -0.07195649296045303,
            0.1384507268667221,
            -0.1113409548997879,
            6.191547870635986,
            0.15773089230060577,
            -0.11599017679691315,
            0.13137400150299072,
            0.22349639236927032,
            0.07843097299337387,
            -0.24292583763599396,
            0.113283671438694,
            0.23800282180309296,
            0.008619246073067188,
            0.06434796750545502,
            -0.03938128799200058,
            -0.04069272056221962,
            0.03898422792553902,
            0.21954825520515442,
            -0.05807232856750488,
            -0.013476049527525902,
            -0.08048920333385468,
            0.005591236054897308,
            0.04192211106419563,
            -0.5381245613098145,
            -0.196457177400589,
            0.17118903994560242,
            0.10054264217615128,
            -0.06085291504859924,
            0.10528447479009628,
            0.1054539903998375,
            -0.09161502867937088,
            -0.007809762842953205,
            -0.05630826950073242,
            -0.1734745055437088,
            0.15497975051403046,
            -0.1546884924173355,
            -0.04742521792650223,
            0.10440371930599213,
            0.19735364615917206,
            0.04717987775802612,
            0.044527214020490646,
            0.14389201998710632,
            -0.10488808155059814,
            0.050431929528713226,
            -0.07294832915067673,
            -0.02267194353044033,
            0.16338637471199036,
            -0.13116894662380219,
            -0.01893160305917263,
            -0.22896794974803925,
            0.09283518046140671,
            0.027354955673217773,
            -0.14310601353645325,
            -0.05725228786468506,
            -0.1361374408006668,
            -0.08064331859350204,
            0.20756717026233673,
            0.05868769809603691,
            0.18726947903633118,
            0.0779009684920311,
            0.20978228747844696,
            -0.09548317641019821,
            -0.02189197763800621,
            -0.07178335636854172,
            0.08255390077829361,
            -0.07651849091053009,
            0.05025257542729378,
            -0.04730036482214928,
            -0.09620323032140732,
            0.0503060445189476,
            -0.1314796656370163,
            -0.10865992307662964,
            -0.06303657591342926,
            0.1872989684343338,
            0.017393989488482475,
            -0.019102465361356735,
            0.00291578471660614,
            0.01784677989780903,
            -0.04800025373697281,
            -0.031025618314743042,
            0.06504768133163452,
            -0.17085719108581543,
            0.14534001052379608,
            0.03548425808548927,
            0.02292010746896267,
            0.17970333993434906,
            -0.05020327493548393,
            -0.19415012001991272,
            0.018953591585159302,
            -0.07010392844676971,
            0.006709945388138294,
            -0.15990345180034637,
            -0.011879743076860905,
            -0.06984531879425049,
            0.17723438143730164,
            -0.052846312522888184,
            0.10709051787853241,
            0.1919310837984085,
            0.03276462107896805,
            0.017752885818481445,
            -0.2687925100326538,
            0.13751472532749176,
            -0.148160919547081,
            -0.12362559139728546,
            0.08969451487064362,
            0.05635293200612068,
            -0.10920237004756927,
            0.03786133602261543,
            0.04066123440861702,
            -0.016527822241187096,
            0.04282236099243164,
            0.08012697845697403,
            0.002194829983636737,
            -0.1591179072856903,
            -0.050799835473299026,
            -0.025392884388566017,
            -0.08705822378396988,
            -0.023516936227679253,
            -0.25605520606040955,
            -0.01250340323895216,
            0.17696738243103027,
            0.17446622252464294,
            -0.039364613592624664,
            -0.24439994990825653,
            -0.03249099478125572,
            -0.2806745171546936,
            0.2432175576686859,
            -0.10171365737915039,
            0.10685187578201294,
            0.08254464715719223,
            0.02173222228884697,
            -0.22558172047138214,
            0.13487617671489716,
            0.11014264822006226,
            -0.1509358137845993,
            0.14347964525222778,
            -0.12196073681116104,
            0.07090530544519424,
            0.09889394044876099,
            -0.007631764747202396,
            0.16760194301605225,
            -0.07141177356243134,
            -0.00044828205136582255,
            -0.010783900506794453,
            0.08283926546573639,
            -0.12871114909648895,
            0.13135819137096405,
            -0.14104238152503967,
            -0.05759687349200249,
            -0.01841595582664013,
            -0.08803603798151016,
            0.09015659242868423,
            -0.19898618757724762,
            -0.03687353804707527,
            -0.05694349482655525,
            0.3102562725543976,
            0.2126455008983612,
            0.051582686603069305,
            -0.2321610450744629,
            -0.15896466374397278,
            0.1039167195558548,
            -0.1759680211544037,
            0.12374000251293182,
            -0.28320416808128357,
            -0.10805954039096832,
            0.07966486364603043,
            -0.09902343899011612,
            0.13801482319831848,
            0.03503292053937912,
            0.06303057074546814,
            -0.012615261599421501,
            0.08663833141326904,
            -0.06695638597011566,
            -0.098719522356987,
            -0.09191150963306427,
            -0.0468902513384819,
            -0.04054086282849312,
            -0.05245796591043472,
            -0.23737700283527374,
            0.09779559075832367,
            -0.12088195979595184,
            0.15228241682052612,
            0.1315939575433731,
            0.06584817916154861,
            -0.0018736079800873995,
            0.07809958606958389,
            -0.0943816676735878,
            -0.008151458576321602,
            -0.1601565182209015,
            0.12914949655532837,
            0.018502099439501762,
            -0.028924908488988876,
            0.09729312360286713,
            0.10657844692468643,
            -0.034450553357601166,
            0.1258762925863266,
            0.055931903421878815,
            0.06196027994155884,
            -0.03960409015417099,
            -0.2529292106628418,
            0.058832038193941116,
            0.008217732422053814,
            0.049492429941892624,
            0.17637303471565247,
            -0.05572494864463806,
            0.032304126769304276,
            -0.04196588322520256,
            -0.042854733765125275,
            -0.027281101793050766,
            0.019979771226644516,
            -0.08897402137517929,
            0.0672319158911705,
            0.1908324807882309,
            0.10037879645824432,
            0.11589012295007706,
            -0.03692799061536789,
            -0.041810303926467896,
            0.009955952875316143,
            -0.04968660697340965,
            -0.03136170655488968,
            0.012948209419846535,
            0.08797533065080643,
            -0.15124870836734772,
            -0.05491754412651062,
            -0.00736341904848814,
            0.013442231342196465,
            0.005598859395831823,
            0.1738085001707077,
            -0.1647765040397644,
            0.05070686340332031,
            -0.01460673101246357,
            -0.01227609533816576,
            0.1246146708726883,
            -0.20044800639152527,
            -0.1321904957294464,
            -0.012037796899676323,
            -0.203578919172287,
            0.04848799854516983,
            -0.19933833181858063,
            0.009926686994731426,
            0.06594839692115784,
            -0.06619960814714432,
            -0.08055835217237473,
            -0.05444047972559929,
            0.012149943970143795,
            0.02238849177956581,
            0.02154545485973358,
            0.07473480701446533,
            0.0633879154920578,
            -0.023049643263220787,
            -0.16536982357501984,
            -0.05986776947975159,
            0.13323499262332916,
            -0.050771988928318024,
            0.05129174143075943,
            0.03633517026901245,
            0.1313750296831131,
            0.039926934987306595,
            0.11421725898981094,
            0.08114363253116608,
            0.12639714777469635,
            0.023692887276411057,
            0.03946606069803238,
            0.07660497725009918,
            0.04789487645030022,
            0.03143412247300148,
            0.14159449934959412,
            0.15479663014411926,
            0.04561498761177063,
            -0.06285560131072998,
            0.11502108722925186,
            0.060310911387205124,
            -0.08195726573467255,
            -0.07619187980890274,
            0.15549030900001526,
            0.07222587615251541,
            0.03078184649348259,
            0.11603596061468124,
            0.024522393941879272,
            -0.11478167772293091,
            0.3234351873397827,
            0.032457128167152405,
            -0.341397225856781,
            0.0817873552441597,
            0.06545288860797882,
            0.06596710532903671,
            0.14975222945213318,
            0.14770808815956116,
            0.19723036885261536,
            0.16991373896598816,
            -0.2037973403930664,
            -0.10181412845849991,
            0.07528935372829437,
            -0.009289638139307499,
            -0.13588830828666687,
            -0.08227845281362534,
            -0.13342903554439545,
            -0.013657662086188793,
            -0.12603667378425598,
            0.06006918102502823,
            -0.29999232292175293,
            0.09348765760660172,
            0.15850622951984406,
            0.26685839891433716,
            -0.03795614838600159,
            0.43811026215553284,
            -0.07431045919656754,
            0.04632285237312317,
            -0.1170736700296402,
            -0.1469598412513733,
            -0.08515973389148712,
            -0.02195080742239952,
            0.011635122820734978,
            -0.07999793440103531,
            -0.029707733541727066,
            0.11393348127603531,
            -0.20553281903266907,
            -0.05678290128707886,
            -0.09715161472558975,
            0.23786106705665588,
            0.06512487679719925,
            -0.2563111186027527,
            -0.03453441336750984,
            -0.10235171020030975,
            -0.1413501650094986,
            0.03971409797668457,
            0.07554945349693298,
            0.07549166679382324,
            0.09018348157405853,
            -0.12909220159053802,
            0.08430683612823486,
            -0.16185832023620605,
            -0.020606692880392075,
            -0.10975699126720428,
            0.02682581916451454,
            -0.08642518520355225,
            -0.019236087799072266,
            0.019842877984046936,
            -0.006710632238537073,
            -0.14237233996391296,
            -0.2010701447725296,
            -0.0714731439948082,
            -0.0812324583530426,
            -0.047413069754838943,
            0.22326484322547913,
            0.056584663689136505,
            0.07810784131288528,
            0.06835366785526276,
            -0.015165945515036583,
            0.05345608666539192,
            0.10068736225366592,
            -0.030296850949525833,
            0.10700073093175888,
            0.027904409915208817,
            0.13080525398254395,
            -0.11000846326351166,
            -0.10142618417739868,
            0.026177534833550453,
            0.016201559454202652,
            -0.2153005301952362,
            0.2537862956523895,
            -0.05872560665011406,
            0.1026376485824585,
            0.14452514052391052,
            0.07562099397182465,
            0.035655826330184937,
            0.06489406526088715,
            0.12987056374549866,
            0.008200056850910187,
            0.05616963654756546,
            -0.12078876793384552,
            0.07719988375902176,
            0.24335002899169922,
            -0.026962334290146828,
            -0.2340501993894577,
            -0.053319185972213745,
            -0.00738032441586256,
            0.08182705193758011,
            -0.06577518582344055,
            -0.10050754249095917,
            -0.034150511026382446,
            -0.03134516626596451,
            0.02156819775700569,
            0.07158169895410538,
            0.06405110657215118,
            0.009179001674056053,
            -0.02701226994395256,
            -0.10851286351680756,
            -0.004681122489273548,
            0.010201478376984596,
            -0.2963300943374634,
            0.18636520206928253,
            -0.02023688144981861,
            0.03782423585653305,
            -0.059163086116313934,
            -0.029390472918748856,
            -0.1548343002796173,
            0.08637101948261261,
            -0.4786232113838196,
            0.125407412648201,
            -0.07369208335876465,
            0.15842050313949585,
            0.0865321233868599,
            -0.037883561104536057,
            0.1033247709274292,
            0.002937124576419592,
            -0.0860772654414177,
            0.09410527348518372,
            -0.0633486732840538,
            -0.13651211559772491,
            -0.1775120198726654,
            -0.2575787901878357,
            -0.021471375599503517,
            -0.031387995928525925,
            0.13936319947242737,
            0.07744070142507553,
            -0.02466287650167942,
            -2.359668254852295,
            0.078097864985466,
            0.05842871218919754,
            0.009668963961303234,
            0.026379534974694252,
            -0.05353798344731331,
            -0.01112231332808733,
            -0.19392693042755127,
            -0.04929600656032562,
            -0.1276622861623764,
            -0.024573717266321182,
            -0.06340952217578888,
            0.1956223100423813,
            -0.04559851065278053,
            -0.1906643807888031,
            -0.07722197473049164,
            0.10945230722427368,
            -0.23365750908851624,
            -0.19207315146923065,
            0.11432752758264542
          ]
        }
      ]
    }
  ]
}


Performance
-----------

- ESM-2 35M is optimized for GPU inference, running efficiently on NVIDIA L4 GPUs.

- Typical inference speed is approximately 1.5 seconds per single-sequence prediction, significantly faster than larger ESM-2 variants (e.g., 650M, 3B, 15B), which typically take between 4 and 10 seconds per sequence on the same hardware.

- Compared to larger ESM-2 models (150M, 650M, and 3B), the 35M parameter model has reduced predictive accuracy on structure prediction benchmarks:
  
  - On CASP14 test set, ESM-2 35M achieves a TM-score of 0.41, compared to 0.47 (150M), 0.51 (650M), and 0.52 (3B).
  
  - On CAMEO test set, ESM-2 35M achieves a TM-score of 0.56, compared to 0.65 (150M), 0.70 (650M), and 0.72 (3B).

- Unsupervised contact prediction accuracy (long-range precision at L) is 0.30, lower than larger ESM-2 models: 0.44 (150M), 0.52 (650M), and 0.54 (3B).

- Despite lower accuracy, ESM-2 35M offers a substantial performance advantage in terms of speed and computational efficiency, making it suitable for high-throughput applications or initial screening tasks where rapid inference is prioritized over maximum predictive accuracy.

- BioLM has optimized deployment of ESM-2 35M to ensure stable GPU utilization, efficient memory management, and minimal latency overhead during inference.


Applications
------------

- Rapid structure prediction for protein engineering workflows, enabling researchers to quickly screen and prioritize candidate proteins for stability, folding, or functional properties; particularly valuable when experimental structure determination is slow or costly; not optimal for proteins undergoing significant conformational changes or intrinsically disordered regions.
- High-throughput structural annotation of metagenomic protein sequences, allowing biotech companies to rapidly identify novel protein folds or domains from environmental samples; useful for discovering new protein scaffolds or functions in enzyme engineering or synthetic biology; may have reduced accuracy for sequences with minimal evolutionary similarity to known proteins.
- Single-sequence structure modeling for protein design tasks, enabling computational design teams to evaluate designed sequences without relying on multiple sequence alignments (MSAs); especially beneficial for novel proteins lacking evolutionary homologs; however, accuracy may decrease relative to MSA-dependent methods for highly divergent sequences or complex multidomain proteins.
- Structural embedding generation for protein function prediction, providing computational biologists with atomic-level embeddings that capture evolutionary and structural information; valuable for clustering proteins by functional similarity or predicting functional sites; less suitable for precise modeling of protein-protein interactions or complexes without additional docking methods.
- Fast structural characterization for protein variant screening, allowing protein engineering teams to quickly assess structural impacts of mutations or sequence modifications; beneficial for stability engineering or affinity maturation workflows; accuracy may be lower for mutations causing significant structural rearrangements or large insertions/deletions.


Limitations
-----------

- **Maximum Sequence Length**: The ESM-2 35M API accepts sequences up to a maximum of ``2048`` amino acids. Longer sequences must be truncated or split into multiple requests.
- **Batch Size**: The maximum ``batch_size`` is ``5`` sequences per request. For larger-scale analyses, parallel requests are required.
- The ESM-2 35M model is optimized for rapid structure prediction directly from single sequences, but accuracy is generally lower compared to AlphaFold2 for proteins with very few evolutionary homologs (low MSA depth). For orphan proteins or highly novel sequences without evolutionary context, predictions may be less reliable.
- Model accuracy (measured by predicted LDDT) correlates with language model perplexity; sequences poorly modeled by the language model (high perplexity) typically yield lower-confidence predictions. Users should interpret predictions cautiously for sequences with high perplexity scores.
- ESM-2 35M is not designed to predict protein complexes or interactions. While it can process artificially concatenated chains, accuracy for protein-protein interfaces and complex structures is significantly lower compared to specialized multimeric predictors such as AlphaFold-Multimer.
- The model does not provide sequence embeddings or encodings suitable for downstream clustering or visualization tasks. For applications requiring embeddings, consider using embedding-focused models such as ProtT5 or ESM-2 embedding variants.


How We Use It
-------------

The ESM-2 35M model enables rapid, scalable exploration of protein sequence space for protein engineering and optimization workflows, allowing researchers to quickly generate accurate sequence embeddings and structural predictions to guide experimental prioritization. By integrating ESM-2 35M embeddings into downstream predictive models and generative AI pipelines, BioLM accelerates tasks such as enzyme design, antibody maturation, and multi-round optimization cycles, resulting in reduced experimental costs and improved hit rates.

- Integrates efficiently with predictive and generative modeling workflows to streamline protein design and optimization.
- Enables rapid sequence-based ranking and filtering, significantly reducing experimental timelines and resource requirements.


Related
-------

- ``AlphaFold2`` – Provides highly accurate structure predictions using MSAs, complementary to ESM-2 35M's fast single-sequence predictions.
- ``ESMFold`` – Uses ESM-2 representations to rapidly predict atomic-level protein structures directly from sequence, ideal for large-scale structural characterization.
- ``ESM-2 150M`` – A larger-scale version of ESM-2 35M, offering improved accuracy at the cost of increased computational resources.
- ``ESM-IF1`` – Inverse folding model utilizing ESM representations, complementary for protein design tasks based on predicted structures from ESM-2 35M.


References
----------

- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2022). `Evolutionary-scale prediction of atomic level protein structure with a language model <https://doi.org/10.1126/science.ade2574>`_. *Science*.

