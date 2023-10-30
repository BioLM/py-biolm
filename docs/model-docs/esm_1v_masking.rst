======
ESM-1V
======

.. article-info::
    :avatar: img/book_icon.png
    :date: Oct 18, 2021
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

On this page, we will show and explain the use of ESM-1V. As well as document the BioLM API for folding, demonstrate no-code and code interfaces to folding.


-----------
Description
-----------

ESM-1v is a part of the ESM (Evolutionary Scale Modeling) series, which encompasses a collection of Transformer-based protein language models such as ESM-2 and ESMFold. This model specializes in executing zero-shot predictions, particularly focusing on determining the impacts of mutations on protein functionality. As articulated by Meier et al., 2021, *"Modeling the effect of sequence variation on function is a fundamental problem for understanding and designing proteins"*. This emphasizes the critical role of ESM-1v in delineating the functional implications of sequence variations in proteins. The models are trained exclusively on functional molecules, facilitating an evaluative capability to discern the functional viability of novel molecules or the deleterious nature of specific mutations.

The architecture of ESM-1v is constructed based on a 'fill-in-the-blank' framework. During the training process, 15% of residues in each sequence are masked, compelling the model to predict the identities of the concealed residues. The weights of the neural network are iteratively updated to optimize the model’s predictive performance.

For prediction tasks, ESM-1v employs a consistent input strategy used during training. It requires a sequence with masked residues, and the model predicts the identities of the masked components, providing a likelihood score associated with each prediction. This likelihood score, ranging from 0 to 1, acts as an indicator of the predicted functionality of a sequence, reflecting the likely accuracy of the unmasked sequence's ability to form a functional protein.


--------
Benefits
--------

* The API can be used by biologists, data scientists, engineers, etc. The key values of the BioLM API is speed, scalability and cost.

* The API has been customized to allow users to easily see how the likelihood of the sequence being functional with the wild-type residue compares to a single-AA mutation at that position.

* The BioLM API allows scientists to programmatically interact with ESM-1V, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

* Our unique API UI Chat allows users to interact with our API and access multiple language models without the need to code!

* The benefit of having access to multiple GPUs is parallel processing.

-----------
Performance
-----------

Graph of average RPS for varying number of sequences

.. note::

   We are in the process of adding a graph.



---------
API Usage
---------

This is the url to use when querying the BioLM ESM-1V Prediction Endpoint: https://biolm.ai/api/v1/models/esm1v_t33_650M_UR90S_1/predict/

The BioLM API endpoint has been customized to return the likelihoods for every AA unmasked at any <mask> position, so you can easily see how the likelihood of the sequence being functional with the wild-type residue compares to a single-AA mutation at that position.
The way to get a straight, “what is the likelihood of function of this sequence” out of this model, is to mask one AA, then get the WT probability for the WT AA, returned by the API.
Furthermore, the BioLM API has 5 endpoints, as there are five models trained randomly on the same data. Hence, the likelihoods coming out of each one for the same input are slightly different.
The best results are achieved by averaging the likelihoods given by all 5 models for a given AA at a given position.


*Definitions*

-Request Keys:

data:
   Inside each instance, there's a key named "data" that holds another dictionary. This dictionary contains the actual input data for the prediction.

text:
   Inside the "data" dictionary, there's a key named "text". The value associated with "text" should be a string containing the amino acid sequence that the user wants to submit for structure prediction.


-Response Keys:

predictions:
   This is the main key in the JSON object that contains an array of prediction results. Each element in the array represents a set of predictions for one input instance.

Score:
   This represents the confidence or probability of the model's prediction for the masked token. A higher score indicates higher confidence.

Token:
   The predicted token's identifier as per the model's tokenization scheme. It's an integer that corresponds to a particular token (in this case, a particular amino acid) in the model's vocabulary.

Token_str:
   Represents the predicted token as a string. That is, the amino acid that was predicted to fill in the masked position in the sequence.

Sequence:
   Represents the complete sequence with the masked position filled in by the predicted token.



^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/esm1v_t33_650M_UR90S_1/predict/' \
               --header "Authorization: Token $BIOLMAI_TOKEN" \
               --header 'Content-Type: application/json' \
               --data '{
                  "instances": [{
                     "data": {"text": "QERLKSIVRILE<mask>SLGYNIVAT"}
                  }]
               }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/esm1v_t33_650M_UR90S_1/predict/"

            payload = json.dumps({
            "instances": [
               {
                  "data": {
                  "text": "QERLKSIVRILE<mask>SLGYNIVAT"
                  }
               }
            ]
            })
            headers = {
            'Authorization': 'Token {}'.format(os.environ['BIOLMAI_TOKEN']),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)


    .. tab-item:: Biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            seqs = ["MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"]

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
            params = "{
            \"instances\": [
               {
                  \"data\": {
                  \"text\": \"MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ\"
                  }
               }
            ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/esm2_t33_650M_UR50D/predict/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)


^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

         {
         "predictions": [
            [
               {
               "score": 0.10117799043655396,
               "token": 4,
               "token_str": "L",
               "sequence": "Q E R L E U T G R L S L G Y N I V A T"
               },
               {
               "score": 0.07831988483667374,
               "token": 8,
               "token_str": "S",
               "sequence": "Q E R L E U T G R S S L G Y N I V A T"
               },
               {
               "score": 0.0764596164226532,
               "token": 10,
               "token_str": "R",
               "sequence": "Q E R L E U T G R R S L G Y N I V A T"
               },
               {
               "score": 0.0663750097155571,
               "token": 7,
               "token_str": "V",
               "sequence": "Q E R L E U T G R V S L G Y N I V A T"
               },
               {
               "score": 0.06510740518569946,
               "token": 12,
               "token_str": "I",
               "sequence": "Q E R L E U T G R I S L G Y N I V A T"
               },
               {
               "score": 0.06203952059149742,
               "token": 6,
               "token_str": "G",
               "sequence": "Q E R L E U T G R G S L G Y N I V A T"
               },
               {
               "score": 0.06067674607038498,
               "token": 5,
               "token_str": "A",
               "sequence": "Q E R L E U T G R A S L G Y N I V A T"
               },
               {
               "score": 0.057782694697380066,
               "token": 15,
               "token_str": "K",
               "sequence": "Q E R L E U T G R K S L G Y N I V A T"
               },
               {
               "score": 0.05674279108643532,
               "token": 11,
               "token_str": "T",
               "sequence": "Q E R L E U T G R T S L G Y N I V A T"
               },
               {
               "score": 0.05069689080119133,
               "token": 9,
               "token_str": "E",
               "sequence": "Q E R L E U T G R E S L G Y N I V A T"
               },
               {
               "score": 0.0472114197909832,
               "token": 18,
               "token_str": "F",
               "sequence": "Q E R L E U T G R F S L G Y N I V A T"
               },
               {
               "score": 0.04127753898501396,
               "token": 13,
               "token_str": "D",
               "sequence": "Q E R L E U T G R D S L G Y N I V A T"
               },
               {
               "score": 0.04123111814260483,
               "token": 17,
               "token_str": "N",
               "sequence": "Q E R L E U T G R N S L G Y N I V A T"
               },
               {
               "score": 0.03877052664756775,
               "token": 14,
               "token_str": "P",
               "sequence": "Q E R L E U T G R P S L G Y N I V A T"
               },
               {
               "score": 0.03758937492966652,
               "token": 16,
               "token_str": "Q",
               "sequence": "Q E R L E U T G R Q S L G Y N I V A T"
               },
               {
               "score": 0.03457427769899368,
               "token": 19,
               "token_str": "Y",
               "sequence": "Q E R L E U T G R Y S L G Y N I V A T"
               },
               {
               "score": 0.025788413360714912,
               "token": 21,
               "token_str": "H",
               "sequence": "Q E R L E U T G R H S L G Y N I V A T"
               },
               {
               "score": 0.02108406089246273,
               "token": 23,
               "token_str": "C",
               "sequence": "Q E R L E U T G R C S L G Y N I V A T"
               },
               {
               "score": 0.020976385101675987,
               "token": 20,
               "token_str": "M",
               "sequence": "Q E R L E U T G R M S L G Y N I V A T"
               },
               {
               "score": 0.015546774491667747,
               "token": 22,
               "token_str": "W",
               "sequence": "Q E R L E U T G R W S L G Y N I V A T"
               }
            ]
         ]
         }




----------
Related
----------

:doc:`/model-docs/esm2_fold`
:doc:`/model-docs/esm2_embeddings`
:doc:`/model-docs/ESM-InverseFold`


------------------
Model Background
------------------



ESM-1v is a large-scale transformer-based protein language model containing 650 million parameters, developed for predicting the effects of genetic variants on protein function (*Meier et al., 2021*). It was pretrained on a dataset of 98 million diverse protein sequences from Uniref90 2020-03, allowing it to learn broad evolutionary sequence variation patterns. The pretraining approach followed that of ESM-1b (*Rives et al., 2020*), using masked language modeling on the amino acid sequences without any task-specific supervised signals. As stated by Meier et al. (2021), "ESM-1v requires no task-specific model training for inference. Moreover, ESM-1v does not require MSA generation."

Inferencing with ESM-1v provides two key advantages over other state-of-the-art methods: (i) it can directly predict mutation impacts without needing additional task-specific training, and (ii) it can estimate fitness landscapes from a single forward pass through the model (*Meier et al., 2021*). This enables more efficient variant effect prediction compared to approaches requiring multiple steps like MSA generation and supervised retraining. By leveraging the self-supervised pretraining on large and diverse protein sequences, ESM-1v acquired generalizable knowledge of sequence-function relationships to allow variant consequence analysis solely from the primary structure.




.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Extraction Method
     - Description
   * - Masked marginal
     - During pretraining, probabilities were derived based on the mask noise. At every position, a mask token was introduced, and the model's predicted probabilities for the tokens at that position were recorded.
   * - Mutant Marginal
     - Probabilities were obtained based on the random token noise during pre-training. Out of the 15% predicted positions in the sequence during pre-training, 10% were randomly altered while 10% remained unchanged. The model aimed to accurately predict the token at these positions. In this extraction method, the researchers adhered to the pre-training approach by inputting mutated tokens and documenting the model's probability of correctness for these tokens.
   * - Wildtype marginal
     - A single forward pass was performed using the wildtype sequence. This method enabled fast scoring as just a single forward pass was used.
   * - Pseudolikelihood
     - The researchers refer to the method outlined in *Salazar et al., 2019.*




-----------------------
Applications of ESM-1V
-----------------------


ESM-1v has great potential in advancing our understanding of protein function and the implications of genetic variations, which is fundamental in many fields including medicine, genetics, and bioengineering.

* Variant Effect Prediction: ESM-1v can be used to predict how specific mutations or variants might affect the function of proteins. For example, in antibody engineering, By masking particular residues in an antibody sequence and using ESM-1v to predict the likely amino acids that could occur at those positions, one can gain insights into how different variants might affect antibody-antigen binding or other functional attributes.

* Drug Discovery: to predict how mutations might affect drug targets or to identify new potential drug targets based on the effect of natural variations.

* Enzyme engineering: to predict how engineered mutations might affect protein function, aiding in the design of proteins with desired properties. Furthermore, Identifying crucial residues in a binding site using ESM-1v with masking techniques holds promise in Enhancing Catalytic Efficiency, Developing Enzyme Inhibitors or Activators

* Predicting protein folding from sequence. Scientists can mask various portions of a sequence and analyze changes in the ESM-1v embedding to predict structural folds. Or mask different sequence regions to identify areas that most significantly alter the embedding away from the native fold.


