================
ESM-2 Embeddings
================

.. article-info::
    :avatar: img/book_icon.png
    :date: Oct 18, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*This page explains the use of ESM-2 for generating embeddings, and documents
the BioLM API to obtain embeddings for any protein sequence.*

-----------
Description
-----------

Large language models (LLMs) like ESM-2 can generate informative and static-length
feature representations of protein sequences. Such models' predictive
embeddings encode relevant biological properties, and are use to make classification,
regression, and other predictions like protein folds. Embeddings are vector representations
that can be extracted and utilized as inputs for a variety of downstream predictive
modeling tasks as an alternative to standard one-hot sequence encodings. 

In biology, feature engineering is often heavily tailored to each application.
However, the embeddings from pretrained language models provide broadly useful
representations across a multitude of applications, such as toxicity prediction,
functional likelihood, thermodynamic properties, structure, and more.

The BioLM API provides easy access to ESM-2 for generating insightful protein
embeddings to model experimental sequences and data. This service accelerates
tasks like sequence similarity detection, therapeutic antibody design, simplifying
the transition from protein sequence data to actionable insights. By cost-optimizing the
infrastructure to compute a variety of protein embeddings, our API lowers barriers to
leveraging advanced LLMs, accelerating the development of predictive
tools from protein sequences.


--------
Benefits
--------

* Always-on, auto-scaling GPU-backed APIs (`Status Page`_); highly-scalable parallelization.
* Save money on infrastructure, GPU costs, and development time.
* Quickly integrate multiple embeddings into your workflows.
* Use our Chat Agents and other Web Apps to interact with bio-LLMs using no code.


---------
API Usage
---------

The endpoint for querying ESM-2 embeddings, protein contact maps, attention
maps, and logits is `https://biolm.ai/api/v1/models/esm2_t33_650M_UR50D/transform/ <https://api.biolm.ai>`_.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/esm2_t33_650M_UR50D/predict/' \
            --header "Authorization: Token ed3fa24ec0432c5ba812a66d7b8931914c73a208d287af387b97bb3ee4cf907e" \
            --header 'Content-Type: application/json' \
            --data '{
            "instances": [{
               "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
            }]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/esm2_t33_650M_UR50D/predict/"

            payload = json.dumps({
            "instances": [
               {
                  "data": {
                  "text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"
                  }
               }
            ]
            })
            headers = {
            'Authorization': 'Token {}'.format(os.environ['ed3fa24ec0432c5ba812a66d7b8931914c73a208d287af387b97bb3ee4cf907e']),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: Biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            seqs = [""MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"]

            cls = biolmai.ESM2Embeddings()
            resp = cls.Transform(seqs)

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



+++++++++++
Definitions
+++++++++++

data:
   Inside each instance, there's a key named "data" that holds another
   dictionary. This dictionary contains the actual input data for the
   prediction.

text:
   Inside the "data" dictionary, there's a key named "text". The value
   associated with "text" should be a string containing the amino acid sequence
   that the user wants to submit for structure prediction.


^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

         {
         "predictions": [
            {
               "name": "0",
               "mean_representations": {
               "33": [
                  0.008923606015741825,
                  -0.005895234644412994,
                  -0.0060966904275119305,
                  -0.016010720282793045,
                  -0.14031203091144562,
                  -0.044720884412527084,


.. note::
  The above response is only a small snippet of the full JSON response. However, all the relevant response keys are included.

+++++++++++
Definitions
+++++++++++

predictions:
   This is the main key in the JSON object that contains an array of prediction results. Each element in the array represents a set of predictions for one input instance.

mean_representations:
   This key holds the embeddings generated by the ESM-2 model for the corresponding input instance. These embeddings represent average values computed over certain dimensions of the model's output.

'33':
   Specifying a particular layer or dimension of the model's output from which the embeddings were derived.



----------
Related
----------

:doc:`/model-docs/ESM-1v`

:doc:`/model-docs/ESM_InverseFold`

:doc:`/model-docs/ESMFold`

------------------
Model Background
------------------

ESM-2 is an expanded transformer-based protein language model that achieves state-of-the-art performance across diverse protein modeling applications compared to previous models like ESM-1v.
As described by *Lin et al., (2022)*, "The resulting ESM-2 model family significantly outperforms previously state-of-the-art ESM-1b (a ∼650 million parameter model) at a comparable number of parameters, and on structure prediction benchmarks it also outperforms other recent protein language models."
ESM-2 was pretrained on the full UniRef50 dataset, comprising 200 million sequences and 120 billion amino acid residues, drastically larger than ESM-1v's subset. The model architecture itself is also much larger, with 33 transformer layers and 1.6 billion parameters, versus 12 layers and 128 million parameters in ESM-1v.
To enable training such a large model, *Lin et al. (2022)* utilized a multi-node setup with per-token batch sizes up to 3.2 million, exploiting the capability of transformer models to leverage large batches. The model architecture applies sparsely-gated mixture-of-experts rather than standard transformers, alongside a multi-task pretraining approach combining language modeling with supervised auxiliary losses. These architectural improvements and training strategies enable ESM-2 to produce superior sequence representations compared to previous models like ESM-1v, providing new state-of-the-art capabilities for predictive modeling tasks in protein science.


-----------------------
Applications of ESM-2
-----------------------

The powerful protein sequence embeddings generated by ESM-2 have wide-ranging applications in protein science. They can aid in predicting protein-protein interactions and designing proteins with specified binding capabilities. Additionally, ESM-2 embeddings facilitate functional annotation of uncharacterized or novel proteins, expanding knowledge of the protein universe.
The embeddings can also be leveraged to anticipate the effects mutations have on protein function and stability, critical for protein design efforts. In drug discovery, they assist target identification by revealing structural and functional similarities with known drug targets. Finally, the high-dimensional sequence representations from ESM-2 expedite comparative analysis of proteins by illuminating conserved domains and regions of interest. This is pivotal for elucidating evolutionary relationships and shared functional attributes among protein families

* Enzyme engineering (enzyme optimization, transfer learning, directed evolution).

* Antibody engineering (machine learning models applied on antibody embeddings may predict affinity, expression, stability without lab assays).

* Protein-protein interaction design - Embeddings can be used to engineer proteins that interact with specific targets, like designing cellular signaling proteins.

* Membrane protein design.



.. _Status Page: https://status.biolm.ai






