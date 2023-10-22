..
   Copyright (c) 2021 Pradyun Gedam
   Licensed under Creative Commons Attribution-ShareAlike 4.0 International License
   SPDX-License-Identifier: CC-BY-SA-4.0


=========
PROGEN-2: MEDIUM
=========

.. article-info::
    :avatar: img/book_icon.png
    :author: Zeeshan Siddiqui
    :date: October 19th, 2023
    :read-time: 7 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of the Progen-2 MEDIUM. As well as document the BioLM API for prediction, and demonstrate no-code and code interfaces for predictions.*

-----------
Description: 
-----------
ProGen2 is one of the  largest protein language models employing self-supervised pretraining on massive protein sequence data to generate useful representations for a variety of protein structure/function prediction and design tasks. It is one of the Attention-based models trained on protein sequences; these models, when trained on protein sequences, apply a mechanism to selectively focus on different parts of the input data to learn relationships and patterns among amino acids in protein sequences. As a protein language model, Progen2 is trained to predict masked amino acids from surrounding context. The model has great potential in generating synthetic libraries of functional proteins for discovery or iterative optimization.
The BioLM API offers access to Progen-2 Medium. Progen2-OAS, and Progen2-BDF90. On this page, the API usage for Progen-2 MEDIUM is provided. 


--------
Benefits
--------

* The BioLM API allows scientists to programmatically interact with Progen-2 MEDIUM, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable. 

* Our unique API UI Chat allows users to interact with our API and access multiple language models without the need to code!

* The benefit of having access to multiple GPUs is parallel processing. Each GPU can handle a different protein folding simulation, allowing for folding dozens of proteins in parallel!


---------
API Usage
---------

This is the url to use when querying the BioLM Progen-2 Prediction Endpoint: https://biolm.ai/api/v1/models/progen2v31/generate/

*Definitions*

Request Keys: These keys together define the conditions and parameters for the sequence generation task requested from the Progen-2 language model via the BioLM API.

t: 
    represents the temperature parameter for the generation process. The temperature affects the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic

p: 
    represent a nucleus sampling parameter, which is a method to control the randomness of the generation by only considering a subset of the most probable tokens for sampling at each step.  Lower nucleus sampling probability, which usually makes sequence generation more conservative, results in sequences more closely matching the training dataset

max_length: 
    The maximum length of the generated sequence. The model will stop generating once this length is reached. 

num_samples:    
    The number of independent sequences the user wants the model to generate for the given prompt. For example, if this value is set to 2, you will get two different generated sequences for the prompt.

model: 
    This specifies which variant of the Progen-2 model to use for the generation. 


-Response Keys:

predictions: 
    This is the main key in the JSON object that contains an array of prediction results. Each element in the array represents a set of predictions for one input instance.

generated: 
    Contains a list of generated sequences and their associated information. Each sequence and its info are represented as a dictionary. The number of dictionaries in this list corresponds to the number of generated sequences the user requested.

text:   
contains the actual generated sequence produced by the model based on the provided prompt and parameters.

ll_sum: 
    Represents the sum of log-likelihoods for each token in the generated sequence. The log-likelihood gives an indication of how probable or confident the model was in generating each token. A higher log-likelihood indicates higher confidence.

ll_mean: 
    This represents the average log-likelihood per token for the generated sequence. It's calculated by taking the mean of the log-likelihoods of all the tokens in the sequence. It provides an indication of the model's confidence in the generation.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/progen2v31/generate/' \
            --header 'Content-Type: application/json' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --data '{
            "instances": [{
                "data": {"text": "M",
                        "t": 0.7,
                        "p": 0.6,
                        "max_length": 1020,
                        "num_samples": 2,
                        "model": "progen2-medium"}
            }]
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/progen2v31/generate/"

            payload = json.dumps({
            "instances": [
                {
                "data": {
                    "text": "M",
                    "t": 0.7,
                    "p": 0.6,
                    "max_length": 1020,
                    "num_samples": 2,
                    "model": "progen2-medium"
                }
                }
            ]
            })
            headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Token {}'.format(os.environ['BIOLMAI_TOKEN']),
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: biolmai SDK
        :sync: sdk

        Content 2

    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            "Content-Type" = "application/json",
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN')),
            )
            params = "{
            \"instances\": [
                {
                \"data\": {
                    \"text\": \"M\",
                    \"t\": 0.7,
                    \"p\": 0.6,
                    \"max_length\": 1020,
                    \"num_samples\": 2,
                    \"model\": \"progen2-medium\"
                }
                }
            ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/progen2v31/generate/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)

^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

        {
        "predictions": {
            "generated": [
            {
                "text": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDSGYSYGPPDYWGQGTLVTVSS",
                "ll_sum": -24.2924747467041,
                "ll_mean": -0.20243728905916214
            },
            {
                "text": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDLGYSSGWYGGAFDYWGQGTLVTVSS",
                "ll_sum": -25.01990509033203,
                "ll_mean": -0.20177342742681503
            }
            ]
        }
        }
---------
Performance
---------

Graph of average RPS for varying number of sequences (Progen-2 MEDIUM)

.. figure:: 
   :scale: 
   :alt: 

   This is the caption of the figure (a simple paragraph).

   The legend consists of all elements after the caption.

.. note::
   This graph will be added very soon. 



----------
Related 
----------
* Progen-2 Medium
* Progen-2 BFDO90

.. note::
    If there is a Progen-2 model you would like to see on the BioLM.ai website, let us know!


------------------
Model Background
------------------

*Madani et al., 2022* trained a suite of models ranging from 151M to 6.4B parameters. The models differ in size and training datasets (collectively comprise over a billion proteins). For more details, refer to Table 1 in here: https://browse.arxiv.org/pdf/2206.13517.pdf

Progen2 was pretrained on a dataset of over 180 million protein sequences from public sources like UniRef50 and the Protein Data Bank, learning contextual representations through masked language modeling. This huge dataset combined with a tokenization scheme (Vocabulary size around 2500), preserves biochemical motifs and enables Progen2 to learn meaningful sequence-structure-function relationships. 

The PROGEN-2 models are autoregressive transformers with next-token prediction language modeling as the learning objective. As the models scale up from 151 million to 6.4 billion parameters, they become more adept at capturing the distribution of protein sequences derived from observed evolutionary data.

As mentioned earlier, the standard PROGEN2 models were pre-trained on a mixture of Uniref90 (*Suzek et al., 2015*) and BFD30 (*Steinegger & Söding, 2018*) databases. The BioLM API offers access to PROGEN2-medium, which has 764M parameters and 27 layers. “Increasing number of parameters allows the model to better capture the distribution of observed evolutionary sequences” -*Madani et al., 2022*. 

In the PROGEN2-BFD90 model, Uniref90 is combined with representative sequences, each having a minimum of 3 cluster members, post clustering of UniprotKB, Metaclust, SRC, and MERC at 90% sequence identity. The BFD90 dataset, thus created, is about double the size of Uniref90. According to Table 8 in *Madani et al., 2022*, Uniref90+BFD90 has a slightly lower perplexity and higher Spearman's rho for "antibody general" tasks, indicating potentially better performance in these areas (antibody developability/enginering). Conversely, Uniref90+BFD30 has a higher Spearman's rho for "antibody binding", suggesting better performance on this specific task.

For protein engineering tasks with narrow fitness landscapes, such as optimizing a specific property like stability, larger protein language models can actually degrade performance compared to smaller models. The additional parameters allow larger models to overfit noise and irrelevant patterns not pertinent to the narrow objective. This was evidenced by ProGen-2’s smaller 151M parameter model outperforming a much larger 1.5B parameter model on targeted protein optimization. Overall, When focusing on a narrow property, model architecture and training methodology seem to matter less than appropriate model size and regularization.Furthermore, smaller models, which capture the observed evolutionary data distribution more poorly, can systematically outperform larger models at zero-shot fitness prediction.

Larger models may be beneficial in  wider fitness landscapes. The larger models might capture more complex relationships between amino acid sequences and their corresponding fitness, which could be crucial in landscapes with a higher level of mutational tolerance. As the model size increases significantly, new, perhaps unexpected, behaviors or capabilities might manifest. In particular, very large models might be better at identifying high-fitness variants in challenging landscapes characterized by low homology (low similarity between sequences) and high epistasis (interactions between different mutations). This could be promising for protein engineering efforts aiming to discover “novel, high-fitness protein variants in a vast and complex sequence space.” -Madani et.al 2022.

For the specialized PROGEN2-OAS training, unpaired antibody sequences from the Observed Antibody Space (OAS) database were utilized. OAS houses a refined assortment of 1.5 billion antibody sequences from eighty immune repertoire sequencing studies, encompassing heavy and light chain sequences from six species including humans, mice, rats, camels, rabbits, and rhesus. Since sequences in OAS possess a certain degree of redundancy, the researchers clustered the OAS sequences at 85% sequence identity using Linclust (Steinegger & Söding, 2018), generating a set of 554M sequences for model training. Note, to overcome bias in the OAS data, and produce full-length antibody sequences, the researchers initiated generation with a three-residue motif commonly found at the beginning of human heavy chain sequences (EVQ).

“For antibody fitness prediction, training on samples from immune repertoire sequencing (OAS) in theory sounds like a good idea, but in practice performs poorly” -Ali Madani. Interestingly, Models trained on universal protein databases perform better in predicting general antibody properties when compared to Progen-2 OAS.  When comparing the models' performance in predicting binding affinity (KD values) of antibodies,  PROGEN2-small performs the best and PROGEN2-OAS the worst. When comparing the models' performance  in predicting general protein properties like expression quality and TM melting temperatures. PROGEN2-Xtra large outperfoms all, but PROGEN2-OAS outperforms Progen-2 small.

.. note::
   The model background above covers information for Progen-2 OAS, Medium and BFD90. 


-----------------------
Applications of Progen-2 
-----------------------

ProGen-2 is capable of generating novel protein sequences, predicting protein functions, and assessing protein fitness without extra fine-tuning. It aids in understanding evolutionary patterns by capturing the distribution of observed evolutionary sequences, facilitating the design of new proteins with desired properties and functionalities, and providing insights into their viability and effectiveness. 

The model has a big use in enzyme engineering, by capturing the distribution of observed evolutionary sequences For instance, by analyzing the evolutionary sequences, one could identify conserved residues or motifs that are crucial for enzyme function or stability. This information could then be used to design novel enzymes with desired properties, such as increased catalytic activity or altered substrate specificity, by mimicking or building upon these conserved evolutionary features. It provides a data-driven approach to identify and understand the fundamental features that could be engineered to achieve desired enzymatic properties.

* Capturing the distribution of observed evolutionary sequences. This can be used in enzyme engineering; by analyzing the evolutionary sequences, scientist can identify conserved residues or motifs that are crucial for enzyme function or stability. In addition, ProGen-2 can be used to complete partial sequences of an enzyme. 

* Generating novel viable protein sequences.

* Predicting protein fitness without requiring additional fine-tuning

* generation of antibody sequence libraries. For instance, if you're aiming to create a library targeting a specific antigen, ProGen-2 could generate a variety of sequences that have desirable properties such as high affinity or specificity, based on patterns learned from known antibody-antigen interactions.

.. note::
   The applications above covers general use-cases for Progen-2 OAS, Medium and BFD90. 