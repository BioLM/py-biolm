..
   Copyright (c) 2021 Pradyun Gedam
   Licensed under Creative Commons Attribution-ShareAlike 4.0 International License
   SPDX-License-Identifier: CC-BY-SA-4.0


=========
PROGEN-2: BFD90
=========

.. article-info::
    :avatar: img/book_icon.png
    :author: Zeeshan Siddiqui
    :date: October 19th, 2023
    :read-time: 7 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of the Progen-2 BFD90. As well as document the BioLM API for prediction, and demonstrate no-code and code interfaces for predictions.*

-----------
Description:
-----------
ProGen2 represents one of the largest protein language models, leveraging self-supervised pretraining on extensive protein sequence data to generate useful representations applicable to diverse protein structure and function prediction and design applications. As an attention-based model trained on protein sequences, ProGen2 employs a mechanism to selectively focus on informative regions of input data, learning intricate patterns and relationships among amino acids within protein sequences. Specifically, ProGen2 is trained via masked language modeling to predict amino acids from surrounding sequence context. As a protein language model, ProGen2 shows considerable promise for generating synthetic libraries of functional proteins to empower discovery and iterative optimization.

The BioLM API offers access to Progen-2 Medium. Progen2-OAS, and Progen2-BDF90. On this page, the API usage for Progen-2 BFD90 is provided.


--------
Benefits
--------

* The BioLM API allows scientists to programmatically interact with Progen-2 BFD90, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

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
                        "model": "progen2-BFD90"}
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
                    "model": "progen2-BFD90"
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
                    \"model\": \"progen2-BFD90\"
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

Graph of average RPS for varying number of sequences (Progen-2 BFD90)


.. note::
   This graph will be added very soon.



--------
Related
--------

:doc:`/model-docs/Progen-2_Medium`
:doc:`/model-docs/Progen-2-OAS`


.. note::
    If there is a Progen-2 model you would like to see on the BioLM.ai website, let us know!


------------------
Model Background
------------------

*Madani et al., 2022* trained a suite of models ranging from 151M to 6.4B parameters. The models differ in size and training datasets (collectively comprise over a billion proteins). For more details, refer to Table 1 in here: https://browse.arxiv.org/pdf/2206.13517.pdf

Progen-2 was pretrained via masked language modeling on an expansive dataset of over 180 million protein sequences from public sources including UniRef50 and the Protein Data Bank. This enables Progen-2 to learn contextual sequence representations that capture motifs and sequence-structure-function relationships. A tokenization scheme with a vocabulary size of approximately 2500 was utilized to retain biochemical motifs within the sequences. In summary, pretraining Progen-2 on a massive and diversified protein sequence dataset empowers the model to learn expressive representations of sequence patterns, motifs, and residues that determine protein structure and function. As states by *-Madani et al., 2022.*, *“Increasing number of parameters allows the model to better capture the distribution of observed evolutionary sequences”*.

Progen-2 utilizes autoregressive transformer architectures trained with next-token prediction as the learning objective for language modeling of protein sequences. As model scale increases from 151 million to 6.4 billion parameters, Progen-2 becomes progressively more proficient at modeling the distribution of protein sequences present in observed evolutionary data. In summary, the combination of autoregressive modeling and large-scale pretraining enables Progen-2 to effectively capture sequence distributions reflective of natural protein evolution.

The standard Progen-2 models were pre-trained on a mixture of Uniref90 *(Suzek et al., 2015)* and BFD30 *(Steinegger & Söding, 2018)* databases.

The ProGen2-BFD90 model supplements Uniref90 with representative sequences clustered from UniprotKB, Metaclust, SRC, and MERC at 90% sequence identity. This generated the BFD90 dataset, approximately double the size of Uniref90. As reported in Table 8 by *Madani et al. (2022)*, Uniref90+BFD90 exhibited slightly lower perplexity and higher Spearman's rho on antibody developability/engineering tasks, potentially indicating superior performance on these objectives. In contrast, Uniref90+BFD30 showed higher Spearman's rho for antibody binding predictions, suggesting enhanced capabilities for this specific task.

For protein engineering endeavors with narrow fitness landscapes, such as optimizing a singular property like stability, larger protein language models can underperform compared to smaller models. The additional parameters enable overfitting to noise and extraneous patterns irrelevant to the focused objective. This was evidenced by the 151M parameter ProGen2 model outperforming a substantially larger 1.5B parameter version on targeted protein optimization. Overall, appropriate model size and regularization appear more crucial than architecture details when concentrating on a narrow property. Moreover, smaller models, which capture the observed protein sequence distribution less accurately, can systematically surpass larger models at zero-shot fitness. For broader fitness landscapes, larger models may confer benefits by capturing more intricate relationships between amino acid sequences and corresponding fitness. This could prove critical in landscapes exhibiting greater mutational tolerance. As model scale grows drastically, new and potentially unexpected capabilities may emerge. Very large models may excel at identifying high-fitness variants within challenging landscapes marked by low homology (sequence similarity) and high epistasis (inter-mutational interactions). This could hold promise for discovery of *"novel, high-fitness protein variants in a vast and complex sequence space"   -Madani et al., 2022.*

For specialized ProGen2-OAS training, unpaired antibody sequences were leveraged from the Observed Antibody Space (OAS) database, which contains a refined set of 1.5 billion heavy and light chain sequences from 80 immune repertoire sequencing studies across 6 species. To reduce redundancy, OAS sequences were clustered at 85% identity using Linclust (Steinegger & Söding, 2018), generating 554 million diverse sequences for training. To mitigate dataset bias and produce full-length antibodies, generation was initiated using a EVQ motif common at the start of human heavy chains. In summary, tailored training on broad antibody space data equips ProGen2-OAS for optimized antibody sequence generation.

As noted by Ali Madani, * "For antibody fitness prediction, training on immune repertoire sequencing samples (OAS) theoretically seems advantageous, yet in practice exhibits inferior performance.”* Interestingly, models trained on universal protein databases surpass Progen2-OAS at predicting general antibody properties. Comparative assessment of binding affinity (KD) prediction reveals Progen-2 small as superior, with Progen-2 OAS the lowest performer. However, for predicting general protein properties such as expression and thermal stability, Progen-2 extra large excels, while Progen-2 OAS outperforms Progen-2 small. In summary, Progen-2 models trained on broad protein sequence space rather than antibody-specific data demonstrate enhanced generalizability for predicting antibody properties, potentially due to the diversity and size of universal protein training data. However, antibody repertoire data provides some specialized benefits evident in predicting select protein engineering objectives.

.. note::
   The model background above covers information for Progen-2 OAS, Medium and BFD90.


-----------------------
Applications of Progen-2
-----------------------

Progen-2 enables generation of novel protein sequences, prediction of protein functions, and assessment of protein fitness without additional fine-tuning. It facilitates comprehension of evolutionary patterns by modeling the distribution of observed evolutionary sequences. This empowers design of proteins with targeted properties and functionalities, while garnering insights into viability and efficacy.

For enzyme engineering, Progen-2's capture of evolutionary sequence distributions has considerable utility. Analysis of conserved residues and motifs within evolutionary sequences can illuminate key determinants of enzyme function and stability. This knowledge enables the design of enzymes with optimized attributes like enhanced catalytic activity or altered substrate specificity by replicating or expanding upon these conserved evolutionary elements.

* Capturing the distribution of observed evolutionary sequences. This can be used in enzyme engineering; by analyzing the evolutionary sequences, scientist can identify conserved residues or motifs that are crucial for enzyme function or stability. In addition, ProGen-2 can be used to complete partial sequences of an enzyme.

* Generating novel viable protein sequences.

* Predicting protein fitness without requiring additional fine-tuning

* generation of antibody sequence libraries. For instance, if you're aiming to create a library targeting a specific antigen, ProGen-2 could generate a variety of sequences that have desirable properties such as high affinity or specificity, based on patterns learned from known antibody-antigen interactions.

.. note::
   The applications above covers general use-cases for Progen-2 OAS, Medium and BFD90.
