================================
AbLang Heavy Chain Seq-coding
================================

.. article-info::
    :avatar: img/book_icon.png
    :date: Feb, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*On this page, we will show and explain the use of AbLang for Antibody Sequence Completion. As well as document the BioLM API, and demonstrate no-code  and code interfaces antibody design*

-----------
Description
-----------

It has been observed in the OAS database that approximately 80% of the sequences lack more than a single residue at the N-terminus, around 43% are devoid of the initial 15 positions, and about 1% possess at least one ambiguous residue at some position within the sequence. Residues may be missing due to sequencing inaccuracies, like ambiguous bases, or the constraints inherent in the sequencing methodologies employed. Completing the sequence by restoring the missing residues provides a great benefit to antibody drug discovery pipelines. 

Antibody sequence restoration can be accomplished through alignment to germline sequences, however this necessitates knowledge or prediction of the correct germline. A key advantage of AbLang is its capacity to restore missing residues without reliance on germline information, streamlining the process and enhancing speed and accuracy compared to germline-dependent approaches. Furthermore, quantitative evaluations demonstrate AbLang's superior ability to restore N-terminal residues in both heavy and light chains with up to 30 missing amino acids versus ESM-1b. AbLang performs relatively well at reconstructing sequences with randomly distributed missing residues, outperforming prior methods reliant on germline alignments.

AbLang also allows prediction of antibody developability, binding affinity, and other key antibody properties directly from sequence for antibody engineering and design applications *((Olsen et al., 2022)*)


--------
Benefits
--------

* Always-on, auto-scaling GPU-backed APIs (`Status Page`_); highly-scalable parallelization.
* Save money on infrastructure, GPU costs, and development time.
* Quickly integrate multiple embeddings into your workflows.
* Use our Chat Agents and other Web Apps to interact with bio-LLMs using no code.


----------------------------------------
API Usage AbLang_Heavy_Chain_Seq-coding
----------------------------------------

API endpoint for `https://biolm.ai/api/v2/ablang-heavy/encode/ <https://api.biolm.ai>`_.


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell
            curl --location 'https://biolm.ai/api/v2/ablang-heavy/encode/' \
            --data '{
            "items": [
               {
                  "sequence": "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS"
               }
            ],
            "params": {
               "include": "seqcoding"
            }
            }'

   .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v2/ablang-heavy/encode/"

            payload = "{\n    \"items\": [\n        {\n            \"sequence\": \"EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS\"\n        }\n    ],\n    \"params\": {\n        \"include\": \"seqcoding\"\n    }\n}"
            headers = {}

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

   .. tab-item:: R
       :sync: r

       .. code:: R
           
           library(RCurl)
           params = "{\n    \"items\": [\n        {\n            \"sequence\": \"EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS\"\n        }\n    ],\n    \"params\": {\n        \"include\": \"seqcoding\"\n    }\n}"
           res <- postForm("https://biolm.ai/api/v2/ablang-heavy/encode/", .opts=list(postfields = params, followlocation = TRUE), style = "httppost")
           cat(res)


+++++++++++
Definitions
+++++++++++

params: 
   Additional parameters for the request.

include: 
   Specifies additional data to be included in the response. "seqcoding" indicates that sequence embeddings should be included.

payload: 
   A string variable containing the JSON payload to be sent in the POST request. It consists of items and sequence.

sequence:
   The amino acid sequence of the antibody heavy chain for which you want to generate embeddings.

items: 
   A list of dictionaries, each representing an item to be processed by the ABLang model. Each dictionary has a key.

^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response
    :open:

    .. code:: json

         {
         "results": [
            {
               "seqcoding": [
               -0.6615958659340097,
               0.13918796144733744,
               -0.9715563959080326,
               -0.24384153723208743,
               0.0955913498129865,
               0.6615201387831495,
               -0.3109214511846215,
               0.4820148539248361,


.. note::
  The above response is only a small snippet of the full JSON response. However, all the relevant response keys are included.

+++++++++++
Definitions
+++++++++++


results:
   A list containing the results of the ABLang model's encoding process. Each element in this list is a dictionary representing the results for one input item (in this case, one antibody heavy chain sequence).


seqcoding: 
   A key within each result dictionary that corresponds to the sequence embeddings generated by the ABLang model for the input antibody heavy chain sequence. The value is a list of floating-point numbers, each representing a dimension in the embedding space. These embeddings capture the characteristics of the input sequence and can be used for various downstream tasks, such as similarity comparisons, clustering, or as input features for machine learning models.


----------
Related
----------

:doc:`/model-docs/AbLang Heavy Chain Res-coding`


------------------
Model Background
------------------

The pre-training data for AbLang consisted of heavy and light chain sequences obtained from the Observed Antibody Space (OAS) database as of October 2021 (millions of unlabeled antibody sequences) *(Olsen et al., 2022)*. To reduce redundancy, sequences occurring at least 3 times were clustered based on identical CDR3 regions, followed by 70% whole sequence identity clustering using the Linclust algorithm. The longest sequence was chosen to represent each cluster. The resulting datasets were randomly split into training sets of 14 million heavy chain and 187,000 light chain sequences, along with evaluation sets of 100,000 and 50,000 sequences respectively. This preprocessed antibody sequence data was then utilized to pretrain the AbLang model parameters through masked language modeling objectives.

The researchers developed two separate AbLang models, one specialized for heavy chains and one for light chains. Each AbLang model is composed of two components. AbRep generates vector representations encoding information from the antibody sequences. AbHead, on the other hand, utilizes these sequence embeddings to calculate the probability of each amino acid appearing at each position, providing values that represent the likelihood of each amino acid at every position in the antibody sequence. These likelihood values could potentially be utilized to explore possible mutations at a given sequence position

By creating dedicated language models for heavy and light chains, AbLang is tailored to capture position-specific patterns within each antibody domain. The coupled AbRep and AbHead architecture enables transforming raw sequences into informed predictions in an end-to-end manner. 

The architecture of AbRep mirrors that of RoBERTa (Liu et al., 2019), with the exception of employing a learned positional embedding layer having a maximum length of 160. Each of its dozen transformer blocks encompasses 12 attenuated heads, boasting an inner hidden size of 3072 and a hidden size of 768. From AbRep, residue codings (768 values for each residue) are derived. AbHead is patterned after the head model design of RoBERTa, featuring a hidden size of 768.

In the training phase, a range of 1% to 25% of residues from each sequence was chosen, wherein 80% were masked, 10% were altered to a different residue randomly, and 10% remained unaltered. One iteration of the AbLang model underwent training on heavy chain sequences across 20 epochs with a batch size of 8192, while another was trained on light chain sequences across 40 epochs with a batch size of 4096. Both model variants were fine-tuned using the Adam optimizer, incorporating a linear warm-up phase for 5% of the steps, a peak learning rate of 0.0002, a cosine function for learning rate reduction, and a weight decay rate of 0.01. A dropout and layer normalization rate of 0.1 and an epsilon of 1e−12 were applied respectively. The chosen hyperparameters were aligned closely with those outlined in the RoBERTa paper (Liu et al., 2019)

.. note::
   The model background above covers information for Ablang Res-coding and Seq-coding.

-----------------------
Applications of AbLang
-----------------------

AbLang is capable of producing three distinct representations or encodings for antibody sequences:

1. Res-codings: Encodings assigned to each residue, consisting of 768 values, facilitating predictions specific to individual residues.

2. Seq-codings: Comprising 768 values designated to each sequence, these encodings cater to predictions that are sequence-specific. Every sequence is given encodings of consistent length, eliminating the need for antibody sequence alignment. Furthermore, AbLang's seq-codings contain knowledge of the germlines, originating cell type and number of mutations (et al)

3. Res-likelihoods: These encodings display the likelihoods of each amino acid at each position in a given antibody sequence, useful for exploring possible mutations. The order of amino acids follows the AbLang vocabulary.

These representations have a wide range of potential applications for antibody design:

* Completing incomplete antibody sequences

* Restoration of an unknown number of missing residues at the ends of antibody sequences.

* Can be used in Therapeutic Antibody design:

* Utilizing the Res-codings and Res-likelihoods, AbLang could facilitate the identification and optimization of antibody sequences that have enhanced affinity and specificity towards specific antigens.

* Through its encodings, AbLang could be harnessed to predict potential immunogenic regions within antibody sequences, aiding in the design of less immunogenic therapeutic antibodies.

* AbLang could be used to generate in silico antibody libraries with diversified sequences for high-throughput screening in therapeutic antibody discovery

* CDR grafting: seq-codings generated by AbLang could be employed for similarity searches to identify CDR loops that are compatible with a given framework

.. note::
   The applications above covers general use-cases for Ablang Res-coding and Seq-coding.

.. _Status Page: https://status.biolm.ai






