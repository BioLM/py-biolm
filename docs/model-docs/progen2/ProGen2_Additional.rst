
===============================
ProGen2 Info
===============================

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Zeeshan Siddiqui
    :date: October 19th, 2023
    :read-time: 8 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of the ProGen2 OAS. As well as document the BioLM API for prediction, and demonstrate no-code and code interfaces for predictions.*

------------------
Model Background
------------------
ProGen2 represents one of the largest protein language models, leveraging self-supervised pretraining on extensive protein sequence data to generate useful representations applicable to diverse protein structure and function prediction and design applications. As an attention-based model trained on protein sequences, ProGen2 employs a mechanism to selectively focus on informative regions of input data, learning intricate patterns and relationships among amino acids within protein sequences. Specifically, ProGen2 is trained via masked language modeling to predict amino acids from surrounding sequence context. As a protein language model, ProGen2 shows considerable promise for generating synthetic libraries of functional proteins to empower discovery and iterative optimization.

*Madani et al., 2022* trained a suite of models ranging from 151M to 6.4B parameters. The models differ in size and training datasets (collectively comprise over a billion proteins). For more details, refer to Table 1 in here: https://browse.arxiv.org/pdf/2206.13517.pdf

ProGen2 was pretrained via masked language modeling on an expansive dataset of over 180 million protein sequences from public sources including UniRef50 and the Protein Data Bank. This enables ProGen2 to learn contextual sequence representations that capture motifs and sequence-structure-function relationships. A tokenization scheme with a vocabulary size of approximately 2500 was utilized to retain biochemical motifs within the sequences. In summary, pretraining ProGen2 on a massive and diversified protein sequence dataset empowers the model to learn expressive representations of sequence patterns, motifs, and residues that determine protein structure and function. As states by *-Madani et al., 2022.*, *“Increasing number of parameters allows the model to better capture the distribution of observed evolutionary sequences”*.

ProGen2 utilizes autoregressive transformer architectures trained with next-token prediction as the learning objective for language modeling of protein sequences. As model scale increases from 151 million to 6.4 billion parameters, ProGen2 becomes progressively more proficient at modeling the distribution of protein sequences present in observed evolutionary data. In summary, the combination of autoregressive modeling and large-scale pretraining enables ProGen2 to effectively capture sequence distributions reflective of natural protein evolution.

The standard ProGen2 models were pre-trained on a mixture of Uniref90 *(Suzek et al., 2015)* and BFD30 *(Steinegger & Söding, 2018)* databases.

The ProGen2-BFD90 model supplements Uniref90 with representative sequences clustered from UniprotKB, Metaclust, SRC, and MERC at 90% sequence identity. This generated the BFD90 dataset, approximately double the size of Uniref90. As reported in Table 8 by *Madani et al. (2022)*, Uniref90+BFD90 exhibited slightly lower perplexity and higher Spearman's rho on antibody developability/engineering tasks, potentially indicating superior performance on these objectives. In contrast, Uniref90+BFD30 showed higher Spearman's rho for antibody binding predictions, suggesting enhanced capabilities for this specific task.

For protein engineering endeavors with narrow fitness landscapes, such as optimizing a singular property like stability, larger protein language models can underperform compared to smaller models. The additional parameters enable overfitting to noise and extraneous patterns irrelevant to the focused objective. This was evidenced by the 151M parameter ProGen2 model outperforming a substantially larger 1.5B parameter version on targeted protein optimization. Overall, appropriate model size and regularization appear more crucial than architecture details when concentrating on a narrow property. Moreover, smaller models, which capture the observed protein sequence distribution less accurately, can systematically surpass larger models at zero-shot fitness. For broader fitness landscapes, larger models may confer benefits by capturing more intricate relationships between amino acid sequences and corresponding fitness. This could prove critical in landscapes exhibiting greater mutational tolerance. As model scale grows drastically, new and potentially unexpected capabilities may emerge. Very large models may excel at identifying high-fitness variants within challenging landscapes marked by low homology (sequence similarity) and high epistasis (inter-mutational interactions). This could hold promise for discovery of *"novel, high-fitness protein variants in a vast and complex sequence space"   -Madani et al., 2022.*

For specialized ProGen2-OAS training, unpaired antibody sequences were leveraged from the Observed Antibody Space (OAS) database, which contains a refined set of 1.5 billion heavy and light chain sequences from 80 immune repertoire sequencing studies across 6 species. To reduce redundancy, OAS sequences were clustered at 85% identity using Linclust (Steinegger & Söding, 2018), generating 554 million diverse sequences for training. To mitigate dataset bias and produce full-length antibodies, generation was initiated using a EVQ motif common at the start of human heavy chains. In summary, tailored training on broad antibody space data equips ProGen2-OAS for optimized antibody sequence generation.

As noted by Ali Madani, * "For antibody fitness prediction, training on immune repertoire sequencing samples (OAS) theoretically seems advantageous, yet in practice exhibits inferior performance.”* Interestingly, models trained on universal protein databases surpass Progen2-OAS at predicting general antibody properties. Comparative assessment of binding affinity (KD) prediction reveals ProGen2 small as superior, with ProGen2 OAS the lowest performer. However, for predicting general protein properties such as expression and thermal stability, ProGen2 extra large excels, while ProGen2 OAS outperforms ProGen2 small. In summary, ProGen2 models trained on broad protein sequence space rather than antibody-specific data demonstrate enhanced generalizability for predicting antibody properties, potentially due to the diversity and size of universal protein training data. However, antibody repertoire data provides some specialized benefits evident in predicting select protein engineering objectives.

.. note::
   The model background above covers information for ProGen2 OAS, Medium, Large and BFD90.


-----------------------
Applications of ProGen2
-----------------------

ProGen2 enables generation of novel protein sequences, prediction of protein functions, and assessment of protein fitness without additional fine-tuning. It facilitates comprehension of evolutionary patterns by modeling the distribution of observed evolutionary sequences. This empowers design of proteins with targeted properties and functionalities, while garnering insights into viability and efficacy.

For enzyme engineering, ProGen2's capture of evolutionary sequence distributions has considerable utility. Analysis of conserved residues and motifs within evolutionary sequences can illuminate key determinants of enzyme function and stability. This knowledge enables the design of enzymes with optimized attributes like enhanced catalytic activity or altered substrate specificity by replicating or expanding upon these conserved evolutionary elements.

* Capturing the distribution of observed evolutionary sequences. This can be used in enzyme engineering; by analyzing the evolutionary sequences, scientist can identify conserved residues or motifs that are crucial for enzyme function or stability. In addition, ProGen2 can be used to complete partial sequences of an enzyme.

* Generating novel viable protein sequences.

* Predicting protein fitness without requiring additional fine-tuning

* generation of antibody sequence libraries. For instance, if you're aiming to create a library targeting a specific antigen, ProGen2 could generate a variety of sequences that have desirable properties such as high affinity or specificity, based on patterns learned from known antibody-antigen interactions.

.. note::
   The applications above covers general use-cases for ProGen2 OAS, Medium, Large and BFD90.

----------------
BioLM Benefits
----------------

* The BioLM API allows scientists to programmatically interact with ProGen2, making it easier to integrate the models into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

----------
Related
----------

:doc:`/model-docs/progen2/ProGen2_API`

