=======================================
ESM-1v Info
=======================================

.. article-info::
    :avatar: ../img/book_icon.png
    :date: Oct 18, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

On this page, we will show and explain the use of ESM-1v. As well as document the BioLM API for folding, demonstrate no-code and code interfaces to folding.

------------------
Model Background
------------------

ESM-1v is part of the ESM (Evolutionary Scale Modeling) series, which encompasses a collection of transformer-based protein language models such as ESM-2 and ESMFold. This model specializes in executing zero-shot predictions, particularly focusing on determining the impacts of mutations on protein functionality. As articulated by *Meier et al., 2021, "Modeling the effect of sequence variation on function is a fundamental problem for understanding and designing proteins"*. This emphasizes the critical role of ESM-1v in delineating the functional implications of sequence variations in proteins. The models are trained exclusively on functional molecules, facilitating an evaluative capability to discern the functional viability of novel molecules or the deleterious nature of specific mutations.

The architecture of ESM-1v is constructed based on a 'fill-in-the-blank' framework. During the training process, 15% of residues in each sequence are masked, compelling the model to predict the identities of the concealed residues. The weights of the neural network are iteratively updated to optimize the model’s predictive performance.

For prediction tasks, ESM-1v employs a consistent input strategy used during training. It requires a sequence with masked residues, and the model predicts the identities of the masked components, providing a likelihood score associated with each prediction. This likelihood score, ranging from 0 to 1, acts as an indicator of the predicted functionality of a sequence, reflecting the likely accuracy of the unmasked sequence's ability to form a functional protein.

ESM-1v is a large-scale transformer-based protein language model containing 650 million parameters, developed for predicting the effects of genetic variants on protein function (*Meier et al., 2021*). It was pretrained on a dataset of 98 million diverse protein sequences from Uniref90 2020-03, allowing it to learn broad evolutionary sequence variation patterns. The pretraining approach followed that of ESM-1b (*Rives et al., 2020*), using masked language modeling on the amino acid sequences without any task-specific supervised signals. As stated by *Meier et al,. (2021), "ESM-1v requires no task-specific model training for inference. Moreover, ESM-1v does not require MSA generation."*

Inferencing with ESM-1v provides two key advantages over other state-of-the-art methods: (i) it can directly predict mutation impacts without needing additional task-specific training, and (ii) it can estimate fitness landscapes from a single forward pass through the model (*Meier et al., 2021*). This enables more efficient variant effect prediction compared to approaches requiring multiple steps like MSA generation and supervised retraining. By leveraging the self-supervised pretraining on large and diverse protein sequences, ESM-1v acquired generalizable knowledge of sequence-function relationships to allow variant consequence analysis solely from the primary structure.




.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Extraction Method
     - Description
   * - Masked Marginal
     - During pretraining, probabilities were derived based on the mask noise. At every position, a mask token was introduced, and the model's predicted probabilities for the tokens at that position were recorded.
   * - Mutant Marginal
     - Probabilities were obtained based on the random token noise during pre-training. Out of the 15% predicted positions in the sequence during pre-training, 10% were randomly altered while 10% remained unchanged. The model aimed to accurately predict the token at these positions. In this extraction method, the researchers adhered to the pre-training approach by inputting mutated tokens and documenting the model's probability of correctness for these tokens.
   * - Wildtype Marginal
     - A single forward pass was performed using the wildtype sequence. This method enabled fast scoring as just a single forward pass was used.
   * - Pseudo Likelihood
     - The researchers refer to the method outlined in *Salazar et al., 2019.*




-----------------------
Applications of ESM-1V
-----------------------


ESM-1v has great potential in advancing our understanding of protein function and the implications of genetic variations, which is fundamental in many fields including medicine, genetics, and bioengineering.

* Variant effect prediction: ESM-1v can be used to predict how specific mutations or variants might affect the function of proteins. For example, in antibody engineering, By masking particular residues in an antibody sequence and using ESM-1v to predict the likely amino acids that could occur at those positions, one can gain insights into how different variants might affect antibody-antigen binding or other functional attributes.

* Drug discovery: to predict how mutations might affect drug targets or to identify new potential drug targets based on the effect of natural variations.

* Enzyme engineering: to predict how engineered mutations might affect protein function, aiding in the design of proteins with desired properties. Furthermore, Identifying crucial residues in a binding site using ESM-1v with masking techniques holds promise in Enhancing Catalytic Efficiency, Developing Enzyme Inhibitors or Activators

* Predicting protein folding from sequence: Scientists can mask various portions of a sequence and analyze changes in the ESM-1v embedding to predict structural folds. Or mask different sequence regions to identify areas that most significantly alter the embedding away from the native fold.


--------
Benefits
--------

* The API can be used by biologists, data scientists, engineers, etc. The key values of the BioLM API is speed, scalability and cost.

* The API has been customized to allow users to easily see how the likelihood of the sequence being functional with the wild-type residue compares to a single-AA mutation at that position.

* The BioLM API allows scientists to programmatically interact with ESM-1v, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

* Our unique API UI Chat allows users to interact with our API and access multiple language models without the need to code!

* The benefit of having access to multiple GPUs is parallel processing.

-------
Related
-------
:doc:`/model-docs/esm1v/ESM-1v_API`

:doc:`/model-docs/esmif/index`

:doc:`/model-docs/esmfold/index`

:doc:`/model-docs/esm2/index`

