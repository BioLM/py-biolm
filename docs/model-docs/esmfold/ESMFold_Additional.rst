========================================
ESMFold Info
========================================

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Article Information
    :date: Oct 24, 2023
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*This page explains the use of ESMFold, as well as documents
its usage on BioLM for protein structure prediction.*

------------------
Model Background
------------------

Recent computational protein folding capability enables myriad of applications
from elucidating structures of novel proteins, designing engineered proteins,
modeling molecular interactions, evaluating impacts of mutations, and assembling
multi-protein complexes.

Advances in large-scale language modeling is moving us closer to achieving a
universal model for proteins. ESMFold, a protein structure prediction tool that
utilizes the ESM-2 language model, is one of the most advanced models currently
available. ESMFold's training data is derived from UniRef, with a focus on
UniRef50 clusters, which are non-redundant sets of protein sequences with at
least 50% sequence identity to each other. The training process included the
selection of sequences from around 43 million UniRef50 training groups, covering
close to 138 million UniRef90 sequences, which amounts to nearly 65 million
distinct sequences throughout the training period. ESMFold achieves a faster
performance compared to AlphaFold as it is capable of conducting end-to-end
atomic structure predictions straight from the sequence, bypassing the need for
a multiple sequence alignment (MSA). These models learn so much about protein
sequences and the evolutionary patterns that relate sequences to function, that
then they don’t need sequence alignments at all in order to fold them. This
leads to a more simplified neural architecture for inference, drastically
reducing the time taken in the inference forward pass and removing the lengthy
search for related proteins, which is a notable part of the process in AlphaFold
-*“This results in an improvement in speed of up to 60x on the inference forward
pass alone, while also removing the search process for related proteins
entirely, which can take over 10 minutes with the high-sensitivity pipelines
used by AlphaFold” -  Lin et al., 2022.* In addition, AlphaFold 2 may struggle
with ‘orphan proteins’, which lack multiple sequence alignments due to
insufficient database sequences. Since ESMFold bypasses alignments, it may model
orphan proteins more effectively. This, in turn, could inform and facilitate the
de novo design of proteins with desired characteristics, thereby extending the
reach and success of de novo protein design efforts.

-----------------------
Applications of Folding
-----------------------

ESMFold is a revolutionary tool for folding that can be used by a diverse range
of topics within biology, ranging from synthetic biology, neuroscience, enzyme
engineering, immunology, virology, industrial biotechnology, etc. A great
starting point for ESMFold is when scientist starts with a single sequence or
library of designed sequences for which they wish to understand the 3D
structure.

* Predict how post-translational modifications affect chaperone protein
  structure.
* Analyze capsid protein folding of viruses like HIV, Influenza, and SARS-CoV-2.
* Design novel self-assembling protein nanostructures by rapidly predicting
  their protein architectures.
* Predict 3D structures of computationally designed enzyme sequences to
  assess if they fold into stable enzymes; by rapidly modeling many designs,
  ESMFold facilitates computational filtering and optimization of the lead de
  novo enzymes. (have a link to a tutorial page here).
* Used in antibody engineering. Once CDR variants are designed computationally,
  scientists can use ESMFold to predict structures to filter and select optimal
  candidates. Can also predict structures for lead antibody variable domains.


----------------
BioLM Benefits
----------------

The BioLM API is democratizing access to 3D structural
modeling, with its rapid ESMFold API,  bringing the power of structural biology
to address diverse questions in protein science, biomedicine, synthetic biology,
and beyond. The API can be used by biologists, data scientists, engineers, etc. The key values of the BioLM API is speed, scalability and cost.

* The API allows 1440 folds per minute, or 2M per day (Figure 1).
* The BioLM API allows scientists to programmatically interact with ESMFold,
  making it easier to integrate the model into their scientific workflows.
  The API accelerates workflow, allows for customization, and is designed to be
  highly scalable.
* Our unique API UI Chat allows users to interact with our API and access
  multiple language models without the need to code!
* The benefit of having access to multiple GPUs is parallel processing. Each
  GPU can handle a different protein folding simulation, allowing for folding
  dozens of proteins in parallel.


-------
Related
-------
:doc:`/model-docs/esmfold/ESMFold_API`

:doc:`/model-docs/esmif/index`

:doc:`/model-docs/esm2/index`

:doc:`/model-docs/esm1v/index`


