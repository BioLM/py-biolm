=====================================
ESM-IF1 Info
=====================================

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Zeeshan Siddiqui
    :date: October 18th, 2023
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*This page explains the use of ESM-IF1, as well as documents
its usage on BioLM for generating de-novo and natural sequences.*


------------------
Model Background
------------------

The limited number of experimentally determined protein structures constrains
the potential of deep learning for protein design applications. To overcome this,
researchers explored leveraging predicted protein structures from AlphaFold2
for training deep learning models, with a focus on modeling the protein
backbone. The goal was to substantially augment the training data available beyond
experimentally known structures. The resulting model is ESM-Inverse Folding (ESM-IF1).

ESM-IF1 is a machine learning model designed for inverse folding tasks, which
involve generating protein sequences that can fold into specified structures.
To perform inverse folding, the model attempts to recover the native
sequence of a protein from the coordinates of its backbone atoms.

A key feature of ESM-IF1 is its capacity to produce sequences with natural,
protein-like attributes, in addition to the desired structure. By
accounting for critical sequence determinants beyond just fold specificity,
ESM-IF1 holds considerable promise as a tool for protein design efforts
requiring the generation of viable sequences with targeted structure and function.

The ESM-IF1 model builds upon the ESM transformer model, which
contains ∼650 million parameters across 33 layers and was pretrained on tens
of millions of sequences subset from UniRef. UniRef50 provides greater
sequence diversity and minimizes redundancy for pre-training protein language
models *(Rives et al., 2021)*, as compared with UniProt.

To generate structures for pretraining ESM-IF1, AlphaFold2 was used to
predict 12 million protein folds based on sequences from UniRef50. The authors
developed an autoregressive inverse folding model optimized for fixed-backbone
protein design, built on top of ESM embeddings, and evaluated generalizability
across distinct backbone topologies.

Larger models demonstrated markedly better sequence recovery, increasing from
about 39% to 51%, highlighting the value of leveraging predicted structures
for training data. Notably, the GVP-Transformer model trained on UniRef50
predicted structures improved sequence recovery by 9.4 percentage points compared to the top model
trained solely on the CATH structural database.


-----------------------
Applications of ESM-IF1
-----------------------

With the advent of large-scale DNA synthesis, researchers can synthesize many
molecules and evaluate each for properties and mutations that
might make some sequences more amenable to manufacture or more stable and active.
ESM-IF1 has applications in enzyme engineering, antibody
development, biosensors design, prediction of stability, insertion effects, and more.

The authors used ESM-IF1 to redesign the Receptor Binding Domain
(RBD) sequence of the SARS-CoV-2 spike protein, which was not present in the
training data. Generated sequences had an average recovery of about 50% to the
natural sequence, and were evaluated computationally with docking. This showcases
the potential for the model to generate viable, functional
sequences and molecules.

*"If ways can be found to continue to leverage predicted structures for
generative models of proteins, it may be possible to create models that learn
to design proteins from an expanded universe of the billions of natural
sequences whose structures are currently unknown." -Hsu et al., 2022.*

----------------
BioLM Benefits
----------------

The BioLM API gives users on-demand, scalable access to ESM-IF1 through a REST interface.
With our cost- and performance-optimized GPUs, millions of *de novo*
sequences can be generated using ESM-IF1, and ranked and screened to identify the
best candidates.

* Programmatic access allows easy integration into applications and workflows.
* Scalable backend allows massively-parallel inverse-folding on low-cost GPUs.
* Interact with ESM-IF1 using natural language, through our no-code chat interface.

-------
Related
-------
:doc:`/model-docs/esmif/ESM_InverseFold_API`

:doc:`/model-docs/esmfold/index`

:doc:`/model-docs/esm2/index`

:doc:`/model-docs/esm1v/index`



