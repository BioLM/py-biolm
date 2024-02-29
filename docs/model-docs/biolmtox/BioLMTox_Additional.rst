======================================
BioLMTox Info
======================================

.. article-info::
    :avatar: ../img/book_icon.png
    :date: Dec 26, 2023
    :read-time: 6 min read
    :author: Chance Challacombe
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1


*This page explains applications of BioLMTox and documents
it's usage for classification and embedding extraction on BioLM.*

-----------
Description
-----------

Toxin classification
has important applications in both industry and research settings and has been a
concern for some time with respect to biosecurity and in the fields of protein, DNA
and drug design. BioLMTox is an application of the pre-train fine-tune paradigm,
honing the ESM-2 Pre-Trained Protein Language Model for general toxin classification.


------------------
Model Background
------------------

BioLMTox is a protein language model fine-tuned for general (different domains of life and sequence lengths)
toxin classification. BioLMTox was trained on a selection of sequences from the UniProt, UniRef50 and
comparable SOTA datasets.

-----------------------
Applications of BioLMTox
-----------------------

BioLMTox classification predictions and embeddings can be

* used to augment biosecurity screening. Incorporate BioLMTox predictions before wet lab testing or alongside other computational screening software.

* used to discriminate between toxin and not toxin homolologs that may bypass standard sequence similarity methods

* incorporated into public facing APIs, we apps and chat agents to reduce dual-use risks

----------------
BioLM Benefits
----------------

* Always-on, auto-scaling GPU-backed APIs; highly-scalable parallelization.
* Save money on infrastructure, GPU costs, and development time.
* Quickly integrate multiple embeddings into your workflows.
* Interact with the endpoint using natural language and our Chat Agents.
* Rapidly screen for biosecurity risks
* Get ahead of potential biosecurity regulation and laws

-------
Related
-------

:doc:`/model-docs/esm2/index`
