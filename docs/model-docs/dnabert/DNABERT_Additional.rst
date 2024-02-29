=====================
DNABERT Info
=====================

.. article-info::
    :avatar: img/book_icon.png
    :date: Nov 7, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

On this page, we will show and explain the use of DNABERT. As well as document the BioLM API for fine-tuning, demonstrate no-code and code interfaces.

------------------
Model Background
------------------

DNABERT utilizes tokenized k-mer sequences as its foundational training data, where a k-mer is a DNA substring of length 'k'. Special tokens such as CLS (Class token), SEP (Separator token), and MASK (Mask token) are integrated into these sequences. The CLS token encapsulates the essence of the entire sequence, the SEP token acts as a delimiter, and MASK tokens represent masked k-mers during preliminary training.

The architecture includes an embedding layer that converts tokenized sequences into vector representations, followed by 12 transformer blocks that capture the relationships between different tokens within the sequences. Symbols like Et, It, and Ot denote positional embedding, input embedding, and the last hidden state at each token, essential for understanding positional interrelationships among tokens.

DNABERT undergoes general-purpose pre-training and task-specific fine-tuning, enhancing its adaptability across diverse genomic analysis tasks. It exhibits global attention patterns, focusing on pivotal regions within sequences, such as known binding sites, optimizing its analytical precision in genomic studies. (Ji et al., 2021)


-----------------------
Applications of DNABERT
-----------------------

* DNABERT can be employed to annotate genomic variants, helping to identify those that may have significant functional impacts. This is critical for understanding the genetic basis of diseases and could also be crucial in personalized medicine.

* By understanding the genomic context and potential regulatory interactions, DNABERT could assist in optimizing the design of synthetic constructs to ensure their functionality and stability within microbial hosts.

* DNABERT can be fine-tuned for identifying cis-regulatory elements or enhancers using ATAC-seq or DAP-seq data.

* DNABERT-Prom (one of the fine-tuned models) successfully predicts proximal and core promoter regions.

* DNABERT-TF accurately identifies transcription factor binding sites. Understanding where transcription factors bind can provide insights into the gene regulatory mechanisms and how they are altered in different diseases.

* DNABERT-viz allows visualization of critical regions, contexts and sequence motifs.

* DNABERT-Splice can be used to accurately recognize canonical and non-canonical splice sites. Accurate splice site prediction can aid in the study of splicing mechanisms and the identification of splicing-related disorders.

------------------
BioLM Benefits
------------------

* The API can be used by biologists, data scientists, engineers, etc. The key values of the BioLM API is speed, scalability and cost.

* The BioLM API allows scientists to programmatically interact with DNABERT, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

* Leverage fine-tuning to quickly and securely discover signal in your data and then access your fine-tuned model with our platform

* The benefit of having access to multiple GPUs is parallel processing.

-------
Related
-------

:doc:`/model-docs/dnabert/classifier_ft`
