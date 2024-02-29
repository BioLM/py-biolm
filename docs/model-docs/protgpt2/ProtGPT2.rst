========================
ProtGPT2 Info
========================

.. article-info::
    :avatar: img/book_icon.png
    :date: Nov 10, 2023
    :read-time: 6 min read
    :author: Zeeshan Siddiqui
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

On this page, we will show and explain the use of ProtGPT2. As well as document the BioLM API for fine-tuning, demonstrate no-code and code interfaces.

-----------
Description
-----------

ProtGPT2 is a language model that can be used for de novo protein design and engineering.

The model is capable of generating de novo protein sequences that maintain amino acid and disorder propensities akin to natural proteins but are evolutionarily distant from known protein space. Secondary structure prediction suggests 88% of its sequences are globular, similar to natural proteins. ProtGPT2 can quickly generate sequences on standard workstations or be fine-tuned for specific protein families, with applications in various scientific fields. It can generate de novo protein sequences that are evolutionarily distinct yet exhibit natural-like properties and stabilities. ProtGPT2 also explores uncharted areas of the protein space, potentially contributing to advancements in biomedical and environmental science applications

By applying state-of-the-art techniques from natural language processing, specifically generative Transformers, ProtGPT2 is an example of how advanced computational methods are being leveraged to push the boundaries of synthetic biology and protein engineering. *Ferruz et al., 2022*

ProtGPT2's ability to produce sequences quickly and its capacity to be fine-tuned for specific families of proteins, makes it a flexible and valuable tool in multiple cutting-edge scientific and engineering domains



------------------
Model Background
------------------

ProtGPT2 is a decoder-only transformer model pre-trained on the protein space database UniRef50 (version 2021_04), and contains 36 layers with a model dimensionality of 1280, totalling 738 million parameters.

The model employs a self-supervised training approach, meaning it learns from raw sequence data without labeled annotations. Specifically, the model is trained using a causal language modeling technique where it predicts the next short sequence fragment based on the preceding context (in this case, oligomer). This training scheme allows the model to develop an implicit understanding of protein language patterns. By successfully predicting subsequent sequence chunks, the model acquires the ability to generate full-length novel protein sequences that conform to the characteristics of natural proteins. The self-supervised strategy circumvents the need for manually annotated data to train the model.

ProtGPT2 was trained using UniRef50, a comprehensive dataset clustering UniProt sequences at 50% identity, which includes a wide variety of protein sequences, even those from the uncharacterized "dark proteome". The model was trained on 44.9 million sequences, with an additional 4.9 million held out for evaluation. The methodology involved using a Transformer architecture optimized by minimizing the negative log-likelihood, allowing the model to predict each amino acid in a sequence based on the preceding sequence context. Key results from ProtGPT2 training indicate that the model can generate protein sequences that are both unique and exhibit natural-like properties. This capability could have significant implications for protein engineering and understanding protein function in unexplored areas of the protein space.

The ProtGPT2 paper indicates that while the model's generated protein sequences share some similarity with natural proteins, they remain distinct and novel. Using HHblits to compare 10,000 generated sequences with a protein database, the findings show that most ProtGPT2 sequences align above a threshold indicating some relatedness to known proteins. However, they significantly diverge in the high-identity range, suggesting that ProtGPT2 generates innovative sequences without simply mimicking existing ones. This demonstrates ProtGPT2's ability to contribute novel designs to the protein space, rather than reproducing what is already known. Furthermore, through evaluation using AlphaFold predictions, Rosetta Relax scores, and molecular dynamics simulations, ProtGPT2 sequences showed a decent mean probability of being ordered and thermodynamic stability comparable to natural sequences. Furthermore, molecular dynamics simulations suggested that ProtGPT2 sequences possess dynamic properties akin to natural proteins, essential for functional interactions in biological systems.

In addition, by incorporating ProtGPT2 sequences into a network representation of protein space, the model demonstrates its capacity to connect disparate 'islands' of protein structures. It successfully generates complex structures across various protein classes, including challenging ones like all-β and membrane proteins. ProtGPT2 not only replicates existing natural folds but also creates novel topologies not found in current databases, exemplifying its potential to design proteins with new functions and interactions.

-------------------------
Applications of ProtGPT2
-------------------------

* Enzyme Design: ProtGPT2’s ability to generate novel protein sequences that are evolutionarily distant from known proteins but structurally sound, makes it a powerful tool for designing enzymes with new functionalities or improved properties for industrial processes.

* Synthetic Biology: In the realm of SynBio, ProtGPT2 can be used to engineer proteins with desirable characteristics, such as enhanced stability or activity, which are essential in the development of biocatalysts, biosensors, and bioremediation agents.

* Environmental Science: By designing proteins that can degrade environmental pollutants or convert waste into valuable products.

* Drug Discovery: ProtGPT2 can expedite the drug discovery process by generating and screening protein sequences that can bind to specific drug targets or by designing novel biologics with therapeutic potential.

* Directed Evolution: The model can simulate the process of directed evolution in-silico, generating a diversity of protein variants for subsequent screening and selection for desired traits, thus speeding up the development of proteins with enhanced or novel functions.

* Industrial Catalysis: ProtGPT2 can be employed to create enzymes or protein-based catalysts that can operate under harsh industrial conditions, which can be used in the manufacturing of chemicals, pharmaceuticals, and materials.

----------------
BioLM Benefits
----------------

* The API can be used by biologists, data scientists, engineers, etc. The key values of the BioLM API is speed, scalability and cost.

* The BioLM API allows scientists to programmatically interact with ProtGPT2, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

* Generate your own sequences and discoveries by fine-tuning your own GPT model with our API.

* The benefit of having access to multiple GPUs is parallel processing.

----------
Related
----------

:doc:`/model-docs/protgpt2/generator_ft`

