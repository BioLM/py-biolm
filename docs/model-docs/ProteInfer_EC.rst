..
   Copyright (c) 2021 Pradyun Gedam
   Licensed under Creative Commons Attribution-ShareAlike 4.0 International License
   SPDX-License-Identifier: CC-BY-SA-4.0


=========
ProteInfer EC
=========

.. article-info::
    :avatar: img/book_icon.png
    :author: Article Information
    :date: Jul 24, 2021
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of ProteInfer for enzyme function prediction. As well as document the BioLM API, and demonstrate no-code  and code interfaces to enzyme function prediction.*


-----------
Description
-----------

Proteins exhibit vast diversity in sequences and functions. Homology-based approaches for functional prediction are inherently limited by availability of closely related sequences. ProteInfer, on the other hand, is able to learn patterns and relationships in protein sequences that are not based on homology, and it has been shown to be effective in predicting the function of proteins with limited homology to known sequences.

*“Here we introduce ProteInfer, which instead employs deep convolutional neural networks to directly predict a variety of protein functions; Enzyme Commission (EC) numbers and Gene Ontology (GO) terms – directly from an unaligned amino acid sequence.” -Sanderson et al., 2023*

The model uses a deep neural network with special convolutional layers (dilated convolutions) to process one-hot encoded protein sequences. The architecture allows the model to capture both local and global hierarchical features of the sequences, and through a series of transformations, including mean-pooling and passing through a fully connected layer, the model outputs probabilities for different functional classifications of the proteins. This architecture enables the model to make nuanced predictions about protein functions based on their amino acid sequences.

ProteInfer implements dilated convolutional layers to extract hierarchical local and global features from one-hot encoded input sequences. Through progressive transformations, including mean-pooling and fully connected layers, ProteInfer produces probabilistic predictions for enzyme commission numbers and gene ontology terms. This architecture enables nuanced modeling of sequence-function relationships beyond homology.

A key component is ProteInfer EC, which predicts enzyme commission numbers from sequence. These standard codes classify enzyme-catalyzed reactions, enabling systematic identification of enzymes and functions. By predicting EC numbers, ProteInfer provides insights into the catalytic reactions and enzymatic roles of proteins, which is crucial for elucidating biological systems.

Comparisons reveal ProteInfer has higher precision while BLASTp alignment shows greater recall. An ensemble approach combining both methods improves overall performance, synergistically integrating strengths of alignment-based homology detection and deep neural network sequence modeling, particularly for challenging remote homology scenarios (for example, the dataset clustered based on UniRef50).

--------
Benefits
--------

* The BioLM API allows scientists to programmatically interact with ProteInfer EC, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

* Our unique API UI Chat allows users to interact with our API and access multiple language models without the need to code!

* The benefit of having access to multiple GPUs is parallel processing. Each GPU can handle a different protein folding simulation, allowing for folding dozens of proteins in parallel!

---------
Performance
---------

Graph of average RPS for varying number of sequences (ProteInfer EC)

.. note::
   We are in the process of adding the graph.


---------
API Usage
---------

This is the url to use when querying the BioLM ProteInfer EC Prediction Endpoint: https://biolm.ai/api/v1/models/enzyme_function/predict/


*Definitions*

-Request Keys:

data:
    Inside each instance, there's a key named "data" that holds another dictionary. This dictionary contains the actual input data for the prediction.

text:
    Inside the "data" dictionary, there's a key named "text". The value associated with "text" should be a string containing the full-length protein sequence that the user wants to submit for structure prediction.


-Response Keys:

predictions:
    This key holds a list of dictionaries, each containing a prediction result. Each item in the list represents a predicted Enzyme Commission (EC) number along with additional information related to the prediction.

sequence_name:
    identifier for the input protein sequence for which the EC numbers are being predicted.

predicted_label:
    represents the predicted EC number. EC numbers are used to classify enzymes and includes four levels of classification, each separated by a dot. ( "EC:3.-.-.-" and "EC:3.2.1.-" are examples of predicted EC numbers).

confidence:
    This is a measure of the model's certainty or confidence in the predicted EC number, ranging from 0 to 1, with higher values indicating higher confidence.

description:
    This provides a textual description or annotation related to the predicted EC number, giving some context or information about the type of reaction the enzyme catalyzes


^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/enzyme_function/predict/' \
            --header 'Content-Type: application/json' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --data '{
                "instances": [
            {"data": {"text": "MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS"}
            }
            ]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/enzyme_function/predict/"

            payload = json.dumps({
            "instances": [
                {
                "data": {
                    "text": "MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS"
                }
                }
            ]
            })
            headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Token {}'.format(os.environ['BIOLMAI_TOKEN'])
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)


    .. tab-item:: R
        :sync: r

        .. code:: R

            library(RCurl)
            headers = c(
            "Content-Type" = "application/json",
            'Authorization' = paste('Token', Sys.getenv('BIOLMAI_TOKEN'))
            )
            params = "{
            \"instances\": [
                {
                \"data\": {
                    \"text\": \"MGAASGRRGPGLLLPLPLLLLLPPQPALALDPGLQPGNFSADEAGAQLFAQSYNSSAEQVLFQSVAASWAHDTNITAENARRQEEAALLSQEFAEAWGQKAKELYEPIWQNFTDPQLRRIIGAVRTLGSANLPLAKRQQYNALLSNMSRIYSTAKVCLPNKTATCWSLDPDLTNILASSRSYAMLLFAWEGWHNAAGIPLKPLYEDFTALSNEAYKQDGFTDTGAYWRSWYNSPTFEDDLEHLYQQLEPLYLNLHAFVRRALHRRYGDRYINLRGPIPAHLLGDMWAQSWENIYDMVVPFPDKPNLDVTSTMLQQGWNATHMFRVAEEFFTSLELSPMPPEFWEGSMLEKPADGREVVCHASAWDFYNRKDFRIKQCTRVTMDQLSTVHHEMGHIQYYLQYKDLPVSLRRGANPGFHEAIGDVLALSVSTPEHLHKIGLLDRVTNDTESDINYLLKMALEKIAFLPFGYLVDQWRWGVFSGRTPPSRYNFDWWYLRTKYQGICPPVTRNETHFDAGAKFHVPNVTPYIRYFVSFVLQFQFHEALCKEAGYEGPLHQCDIYRSTKAGAKLRKVLQAGSSRPWQEVLKDMVGLDALDAQPLLKYFQPVTQWLQEQNQQNGEVLGWPEYQWHPPLPDNYPEGIDLVTDEAEASKFVEEYDRTSQVVWNEYAEANWNYNTNITTETSKILLQKNMQIANHTLKYGTQARKFDVNQLQNTTIKRIIKKVQDLERAALPAQELEEYNKILLDMETTYSVATVCHPNGSCLQLEPDLTNVMATSRKYEDLLWAWEGWRDKAGRAILQFYPKYVELINQAARLNGYVDAGDSWRSMYETPSLEQDLERLFQELQPLYLNLHAYVRRALHRHYGAQHINLEGPIPAHLLGNMWAQTWSNIYDLVVPFPSAPSMDTTEAMLKQGWTPRRMFKEADDFFTSLGLLPVPPEFWNKSMLEKPTDGREVVCHASAWDFYNGKDFRIKQCTTVNLEDLVVAHHEMGHIQYFMQYKDLPVALREGANPGFHEAIGDVLALSVSTPKHLHSLNLLSSEGGSDEHDINFLMKMALDKIAFIPFSYLVDQWRWRVFDGSITKENYNQEWWSLRLKYQGLCPPVPRTQGDFDPGAKFHIPSSVPYIRYFVSFIIQFQFHEALCQAAGHTGPLHKCDIYQSKEAGQRLATAMKLGFSRPWPEAMQLITGQPNMSASAMLSYFKPLLDWLRTENELHGEKLGWPQYNWTPNSARSEGPLPDSGRVSFLGLDLDAQQARVGQWLLLFLGIALLVATLGLSQRLFSIRHRSLHRHSHGPQFGSEVELRHS\"
                }
                }
            ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/enzyme_function/predict/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)


^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

        {
        "predictions": [
            {
            "sequence_name": "seq_0",
            "predicted_label": "EC:3.-.-.-",
            "confidence": 1,
            "description": "Hydrolases."
            },
            {
            "sequence_name": "seq_0",
            "predicted_label": "EC:3.2.-.-",
            "confidence": 1,
            "description": "Glycosylases."
            },
            {
            "sequence_name": "seq_0",
            "predicted_label": "EC:3.2.1.-",
            "confidence": 1,
            "description": "and S-glycosyl compounds."
            },
            {
            "sequence_name": "seq_0",
            "predicted_label": "EC:3.4.-.-",
            "confidence": 1,
            "description": "Acting on peptide bonds (peptidases)."
            },
            {
            "sequence_name": "seq_0",
            "predicted_label": "EC:3.4.15.-",
            "confidence": 1,
            "description": "Peptidyl-dipeptidases."
            },
            {
            "sequence_name": "seq_0",
            "predicted_label": "EC:3.4.15.1",
            "confidence": 1,
            "description": "Peptidyl-dipeptidase A."
            }
        ]
        }



----------
Related
----------
:doc:`/model-docs/ProteInfer_GO`


------------------
Model Background
------------------

ProteInfer utilizes deep dilated convolutional neural networks to model mappings between full-length protein sequences and functional labels. As described by *Sanderson et al. (2023)*, ProteInfer models were trained on high-quality Swiss-Prot entries within UniProtKB (UniProt Consortium: https://academic.oup.com/nar/article/47/D1/D506/5160987), representing a well-curated subset of the known protein universe. Swiss-Prot contains 570,157 expertly annotated sequences from 294,587 unique references, totaling 206 million amino acids. Within UniProtKB, protein functions are captured via cross-references to ontologies like Enzyme Commission (EC) numbers, denoting enzymatic activity, and Gene Ontology (GO) terms, describing molecular function, biological process, and subcellular localization. By linking to standardized ontologies, UniProt systematically associates proteins with functional descriptors.

The ProteInfer enzyme function predictor uses a deep neural network to predict EC numbers from sequence. Proteins may have multiple EC numbers mapping to over 8,000 classified reactions (EC-IUBMB, ExplorEnz, BRENDA databases). The optimized 5-block convolutional model achieves a maximum F1 score of 0.977 on randomly split test data, correctly predicting 96.7% of EC labels with a 1.4% false positive rate, indicating reliable EC number prediction from sequence alone. Performance was relatively consistent across EC classes, with minor variations in F1 scores between categories like ligases and oxidoreductases. Precision exceeded recall at optimal thresholds, suggesting accurate positive predictions but difficulty capturing all functional associations. Varying confidence thresholds enables balancing precision and recall based on use case. While improvements remain possible, ProteInfer EC exhibits robust sequence-based EC prediction that could enable high-throughput annotation of uncharacterized proteins.


-----------------------------
Applications of ProteInfer EC
-----------------------------

By linking protein sequence to catalytic function, ProteInfer EC can provide useful insights to guide rational design and accelerate characterization of engineered enzymes.

* Predicting function of engineered enzymes

* Guiding site-directed mutagenesis

* Assessing fitness landscapes

* Drug discovery

* Systems and Synthetic Biology

ProteInfer is adept at identifying regions within a protein sequence that are pivotal for specific reactions. This facilitates the understanding of functional correlations in multi-domain enzymes by bridging sequence attributes to functional outcomes.  A specific protein, “fol1” from Saccharomyces cerevisiae, which is not included in the training data, is highlighted as an important example due to its multiple domains that each perform different roles in tetrahydrofolate synthesis. The model predicts these regions as being highly involved or essential in carrying out certain reactions or functions of the protein. These predicted regions align with existing scientific knowledge.

The ProteInfer EC model enables prediction of conditional enzyme activity by identifying sequence motifs and features associated with activity under different conditions. For example, motifs present in thermophilic enzymes may indicate thermostability if also found in the query sequence. Identified similarities and differences in sequence could reveal structural factors modulating activity. By leveraging ProteInfer EC's learned sequence representations, researchers can elucidate sequence-function relationships and patterns that determine an enzyme's conditional activity in varying contexts.





