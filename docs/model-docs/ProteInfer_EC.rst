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

Proteins are highly diverse and can have a wide range of functions, and traditional methods for predicting protein function, such as homology-based approaches, can be limited by the availability of closely related sequences. ProteInfer, on the other hand, is able to learn patterns and relationships in protein sequences that are not based on homology, and it has been shown to be effective in predicting the function of proteins with limited homology to known sequences.

““Here we introduce ProteInfer, which instead employs deep convolutional neural networks to directly predict a variety of protein functions – Enzyme Commission (EC) numbers and Gene Ontology (GO) terms – directly from an unaligned amino acid sequence.” *-Sanderson et al., 2023*

The model uses a deep neural network with special convolutional layers (dilated convolutions) to process one-hot encoded protein sequences. The architecture allows the model to capture both local and global hierarchical features of the sequences, and through a series of transformations, including mean-pooling and passing through a fully connected layer, the model outputs probabilities for different functional classifications of the proteins. This architecture enables the model to make nuanced predictions about protein functions based on their amino acid sequences.

ProteInfer EC refers to the aspect of the ProteInfer model that predicts the Enzyme Commission (EC) numbers of a given protein based on its amino acid sequence. The EC numbers are standard codes assigned to enzymes based on the reactions they catalyze, providing a systematic way of identifying enzymes and their functions. ProteInfer can model complex relationships within the protein sequence data to provide insightful functional predictions. ProteInfer's EC prediction feature helps in determining the enzymatic activities of proteins. By predicting the EC numbers, ProteInfer provides insights into the reactions a given protein can catalyze, which is crucial in understanding and studying enzyme function and interactions in biological systems.

The researchers also compared the performance of ProteInfer and BLASTp in protein function prediction. BLASTp has higher recall, while ProteInfer has higher precision. An ensemble approach, combining both methods, enhances overall performance, providing a synergy by leveraging the unique strengths of alignment-based and neural network-based strategies, especially in more challenging tasks involving remote homologies (dataset clustered based on UniRef50).

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

.. figure:: 
   :scale: 
   :alt: 

   This is the caption of the figure (a simple paragraph).

   The legend consists of all elements after the caption.

.. note::
   We are in the process of adding a graph. 



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

    .. tab-item:: biolmai SDK
        :sync: sdk

        Content 2

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
* ProteIndfer GO: :ref:`` 


------------------
Model Background
------------------

The model employs deep dilated convolutional networks to learn the mapping between full-length protein sequences and functional annotations. The resulting ProteInfer models take amino acid sequences as input and are trained on the well-curated portion of the protein universe annotated by Swiss-Prot (UniProt Consortium: https://academic.oup.com/nar/article/47/D1/D506/5160987)

The ProteInfer models accept amino acid sequences as input data. They were trained on high-quality protein sequences annotated by Swiss-Prot, representing a well-curated subset of known protein space. As of 2023, the Swiss-Prot section of the UniProtKB database contains 570,157 sequence entries. These entries are curated from 294,587 unique references and consist of a total of 206,173,379 amino acids.

In the UniProt database, protein functional information is captured through cross-references to external ontologies. These cross-references connect a protein to descriptive labels like Enzyme Commission (EC) numbers denoting enzymatic function or Gene Ontology (GO) terms describing molecular function, biological process, and subcellular localization. By linking to standardized ontologies, UniProt associates each protein with functional annotations in a structured manner.

The ProteInfer enzyme function prediction model utilizes a deep neural network to predict Enzyme Commission (EC) numbers for protein sequences. Proteins may have zero, one, or multiple associated EC numbers mapping to over 8,000 classified chemical reactions. (Note: The Enzyme Commission (EC) numbers and their associated chemical reactions are catalogued in specialized databases such as EC-IUBMB, ExplorEnz, and BRENDA).

The optimized model design contains 5 residual blocks with 1100 filters per block, converging after 500,000 training steps. On a test set split randomly, it achieved an impressive maximum Fmax score of 0.977. This high ‘Fmax’ score indicates the model correctly predicted 96.7% of true EC labels with only a 1.4% false positive rate, demonstrating reliable performance in identifying enzymatic functions from sequence.

Analysis of prediction efficacy across EC classes showed relatively consistent results, with minor variations in Fmax scores between categories like ligases and oxidoreductases. Precision exceeded recall at the optimal threshold, suggesting accurate positive predictions but some difficulty in capturing all true functional associations. Varying the confidence threshold allows trading off between precision and recall, providing flexibility based on application needs. While room for improvement remains, its robust performance could enable high-throughput annotation of uncharacterized protein sequences.


-----------------------
Applications of ProteInfer EC
-----------------------

By linking protein sequence to catalytic function, ProtInfer EC could provide useful insights to guide rational design and accelerate characterization of engineered enzymes.

* Predicting function of engineered enzymes 

* Guiding site-directed mutagenesis

* Assessing fitness landscapes 

* Drug discovery

* Systems and Synthetic Biology

ProteInfer’s ability to identify regions within a protein sequence crucial for specific reactions. The model can be used to link sequence to function in multi-domain enzymes. A specific protein, "fol1" from Saccharomyces cerevisiae, which is not included in the training data, is highlighted as an important example due to its multiple domains that each perform different roles in tetrahydrofolate synthesis.  The model predicts these regions as being highly involved or essential in carrying out certain reactions or functions of the protein. These predicted regions align with existing scientific knowledge. 

The ProtInfer EC model can predict an enzyme's activity under different conditions. For instance, motifs present in thermophilic enzymes may suggest thermostability if also present in the query. Sequence similarities and differences could reveal structural factors influencing conditional activity. By leveraging ProtInfer EC's sequence representations, researchers can uncover sequence-function patterns that modulate an enzyme's conditional activity.

Overall, ProtInfer EC can be a useful tool for predicting the activity of enzymes, offering valuable insights into enzyme properties crucial for specific activities. This information can be used to optimize the enzyme for specific applications, or to design new enzymes with specific properties.







