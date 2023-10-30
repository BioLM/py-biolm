..
   Copyright (c) 2021 Pradyun Gedam
   Licensed under Creative Commons Attribution-ShareAlike 4.0 International License
   SPDX-License-Identifier: CC-BY-SA-4.0


=========
ProteInfer GO
=========

.. article-info::
    :avatar: img/book_icon.png
    :author: Zeeshan Siddiqui
    :date: October 18th, 2023
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*On this page, we will show and explain the use of ProteInfer (GO) for GO-TERM predictions. As well as document the BioLM API, and demonstrate no-code  and code interfaces to GO-TERM predictions.*


-----------
Description:
-----------
Proteins exhibit vast diversity in sequences and functions. Homology-based approaches for functional prediction are inherently limited by availability of closely related sequences. ProteInfer, on the other hand, is able to learn patterns and relationships in protein sequences that are not based on homology, and it has been shown to be effective in predicting the function of proteins with limited homology to known sequences.

*“Here we introduce ProteInfer, which instead employs deep convolutional neural networks to directly predict a variety of protein functions; Enzyme Commission (EC) numbers and Gene Ontology (GO) terms – directly from an unaligned amino acid sequence.” -Sanderson et al., 2023*

.. note::
   for a more detailed description, please refer to the ProteInfer EC API docs.
--------
Benefits
--------

* The BioLM API allows scientists to programmatically interact with ProteInfer GO, making it easier to integrate the model into their scientific workflows. The API accelerates workflow, allows for customization, and is designed to be highly scalable.

* Our unique API UI Chat allows users to interact with our API and access multiple language models without the need to code!

* The benefit of having access to multiple GPUs is parallel processing. Each GPU can handle a different protein folding simulation, allowing for folding dozens of proteins in parallel!


---------
API Usage
---------

This is the url to use when querying the BioLM ProteInfer GO-terms Prediction Endpoint: https://biolm.ai/api/v1/models/protein_go_function/predict/

*Definitions*

-Request Keys:

data:
    Inside each instance, there's a key named "data" that holds another dictionary. This dictionary contains the actual input data for the prediction.

text:
    Inside the "data" dictionary, there's a key named "text". The value associated with "text" should be a string containing the full-length protein sequence that the user wants to submit for structure prediction.



-Response Keys:

predictions:
    holds a list of dictionaries, each containing a prediction result. Each item in the list represents a predicted Gene Ontology (GO) term

sequence_name:
    identifier for the input protein sequence for which the GO terms are being predicted.

predicted_label:
    represents the predicted GO term ID. "GO:0008150" and "GO:0003674" are examples of predicted GO term IDs in the response.

confidence:
    This is a measure of the model's certainty or confidence in the predicted EC number, ranging from 0 to 1, with higher values indicating higher confidence.

description:
    textual description or name of the predicted GO term.  Descriptions like "biological_process" and "molecular_function" provide a brief understanding of what each GO term represents in biological terminology.



^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/protein_go_function/predict/' \
            --header 'Content-Type: application/json' \
            --header "Authorization: Token $BIOLMAI_TOKEN" \
            --data '{
                "instances": [
            {"data": {"text": "MPKYVEGVELTQEGMHAIFARMGYGDITSGSIYNGVPTIDTGALNRQGFMPVLTGVGPHRDSGHWIMLIKGPGNQYYLFDPLGKTSGEGYQNILAAQLPMGSTLSVIPNGSGLNMGLCGYWVASAGLRAHQALNQHNPPTLLNVGQTITNEMRNELDHDGYRKITGWLRAVADEFPEGDPQLDGKALRENTEKDLKIEIPTLVLPGKDTSPKEMSVKPTAPQDKSVPVWNGFSLYTDDTVKAAAQYAYDNYLGKPYTGSVESAPANFGGRMVYRQHHGLSHTLRTMAYAELIVEEARKAKLRGETLGKFKDGRTIADVTPQELKKIMIAQAFFVAGRDDEASDAKNYQKYHEQSRDAFLKYVKDNESTLIPDVFKDQEDVNFYARVIEDKSHDWESTPAHVLINQGHMVDLVRVKQPPESFLQRYFSSMQRWIGSQATEAVFGIQRQFFHATYEVVAGFDSDNKEPHLVVSGLGRYVIGEDGQPIREAPKKGQKEGDLKVFPQTYKLKENERLMRVDEFLKLPEIQNTFPGSGKHLQGGMPGMNEMDYWNRLNSLNRARCENDVDFCLKQLQTAHDKAKIEPIKQAFQSSKGKERRQPNVDEIAAARIIQQILANPDCIHDDHVLINGQKLEQQFFRDLLAKCEMAVVGSLLNDTDIGNIDTLMRHEKDTEFHSTNPEAVPVKIGEYWINDQRINNSSGNITQKKHDLIFLMQNDAWYFSRVNAIAQNRDKGSTFKEVLITTLMTPLTSKALVDTSQAKPPTRLFRGLNLSEEFTKGLIDQANAMIANTTERLFTDHSPEAFKQIKLNDLSKMSGRTNASTTTEIKLVKETWDSNVIFEMLDPDGLLHSKQVGRHGEGTESEFSVYLPEDVALVPVKVTLDGKTQKGENRYVFTFVAVKSPDFTPRHESGYAVEPFLRMQAAKLAEVKSSIEKAQRAPDLETIFNLQNEVEAVQYSHLSTGYKNFLKNTVGPVLENSLSGLMESDTDTLSKALAAFPSDTQWSAFNFEEARQAKRQMDAIKQMVGNKVVLDALTQCQDALEKQNIAGALDALKKIPSEKEMGTIRRELREQIQSARQELESLQRAVVTPVVTDEKKVRERYDALIENTSKKITELETGKLPNLDAVKKGISNLSNLKQEVTVLRNEKIRMHVGTDKVDFSDVEKLEQQIQVIDTKLADAYLLEVTKQISALDNTKPKNQTELKTKIAAFLDRTTDIEMLRNERIKKHGSSKDPLDLSDLDKLSGSLQRINQSLVSDLITTIRVSINQMEAKTFHEQEKEIQQNFELLAKLEKTLDKSKTSEKLREDIPKLNDLLVAKQKAYPQMVQMQLKSEVFVTQLREVCQANHDDLDKTRNARLRELDRLDREAGITRMVGNLIWGLTNKVGLTTDERLDIRTKQQSLARFKNELFNDKIDTDQLISNLARKRPSELQEGLGISTDNAMELHLLLTELAGKTTSPDELEERMKAIDDISTKIGREPEHLKFVMVEEDESNKKTIGF"}
            }
            ]
            }'


    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/protein_go_function/predict/"

            payload = json.dumps({
            "instances": [
                {
                "data": {
                    "text": "MPKYVEGVELTQEGMHAIFARMGYGDITSGSIYNGVPTIDTGALNRQGFMPVLTGVGPHRDSGHWIMLIKGPGNQYYLFDPLGKTSGEGYQNILAAQLPMGSTLSVIPNGSGLNMGLCGYWVASAGLRAHQALNQHNPPTLLNVGQTITNEMRNELDHDGYRKITGWLRAVADEFPEGDPQLDGKALRENTEKDLKIEIPTLVLPGKDTSPKEMSVKPTAPQDKSVPVWNGFSLYTDDTVKAAAQYAYDNYLGKPYTGSVESAPANFGGRMVYRQHHGLSHTLRTMAYAELIVEEARKAKLRGETLGKFKDGRTIADVTPQELKKIMIAQAFFVAGRDDEASDAKNYQKYHEQSRDAFLKYVKDNESTLIPDVFKDQEDVNFYARVIEDKSHDWESTPAHVLINQGHMVDLVRVKQPPESFLQRYFSSMQRWIGSQATEAVFGIQRQFFHATYEVVAGFDSDNKEPHLVVSGLGRYVIGEDGQPIREAPKKGQKEGDLKVFPQTYKLKENERLMRVDEFLKLPEIQNTFPGSGKHLQGGMPGMNEMDYWNRLNSLNRARCENDVDFCLKQLQTAHDKAKIEPIKQAFQSSKGKERRQPNVDEIAAARIIQQILANPDCIHDDHVLINGQKLEQQFFRDLLAKCEMAVVGSLLNDTDIGNIDTLMRHEKDTEFHSTNPEAVPVKIGEYWINDQRINNSSGNITQKKHDLIFLMQNDAWYFSRVNAIAQNRDKGSTFKEVLITTLMTPLTSKALVDTSQAKPPTRLFRGLNLSEEFTKGLIDQANAMIANTTERLFTDHSPEAFKQIKLNDLSKMSGRTNASTTTEIKLVKETWDSNVIFEMLDPDGLLHSKQVGRHGEGTESEFSVYLPEDVALVPVKVTLDGKTQKGENRYVFTFVAVKSPDFTPRHESGYAVEPFLRMQAAKLAEVKSSIEKAQRAPDLETIFNLQNEVEAVQYSHLSTGYKNFLKNTVGPVLENSLSGLMESDTDTLSKALAAFPSDTQWSAFNFEEARQAKRQMDAIKQMVGNKVVLDALTQCQDALEKQNIAGALDALKKIPSEKEMGTIRRELREQIQSARQELESLQRAVVTPVVTDEKKVRERYDALIENTSKKITELETGKLPNLDAVKKGISNLSNLKQEVTVLRNEKIRMHVGTDKVDFSDVEKLEQQIQVIDTKLADAYLLEVTKQISALDNTKPKNQTELKTKIAAFLDRTTDIEMLRNERIKKHGSSKDPLDLSDLDKLSGSLQRINQSLVSDLITTIRVSINQMEAKTFHEQEKEIQQNFELLAKLEKTLDKSKTSEKLREDIPKLNDLLVAKQKAYPQMVQMQLKSEVFVTQLREVCQANHDDLDKTRNARLRELDRLDREAGITRMVGNLIWGLTNKVGLTTDERLDIRTKQQSLARFKNELFNDKIDTDQLISNLARKRPSELQEGLGISTDNAMELHLLLTELAGKTTSPDELEERMKAIDDISTKIGREPEHLKFVMVEEDESNKKTIGF"
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
                    \"text\": \"MPKYVEGVELTQEGMHAIFARMGYGDITSGSIYNGVPTIDTGALNRQGFMPVLTGVGPHRDSGHWIMLIKGPGNQYYLFDPLGKTSGEGYQNILAAQLPMGSTLSVIPNGSGLNMGLCGYWVASAGLRAHQALNQHNPPTLLNVGQTITNEMRNELDHDGYRKITGWLRAVADEFPEGDPQLDGKALRENTEKDLKIEIPTLVLPGKDTSPKEMSVKPTAPQDKSVPVWNGFSLYTDDTVKAAAQYAYDNYLGKPYTGSVESAPANFGGRMVYRQHHGLSHTLRTMAYAELIVEEARKAKLRGETLGKFKDGRTIADVTPQELKKIMIAQAFFVAGRDDEASDAKNYQKYHEQSRDAFLKYVKDNESTLIPDVFKDQEDVNFYARVIEDKSHDWESTPAHVLINQGHMVDLVRVKQPPESFLQRYFSSMQRWIGSQATEAVFGIQRQFFHATYEVVAGFDSDNKEPHLVVSGLGRYVIGEDGQPIREAPKKGQKEGDLKVFPQTYKLKENERLMRVDEFLKLPEIQNTFPGSGKHLQGGMPGMNEMDYWNRLNSLNRARCENDVDFCLKQLQTAHDKAKIEPIKQAFQSSKGKERRQPNVDEIAAARIIQQILANPDCIHDDHVLINGQKLEQQFFRDLLAKCEMAVVGSLLNDTDIGNIDTLMRHEKDTEFHSTNPEAVPVKIGEYWINDQRINNSSGNITQKKHDLIFLMQNDAWYFSRVNAIAQNRDKGSTFKEVLITTLMTPLTSKALVDTSQAKPPTRLFRGLNLSEEFTKGLIDQANAMIANTTERLFTDHSPEAFKQIKLNDLSKMSGRTNASTTTEIKLVKETWDSNVIFEMLDPDGLLHSKQVGRHGEGTESEFSVYLPEDVALVPVKVTLDGKTQKGENRYVFTFVAVKSPDFTPRHESGYAVEPFLRMQAAKLAEVKSSIEKAQRAPDLETIFNLQNEVEAVQYSHLSTGYKNFLKNTVGPVLENSLSGLMESDTDTLSKALAAFPSDTQWSAFNFEEARQAKRQMDAIKQMVGNKVVLDALTQCQDALEKQNIAGALDALKKIPSEKEMGTIRRELREQIQSARQELESLQRAVVTPVVTDEKKVRERYDALIENTSKKITELETGKLPNLDAVKKGISNLSNLKQEVTVLRNEKIRMHVGTDKVDFSDVEKLEQQIQVIDTKLADAYLLEVTKQISALDNTKPKNQTELKTKIAAFLDRTTDIEMLRNERIKKHGSSKDPLDLSDLDKLSGSLQRINQSLVSDLITTIRVSINQMEAKTFHEQEKEIQQNFELLAKLEKTLDKSKTSEKLREDIPKLNDLLVAKQKAYPQMVQMQLKSEVFVTQLREVCQANHDDLDKTRNARLRELDRLDREAGITRMVGNLIWGLTNKVGLTTDERLDIRTKQQSLARFKNELFNDKIDTDQLISNLARKRPSELQEGLGISTDNAMELHLLLTELAGKTTSPDELEERMKAIDDISTKIGREPEHLKFVMVEEDESNKKTIGF\"
                }
                }
            ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/protein_go_function/predict/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
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

            "predicted_label": "GO:0008150",

            "confidence": 0.9999891519546509,

            "description": "biological_process"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0003674",

            "confidence": 0.9999692440032959,

            "description": "molecular_function"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0005488",

            "confidence": 0.9993295073509216,

            "description": "binding"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0005575",

            "confidence": 0.9987098574638367,

            "description": "cellular_component"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0065007",

            "confidence": 0.9962041974067688,

            "description": "biological regulation"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0016787",

            "confidence": 0.9938335418701172,

            "description": "hydrolase activity"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0044464",

            "confidence": 0.9935839176177979,

            "description": "cell part"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0009987",

            "confidence": 0.9862375259399414,

            "description": "cellular process"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0050789",

            "confidence": 0.984720766544342,

            "description": "regulation of biological process"

            },

            {

            "sequence_name": "seq_0",

            "predicted_label": "GO:0043228",

            "confidence": 0.9805465340614319,

            "description": "non-membrane-bounded organelle"

            }

        ]

        }

---------
Performance
---------

Graph of average RPS for varying number of sequences (ProteInfer GO)


.. note::
   This graph will be added soon.



----------
Related
----------
:doc:`/model-docs/ProteInfer_EC`


------------------
Model Background
------------------

Th ProteInfer deep learning model exhibits high precision on over five thousand enzyme labels within the Swiss-Prot database, prompting the researchers to explore its capability in predicting protein attributes using a broader label vocabulary, under a similar train-test arrangement to that applied in enzyme function prediction. The Gene Ontology (GO) terms offer a detailed depiction of protein functional characteristics, encompassing 32,109 such labels in Swiss-Prot, which delineate the molecular functions (for instance, DNA-binding, amylase activity), biological processes involved (like DNA replication, meiosis), and the cellular compartments they are associated with (such as mitochondrion, cytosol). These terms are organized within a complex directed acyclic graph, where certain nodes possess up to 12 ancestral nodes.

The researcher’s compared the model's performance to other methods including BLAST and an ensemble of both BLAST and ProteInfer, as well as InterProScan, which is another method of assigning functional annotations to proteins. The amalgamation of ProteInfer and BLAST emerged as the superior performer in terms of the F1 score among the individual methods, indicating an optimal balance between precision and recall.

The paper further delineates the variance in performance across diverse categories of GO terms: molecular function, biological process, and cellular component, with molecular function securing the highest F1 score. When assessed against InterProScan, a signature-based methodology, ProteInfer displayed an elevated recall at a comparable level of precision, signifying a superior capability to pinpoint true positive GO annotations. Several challenges and caveats in the comparative analysis are also highlighted, such as the comprehensive nature of GO term annotations in Swiss-Prot and discrepancies in GO term allocations between InterPro and UniProt labeling frameworks.

The researchers also assess the model's ability to recall a subset of GO term annotations not associated with the "inferred from electronic annotation" (IEA) evidence code, showcasing that a considerable portion of annotations could be successfully recalled across different GO categories at a certain threshold. This part underscores the model's performance in recalling annotations derived from either experimental work or more rigorously curated evidence, which is a crucial aspect in evaluating the model's applicability and reliability in a real-world biological context.

.. note::
   Note, for a more information about the ProteInfer model background, please refer to the ProteInfer EC API docs.


-----------------------------
Applications of ProteInfer GO
-----------------------------

ProteInfer GO effectively predicts Gene Ontology terms directly from protein sequence using a powerful multi-label transformer classifier tailored for functional annotation applications. ProteInfer GO predictions can potentially aid genome annotation, protein characterization, system biology, engineering, and biomedical applications involving analyzing protein and gene function.

* Protein function prediction - Predicting GO terms can help annotate the molecular functions, biological processes, and cellular locations of uncharacterized proteins. This can aid discovery of the roles proteins play in biological systems.

* Protein interaction prediction - Knowing the GO terms for proteins can help predict if proteins may interact based on if they share similar functions and processes. This can guide experiments to validate interactions.

* Functional Annotation: ProteInfer GO can be used to annotate novel or uncharacterized protein sequences with functional labels. This can be particularly helpful in large-scale genomics projects where a myriad of new sequences are generated.



