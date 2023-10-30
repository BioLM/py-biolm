=======
ESMFold
=======

.. article-info::
    :avatar: ../img/book_icon.png
    :author: Article Information
    :date: Oct 24, 2023
    :read-time: 5 min read
    :class-container: sd-p-2 sd-outline-muted sd-rounded-1

*This page explains the use of ESMFold, as well as documents
its usage on BioLM for protein structure prediction.*

-----------
Description
-----------

Recent computational protein folding capability enables myriad of applications
from elucidating structures of novel proteins, , designing engineered proteins,
modeling molecular interactions, evaluating impacts of mutations, and assembling
multi-protein complexes. The BioLM API is democratizing access to 3D structural
modeling, with its rapid ESMfold API,  bringing the power of structural biology
to address diverse questions in protein science, biomedicine, synthetic biology,
and beyond.

--------
Benefits
--------

The API can be used by biologists, data scientists, engineers, etc. The key values of the BioLM API is speed, scalability and cost.

* The API allows 1440 folds per minute, or 2M per day (Figure 1).
* The BioLM API allows scientists to programmatically interact with ESMfold,
  making it easier to integrate the model into their scientific workflows.
  The API accelerates workflow, allows for customization, and is designed to be
  highly scalable.
* Our unique API UI Chat allows users to interact with our API and access
  multiple language models without the need to code!
* The benefit of having access to multiple GPUs is parallel processing. Each
  GPU can handle a different protein folding simulation, allowing for folding
  dozens of proteins in parallel

---------
API Usage
---------

This is the url to use when querying the BioLM ESMfold Prediction Endpoint: https://biolm.ai/api/v1/models/esmfold-singlechain/predict/

^^^^^^^^^^^^^^^
Making Requests
^^^^^^^^^^^^^^^

*Definitions*

-Request Keys:

data:
  Inside each instance, there's a key named "data" that holds another dictionary. This dictionary contains the actual input data for the prediction.

text:
  Inside the "data" dictionary, there's a key named "text". The value associated with "text" should be a string containing the amino acid sequence that the user wants to submit for structure prediction.


-Response Keys:

predictions:
  This is the main key in the JSON object that contains an array of prediction results. Each element in the array represents a set of predictions for one input instance.

pdb:
  Contains a string representing the 3D structure of the protein predicted by the model in PDB (Protein Data Bank) format

mean_plddt:
  Contains a string representing the mean pLDDT score of the predicted structure. The pLDDT (predicted Local Distance Difference Test) score is a measure of the accuracy of the predicted structure, with values ranging from 0 to 100. Higher scores indicate higher confidence in the prediction.

durations:
  Contains a string that represents the total time taken for the request to be processed and the response to be generated


.. tab-set::

    .. tab-item:: Curl
        :sync: curl

        .. code:: shell

            curl --location 'https://biolm.ai/api/v1/models/esmfold-singlechain/predict/' \
              --header 'Cookie: access=MY_ACCESS_TOKEN;refresh=MY_REFRESH_TOKEN' \
              --header 'Content-Type: application/json' \
              --data '{
                "instances": [{
                  "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"}
                }]
              }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests
            import json

            url = "https://biolm.ai/api/v1/models/esmfold-singlechain/predict/"

            payload = json.dumps({
              "instances": [
                {
                  "data": {
                    "text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ"
                  }
                }
              ]
            })
            headers = {
              'Cookie': 'access=MY_ACCESS_TOKEN;refresh=MY_REFRESH_TOKEN',
              'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)

    .. tab-item:: Biolmai SDK
        :sync: sdk

        .. code:: sdk

            import biolmai
            seqs = ["MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ""]

            cls = biolmai.ESMFoldSingleChain()
            resp = cls.predict(seqs)

    .. tab-item:: R
        :sync: r

        .. code:: shell

            library(RCurl)
            headers = c(
              "Cookie" = "access=MY_ACCESS_TOKEN;refresh=MY_REFRESH_TOKEN",
              "Content-Type" = "application/json"
            )
            params = "{
              \"instances\": [
                {
                  \"data\": {
                    \"text\": \"MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQHAVAFAQSQLHQQDRKWPRLPDYFAIGRTTALALHTVSGQKILYPQDREISEVLLQLPELQNIAGKRALILRGNGGRELIGDTLTARGAEVTFCECYQRCAIHYDGAEEAMRWQAREVTMVVVTSGEMLQQLWSLIPQWYREHWLLHCRLLVVSERLAKLARELGWQDIKVADNADNDALLRALQ\"
                  }
                }
              ]
            }"
            res <- postForm("https://biolm.ai/api/v1/models/esmfold-singlechain/predict/", .opts=list(postfields = params, httpheader = headers, followlocation = TRUE), style = "httppost")
            cat(res)

^^^^^^^^^^^^^
JSON Response
^^^^^^^^^^^^^

.. dropdown:: Expand Example Response

    .. code:: json

        {
          "predictions": [
            {
              "pdb": [
                "PARENT N/A\nATOM      1  N   MET A   1      -4.572  14.264  12.502  1.00 84.99           N  \nATOM      2  CA  MET A   1      -5.476  13.273  11.925  1.00 85.61           C  \nATOM      3  C   MET A   1      -5.150  13.031  10.454  1.00 87.65           C  \nATOM      4  CB  MET A   1      -6.931  13.721  12.071  1.00 80.07           C  \nATOM      5  O   MET A   1      -5.177  13.961   9.647  1.00 81.61           O  \nATOM      6  CG  MET A   1      -7.942  12.668  11.646  1.00 71.48           C  \nATOM      7  SD  MET A   1      -9.343  12.524  12.823  1.00 64.78           S  \nATOM      8  CE  MET A   1     -10.658  13.312  11.853  1.00 67.33           C  \nATOM      9  N   SER A   2      -4.501  12.059   9.963  1.00 89.83           N  \nATOM     10  CA  SER A   2      -4.106  11.761   8.590  1.00 89.80           C  \nATOM     11  C   SER A   2      -5.110  10.833   7.914  1.00 89.51           C  \nATOM     12  CB  SER A   2      -2.714  11.131   8.556  1.00 86.34           C  \nATOM     13  O   SER A   2      -5.761  10.025   8.580  1.00 85.88           O  \nATOM     14  OG  SER A   2      -1.762  11.981   9.173  1.00 77.03           O  \nATOM     15  N   ILE A   3      -5.828  11.200   6.932  1.00 89.91           N  \nATOM     16  CA  ILE A   3      -6.772  10.401   6.158  1.00 89.61           C  \nATOM     17  C   ILE A   3      -6.011   9.415   5.275  1.00 89.11           C  \nATOM     18  CB  ILE A   3      -7.694  11.292   5.296  1.00 87.28           C  \nATOM     19  O   ILE A   3      -5.106   9.806   4.534  1.00 85.57           O  \nATOM     20  CG1 ILE A   3      -8.442  12.298   6.178  1.00 77.74           C  \nATOM     21  CG2 ILE A   3      -8.674  10.435   4.489  1.00 77.96           C  \nATOM     22  CD1 ILE A   3      -9.185  13.373   5.397  1.00 75.75           C  \nATOM     23  N   LEU A   4      -6.151   8.179   5.566  1.00 86.89           N  \nATOM     24  CA  LEU A   4      -5.565   7.099   4.780  1.00 86.42           C  \nATOM     25  C   LEU A   4      -6.379   6.844   3.516  1.00 85.95           C  \nATOM     26  CB  LEU A   4      -5.478   5.817   5.612  1.00 83.89           C  \nATOM     27  O   LEU A   4      -7.589   6.617   3.586  1.00 82.52           O  \nATOM     28  CG  LEU A   4      -4.768   4.631   4.958  1.00 78.16           C  \nATOM     29  CD1 LEU A   4      -3.295   4.954   4.732  1.00 72.95           C  \nATOM     30  CD2 LEU A   4      -4.920   3.377   5.814  1.00 73.90           C  \nATOM     31  N   VAL A   5      -5.997   7.135   2.383  1.00 86.17           N  \nATOM     32  CA  VAL A   5      -6.700   6.922   1.121  1.00 85.29           C  \nATOM     33  C   VAL A   5      -6.282   5.583   0.517  1.00 84.88           C  \nATOM     34  CB  VAL A   5      -6.428   8.066   0.120  1.00 82.82           C  \nATOM     35  O   VAL A   5      -5.104   5.370   0.219  1.00 81.93           O  \nATOM     36  CG1 VAL A   5      -7.684   8.384  -0.690  1.00 74.66           C  \nATOM     37  CG2 VAL A   5      -5.934   9.311   0.855  1.00 76.11           C  \nATOM     38  N   THR A   6      -7.077   4.563   0.404  1.00 83.42           N  \nATOM     39  CA  THR A   6      -6.710   3.268  -0.158  1.00 82.83           C  \nATOM     40  C   THR A   6      -6.905   3.260  -1.671  1.00 82.98           C  \nATOM     41  CB  THR A   6      -7.536   2.131   0.475  1.00 81.04           C  \nATOM     42  O   THR A   6      -7.820   3.903  -2.188  1.00 80.65           O  \nATOM     43  CG2 THR A   6      -7.221   1.984   1.960  1.00 76.38           C  \nATOM     44  OG1 THR A   6      -8.931   2.419   0.319  1.00 76.50           O  \nATOM     45  N   ARG A   7      -5.915   2.802  -2.506  1.00 81.72           N  \nATOM     46  CA  ARG A   7      -6.109   2.453  -3.910  1.00 81.61           C  \nATOM     47  C   ARG A   7      -6.893   1.153  -4.047  1.00 81.93           C  \nATOM     48  CB  ARG A   7      -4.762   2.331  -4.624  1.00 79.30           C  \nATOM     49  O   ARG A   7      -6.836   0.291  -3.168  1.00 79.94           O  \nATOM     50  CG  ARG A   7      -3.848   3.530  -4.427  1.00 75.29           C  \nATOM     51  CD  ARG A   7      -2.513   3.345  -5.134  1.00 76.30           C  \nATOM     52  NE  ARG A   7      -1.620   2.471  -4.380  1.00 69.12           N  \nATOM     53  NH1 ARG A   7      -0.793   1.370  -6.235  1.00 63.04           N  \nATOM     54  NH2 ARG A   7      -0.050   0.812  -4.139  1.00 61.29           N  \nATOM     55  CZ  ARG A   7      -0.823   1.553  -4.920  1.00 71.19           C  \nATOM     56  N   PRO A   8      -7.862   1.016  -5.158  1.00 80.07           N  \nATOM     57  CA  PRO A   8      -8.517  -0.273  -5.389  1.00 80.20           C  \nATOM     58  C   PRO A   8      -7.522  -1.404  -5.641  1.00 80.40           C  \nATOM     59  CB  PRO A   8      -9.375  -0.015  -6.631  1.00 78.26           C  \nATOM     60  O   PRO A   8      -6.450  -1.174  -6.205  1.00 78.23           O  \nATOM     61  CG  PRO A   8      -8.721   1.144  -7.312  1.00 76.81           C  \nATOM     62  CD  PRO A   8      -8.028   1.979  -6.274  1.00 77.74           C  \nATOM     63  N   SER A   9      -7.740  -2.632  -4.906  1.00 82.81           N  \nATOM     64  CA  SER A   9      -6.892  -3.791  -5.164  1.00 83.46           C  \nATOM     65  C   SER A   9      -7.081  -4.309  -6.586  1.00 83.75           C  \nATOM     66  CB  SER A   9      -7.190  -4.907  -4.162  1.00 80.52           C  \nATOM     67  O   SER A   9      -8.139  -4.115  -7.187  1.00 81.17           O  \nATOM     68  OG  SER A   9      -7.716  -6.047  -4.820  1.00 74.65           O  \nATOM     69  N   PRO A  10      -5.942  -4.628  -7.300  1.00 75.88           N  \nATOM     70  CA  PRO A  10      -6.115  -5.229  -8.624  1.00 75.12           C  \nATOM     71  C   PRO A  10      -7.209  -6.294  -8.650  1.00 74.87           C  \nATOM     72  CB  PRO A  10      -4.744  -5.846  -8.913  1.00 72.15           C  \nATOM     73  O   PRO A  10      -7.908  -6.442  -9.656  1.00 72.16           O  \nATOM     74  CG  PRO A  10      -3.806  -5.146  -7.984  1.00 69.24           C  \nATOM     75  CD  PRO A  10      -4.570  -4.731  -6.759  1.00 69.04           C  \nATOM     76  N   ALA A  11      -7.448  -7.081  -7.506  1.00 74.89           N  \nATOM     77  CA  ALA A  11      -8.494  -8.101  -7.518  1.00 73.72           C  \nATOM     78  C   ALA A  11      -9.881  -7.465  -7.529  1.00 73.35           C  \nATOM     79  CB  ALA A  11      -8.346  -9.029  -6.314  1.00 70.06           C  \nATOM     80  O   ALA A  11     -10.849  -8.079  -7.984  1.00 69.65           O  \nATOM     81  N   GLU A  12      -9.854  -6.213  -7.084  1.00 73.39           N  \nATOM     82  CA  GLU A  12     -11.136  -5.517  -7.150  1.00 73.14           C  \nATOM     83  C   GLU A  12     -11.459  -5.089  -8.579  1.00 71.50           C  \nATOM     84  CB  GLU A  12     -11.134  -4.299  -6.223  1.00 68.99           C  \nATOM     85  O   GLU A  12     -12.552  -4.588  -8.850  1.00 69.09           O  \nATOM     86  CG  GLU A  12     -11.168  -4.651  -4.743  1.00 65.85           C  \nATOM     87  CD  GLU A  12     -10.828  -3.477  -3.839  1.00 64.45           C  \nATOM     88  OE1 GLU A  12     -10.715  -2.336  -4.342  1.00 65.88           O  \nATOM     89  OE2 GLU A  12     -10.673  -3.700  -2.618  1.00 64.49           O  \nATOM     90  N   LEU A  13     -10.374  -5.204  -9.394  1.00 66.16           N  \nATOM     91  CA  LEU A  13     -10.505  -4.781 -10.784  1.00 65.21           C  \nATOM     92  C   LEU A  13     -10.886  -5.958 -11.677  1.00 64.60           C  \nATOM     93  CB  LEU A  13      -9.200  -4.153 -11.279  1.00 62.50           C  \nATOM     94  O   LEU A  13     -11.125  -5.782 -12.874  1.00 63.39           O  \nATOM     95  CG  LEU A  13      -8.821  -2.804 -10.665  1.00 61.37           C  \nATOM     96  CD1 LEU A  13      -7.376  -2.453 -11.004  1.00 59.14           C  \nATOM     97  CD2 LEU A  13      -9.769  -1.711 -11.148  1.00 60.20           C  \nATOM     98  N   VAL A  14     -11.164  -7.141 -11.026  1.00 73.05           N  \nATOM     99  CA  VAL A  14     -11.658  -8.212 -11.885  1.00 72.60           C  \nATOM    100  C   VAL A  14     -13.052  -8.638 -11.431  1.00 71.57           C  \nATOM    101  CB  VAL A  14     -10.702  -9.426 -11.884  1.00 67.97           C  \nATOM    102  O   VAL A  14     -13.337  -8.675 -10.232  1.00 67.73           O  \nATOM    103  CG1 VAL A  14     -11.207 -10.511 -12.834  1.00 61.45           C  \nATOM    104  CG2 VAL A  14      -9.288  -8.993 -12.266  1.00 61.82           C  \nTER     105      VAL A  14\nEND\n"
              ],
              "mean_plddt": "76.2",
              "ptm": "0.017",
              "duration": "3.7s"
            }
          ]
        }


^^^^^^^^^^^
Performance
^^^^^^^^^^^

Graph of average RPS for varying number of sequences

.. figure:: ../img/esmfold_perf.png
   :scale: 50 %
   :alt: map to buried treasure

   This is the caption of the figure (a simple paragraph).

   The legend consists of all elements after the caption.

.. note::
   The duration for folding predominantly depends on sequence length.

   A sequence of length 60 might fold in 6 seconds, however a sequence of
   length 500 might fold in 400 seconds. Above, we plot the performance of a
   single sequence length.

--------
Related
--------

:doc:`/model-docs/embeddings`
:doc:`/model-docs/esm_1v_masking`
:doc:`/model-docs/ESM-InverseFold`

------------------
ESMFold Background
------------------

Advances in large-scale language modeling is moving us closer to achieving a
universal model for proteins. Esmfold, a protein structure prediction tool that
utilizes the ESM-2 language model, is one of the most advanced models currently
available. Esmfold's training data is derived from UniRef, with a focus on
UniRef50 clusters, which are non-redundant sets of protein sequences with at
least 50% sequence identity to each other. The training process included the
selection of sequences from around 43 million UniRef50 training groups, covering
close to 138 million UniRef90 sequences, which amounts to nearly 65 million
distinct sequences throughout the training period. Esmfold achieves a faster
performance compared to AlphaFold as it is capable of conducting end-to-end
atomic structure predictions straight from the sequence, bypassing the need for
a multiple sequence alignment (MSA). These models learn so much about protein
sequences and the evolutionary patterns that relate sequences to function, that
then they don’t need sequence alignments at all in order to fold them. This
leads to a more simplified neural architecture for inference, drastically
reducing the time taken in the inference forward pass and removing the lengthy
search for related proteins, which is a notable part of the process in AlphaFold
-“This results in an improvement in speed of up to 60x on the inference forward
pass alone, while also removing the search process for related proteins
entirely, which can take over 10 minutes with the high-sensitivity pipelines
used by AlphaFold” -  Lin et al., 2022. In addition, AlphaFold 2 may struggle
with ‘orphan proteins’, which lack multiple sequence alignments due to
insufficient database sequences. Since esmfold bypasses alignments, it may model
orphan proteins more effectively. This, in turn, could inform and facilitate the
de novo design of proteins with desired characteristics, thereby extending the
reach and success of de novo protein design efforts.

-----------------------
Applications of Folding
-----------------------

Esmfold is a revolutionary tool for folding that can be used by a diverse range
of topics within biology, ranging from synthetic biology, neuroscience, enzyme
engineering, immunology, virology, industrial biotechnology, etc. A great
starting point for Esmfold is when scientist starts with a single sequence or
library of designed sequences for which they wish to understand the 3D
structure.

* Predict how post-translational modifications affect chaperone protein
  structure.
* Analyze capsid protein folding of viruses like HIV, influenza, and SARS-CoV-2.
* Design novel self-assembling protein nanostructures by rapidly predicting
  their protein architectures.
* Predict 3D structures of computationally designed enzyme sequences to
  assess if they fold into stable enzymes; by rapidly modeling many designs,
  ESMFold facilitates computational filtering and optimization of the lead de
  novo enzymes. (have a link to a tutorial page here).
* Used in antibody engineering. Once CDR variants are designed computationally,
  scientists can use Esmfold to predict structures to filter and select optimal
  candidates. Can also predict structures for lead antibody variable domains.
