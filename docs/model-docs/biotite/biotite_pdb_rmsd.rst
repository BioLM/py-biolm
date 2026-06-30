Biotite PDB RMSD API
====================

The Biotite PDB RMSD service computes optimal backbone or all-atom root-mean-square deviation between pairs of protein (or other macromolecular) structures using Biotite’s NumPy/Cython-accelerated structure routines. Given two validated PDB strings and user-specified chain mappings, it applies Kabsch-based superposition and returns RMSD values in Å for each request item, supporting batches of up to 8 structure pairs. Typical applications include clustering MD snapshots, comparing designed variants to references, and ranking candidate models in protein engineering workflows.

Predict
-------

Compute RMSD between pairs of protein structures, with single-chain and multi-chain mappings.

.. tab-set::

    .. tab-item:: Python (biolmai)
        :sync: sdk

        .. code:: python

            from biolmai import BioLM
            response = BioLM(
                entity="biotite",
                action="predict",
                params={},
                items=[
                  {
                    "pdb_a": "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C\nATOM      3  C   ALA A   1      12.000  11.458  10.000  1.00 20.00           C\nATOM      4  O   ALA A   1      11.458  12.000  11.000  1.00 20.00           O\nATOM      5  N   GLY A   2      13.458  11.458  10.000  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.000  12.916  10.000  1.00 20.00           C\nATOM      7  C   GLY A   2      15.458  12.916  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.000  14.374  10.000  1.00 20.00           O\nATOM      9  N   SER B   1      20.000  20.000  20.000  1.00 20.00           N\nATOM     10  CA  SER B   1      21.458  20.000  20.000  1.00 20.00           C\nATOM     11  C   SER B   1      22.000  21.458  20.000  1.00 20.00           C\nATOM     12  O   SER B   1      21.458  22.000  21.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                    "pdb_b": "ATOM      1  N   ALA A   1      10.200  10.100   9.900  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.658  10.050  10.050  1.00 20.00           C\nATOM      3  C   ALA A   1      12.150  11.550   9.950  1.00 20.00           C\nATOM      4  O   ALA A   1      11.600  12.050  10.900  1.00 20.00           O\nATOM      5  N   GLY A   2      13.650  11.500  10.050  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.150  12.950  10.050  1.00 20.00           C\nATOM      7  C   GLY A   2      15.600  12.900  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.050  14.350   9.950  1.00 20.00           O\nATOM      9  N   SER B   1      19.900  19.900  20.100  1.00 20.00           N\nATOM     10  CA  SER B   1      21.350  19.950  19.950  1.00 20.00           C\nATOM     11  C   SER B   1      21.950  21.400  19.950  1.00 20.00           C\nATOM     12  O   SER B   1      21.350  21.950  20.950  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                    "chain_ids": {
                      "a": [
                        "A"
                      ],
                      "b": [
                        "A"
                      ]
                    }
                  },
                  {
                    "pdb_a": "ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           N\nATOM      2  CA  ALA A   1       6.458   5.000   5.000  1.00 20.00           C\nATOM      3  C   ALA A   1       7.000   6.458   5.000  1.00 20.00           C\nATOM      4  O   ALA A   1       6.458   7.000   6.000  1.00 20.00           O\nATOM      5  N   GLY A   2       8.458   6.458   5.000  1.00 20.00           N\nATOM      6  CA  GLY A   2       9.000   7.916   5.000  1.00 20.00           C\nATOM      7  C   GLY A   2      10.458   7.916   5.000  1.00 20.00           C\nATOM      8  O   GLY A   2      11.000   9.374   5.000  1.00 20.00           O\nATOM      9  N   SER B   1      15.000  15.000  15.000  1.00 20.00           N\nATOM     10  CA  SER B   1      16.458  15.000  15.000  1.00 20.00           C\nATOM     11  C   SER B   1      17.000  16.458  15.000  1.00 20.00           C\nATOM     12  O   SER B   1      16.458  17.000  16.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                    "pdb_b": "ATOM      1  N   ALA C   1       5.100   5.050   5.050  1.00 20.00           N\nATOM      2  CA  ALA C   1       6.550   5.050   5.050  1.00 20.00           C\nATOM      3  C   ALA C   1       7.050   6.500   5.050  1.00 20.00           C\nATOM      4  O   ALA C   1       6.500   7.050   6.050  1.00 20.00           O\nATOM      5  N   GLY C   2       8.550   6.450   5.050  1.00 20.00           N\nATOM      6  CA  GLY C   2       9.050   7.900   5.050  1.00 20.00           C\nATOM      7  C   GLY C   2      10.500   7.900   5.000  1.00 20.00           C\nATOM      8  O   GLY C   2      11.050   9.350   5.000  1.00 20.00           O\nATOM      9  N   SER D   1      14.900  14.950  14.950  1.00 20.00           N\nATOM     10  CA  SER D   1      16.350  14.950  14.950  1.00 20.00           C\nATOM     11  C   SER D   1      16.950  16.400  14.950  1.00 20.00           C\nATOM     12  O   SER D   1      16.350  16.950  15.950  1.00 20.00           O\nTER      13      SER D   1\nEND\n",
                    "chain_ids": {
                      "a": [
                        "A",
                        "B"
                      ],
                      "b": [
                        "C",
                        "D"
                      ]
                    }
                  }
                ]
            )
            print(response)

    .. tab-item:: cURL
        :sync: curl

        .. code:: bash

            curl -X POST https://biolm.ai/api/v3/biotite/predict/ \
              -H "Authorization: Token YOUR_API_KEY" \
              -H "Content-Type: application/json" \
              -d '{
              "items": [
                {
                  "pdb_a": "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C\nATOM      3  C   ALA A   1      12.000  11.458  10.000  1.00 20.00           C\nATOM      4  O   ALA A   1      11.458  12.000  11.000  1.00 20.00           O\nATOM      5  N   GLY A   2      13.458  11.458  10.000  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.000  12.916  10.000  1.00 20.00           C\nATOM      7  C   GLY A   2      15.458  12.916  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.000  14.374  10.000  1.00 20.00           O\nATOM      9  N   SER B   1      20.000  20.000  20.000  1.00 20.00           N\nATOM     10  CA  SER B   1      21.458  20.000  20.000  1.00 20.00           C\nATOM     11  C   SER B   1      22.000  21.458  20.000  1.00 20.00           C\nATOM     12  O   SER B   1      21.458  22.000  21.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                  "pdb_b": "ATOM      1  N   ALA A   1      10.200  10.100   9.900  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.658  10.050  10.050  1.00 20.00           C\nATOM      3  C   ALA A   1      12.150  11.550   9.950  1.00 20.00           C\nATOM      4  O   ALA A   1      11.600  12.050  10.900  1.00 20.00           O\nATOM      5  N   GLY A   2      13.650  11.500  10.050  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.150  12.950  10.050  1.00 20.00           C\nATOM      7  C   GLY A   2      15.600  12.900  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.050  14.350   9.950  1.00 20.00           O\nATOM      9  N   SER B   1      19.900  19.900  20.100  1.00 20.00           N\nATOM     10  CA  SER B   1      21.350  19.950  19.950  1.00 20.00           C\nATOM     11  C   SER B   1      21.950  21.400  19.950  1.00 20.00           C\nATOM     12  O   SER B   1      21.350  21.950  20.950  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                  "chain_ids": {
                    "a": [
                      "A"
                    ],
                    "b": [
                      "A"
                    ]
                  }
                },
                {
                  "pdb_a": "ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           N\nATOM      2  CA  ALA A   1       6.458   5.000   5.000  1.00 20.00           C\nATOM      3  C   ALA A   1       7.000   6.458   5.000  1.00 20.00           C\nATOM      4  O   ALA A   1       6.458   7.000   6.000  1.00 20.00           O\nATOM      5  N   GLY A   2       8.458   6.458   5.000  1.00 20.00           N\nATOM      6  CA  GLY A   2       9.000   7.916   5.000  1.00 20.00           C\nATOM      7  C   GLY A   2      10.458   7.916   5.000  1.00 20.00           C\nATOM      8  O   GLY A   2      11.000   9.374   5.000  1.00 20.00           O\nATOM      9  N   SER B   1      15.000  15.000  15.000  1.00 20.00           N\nATOM     10  CA  SER B   1      16.458  15.000  15.000  1.00 20.00           C\nATOM     11  C   SER B   1      17.000  16.458  15.000  1.00 20.00           C\nATOM     12  O   SER B   1      16.458  17.000  16.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                  "pdb_b": "ATOM      1  N   ALA C   1       5.100   5.050   5.050  1.00 20.00           N\nATOM      2  CA  ALA C   1       6.550   5.050   5.050  1.00 20.00           C\nATOM      3  C   ALA C   1       7.050   6.500   5.050  1.00 20.00           C\nATOM      4  O   ALA C   1       6.500   7.050   6.050  1.00 20.00           O\nATOM      5  N   GLY C   2       8.550   6.450   5.050  1.00 20.00           N\nATOM      6  CA  GLY C   2       9.050   7.900   5.050  1.00 20.00           C\nATOM      7  C   GLY C   2      10.500   7.900   5.000  1.00 20.00           C\nATOM      8  O   GLY C   2      11.050   9.350   5.000  1.00 20.00           O\nATOM      9  N   SER D   1      14.900  14.950  14.950  1.00 20.00           N\nATOM     10  CA  SER D   1      16.350  14.950  14.950  1.00 20.00           C\nATOM     11  C   SER D   1      16.950  16.400  14.950  1.00 20.00           C\nATOM     12  O   SER D   1      16.350  16.950  15.950  1.00 20.00           O\nTER      13      SER D   1\nEND\n",
                  "chain_ids": {
                    "a": [
                      "A",
                      "B"
                    ],
                    "b": [
                      "C",
                      "D"
                    ]
                  }
                }
              ],
              "params": {}
            }'

    .. tab-item:: Python Requests
        :sync: python

        .. code:: python

            import requests

            url = "https://biolm.ai/api/v3/biotite/predict/"
            headers = {
                "Authorization": "Token YOUR_API_KEY",
                "Content-Type": "application/json"
            }
            payload = {
                  "items": [
                    {
                      "pdb_a": "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C\nATOM      3  C   ALA A   1      12.000  11.458  10.000  1.00 20.00           C\nATOM      4  O   ALA A   1      11.458  12.000  11.000  1.00 20.00           O\nATOM      5  N   GLY A   2      13.458  11.458  10.000  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.000  12.916  10.000  1.00 20.00           C\nATOM      7  C   GLY A   2      15.458  12.916  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.000  14.374  10.000  1.00 20.00           O\nATOM      9  N   SER B   1      20.000  20.000  20.000  1.00 20.00           N\nATOM     10  CA  SER B   1      21.458  20.000  20.000  1.00 20.00           C\nATOM     11  C   SER B   1      22.000  21.458  20.000  1.00 20.00           C\nATOM     12  O   SER B   1      21.458  22.000  21.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                      "pdb_b": "ATOM      1  N   ALA A   1      10.200  10.100   9.900  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.658  10.050  10.050  1.00 20.00           C\nATOM      3  C   ALA A   1      12.150  11.550   9.950  1.00 20.00           C\nATOM      4  O   ALA A   1      11.600  12.050  10.900  1.00 20.00           O\nATOM      5  N   GLY A   2      13.650  11.500  10.050  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.150  12.950  10.050  1.00 20.00           C\nATOM      7  C   GLY A   2      15.600  12.900  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.050  14.350   9.950  1.00 20.00           O\nATOM      9  N   SER B   1      19.900  19.900  20.100  1.00 20.00           N\nATOM     10  CA  SER B   1      21.350  19.950  19.950  1.00 20.00           C\nATOM     11  C   SER B   1      21.950  21.400  19.950  1.00 20.00           C\nATOM     12  O   SER B   1      21.350  21.950  20.950  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                      "chain_ids": {
                        "a": [
                          "A"
                        ],
                        "b": [
                          "A"
                        ]
                      }
                    },
                    {
                      "pdb_a": "ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           N\nATOM      2  CA  ALA A   1       6.458   5.000   5.000  1.00 20.00           C\nATOM      3  C   ALA A   1       7.000   6.458   5.000  1.00 20.00           C\nATOM      4  O   ALA A   1       6.458   7.000   6.000  1.00 20.00           O\nATOM      5  N   GLY A   2       8.458   6.458   5.000  1.00 20.00           N\nATOM      6  CA  GLY A   2       9.000   7.916   5.000  1.00 20.00           C\nATOM      7  C   GLY A   2      10.458   7.916   5.000  1.00 20.00           C\nATOM      8  O   GLY A   2      11.000   9.374   5.000  1.00 20.00           O\nATOM      9  N   SER B   1      15.000  15.000  15.000  1.00 20.00           N\nATOM     10  CA  SER B   1      16.458  15.000  15.000  1.00 20.00           C\nATOM     11  C   SER B   1      17.000  16.458  15.000  1.00 20.00           C\nATOM     12  O   SER B   1      16.458  17.000  16.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
                      "pdb_b": "ATOM      1  N   ALA C   1       5.100   5.050   5.050  1.00 20.00           N\nATOM      2  CA  ALA C   1       6.550   5.050   5.050  1.00 20.00           C\nATOM      3  C   ALA C   1       7.050   6.500   5.050  1.00 20.00           C\nATOM      4  O   ALA C   1       6.500   7.050   6.050  1.00 20.00           O\nATOM      5  N   GLY C   2       8.550   6.450   5.050  1.00 20.00           N\nATOM      6  CA  GLY C   2       9.050   7.900   5.050  1.00 20.00           C\nATOM      7  C   GLY C   2      10.500   7.900   5.000  1.00 20.00           C\nATOM      8  O   GLY C   2      11.050   9.350   5.000  1.00 20.00           O\nATOM      9  N   SER D   1      14.900  14.950  14.950  1.00 20.00           N\nATOM     10  CA  SER D   1      16.350  14.950  14.950  1.00 20.00           C\nATOM     11  C   SER D   1      16.950  16.400  14.950  1.00 20.00           C\nATOM     12  O   SER D   1      16.350  16.950  15.950  1.00 20.00           O\nTER      13      SER D   1\nEND\n",
                      "chain_ids": {
                        "a": [
                          "A",
                          "B"
                        ],
                        "b": [
                          "C",
                          "D"
                        ]
                      }
                    }
                  ],
                  "params": {}
                }

            response = requests.post(url, headers=headers, json=payload)
            print(response.json())

    .. tab-item:: R
        :sync: r

        .. code:: r

            library(httr)

            url <- "https://biolm.ai/api/v3/biotite/predict/"
            headers <- c("Authorization" = "Token YOUR_API_KEY", "Content-Type" = "application/json")
            body <- list(
              items = list(
                list(
                  pdb_a = "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
            ATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C
            ATOM      3  C   ALA A   1      12.000  11.458  10.000  1.00 20.00           C
            ATOM      4  O   ALA A   1      11.458  12.000  11.000  1.00 20.00           O
            ATOM      5  N   GLY A   2      13.458  11.458  10.000  1.00 20.00           N
            ATOM      6  CA  GLY A   2      14.000  12.916  10.000  1.00 20.00           C
            ATOM      7  C   GLY A   2      15.458  12.916  10.000  1.00 20.00           C
            ATOM      8  O   GLY A   2      16.000  14.374  10.000  1.00 20.00           O
            ATOM      9  N   SER B   1      20.000  20.000  20.000  1.00 20.00           N
            ATOM     10  CA  SER B   1      21.458  20.000  20.000  1.00 20.00           C
            ATOM     11  C   SER B   1      22.000  21.458  20.000  1.00 20.00           C
            ATOM     12  O   SER B   1      21.458  22.000  21.000  1.00 20.00           O
            TER      13      SER B   1
            END
            ",
                  pdb_b = "ATOM      1  N   ALA A   1      10.200  10.100   9.900  1.00 20.00           N
            ATOM      2  CA  ALA A   1      11.658  10.050  10.050  1.00 20.00           C
            ATOM      3  C   ALA A   1      12.150  11.550   9.950  1.00 20.00           C
            ATOM      4  O   ALA A   1      11.600  12.050  10.900  1.00 20.00           O
            ATOM      5  N   GLY A   2      13.650  11.500  10.050  1.00 20.00           N
            ATOM      6  CA  GLY A   2      14.150  12.950  10.050  1.00 20.00           C
            ATOM      7  C   GLY A   2      15.600  12.900  10.000  1.00 20.00           C
            ATOM      8  O   GLY A   2      16.050  14.350   9.950  1.00 20.00           O
            ATOM      9  N   SER B   1      19.900  19.900  20.100  1.00 20.00           N
            ATOM     10  CA  SER B   1      21.350  19.950  19.950  1.00 20.00           C
            ATOM     11  C   SER B   1      21.950  21.400  19.950  1.00 20.00           C
            ATOM     12  O   SER B   1      21.350  21.950  20.950  1.00 20.00           O
            TER      13      SER B   1
            END
            ",
                  chain_ids = list(
                    a = list(
                      "A"
                    ),
                    b = list(
                      "A"
                    )
                  )
                ),
                list(
                  pdb_a = "ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           N
            ATOM      2  CA  ALA A   1       6.458   5.000   5.000  1.00 20.00           C
            ATOM      3  C   ALA A   1       7.000   6.458   5.000  1.00 20.00           C
            ATOM      4  O   ALA A   1       6.458   7.000   6.000  1.00 20.00           O
            ATOM      5  N   GLY A   2       8.458   6.458   5.000  1.00 20.00           N
            ATOM      6  CA  GLY A   2       9.000   7.916   5.000  1.00 20.00           C
            ATOM      7  C   GLY A   2      10.458   7.916   5.000  1.00 20.00           C
            ATOM      8  O   GLY A   2      11.000   9.374   5.000  1.00 20.00           O
            ATOM      9  N   SER B   1      15.000  15.000  15.000  1.00 20.00           N
            ATOM     10  CA  SER B   1      16.458  15.000  15.000  1.00 20.00           C
            ATOM     11  C   SER B   1      17.000  16.458  15.000  1.00 20.00           C
            ATOM     12  O   SER B   1      16.458  17.000  16.000  1.00 20.00           O
            TER      13      SER B   1
            END
            ",
                  pdb_b = "ATOM      1  N   ALA C   1       5.100   5.050   5.050  1.00 20.00           N
            ATOM      2  CA  ALA C   1       6.550   5.050   5.050  1.00 20.00           C
            ATOM      3  C   ALA C   1       7.050   6.500   5.050  1.00 20.00           C
            ATOM      4  O   ALA C   1       6.500   7.050   6.050  1.00 20.00           O
            ATOM      5  N   GLY C   2       8.550   6.450   5.050  1.00 20.00           N
            ATOM      6  CA  GLY C   2       9.050   7.900   5.050  1.00 20.00           C
            ATOM      7  C   GLY C   2      10.500   7.900   5.000  1.00 20.00           C
            ATOM      8  O   GLY C   2      11.050   9.350   5.000  1.00 20.00           O
            ATOM      9  N   SER D   1      14.900  14.950  14.950  1.00 20.00           N
            ATOM     10  CA  SER D   1      16.350  14.950  14.950  1.00 20.00           C
            ATOM     11  C   SER D   1      16.950  16.400  14.950  1.00 20.00           C
            ATOM     12  O   SER D   1      16.350  16.950  15.950  1.00 20.00           O
            TER      13      SER D   1
            END
            ",
                  chain_ids = list(
                    a = list(
                      "A",
                      "B"
                    ),
                    b = list(
                      "C",
                      "D"
                    )
                  )
                )
              ),
              params = list()
            )

            res <- POST(url, add_headers(.headers = headers), body = body, encode = "json")
            print(content(res))

.. http:post:: /api/v3/biotite/predict/

   Predict endpoint for Biotite PDB RMSD.

   :reqheader Content-Type: application/json
   :reqheader Authorization: Token YOUR_API_KEY

   .. container:: field-heading

      Request

   .. container:: field-definition

      - **params** (*object*, optional) --- Configuration parameters:

      - **items** (*array of objects*, min: 1, max: 8, required) --- RMSD input items:

        - **pdb_a** (*string*, min length: 1, required) — First PDB structure as string

        - **pdb_b** (*string*, min length: 1, required) — Second PDB structure as string

        - **chain_ids** (*object*, required) — Mapping of PDB identifier to chain ID list:

          - **a** (*array of strings*, required) — Chain IDs for pdb_a

          - **b** (*array of strings*, required) — Chain IDs for pdb_b

   **Example request**:

   .. sourcecode:: http

      POST /api/v3/biotite/predict/ HTTP/1.1
      Host: biolm.ai
      Authorization: Token YOUR_API_KEY
      Content-Type: application/json

            {
        "items": [
          {
            "pdb_a": "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C\nATOM      3  C   ALA A   1      12.000  11.458  10.000  1.00 20.00           C\nATOM      4  O   ALA A   1      11.458  12.000  11.000  1.00 20.00           O\nATOM      5  N   GLY A   2      13.458  11.458  10.000  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.000  12.916  10.000  1.00 20.00           C\nATOM      7  C   GLY A   2      15.458  12.916  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.000  14.374  10.000  1.00 20.00           O\nATOM      9  N   SER B   1      20.000  20.000  20.000  1.00 20.00           N\nATOM     10  CA  SER B   1      21.458  20.000  20.000  1.00 20.00           C\nATOM     11  C   SER B   1      22.000  21.458  20.000  1.00 20.00           C\nATOM     12  O   SER B   1      21.458  22.000  21.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
            "pdb_b": "ATOM      1  N   ALA A   1      10.200  10.100   9.900  1.00 20.00           N\nATOM      2  CA  ALA A   1      11.658  10.050  10.050  1.00 20.00           C\nATOM      3  C   ALA A   1      12.150  11.550   9.950  1.00 20.00           C\nATOM      4  O   ALA A   1      11.600  12.050  10.900  1.00 20.00           O\nATOM      5  N   GLY A   2      13.650  11.500  10.050  1.00 20.00           N\nATOM      6  CA  GLY A   2      14.150  12.950  10.050  1.00 20.00           C\nATOM      7  C   GLY A   2      15.600  12.900  10.000  1.00 20.00           C\nATOM      8  O   GLY A   2      16.050  14.350   9.950  1.00 20.00           O\nATOM      9  N   SER B   1      19.900  19.900  20.100  1.00 20.00           N\nATOM     10  CA  SER B   1      21.350  19.950  19.950  1.00 20.00           C\nATOM     11  C   SER B   1      21.950  21.400  19.950  1.00 20.00           C\nATOM     12  O   SER B   1      21.350  21.950  20.950  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
            "chain_ids": {
              "a": [
                "A"
              ],
              "b": [
                "A"
              ]
            }
          },
          {
            "pdb_a": "ATOM      1  N   ALA A   1       5.000   5.000   5.000  1.00 20.00           N\nATOM      2  CA  ALA A   1       6.458   5.000   5.000  1.00 20.00           C\nATOM      3  C   ALA A   1       7.000   6.458   5.000  1.00 20.00           C\nATOM      4  O   ALA A   1       6.458   7.000   6.000  1.00 20.00           O\nATOM      5  N   GLY A   2       8.458   6.458   5.000  1.00 20.00           N\nATOM      6  CA  GLY A   2       9.000   7.916   5.000  1.00 20.00           C\nATOM      7  C   GLY A   2      10.458   7.916   5.000  1.00 20.00           C\nATOM      8  O   GLY A   2      11.000   9.374   5.000  1.00 20.00           O\nATOM      9  N   SER B   1      15.000  15.000  15.000  1.00 20.00           N\nATOM     10  CA  SER B   1      16.458  15.000  15.000  1.00 20.00           C\nATOM     11  C   SER B   1      17.000  16.458  15.000  1.00 20.00           C\nATOM     12  O   SER B   1      16.458  17.000  16.000  1.00 20.00           O\nTER      13      SER B   1\nEND\n",
            "pdb_b": "ATOM      1  N   ALA C   1       5.100   5.050   5.050  1.00 20.00           N\nATOM      2  CA  ALA C   1       6.550   5.050   5.050  1.00 20.00           C\nATOM      3  C   ALA C   1       7.050   6.500   5.050  1.00 20.00           C\nATOM      4  O   ALA C   1       6.500   7.050   6.050  1.00 20.00           O\nATOM      5  N   GLY C   2       8.550   6.450   5.050  1.00 20.00           N\nATOM      6  CA  GLY C   2       9.050   7.900   5.050  1.00 20.00           C\nATOM      7  C   GLY C   2      10.500   7.900   5.000  1.00 20.00           C\nATOM      8  O   GLY C   2      11.050   9.350   5.000  1.00 20.00           O\nATOM      9  N   SER D   1      14.900  14.950  14.950  1.00 20.00           N\nATOM     10  CA  SER D   1      16.350  14.950  14.950  1.00 20.00           C\nATOM     11  C   SER D   1      16.950  16.400  14.950  1.00 20.00           C\nATOM     12  O   SER D   1      16.350  16.950  15.950  1.00 20.00           O\nTER      13      SER D   1\nEND\n",
            "chain_ids": {
              "a": [
                "A",
                "B"
              ],
              "b": [
                "C",
                "D"
              ]
            }
          }
        ],
        "params": {}
      }

   :statuscode 200: Successful response
   :statuscode 400: Invalid input
   :statuscode 500: Internal server error

   .. container:: field-heading

      Response

   .. container:: field-definition

      - **results** (*array of objects*) --- One result per input item, in the order requested:

        - **rmsd** (*float*, units: Å) — Root mean square deviation between superimposed atom coordinates

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: application/json

            {
        "results": [
          {
            "rmsd": 0.022410493344068527
          },
          {
            "rmsd": 0.09871604293584824
          }
        ]
      }


Performance
-----------

- Computational method and scaling
  - Performs rigid-body superposition using a Kabsch-based rotation/translation followed by RMSD in Ångström
  - Uses Biotite’s NumPy-style AtomArray/AtomArrayStack backend and Cython-accelerated kernels, giving near-C performance with time and memory scaling linearly in the number of selected atoms
  - For typical antibody/enzyme-sized selections (∼1,000–3,000 atoms), RMSD evaluation is dominated by PDB parsing; the superposition and RMSD computation itself is effectively instantaneous on CPU

- Relative speed vs. other BioLM structure services
  - Orders of magnitude faster than 3D structure prediction models (AlphaFold2, ESMFold, Chai-1) because it only operates on provided coordinates instead of running a deep generative or diffusion model
  - Typically contributes <1–2% of total wall-clock time when used as a post-processing step in 3D generative or design pipelines (e.g. ESM-IF1, ProstT5 AA2Fold, LigandMPNN/ProteinMPNN loops) for ranking or trajectory-style analyses

- Numerical performance and accuracy
  - Uses double-precision floating point for both Kabsch fit and RMSD, providing numerically stable results even for large complexes
  - Benchmarks from the Biotite publication show superposition+RMSD on ∼1,000-atom proteins is roughly an order of magnitude faster than Biopython’s object-graph-based implementations and comparable to other array-based libraries (MDTraj, MDAnalysis)
  - RMSD values are consistent to machine precision with other Kabsch-based implementations used in BioLM’s stack for backbone/coordinate comparisons

- Deployment characteristics
  - Runs on CPU-focused infrastructure tuned for high-throughput numeric workloads; GPU acceleration is unnecessary because computation is inexpensive relative to I/O for typical biomolecular sizes
  - Internal caching and streaming PDB parsing reduce overhead when processing multiple RMSD items per batch, and shared Biotite-based preprocessing across related services (e.g. chain extraction) further amortizes parsing and coordinate conversion costs

Applications
------------

- Structural similarity screening in protein engineering campaigns, by computing backbone or Cα RMSD between designed variants and a reference PDB structure to quickly filter out grossly misfolded or rearranged designs; this helps narrow large sets of AI-generated protein sequences or structures to those that maintain a desired fold before more expensive physics-based modeling or wet-lab screening
- Conformational stability assessment in enzyme optimization, by measuring RMSD across MD-derived PDB snapshots or between room-temperature and cryo structures for the same variant, helping industrial enzyme developers flag mutations that induce large backbone rearrangements, stabilize productive active-site conformations, or alter domain orientations; RMSD is not a full stability metric, but serves as a fast geometric proxy alongside energy or dynamics analyses
- Antibody structure benchmarking for developability workflows, by comparing predicted or experimentally solved 3D structures of humanized or affinity-matured antibodies against a reference antibody PDB and quantifying RMSD on selected chains (e.g., heavy vs light); this enables therapeutic teams to check that engineering steps preserve overall framework geometry while allowing targeted loop rearrangements, while recognizing that RMSD alone does not report on antigen-binding affinity or liability profiles
- Clone and lot comparability in biologics manufacturing, by computing chain-resolved RMSD between a reference therapeutic antibody or protein PDB and structures obtained from alternative expression systems, formulations, or manufacturing sites, supporting structural comparability checks in CMC workflows; RMSD offers a concise backbone-geometry metric, but should be interpreted together with biophysical characterization and functional assays for regulatory decisions
- Model selection and clustering for generative protein design, by using PDB RMSD to cluster large sets of generated structures into distinct conformational families and selecting diverse representatives for downstream in silico or in vitro testing; this reduces redundant wet-lab work on near-duplicate 3D designs and focuses resources on structurally distinct candidates, with the caveat that low RMSD does not guarantee similar activity and should be combined with task-specific scoring (e.g., binding or catalytic performance predictors)

Limitations
-----------

- **Batch Size**: Each ``BiotiteRMSDRequest`` or ``BiotiteExtractChainsRequest`` can contain at most ``BiotiteParams.batch_size`` items (currently ``8``). Larger workloads (e.g. RMSD matrices, high-throughput chain extraction) must be split client-side into multiple requests.
- **Maximum Sequence Length**: Extracted chain sequences are expected to be within ``BiotiteParams.max_sequence_len`` (``2048``) residues. Very large chains or mega-complexes can increase runtime and memory and may fail validation; they are generally not suitable for simple, high-throughput RMSD screening with this endpoint.
- **Input Structures and Chains**: ``pdb_a``, ``pdb_b`` and ``pdb_string`` must be valid PDB-formatted strings accepted by ``validate_pdb``; missing coordinates, severe alternate locations, or inconsistent residue/atom naming can cause parsing failures or unreliable RMSD values. The ``chain_ids`` fields must reference existing chain IDs only; the API does not infer chains, renumber residues, or perform sequence-based matching.
- **RMSD Definition and Alignment**: The returned ``rmsd`` is a single global, coordinate-based RMSD in Å computed after rigid-body superimposition (Kabsch-style) on the selected atoms. The service assumes atom-order compatibility between the specified chains and does not perform sequence alignment, atom remapping, or per-domain/local superposition, so it is not appropriate for cross-topology comparisons, domain-only RMSD without manual selection, or highly flexible multi-domain systems where local alignment is required.
- **Scope of Structural Analysis**: Biotite’s structure algorithms are tuned for typical protein structures. Nucleic acids, very large assemblies, membrane systems, and exotic residues receive only generic handling. RMSD is computed on the input coordinates “as is” (no protonation, refinement, or minimization) and should not be treated as a surrogate for detailed biophysical similarity (e.g. binding energetics, stability) without additional modeling.
- **Pipeline Role**: This API extracts chains and computes RMSD; it does not generate structures, predict 3D folds, or produce embeddings/functional scores. For large protein engineering or library design campaigns, it is best used as a downstream structural check or validation step rather than as the primary design, ranking, or screening method.

How We Use It
-------------

Biotite-based PDB RMSD gives a consistent, alignment-aware measure of backbone similarity that fits directly into protein design and optimization loops, enabling large panels of predicted or experimental structures to be ranked against a reference scaffold or state. Exposed as a scalable API metric, it slots into multi-model workflows that combine generative sequence design, structure prediction, and property scoring to enforce geometry constraints across design rounds, monitor structural drift, and prune candidates before synthesis, which reduces wet-lab screening and shortens design–build–test cycles.

- Integrates with structure prediction and generative design services to rank or cluster variants by chain-resolved structural similarity to target folds, interfaces, or multichain assemblies.
- Supports API-driven workflows where RMSD is one of several standardized scores (e.g., RMSD alongside stability or developability metrics) used to prioritize enzyme, antibody, and other protein engineering candidates for experimental validation.

Related
-------

- ``AlphaFold2`` – Predict high-confidence 3D protein structures from sequence, which can then be compared with experimental structures using Biotite PDB RMSD to quantify structural differences.
- ``ESMFold`` – Generate protein structures directly from language models, providing alternative conformations that can be systematically compared or clustered using Biotite PDB RMSD.
- ``ESM-IF1`` – Design or evaluate sequences on a fixed backbone; Biotite PDB RMSD can then assess how sequence changes affect backbone deviations across designs.
- ``TEMPRO 3B`` – Model protein conformational ensembles and dynamics, where Biotite PDB RMSD is useful for measuring structural drift, frame-to-frame deviations, or clustering trajectory snapshots.

References
----------

- Kunzmann, P., & Hamacher, K. (2018). `Biotite: a unifying open source computational biology framework in Python <https://doi.org/10.1186/s12859-018-2367-z>`_. *BMC Bioinformatics*, 19, 346.
