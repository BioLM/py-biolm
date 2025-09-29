# Python SDK for BioLM API

## Development

First, set up the Python environment using `pyenv`:
```shell
pyenv install 3.11.5
pyenv virtualenv 3.11.5 py-biolm
pyenv activate py-biolm
```

Then, install the development dependencies:
```shell
pip install -r requirements_dev.txt
```

Finally, run the Makefile to complete the setup:
```shell
make install
```

This installs the biolm package itself in editable mode. This allows you to make changes to the package and see those changes reflected without needing to reinstall the package:
It runs the command: `pip install -e .`. The -e flag stands for "editable mode". This is a useful setup for development as it means any changes you make to the package code will be immediately available in your environment, without needing to reinstall the package each time you make a change.


## Set the `BIOLM_TOKEN` environment variable
Set the token in your shell to authenticate your requests to the BioLM API:
```shell
export BIOLM_TOKEN=<your_token>
```
The token can be obtained from your BioLM account: [Get your BioLM API token](https://biolm.ai/ui/accounts/user-api-tokens/).
Run `biolm status` to verify that the token is set correctly.

## Run tests with a specific seed
Use this command to run the automated tests:
```shell
RS=118 make test
```
The `RS` environment variable sets the seed for any random operations in the tests, ensuring consistent results.
