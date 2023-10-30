# biolm-sdk
Python SDK for BioLM API

## Development

```shell
pyenv install 3.11.5
pyenv virtualenv 3.11.5 py-biolm
pyenv activate py-biolm
```

```shell
pip install -r requirements_dev.txt
make install
```

Set `export BIOLMAI_TOKEN=<your_token>` using [a BioLM API token](https://biolm.ai/ui/accounts/user-api-tokens/).

```shell
RS=118 make test
```
