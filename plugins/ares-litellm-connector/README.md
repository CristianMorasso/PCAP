# LiteLLM Connector Plugin (ares-litellm-connector)

This package provides a plugin for ARES to allow the connection to models deployed via LiteLLM.


### Get started
1. Clone the repository and install in your environment
    ```bash
    cd plugins/ares-litellm-connector
    pip install .
    ```

2. Execute a direct request probe against LiteLLM connector target using ARES with an example configuration provided:
    ```bash
    ares evaluate examples/plugins/ares-litellm-connector/litellm-connector-example-ollama.yaml
    ```

### Contributors
The following notes are for contributing developers to this plugin.
1. Editable install of ares-litellm-connector:
    ```bash
    pip install -r plugins/ares-litellm-connector/requirements.txt
    ```
2. Running tests, linting, formatting and static type checking:
    ```bash
    cd ares-litellm-connector
    pytest --cov ares_litellm tests # tests
    black --line-length 120 src # formatting
    pylint src # linting
    mypy --install-types --non-interactive src # static type checks
    bandit -c bandit.yaml -r src # security vulnerabilities
    ```
    > `isort src` can be used to help manage import ordering.
