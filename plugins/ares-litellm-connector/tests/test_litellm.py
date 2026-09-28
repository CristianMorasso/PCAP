"""
Tests for LiteLLM Connector
"""

import pytest  # type: ignore
from ares_litellm import LiteLLMConnector

from ares.utils import Status


def test_connector_validates() -> None:
    connector = LiteLLMConnector(LiteLLMConnector.template())

    assert connector


@pytest.mark.skip("Requires ollama")
def test_generate_ollama() -> None:
    connector = LiteLLMConnector(
        {
            "type": "ares_litellm.LiteLLMConnector",
            "name": "foobar",
            "model": "ollama/granite3.3",
            "endpoint": "http://localhost:11434",
            "endpoint-type": "ollama",
        }
    )

    assert connector
    response = connector.generate([{"role": "user", "content": "Hi how are you?"}])

    assert response
    assert response.status == Status.SUCCESS

    response = connector.generate("Hi how are you doing?")

    assert response
    assert response.status == Status.SUCCESS
