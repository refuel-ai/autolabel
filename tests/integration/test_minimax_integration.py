"""Integration tests for the MiniMax LLM provider.

These tests make real API calls to the MiniMax API and require:
- MINIMAX_API_KEY environment variable to be set

Run with: pytest tests/integration/test_minimax_integration.py -v
Skip these in CI by running: pytest -m "not integration"
"""
import os

import pytest

from autolabel.configs import AutolabelConfig
from autolabel.models.minimax import MiniMaxLLM

pytestmark = pytest.mark.skipif(
    os.getenv("MINIMAX_API_KEY") is None,
    reason="MINIMAX_API_KEY not set",
)

INTEGRATION_CONFIG = "tests/assets/banking/config_integration_minimax.json"
INTEGRATION_CONFIG_M25HS = "tests/assets/banking/config_integration_minimax_m25hs.json"


@pytest.mark.integration
def test_minimax_real_api_label():
    """Test real API call for synchronous labeling."""
    model = MiniMaxLLM(config=AutolabelConfig(config=INTEGRATION_CONFIG))
    result = model._label(
        ["Classify the sentiment: 'I love this product!' Answer with one word: positive, negative, or neutral."],
        {},
    )
    assert len(result.generations) == 1
    assert result.errors[0] is None
    assert len(result.generations[0][0].text) > 0
    assert result.latencies[0] > 0


@pytest.mark.integration
def test_minimax_real_api_m25_highspeed():
    """Test real API call with MiniMax-M2.5-highspeed model."""
    model = MiniMaxLLM(config=AutolabelConfig(config=INTEGRATION_CONFIG_M25HS))
    result = model._label(
        ["What is 2+2? Answer with just the number."],
        {},
    )
    assert len(result.generations) == 1
    assert result.errors[0] is None
    assert "4" in result.generations[0][0].text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_minimax_real_api_async_label():
    """Test real API call for async labeling."""
    model = MiniMaxLLM(config=AutolabelConfig(config=INTEGRATION_CONFIG))
    result = await model._alabel(
        ["Classify the sentiment: 'I love this product!' Answer with one word: positive, negative, or neutral."],
        {},
    )
    assert len(result.generations) == 1
    assert result.errors[0] is None
    assert result.latencies[0] > 0
