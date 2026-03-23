"""Unit tests for the MiniMax LLM provider."""
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Generation
from pytest import approx

from autolabel.configs import AutolabelConfig
from autolabel.models.minimax import MiniMaxLLM
from autolabel.schema import ErrorType, RefuelLLMResult


# ---- Config paths ----

MINIMAX_CONFIG_PATH = "tests/assets/banking/config_banking_minimax.json"
MINIMAX_M25HS_CONFIG = "tests/assets/banking/config_banking_minimax_m25hs.json"
MINIMAX_CUSTOM_CONFIG = "tests/assets/banking/config_banking_minimax_custom.json"
MINIMAX_HIGHTEMP_CONFIG = "tests/assets/banking/config_banking_minimax_hightemp.json"
MINIMAX_ZEROTEMP_CONFIG = "tests/assets/banking/config_banking_minimax_zerotemp.json"
MINIMAX_FUTURE_CONFIG = "tests/assets/banking/config_banking_minimax_future.json"


@pytest.fixture(autouse=True)
def set_minimax_api_key():
    """Set MINIMAX_API_KEY for all tests."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
        yield


@pytest.fixture
def minimax_model():
    """Create a MiniMaxLLM instance for testing."""
    return MiniMaxLLM(
        config=AutolabelConfig(config=MINIMAX_CONFIG_PATH),
    )


# ---- Initialization Tests ----

def test_minimax_initialization(minimax_model):
    """Test that MiniMaxLLM initializes correctly with default config."""
    assert minimax_model.model_name == "MiniMax-M2.7"
    assert minimax_model.url == "https://api.minimax.io/v1/chat/completions"
    assert minimax_model.model_params["temperature"] == 0.1
    assert minimax_model.model_params["max_tokens"] == 1000


def test_minimax_initialization_missing_api_key():
    """Test that MiniMaxLLM raises ValueError when MINIMAX_API_KEY is not set."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("MINIMAX_API_KEY", None)
        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            MiniMaxLLM(
                config=AutolabelConfig(config=MINIMAX_CONFIG_PATH),
            )


def test_minimax_custom_model_name():
    """Test initialization with a custom model name."""
    model = MiniMaxLLM(config=AutolabelConfig(config=MINIMAX_M25HS_CONFIG))
    assert model.model_name == "MiniMax-M2.5-highspeed"


def test_minimax_custom_params():
    """Test initialization with custom model parameters."""
    model = MiniMaxLLM(config=AutolabelConfig(config=MINIMAX_CUSTOM_CONFIG))
    assert model.model_params["max_tokens"] == 2000
    assert model.model_params["temperature"] == 0.5
    assert model.timeout == 90
    assert "request_timeout" not in model.model_params


def test_minimax_temperature_clamping():
    """Test that temperature is clamped to [0, 1] range."""
    model = MiniMaxLLM(config=AutolabelConfig(config=MINIMAX_HIGHTEMP_CONFIG))
    assert model.model_params["temperature"] == 1.0


def test_minimax_temperature_zero():
    """Test that temperature=0 is accepted."""
    model = MiniMaxLLM(config=AutolabelConfig(config=MINIMAX_ZEROTEMP_CONFIG))
    assert model.model_params["temperature"] == 0.0


# ---- Think Tag Stripping Tests ----

def test_strip_think_tags(minimax_model):
    """Test that <think>...</think> tags are stripped from model output."""
    text = "<think>Let me analyze this...</think>positive"
    assert minimax_model._strip_think_tags(text) == "positive"


def test_strip_think_tags_multiline(minimax_model):
    """Test stripping multi-line think tags."""
    text = "<think>\nStep 1: Analyze\nStep 2: Classify\n</think>\nnegative"
    assert minimax_model._strip_think_tags(text) == "negative"


def test_strip_think_tags_no_tags(minimax_model):
    """Test that text without think tags is returned unchanged."""
    text = "positive"
    assert minimax_model._strip_think_tags(text) == "positive"


# ---- Label Tests ----

def test_minimax_label_sync(minimax_model):
    """Test synchronous labeling."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "positive"}}],
    }

    with patch("requests.post", return_value=mock_response):
        result = minimax_model._label(["test prompt"], {})

    assert len(result.generations) == 1
    assert result.generations[0][0].text == "positive"
    assert result.errors[0] is None
    assert len(result.latencies) == 1


def test_minimax_label_with_think_tags(minimax_model):
    """Test that think tags are stripped from sync label results."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "<think>reasoning here</think>negative"}}
        ],
    }

    with patch("requests.post", return_value=mock_response):
        result = minimax_model._label(["test prompt"], {})

    assert result.generations[0][0].text == "negative"


def test_minimax_label_multiple_prompts(minimax_model):
    """Test labeling multiple prompts."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "label_result"}}],
    }

    with patch("requests.post", return_value=mock_response):
        result = minimax_model._label(["prompt1", "prompt2", "prompt3"], {})

    assert len(result.generations) == 3
    assert all(g[0].text == "label_result" for g in result.generations)
    assert all(e is None for e in result.errors)


def test_minimax_label_error_handling(minimax_model):
    """Test error handling during labeling."""
    with patch("requests.post", side_effect=Exception("API error")):
        result = minimax_model._label(["test prompt"], {})

    assert len(result.generations) == 1
    assert result.generations[0][0].text == ""
    assert result.errors[0] is not None
    assert result.errors[0].error_type == ErrorType.LLM_PROVIDER_ERROR


@pytest.mark.asyncio
async def test_minimax_alabel(minimax_model):
    """Test async labeling."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "positive"}}],
    }

    with patch.object(minimax_model, "_alabel_with_retry", return_value=(mock_response, 0.5)):
        result = await minimax_model._alabel(["test prompt"], {})

    assert len(result.generations) == 1
    assert result.generations[0][0].text == "positive"


@pytest.mark.asyncio
async def test_minimax_alabel_with_think_tags(minimax_model):
    """Test that think tags are stripped from async label results."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "<think>step by step</think>neutral"}}
        ],
    }

    with patch.object(minimax_model, "_alabel_with_retry", return_value=(mock_response, 0.5)):
        result = await minimax_model._alabel(["test prompt"], {})

    assert result.generations[0][0].text == "neutral"


@pytest.mark.asyncio
async def test_minimax_alabel_error_handling(minimax_model):
    """Test async error handling."""
    with patch.object(
        minimax_model, "_alabel_with_retry", side_effect=Exception("Async API error")
    ):
        result = await minimax_model._alabel(["test prompt"], {})

    assert len(result.generations) == 1
    assert result.generations[0][0].text == ""
    assert result.errors[0] is not None
    assert result.errors[0].error_type == ErrorType.LLM_PROVIDER_ERROR


# ---- Cost Tests ----

def test_minimax_get_cost_m27(minimax_model):
    """Test cost calculation for MiniMax-M2.7."""
    prompt = "TestingExamplePrompt"  # 20 chars -> 5 tokens
    label = "positive"  # 8 chars -> 2 tokens
    cost = minimax_model.get_cost(prompt, label)
    expected = (1.0 / 1_000_000) * 5 + (4.0 / 1_000_000) * 2
    assert cost == approx(expected, rel=1e-3)


def test_minimax_get_cost_no_label(minimax_model):
    """Test cost calculation with no label (prompt only)."""
    prompt = "TestingExamplePrompt"  # 20 chars -> 5 tokens
    cost = minimax_model.get_cost(prompt)
    expected = (1.0 / 1_000_000) * 5
    assert cost == approx(expected, rel=1e-3)


def test_minimax_get_cost_m25_highspeed():
    """Test cost calculation for MiniMax-M2.5-highspeed."""
    model = MiniMaxLLM(config=AutolabelConfig(config=MINIMAX_M25HS_CONFIG))
    prompt = "TestingExamplePrompt"  # 20 chars -> 5 tokens
    label = "positive"  # 8 chars -> 2 tokens
    cost = model.get_cost(prompt, label)
    expected = (0.5 / 1_000_000) * 5 + (2.0 / 1_000_000) * 2
    assert cost == approx(expected, rel=1e-3)


def test_minimax_get_cost_unknown_model():
    """Test cost calculation falls back to M2.7 pricing for unknown models."""
    model = MiniMaxLLM(config=AutolabelConfig(config=MINIMAX_FUTURE_CONFIG))
    prompt = "TestingExamplePrompt"
    cost = model.get_cost(prompt, "positive")
    # Should use M2.7 fallback pricing
    expected = (1.0 / 1_000_000) * 5 + (4.0 / 1_000_000) * 2
    assert cost == approx(expected, rel=1e-3)


# ---- Capability Tests ----

def test_minimax_returns_token_probs(minimax_model):
    """Test that MiniMax does not return token probabilities."""
    assert minimax_model.returns_token_probs() is False


def test_minimax_get_num_tokens(minimax_model):
    """Test token count estimation."""
    prompt = "This is a test prompt with twenty characters"
    expected = len(prompt) // 4
    assert minimax_model.get_num_tokens(prompt) == expected


# ---- Request Format Tests ----

def test_minimax_request_format(minimax_model):
    """Test that the request is formatted correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "result"}}],
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        minimax_model._label(["test prompt"], {})

    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.minimax.io/v1/chat/completions"
    request_data = call_args[1]["json"]
    assert request_data["model"] == "MiniMax-M2.7"
    assert request_data["messages"] == [{"role": "user", "content": "test prompt"}]
    assert "temperature" in request_data
    assert "max_tokens" in request_data
    headers = call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test-key-123"


# ---- Registry Tests ----

def test_minimax_in_model_registry():
    """Test that MiniMax is registered in the MODEL_REGISTRY."""
    from autolabel.models import MODEL_REGISTRY
    from autolabel.schema import ModelProvider
    assert ModelProvider.MINIMAX in MODEL_REGISTRY


def test_minimax_model_provider_enum():
    """Test that MiniMax is in the ModelProvider enum."""
    from autolabel.schema import ModelProvider
    assert ModelProvider.MINIMAX.value == "minimax"
    assert ModelProvider("minimax") == ModelProvider.MINIMAX
