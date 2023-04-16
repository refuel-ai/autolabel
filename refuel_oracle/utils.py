import json

import regex
import tiktoken
from anthropic import tokenizer as anthropic_tokenizer
from langchain.document_loaders import WebBaseLoader
from refuel_oracle.config import Config
from refuel_oracle.llm import LLMFactory, LLMProvider
from transformers import AutoTokenizer

PROVIDER_TO_COST_PER_TOKEN = {
    LLMProvider.openai: {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
    },
    LLMProvider.openai_chat: {
        "gpt-3.5-turbo": 0.002 / 1000,
        "gpt-4": 0.03 / 1000,  # $0.03 per 1000 tokens for prompts
    },
    LLMProvider.anthropic: {
        # $2.90 per million characters for prompts
        # For Claude the average token is about 3.5 characters
        "claude-v1": (2.90 / 1000000)
        * 3.5,
    },
}
PROVIDER_TO_COST_OF_COMPLETION = {
    LLMProvider.openai: {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
    },
    LLMProvider.openai_chat: {
        "gpt-3.5-turbo": 0.002 / 1000,
        "gpt-4": 0.06 / 1000,  # $0.06 per 1000 tokens in response
    },
    LLMProvider.anthropic: {
        # $8.60 per million characters in response
        # For Claude the average token is about 3.5 characters
        "claude-v1": (8.60 / 1000000)
        * 3.5,
    },
}


def calculate_num_tokens(config: Config, string: str) -> int:
    """Returns the number of tokens in a text string

    Args:
        config (Config): Task config passed to the library
        string (str): string input for which to calculate num tokens

    Returns:
        int: num tokens
    """
    if config.get_provider() == "anthropic":
        return anthropic_tokenizer.count_tokens(string)
    elif config.get_provider() == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(config.get_model_name())
        return len(tokenizer.encode(string))
    encoding = tiktoken.encoding_for_model(config.get_model_name())
    num_tokens = len(encoding.encode(string))
    return num_tokens


def calculate_cost(config: Config, num_tokens: int) -> float:
    """Calculate total usage cost from num_tokens

    Args:
        config (Config): Task config passed to the library
        num_tokens (int): num tokens to compute cost

    Returns:
        float: total cost from current usage (num_tokens)
    """
    llm_provider = config.get_provider()
    llm_model = config.get_model_name()
    cost_per_prompt_token = PROVIDER_TO_COST_PER_TOKEN[llm_provider][llm_model]
    cost_per_completion_token = PROVIDER_TO_COST_OF_COMPLETION[llm_provider][llm_model]
    if llm_provider == "anthropic":
        max_tokens = LLMFactory.PROVIDER_TO_DEFAULT_PARAMS[llm_provider].get(
            "max_tokens_to_sample", 0
        )
    else:
        max_tokens = LLMFactory.PROVIDER_TO_DEFAULT_PARAMS[llm_provider].get(
            "max_tokens", 0
        )
    return (num_tokens * cost_per_prompt_token) + (
        cost_per_completion_token * max_tokens
    )


def extract_valid_json_substring(string):
    pattern = (
        r"{(?:[^{}]|(?R))*}"  # Regular expression pattern to match a valid JSON object
    )
    match = regex.search(pattern, string)
    if match:
        json_string = match.group(0)
        try:
            json.loads(json_string)
            return json_string
        except ValueError:
            pass
    return None


def get_web_content(input_text):
    if "web(" not in input_text:
        return input_text

    url = input_text[input_text.find("web(") + 4 : input_text.find(")")]

    loader = WebBaseLoader(url)
    data = loader.load()
    return data[0].page_content
