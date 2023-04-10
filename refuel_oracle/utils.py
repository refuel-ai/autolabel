import tiktoken

from refuel_oracle.config import Config
from refuel_oracle.llm import LLMProvider
from anthropic import tokenizer

PROVIDER_TO_COST_PER_TOKEN = {
    LLMProvider.openai: {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
    },
    LLMProvider.openai_chat: {
        "gpt-3.5-turbo": 0.002 / 1000,
        # TODO: gpt-4 needs a better pricing computation
        # since prompt vs completion tokens are priced differently
        "gpt-4": 0.03 / 1000,
    },
    LLMProvider.anthropic: {
        # TODO prompt vs completion are priced differently
        # price is per character, not tokens
        "claude-v1": 8.60
        / 1000000,  # $8.60 per million characters for completion
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
        return tokenizer.count_tokens(string)
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
    cost_per_token = PROVIDER_TO_COST_PER_TOKEN[llm_provider][llm_model]
    return num_tokens * cost_per_token
