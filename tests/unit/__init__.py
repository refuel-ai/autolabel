"""Set Dummy Environments"""

import os

os.environ.update(
    {
        "REFUEL_API_KEY": "dummy_refuel_api_key",
        "OPENAI_API_KEY": "dummy_open_api_key",
        "ANTHROPIC_API_KEY": "dummy_anthropic_api_key",
    }
)
