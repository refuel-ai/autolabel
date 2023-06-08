# LLM Output Caching

To help reduce time and cost when iterating the prompt for better labeling accuracy, we cache the calls made to the LLM.

## Cache Entry

A cache entry has the following attributes:

- `Model Name`
- `Prompt`
- `Model Params`

This means that anytime there are changes to either the language model or the prompt, the model will be called for producing label. Also, changes to the model parameters like the `max_tokens` or `temperature` could affect the label output and therefore modifying such parameters result in new calls to the model instead of using cached calls.

## Caching Storage

The cached entries are stored in a SQLite database. We will be adding support for In Memory cache and Redis cache in future.

## Disable Caching

The cache is enabled by default and if you wish to disable it, you can set `cache=False` when initializing the LabelingAgent.

```python

from autolabel import LabelingAgent

agent = LabelingAgent(config='examples/configs/civil_comments.json', cache=False)
```
