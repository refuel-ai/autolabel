One of the biggest criticisms of using a LLMs so far has been hallucinations - LLMs can seem very confidence in their language even when they are completely incorrect. To work towards this, the `autolabel` library uses confidence scores to evaluate LLM outputs and obtain a score that is correlated with the output's likelihood of being incorrect. 

The `autolabel` library today heavily relies on token level probabilities to compute confidence scores. However, very few models today return token level probabilities alongside prediction. Hence, Refuel has setup an in-house API to generate token level probabilities for a specific prediction given an input, regardless of the language model that was originally used to query for the prediction. 

Generating confidence scores is simple - setting the key `compute_confidence` to `True` in the `model` dictionary of the config should initiate confidence score retrieval. Here is an example:

```python
{
    "task_name": "PersonLocationOrgMiscNER",
    "task_type": "named_entity_recognition",
    "dataset": {
        "label_column": "CategorizedLabels",
        "text_column": "example",
        "delimiter": "%"
    },
    "model": {
        "provider": "anthropic",
        "name": "claude-v1",
        "compute_confidence": True
    },
    "prompt": {
        "task_guidelines": "You are an expert at extracting entities from text.",
        "labels": [
            "Location",
            "Organization",
            "Person",
            "Miscellaneous"
        ],
        "example_template": "Example: {example}\nOutput: {CategorizedLabels}",
        "few_shot_examples": "data/conll2003_seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5
    }
}
```

In the above example, by setting `compute_confidence` to True, `autolabel` will start calling Refuel's api to generate token level probabilities and compute confidence scores for each prediction. In order for this to run successfully, ensure that the following setup has been completed:

Install the relevant packages by running:
```bash
pip install refuel-autolabel[refuel]
```
and also set the following environment variable:
```
export REFUEL_API_KEY=<your-refuel-key>
```
replacing `<your-refuel-key>` with your API key, which you can get from [here](TBD). [TODO] INSTRUCTIONS FOR GETTING THE API KEY
