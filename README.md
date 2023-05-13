# Autolabel: Using LLMs to label data

First, specify a config file with task instructions. Let's say we use the example file located at `examples/config_chatgpt.json`.

Now, let's read this config file and see how much would it cost:
```python

from autolabel import LabelingAgent

agent = LabelingAgent(
    'examples/configs/task_configs/civil_comments_classification.json',
    'examples/configs/llm_configs/anthropic.json'
)

agent.plan(
    'data/civil_comments_test.csv',
    'examples/configs/dataset_configs/civil_comments.json'
)
```

This prints:

```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 175.31it/s]
Total Estimated Cost: $11.866
Number of examples to label: 2000
Average cost per example: $0.00593


A prompt example:

You are an expert at identifying toxic comments and understanding if a comment is sexually explicit, obscene, toxic, insults a person, demographic or race.
Your job is to correctly label the provided input example into one of the following 2 categories.
Categories:
toxic
not toxic


You will return the answer in JSON format with one key: {"label": "the correct label"}

Some examples with their output answers are provided below:
Example: It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.
Output:
{"label": "toxic"}

Example: This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!
Output:
{"label": "not toxic"}

Now I want you to label the following example: Example: [ Integrity means that you pay your debts.]

Does this apply to President Trump too?
Output:
```

Now, let's run annotation on a subset of the dataset:
```python
labels, output_df, metrics = agent.run(
    'data/civil_comments_test.csv',
    'examples/configs/dataset_configs/civil_comments.json',
    max_items=10
)
# This will also output a file with the labels per row in `data/civil_comments_test_labeled.csv`
```

This prints the following:

```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.87s/it]
Metric: support: [(10, 'index=0')]
Metric: threshold: [(-inf, 'index=0')]
Metric: accuracy: [(0.5, 'index=0')]
Metric: completion_rate: [(1.0, 'index=0')]
Total number of failures: 0
````
