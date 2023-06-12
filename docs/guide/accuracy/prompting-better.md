Like most LLM tasks, a critical part of improving LLM performance in autolabeling tasks is selecting a good prompt. Often, this entails finding a good balance between a descriptive set of instructions, while still remaining concise and clear. 

Consider the following example of refining a prompt used for a classification task on the civil-comments dataset. Each labeling run below included 500 examples and used the same LLM: gpt-3.5-turbo and used a fixed-shot example selection strategy with 4 seed examples.

First attempt:
```json
from autolabel import LabelingAgent

config = {
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "compute_confidence": True
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying toxic comments and understanding if a comment is sexually explicit, obscene, toxic, insults a person, demographic or race. \nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
        "labels": [
            "toxic",
            "not toxic"
        ],
        "few_shot_examples": "../data/civil_comments_seed.csv",
        "few_shot_selection": "fixed",
        "few_shot_num": 4,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}

agent = LabelingAgent(config=config)
labels, df, metrics_list = agent.run('../data/civil_comments_test.csv', max_items = 500)
```

Accuracy: 68%

This first basic prompt is clear and concise, but lacks details that should specify how to distinguish comments that are toxic vs. civil and thus attains a baseline accuracy of just 68%.

Adding detail:

```json
"task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide fair criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy and politicians are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'.\nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
```

Accuracy: 76%

In this second iteration, more detail is added in the prompt such as addressing the nuances between "fair criticisms" vs. toxic comments and also mentions specific types of toxic comments such as "obscenities, racial slurs, or sexually explicit material". This increased level of detail leads to better performance, reaching 76% accuracy.

Final version:

```json
"task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy, politicians and companies are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'. \nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
```

Accuracy: 78%

After subsequently experimenting with a few different variations to the prompt. the final, best-performing version is almost identical to the second iteration, with a slight change of the term "fair criticism" to "criticism" and this very slightly improves accuracy to 78%. Generally, when refining the prompt leads to very marginal differences in performance, it is a likely sign that we are reaching the limits of the LLM's capability for the task. 

As a result, after sufficient iteration of the prompt, it is worth comparing different LLM's as there can often be significant differences in performance. With the same final prompt above, the text-davinci-003 model achieved 88% accuracy, a 10% increase compared to gpt-turbo-3.5.
