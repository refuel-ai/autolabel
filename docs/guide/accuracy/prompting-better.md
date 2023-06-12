Like most LLM tasks, a critical part of improving LLM performance in autolabeling tasks is selecting a good prompt. Often, this entails finding a good balance between a descriptive set of instructions, while still remaining concise and clear. 

Consider the following example of refining a prompt used for a classification task on the civil-comments dataset. Each labeling run below included 500 examples and used the same LLM: gpt-3.5-turbo and used a fixed-shot example selection strategy with 4 seed examples.

First attempt:
```json
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
```

```py
from autolabel import LabelingAgent

agent = LabelingAgent(config=config)
labels, df, metrics_list = agent.run('../data/civil_comments_test.csv', max_items = 500)
```

Accuracy: 68%

This first basic prompt seems clear and concise, but only attains a baseline accuracy of 68%. We can analyze some of the errors the LLM is making to get a better idea of how to improve our prompt. 

```py
df[df['label'] != df['ToxicCommentClassification_llm_label']]
```

In doing so, we notice that a vast majority of the errors (97.2%) are misclassifications of civil comments as toxic by the LLM. For instance, one such example comment is:

```json
'This is malfeasance by the Administrator and the Board. They are wasting our money!'
```

The presence of generally negative words such as "malfeasance" and "wasting" may be misleading the LLM. Our prompt may need to include details that guide the LLM to correctly identify cases where the vocabulary used could be mistaken as toxic, but the surrounding context suggests that the comment is actually civil.


Adding nuance to the prompt:

We can replace the prompt in the above config with the following updated task guidelines.

```json
"task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide fair criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy and politicians are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'.\nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
```

Accuracy: 74%

In this second iteration, we added more detail to the prompt such as addressing the nuances between "fair criticisms" vs. toxic comments. These additional details lead to better performance, reaching 74% accuracy. From a similar analysis of the LLM errors, we see that the previous misclassification example, along with several other similar ones, has now been correctly labeled.

Further improvements:

After subsequently experimenting with a few different variations to this prompt, we do not see significant improvements in performance for this task. As a result, after sufficient iteration of the prompt, it is better to look for performance gains through other modifications to the task configuration. For example, comparing different LLM's can often lead to significant improvements. With the same final prompt above, the text-davinci-003 model achieved 88% accuracy, a 14% increase compared to gpt-turbo-3.5.
