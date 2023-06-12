<figure markdown>
  ![Chain-of-Thought prompting](/assets/standardvscotprompt.png){ width="600" }
  <figcaption>Chain of Thought Prompting (Wei et al)</figcaption>
</figure>

LLMs find it hard to perform well on complex reasoning tasks. We can unlock the reasoning abilities of LLMs using chain of thought prompting. This involves giving the LLM a few reasoning explanations along with the questions and answers and then asking the model to produce the reasoning before producing the answer. The hope is that the seed examples with explanations help the model understand the reasoning behind answer and prods it to use similar reasoning before arriving to the answer.

Chain of thought makes LLMs more effective at reasoning tasks like mathematical word problems, commonsense reasoning questions and complex medical questions. It also provides a window into the thought process of the LLM, though some research points the link between the generated explanation and the final answer may be weak.

## Using Chain Of Thought in Autolabel

Enabling chain-of-thought prompting for your task is straightforward with Autolabel. It works best when provided with a few seed examples with explanations. Thus enabling chain of thought requires a few things -  

1. Writing up explanations or generating explanations for your seed examples automatically by using an LLM
2. Specifying the `explanation_column` in the dataset part of the config.
3. Altering the task guidelines to tell the model to generate an explanation before generating the final answer.
4. Altering the `example_template` to reflect the presence of an explanation.

We will go through using chain of thought on a dataset where it shows improvement, like the Squad question answering dataset.

Let's see a datapoint before there is any explanation added to it.

{{ read_csv('docs/assets/squad_preview.csv') }}


Now we can manually write the explanation for this or a couple of seed examples easily. But this will be tiresome for > 10 examples. LLMs come to the rescue yet again! We can just define the config and ask the agent to generate explanations as well!

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "multi_choice_question_answering",
    "dataset": {
        "label_column": "answer",
        "explanation_column": "explanation",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions based on wikipedia articles. Your job is to answer the following questions using the context provided with the question. Use the context to answer the question - the answer is a continuous span of words from the context.\n",
        "output_guidelines": "Your answer will consist of an explanation, followed by the correct answer. The last line of the response should always be is JSON format with one key: {\"label\": \"the correct answer\"}.\n If the question cannot be answered using the context and the context alone without any outside knowledge, the question is unanswerable. If the question is unanswerable, return the answer as {\"label\": \"unanswerable\"}\n",
        "few_shot_examples": "../data/squad_v2_seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 3,
        "example_template": "Context: {context}\nQuestion: {question}\nAnswer: Let's think step by step.\n{explanation}\n{answer}"
    }
}
```

Notice the changes that we have made to the config compared to the config without Chain-of-Thought [here](/guide/tasks/question_answering_task). We have added two new fields to the config

* `explanation_column`
* `example_template`

`explanation_column` is the column where the explanation for the seed examples will reside. Next, notice that the `example_template` key contains the explanation column as well. This tells the config where the explanation should be put when using the seed examples. We use the `Let's think step by step` prompt to initiate the chain of thought in the model.

Now, in order to generate explanations for the seed examples, in case they were not manually generated is,
```py
from autolabel import LabelingAgent
agent = LabelingAgent(config)
agent.generate_explanations("path_to_seed_examples.csv")
```

Once these explanations are generated, the dataset looks like

{{ read_csv('docs/assets/squad_with_explanation_preview.csv') }}

Now to generate labels for this dataset, all we have to do is,
```py
agent.plan('data/squad_v2_test.csv')
agent.run('data/squad_v2_test.csv', max_items = 100)
```