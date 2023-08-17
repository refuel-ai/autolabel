<figure markdown>
  ![Chain-of-Thought prompting](/assets/standardvscotprompt.png){ width="600" }
  <figcaption>Chain of Thought Prompting (Wei et al)</figcaption>
</figure>

LLMs find it hard to perform well on complex reasoning tasks. We can unlock the reasoning abilities of LLMs using chain of thought prompting. This involves asking the LLM to produce the reasoning before producing the answer (roughly analogous to "show me your work").

Chain of thought makes LLMs more effective at reasoning tasks like mathematical word problems, commonsense reasoning questions and complex medical questions. It also provides a window into the thought process of the LLM, though some research points the link between the generated explanation and the final answer may be weak.

## Using Chain Of Thought in Autolabel [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GYs0_4k8vhGk1LOJISppNN98DRq_Bur1#scrollTo=6xqMfKxa92Sj)

Enabling chain-of-thought prompting for your task is straightforward with Autolabel. It works best when provided with a few seed examples with explanations. Thus enabling chain of thought requires a few things:

1. Setting `chain_of_thought` flag in the labeling config.
2. Providing explanations or generating explanations for your seed examples automatically by using an LLM
3. Setting the `explanation_column` in the labeling config.
4. Altering the task guidelines and `example_template` to tell the model to generate an explanation before generating the final answer.

We will go through using chain of thought on a dataset where it shows improvement, like the SQuAD question answering dataset.

Let's see a datapoint before there is any explanation added to it.

{{ read_csv('docs/assets/squad_preview.csv') }}

Now we can manually write the explanation for this or a couple of seed examples easily. But this will be tiresome for > 10 examples. LLMs come to the rescue yet again! We can just define the config and ask the agent to generate explanations as well!

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "question_answering",
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
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 3,
        "example_template": "Context: {context}\nQuestion: {question}\nAnswer: Let's think step by step.\n{explanation}\n{answer}",
        "chain_of_thought": True
    }
}
```

Notice the changes that we have made to the config compared to the config without Chain-of-Thought [here](/guide/tasks/question_answering_task):

- `chain_of_thought` flag - this tells labeling agent to expect an explanation for the answer, in the seed dataset as well as LLM generated responses.
- `explanation_column` - this is the column where the explanation for the seed examples will reside.
- `example_template` - Notice that the template contains contains the explanation column as well. This tells the config where the explanation should be put when using the seed examples. We use the `Let's think step by step` prompt to initiate the chain of thought in the model.
- `output_guidelines` - We are explicitly prompting the LLM to first output an explanation, and then the final answer.

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
from autolabel import AutolabelDataset
ds = AutolabelDatset('data/squad_v2_test.csv', config = config)
agent.plan(ds)
agent.run(ds, max_items = 100)
```

Autolabel currently supports Chain-of-thought prompting for the following tasks:

1. Classifcation ([example](https://github.com/refuel-ai/autolabel/blob/main/examples/civil_comments/example_civil_comments.ipynb))
2. Entity Match
3. Question Answering ([example](https://github.com/refuel-ai/autolabel/blob/main/examples/squad_v2/example_squad_v2.ipynb))

Support for other tasks coming soon!
