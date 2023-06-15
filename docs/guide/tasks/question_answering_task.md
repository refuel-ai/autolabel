## Introduction

Question answering is the most fundamental task that can be solved using LLMs. Most tasks can be reduced to some form of question answering where the model is optionally given some context and then asked to answer a question. There can be a broad classification of question answering tasks into 2 categories -  

1. Open Book QA - In this variant, the model is given a context along with a question and then asked to answer using the context. Here, we do not rely on knowledge present in the model parameters and instead rely on the reasoning abilities and commonsense properties of the model to answer correctly.

2. Closed Book QA - In this variant, the model is just given a question, without any context or knowledge source and asked to answer based on pretrained knowledge. This requires more knowledge to be present in the model parameters and thus favours bigger LLMs.

In addition to context, question answering tasks can also differ in the way that the answers are generated. The easiest form is one where there is a predefined set of options (for eg. yes or no) and the model needs to choose from one of these options. Another variant allows separate options for each question similar to SAT questions. The last variant is one where the model is free to generate its own answers. This variant is harder to evaluate because multiple answers could mean the same thing.

## Example [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13DiE1dfG7pYGV2FLWkxPSbyTbABIm34I#scrollTo=c93fae0b)

### Dataset

Lets walk through using Autolabel for question answering on the Squad dataset. The Squad dataset comprises of 100k questions and answers along with a context for each question which contains the answer for the question. Additionally, the correct answer is a continuous text span from the context. However, in addition to correct answers, it also contains 50k pairs where the question is unanswerable given the context, that is, the context does not have enough information to answer the question correctly. Here is an example datapoint from the dataset,

```json
{
    "question": "When did Beyonce start becoming popular?",
    "context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles 'Crazy in Love' and 'Baby Boy'.",
    "answer": "in the late 1990s"
}
```

Thus the dataset consists of the `question`, `context` and `answer`. For datasets like SciQ, there may be an additional field called `options` which is a list of strings which are possible answers for a particular question.

### Config

In order to run Autolabel, we need a config defining the 3 important things - task, llm and dataset. Let's assume gpt-3.5-turbo as the LLM for this section.

```json
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions based on wikipedia articles. Your job is to answer the following questions using the context provided with the question. The answer is a continuous span of words from the context. Use the context to answer the question. If the question cannot be answered using the context, answer the question as unanswerable.",
        "few_shot_examples": [
            {
                "question": "What was created by the modern Conservative Party in 1859 to define basic Conservative principles?",
                "answer": "unanswerable",
                "context": "The modern Conservative Party was created out of the 'Pittite' Tories of the early 19th century. In the late 1820s disputes over political reform broke up this grouping. A government led by the Duke of Wellington collapsed amidst dire election results. Following this disaster Robert Peel set about assembling a new coalition of forces. Peel issued the Tamworth Manifesto in 1834 which set out the basic principles of Conservatism; – the necessity in specific cases of reform in order to survive, but an opposition to unnecessary change, that could lead to 'a perpetual vortex of agitation'. Meanwhile, the Whigs, along with free trade Tory followers of Robert Peel, and independent Radicals, formed the Liberal Party under Lord Palmerston in 1859, and transformed into a party of the growing urban middle-class, under the long leadership of William Ewart Gladstone."
            },
            {
                "question": "When is King Mom symbolically burnt?",
                "answer": "On the evening before Lent",
                "context": "Carnival means weeks of events that bring colourfully decorated floats, contagiously throbbing music, luxuriously costumed groups of celebrants of all ages, King and Queen elections, electrifying jump-ups and torchlight parades, the Jouvert morning: the Children's Parades and finally the Grand Parade. Aruba's biggest celebration is a month-long affair consisting of festive 'jump-ups' (street parades), spectacular parades and creative contests. Music and flamboyant costumes play a central role, from the Queen elections to the Grand Parade. Street parades continue in various districts throughout the month, with brass band, steel drum and roadmarch tunes. On the evening before Lent, Carnival ends with the symbolic burning of King Momo."
            },
            {
                "question": "How far does the Alps range stretch?",
                "answer": "the Mediterranean Sea north above the Po basin, extending through France from Grenoble, eastward through mid and southern Switzerland",
                "context": "The Alps are a crescent shaped geographic feature of central Europe that ranges in a 800 km (500 mi) arc from east to west and is 200 km (120 mi) in width. The mean height of the mountain peaks is 2.5 km (1.6 mi). The range stretches from the Mediterranean Sea north above the Po basin, extending through France from Grenoble, eastward through mid and southern Switzerland. The range continues toward Vienna in Austria, and east to the Adriatic Sea and into Slovenia. To the south it dips into northern Italy and to the north extends to the south border of Bavaria in Germany. In areas like Chiasso, Switzerland, and Neuschwanstein, Bavaria, the demarcation between the mountain range and the flatlands are clear; in other places such as Geneva, the demarcation is less clear. The countries with the greatest alpine territory are Switzerland, France, Austria and Italy."
            }
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 3,
        "example_template": "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    }
}
```
The `task_type` sets up the config for a specific task, question_answering in this case.

Take a look at the prompt section of the config. This defines the settings related to defining the task and the machinery around it.  

The `task_guidelines` key is the most important key, it defines the task for the LLM to understand and execute on. In this case, we first set up the task and tell the model the kind of data present in the dataset, by telling it that it is an expert at understanding wikipedia articles. Next, we define the task more concretely by telling the model how to answer the question given the context. We tell the model that the answer is a continuous text span from the context and that in some cases, the answer can be unanswerable and how the model should handle such questions.  

The `example_template` is one of the most important keys to set for a task. This defines the format of every example that will be sent to the LLM. This creates a prompt using the columns from the input dataset, and sends this prompt to the LLM hoping for the llm to generate the column defined under the `label_column`, which is answer in our case. For every input, the model will be given the example with all the columns from the datapoint filled in according to the specification in the `example_template`. The `label_column` will be empty, and the LLM will generate the label. The `example_template` will be used to format all seed examples. Here we also see the ordering of the context followed by question and answer, and also see the `Context: ` string to inform the model which part of the text is the context.

The `few_shot_examples` here is a list of json inputs which define handpicked examples to use as seed examples for the model. These labeled examples help the model understand the task better and how it supposed to answer a question. If there is a larger number of examples, we can specify a path to a csv instead of a list of examples.

`few_shot_num` defines the number of examples selected from the seed set and sent to the LLM. Experiment with this number based on the input token budget and performance degradation with longer inputs.

`few_shot_selection` is set to fixed in this case as we want to use all examples as seed examples. However, if we want to use a subset of examples as seed examples from a larger set, we can set the appropriate strategy like `semantic_similarity` here to get dynamic good seed examples.

### Alternate config for ClosedBook QA

Let's consider a dataset like sciq which is a closed book QA with multiple choice questions. Here we have an example config for this dataset,

```json
config = {
    "task_name": "ClosedBookQAScienceQuestions",
    "task_type": "question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering science questions. Choose an answer from the given options. Use your knowledge of science and common sense to best answer the question.",
        "few_shot_examples": "../examples/squad_v2/seed.csv",
        "few_shot_selection": "fixed",
        "few_shot_num": 3,
        "example_template": "Question: {question}\nOptions: {options}\nAnswer: {answer}"
    }
}
```

Notice in this case we don't have the `context` and pass in the `options` as list of string options. These are present in the dataset and are appropriately called in the example template.

### Run the task

```py
from autolabel import LabelingAgent
agent = LabelingAgent(config)
agent.plan('data/squad_v2_test.csv')
agent.run('data/squad_v2_test.csv', max_items = 100)
```

### Evaluation metrics

On running the above config, this is an example output expected for labeling 100 items.
```
Actual Cost: 0.13500600000000001
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ f1                 ┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 0.7018720299348971 │ 100     │ -inf      │ 0.59     │ 1.0             │
└────────────────────┴─────────┴───────────┴──────────┴─────────────────
```

**Accuracy** - This is the exact match performance based on the reference answer. Here we give the model 1 if the answer matches exactly with the correct answer and 0 otherwise. This is particularly harsh for the model in cases where there isnt a multi choice given to the model for eg. Squad. Even if the model gets one word wrong without changing the meaning, the model will get penalized.

**F1** - This is calculated by treating the predicted and the ground truth tokens as a list of tokens. Using this, an F1 score is calculated for every examples. This score can then be averaged over the entire dataset to get the final score. An exact match would get an F1 score of 1. This metric allows the model to make small mistakes in the predicted tokens and might be a more accurate metric for cases where the answers are not restricted to a set of options.

**Completion Rate** - There can be errors while running the LLM related to labeling for eg. the LLM may give a label which is not in the label list or provide an answer which is not parsable by the library. In this cases, we mark the example as not labeled successfully. The completion rate refers to the proportion of examples that were labeled successfully.
