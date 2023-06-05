# Refuel Autolabel

Refuel helps you label datasets at the speed of thought and at human-level accuracy using LLMs. 


### Using Autolabel for classifying unlabeled data with ChatGPT

The LabelingAgent class is initialized with a configuration file, which defines the task you would like the LLM to perform on your data (i.e. classification, entity recognition, etc.)
``` py
annotator = LabelingAgent('examples/config_chatgpt.json')
```

Many sample configuration files can be found in the examples directory of the repository. These can act as a good starting template for other projects.

In this example, we are using ChatGPT to classify news articles into the appropriate category.
``` py
annotator.run(
    dataset='examples/ag_news_filtered_labels_sampled.csv',
    max_items=100,
)
```

## Concepts

### Chain of Thought (CoT)
Chain of thought prompting encourages the model to reason about the output before generating it. This is done by adding the sentence 'Lets think step by step' before the model generates the explanation and the output. This prompt helps the model in reasoning tasks where it reasons through subtasks to get to the final answer, instead of directly predicting the answer.

### Self Consistency
Self consistency is a variant of ensembling used on model outputs. Here, the model is asked to predict the answer of a single question, multiple times, coming up with different reasonings or 'explanations' (from CoT) each time. In the end, a majority vote is taken over the outputs generating, marginalizing the reasoning text. This helps the model try out different reasoning paths, and we pick the answer having the most reasoning paths leading to it.

### Confidence Score
Along with the answer, the model can produce a score which measures the 'confidence' that the model has in its output. This is a measure of the calibration of the model. A perfect model would be able to tell its probability of being correct accurately, and a graph between the number of correct examples above a confidence threshold would be equal to the confidence score itself.

### Example Selection
It has been shown that the specific seed examples used while constructing the initial prompt have an impact on the performance of the model. Seed examples are the dataset examples which the model is shown to help it understand the task better. Selecting the seed example per datapoint which needs to be labeled can help boost performance. We support the following example selection techniques -

1. Fixed_few_shot - Here the same set of seed examples are used for every input data point.
2. Semantic_similarity - Here the model creates language embeddings for all the examples in the seed set and finds the few shot examples which are closest to the input datapoint. The hope is that closer datapoints from the seed set will give the model more context on how similar examples have been labeled, helping it improve performance.
3. Max_marginal_relevance - 

## Tutorial

Take a look at the example usage to get a better idea of how to use different aspects of the module.