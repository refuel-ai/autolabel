Few shot learning is one of the most powerful tools that autolabel offers to improve the accuracy of LLM generated labels. However, curating a seed dataset to use for few shot learning can be a time consuming and tedious process. To make this process easier, autolabel's LabelingAgent provides a method to generate synthetic datasets. These datasets can be used as seed datasets for few shot learning or any other purpose. This guide will walk you through the process of generating a synthetic dataset using autolabel.

Currently, autolabel supports synthetic dataset generation for classification and entity matching tasks. We plan to add support for other task types in the future.

### **Walkthrough: Creating a Synthetic Dataset for Banking**

<ol>
<li>The first step is to import the LabelingAgent from autolabel. This is the main class that we will use to generate the synthetic dataset.

```python
from autolabel import LabelingAgent
```

</li>
<li>The next step is to create the task config. Make sure to add the <code>dataset_generation</code> section to the config. This section contains the parameters for the dataset generation process. The <code>guidelines</code> parameter is a string containing the guidelines for the dataset generation task. The <code>num_rows</code> parameter is an integer indicating the number of rows <em><strong>per label</strong></em> to generate in the dataset.

```python
config = {
  "task_name": "BankingComplaintsClassification",
  "task_type": "classification",
  "dataset": {
    "label_column": "label",
    "delimiter": ","
  },
  "model": {
    "provider": "openai",
    "name": "gpt-3.5-turbo"
  },
  "prompt": {
    "task_guidelines": "You are an expert at understanding bank customers support complaints and queries.\nYour job is to correctly classify the provided input example into one of the following categories.\nCategories:\n{labels}",
    "output_guidelines": "You will answer with just the the correct output label and nothing else.",
    "labels": {
        "activate_my_card": "the customer cannot activate their credit or debit card",
        "age_limit": "the customer is under the age limit",
        "apple_pay_or_google_pay": "the customer is having trouble using apple pay or google pay",
        ... # more labels
    },
    "example_template": "Input: {example}\nOutput: {label}"
  },
  "dataset_generation": {
    "num_rows": 5,
    "guidelines": "You are an expert at generating synthetic data. You will generate a dataset that satisfies the following criteria:\n1. The data should be diverse and cover a wide range of scenarios.\n2. The data should be as realistic as possible, closely mimicking real-world data.\n3. The data should vary in length, some shorter and some longer.\n4. The data should be generated in a csv format.\n\nEach row should contain a realistic bank complaint. Use CSV format, with each line containing just the complaint and nothing else."
  }
}
```

Note that here, we defined <code>labels</code> as a dictionary where the keys are the valid labels and the values are descriptions for those labels. This helps the LLM understand what each label means and can result in a higher quality dataset.

</li>
<li>Now all that's left is to run the code that generates the dataset!

```python
agent = LabelingAgent(config)
ds = agent.generate_synthetic_dataset()
```

</li>
</ol>

That's it! You now have a synthetic dataset that you can use for few shot learning or for any other purpose. You can save the dataset to a csv file using the following code:

```python
ds.save("synthetic_dataset.csv")
```

### Model and Model Parameters

To edit the model used for synthetic dataset generation, simply change the `model` section of the config. We've found that setting a higher temperature for this task generally results in more realistic datasets. We recommend experimenting with different models and model parameters to see what works best for your use case.
