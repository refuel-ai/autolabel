# Autolabel: Using LLMs to label data

First, specify a config file with task instructions. Let's say we use the example file located at `examples/config_chatgpt.json`.

Now, let's read this config file and see how much would it cost:
```python

from autolabel import LabelingAngent

l = LabelingAngent('examples/config_chatgpt.json', debug=True)

l.plan('examples/ag_news_filtered_labels_sampled.csv')
```

This prints:

```
Total Estimated Cost: $2.062
Number of examples to label: 2997
Average cost per example: 0.0006918845512178845

A prompt example:

You are an expert at understanding news articles.
Your job is to correctly label the provided input example into one of the following 9 categories.
Categories:
Sports
Health
Business
Entertainment
Sci/Tech
Italia
Software
Music Feeds
Toons


You will return the answer in JSON format with two keys: {"answered": "can you answer this question. say YES or NO", "label": "the correct label"}

Some examples with their output answers are provided below:
Example: The NHL and the players' association appeared headed toward a lockout when talks broke off yesterday after the union's first new proposal in nearly a year.
Output: {"answered": "yes", "label": "Sports"}

Example: Computer Associates International Inc. said Monday it has revoked home security and office support benefits to former Chief Executive Officer Sanjay Kumar, who was indicted last week on
Output: {"answered": "yes", "label": "Business"}

Example: Big Blue adds "innovaton center" to its China Research Lab to develop technology catering to small and midsize businesses.
Output: {"answered": "yes", "label": "Sci/Tech"}

Example: Before you wear your your cool yellow LiveStrong wristband at the hospital, think twice. Morton Plant Mease hospitals, St. Joseph #39;s and St Anthony #39;s are putting the brakes on Lance Armstrong #39;s
Output: {"answered": "yes", "label": "Health"}

Example: Energy Secretary Vince Perez Thursday called on oil companies to reduce oil prices in two weeks #39; time following a decision by the Organization of Oil Exporting Countries (OPEC) to raise its supply quota by 1 million barrels a day.
Output:
```

Now, let's run annotation on a subset of the dataset:
```python
l.run(
    'examples/ag_news_filtered_labels_sampled.csv',
    max_items=10
)
# This will also output a file with the labels per row in `examples/ag_news_filtered_labels_sampled_labeled.csv`
```

This prints the following:

```
Metric: support: 10
Metric: completion_rate: 1.0
Metric: accuracy: 0.9
Metric: confusion_matrix: {
    'labels': ['Sports', 'Health', 'Business', 'Entertainment', 'Sci/Tech', 'Italia', 'Software', 'Music Feeds', 'Toons', 'NO_LABEL'],
    'value': array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 9, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    }
````
