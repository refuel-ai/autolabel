# refuel-oracle
Using LLMs to label data

```python

from refuel_oracle.oracle import Oracle

o = Oracle('examples/config_chatgpt.json', debug=True)

o.plan('examples/ag_news_filtered_labels_sampled.csv', input_column='description')
Total Estimated Cost: 2.073578
Number of examples to label: 2997
Average cost per example: 0.0006918845512178845

o.annotate('examples/ag_news_filtered_labels_sampled.csv', input_column='description', ground_truth_column='label', max_items=10)

# This will also output a file with the labels per row in `examples/ag_news_filtered_labels_sampled_labeled.csv`

Metric: support: 10
Metric: completion_rate: 1.0
Metric: accuracy: 0.9
```

