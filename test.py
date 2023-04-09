import os

from refuel_oracle.oracle import Oracle

curr_directory = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(curr_directory, "examples/config2.json")
ag_news_file_name = "examples/ag_news_filtered_labels_sampled.csv"

annotator = Oracle(config_path, debug=True)

print("Running Oracle.plan() to calculate expected cost of labeling dataset.")
annotator.plan(dataset=os.path.join(curr_directory, ag_news_file_name))

print("Running Oracle.annotate() on a subset of items in dataset")
annotator.annotate(
    dataset=os.path.join(curr_directory, ag_news_file_name),
    max_items=15,
)
