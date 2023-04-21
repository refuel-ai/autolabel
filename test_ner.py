import os

from refuel_oracle.oracle import Oracle

curr_directory = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(curr_directory, "examples/config_wikiann.json")
wikiann_dataset = "examples/wikiann.csv"

annotator = Oracle(config_path, debug=True)

# print("Running Oracle.plan() to calculate expected cost of labeling dataset.")
# annotator.plan(dataset=os.path.join(curr_directory, wikiann_dataset))

print("Running Oracle.annotate() on a subset of items in dataset")
annotator.annotate(
    dataset=os.path.join(curr_directory, wikiann_dataset),
    max_items=50,
)
