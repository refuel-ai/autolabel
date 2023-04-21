import os

from refuel_oracle.oracle import Oracle

curr_directory = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(curr_directory, "examples/config_emotion_oai.json")
data_file_name = "examples/filtered_ledgar.csv"

# config_path = os.path.join(curr_directory, "examples/config_wikiann.json")
# data_file_name = "examples/wikiann.csv"

annotator = Oracle(config_path, debug=True)
plan_first = False
if plan_first:
    print("Running Oracle.plan() to calculate expected cost of labeling dataset.")
    annotator.plan(dataset=os.path.join(curr_directory, data_file_name))

print("Running Oracle.annotate() on a subset of items in dataset")
annotator.annotate(
    dataset=os.path.join(curr_directory, data_file_name),
    max_items=5,
)
