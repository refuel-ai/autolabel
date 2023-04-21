import os

from refuel_oracle.oracle import Oracle

curr_directory = os.path.dirname(os.path.abspath(__file__))

# config_path = os.path.join(curr_directory, "examples/config_imdb_oai.json")
# data_file_name = "examples/filtered_imdb.csv"

config_path = os.path.join(curr_directory, "examples/config_medqa_oai.json")
data_file_name = "examples/filtered_medqa.csv"

# config_path = os.path.join(curr_directory, "examples/config_walmart_amazon_oai.json")
# data_file_name = "examples/filtered_walmart_amazon.csv"

# config_path = os.path.join(curr_directory, "examples/config_sciq_oai.json")
# data_file_name = "examples/filtered_sciq.csv"

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
