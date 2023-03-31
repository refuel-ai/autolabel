import os

from refuel_oracle.oracle import Oracle

curr_directory = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(curr_directory, "examples/config.json")
ag_news_file_name = "examples/ag_news_filtered_labels_sampled.csv"

annotator = Oracle(config_path, debug=True)

print("Running Oracle.plan() to calculate expected cost of labeling dataset.")
annotator.plan(
    dataset=os.path.join(curr_directory, ag_news_file_name),
    input_column="description",
)

print("Running Oracle.annotate() on a subset of items in dataset")
annotation_obj = annotator.annotate(
    dataset=os.path.join(curr_directory, ag_news_file_name),
    input_column="description",
    ground_truth_column="label",
    output_column="llm_labels",
    max_items=10,
)

for i in range(5):
    print(f"\n\n#### output[{i}] ####")
    print("Data:")
    print(annotation_obj.data[i])
    print("Label:")
    print(annotation_obj.labels[i])
    print("Confidence:")
    print(annotation_obj.confidence[i])
