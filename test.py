from refuel_oracle.oracle import Oracle

config_path = "examples/config.json"

annotator = Oracle(config_path)

print("Running Oracle.plan() to calculate expected cost of labeling dataset.")
annotator.plan(
    dataset="/Users/pc/refuel/repos/refuel-oracle/examples/ag_news_filtered_labels_sampled.csv",
    input_column="description",
)

print("Running Oracle.annotate() on a subset of items in dataset")
annotator.annotate(
    dataset="/Users/pc/refuel/repos/refuel-oracle/examples/ag_news_filtered_labels_sampled.csv",
    input_column="description",
    output_column="llm_labels",
    max_items=10,
)
