from refuel_oracle.oracle import Oracle

config_path = "examples/config.json"

annotator = Oracle(config_path)

annotator.annotate(
    dataset="/Users/pc/refuel/repos/refuel-oracle/examples/ag_news_filtered_labels_sampled.csv",
    input_column="description",
    output_column="llm_labels",
    max_items=10,
)
