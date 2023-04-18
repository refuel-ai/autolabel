from datasets import load_dataset
from transformers import AutoTokenizer
import json

dataset_csv = open("examples/wikiann.csv", "w")
dataset_csv.write("Text%IndividualLabels%CategorizedLabels\n")
dataset = load_dataset("wikiann", "en")
entity_category_mapping = {"LOC": "Location", "ORG": "Organization", "PER": "Person"}
for item in dataset["test"]:
    curr_text = " ".join(item["tokens"]).replace("%", " ")
    curr_labels = item["spans"]
    individual_labels = []
    individual_entity_categories = {"Location": [], "Organization": [], "Person": []}

    for label in curr_labels:
        entity_category = entity_category_mapping[label.split(":")[0]]
        entity_text = (
            ":".join(label.split(":")[1:]).strip(" ").strip("\n").replace("%", " ")
        )
        individual_entity_categories[entity_category].append(entity_text)
        individual_labels.append({"Description": entity_category, "Text": entity_text})

    dataset_csv.write(
        f"{curr_text}%{json.dumps(individual_labels)}%{json.dumps(individual_entity_categories)}\n"
    )

dataset_csv.close()
