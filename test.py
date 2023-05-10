import os

from autolabel.labeler import LabelingAgent

model_config_path = "examples/configs/llm_configs/flan_t5_xxl.json"

# config_path = os.path.join(curr_directory, "examples/config_imdb_oai.json")
# data_file_name = "examples/filtered_imdb.csv"

# config_path = os.path.join(curr_directory, "examples/config_banking_oai.json")
# data_file_name = "examples/filtered_banking.csv"

# config_path = os.path.join(curr_directory, "examples/config_emotion_oai.json")
# data_file_name = "examples/filtered_emotion.csv"

# config_path = os.path.join(curr_directory, "examples/config_ledgar_oai.json")
# data_file_name = "examples/filtered_ledgar.csv"

# config_path = os.path.join(curr_directory, "examples/config_medqa_oai.json")
# data_file_name = "examples/filtered_medqa.csv"

# config_path = os.path.join(curr_directory, "examples/config_walmart_amazon_oai.json")
# data_file_name = "examples/filtered_walmart_amazon.csv"

# config_path = os.path.join(curr_directory, "examples/config_sciq_oai.json")
# data_file_name = "examples/filtered_sciq.csv"

# config_path = os.path.join(curr_directory, "examples/config_medqa_oai.json")
# data_file_name = "examples/filtered_medqa.csv"

# task_config_path = "examples/configs/task_configs/banking_classification.json"
# data_file_name = "data/banking_test.csv"
# data_config_path = "examples/configs/dataset_configs/banking.json"

# task_config_path = "examples/configs/task_configs/sciq_qa.json"
# data_file_name = "data/sciq_test.csv"
# data_config_path = "examples/configs/dataset_configs/sciq.json"

# task_config_path = "examples/configs/task_configs/conll2003_ner.json"
# data_file_name = "data/conll2003_test.csv"
# data_config_path = "examples/configs/dataset_configs/conll2003.json"

# task_config_path = "examples/configs/task_configs/walmart_amazon_matching.json"
# data_file_name = "data/walmart_amazon_test.csv"
# data_config_path = "examples/configs/dataset_configs/walmart_amazon.json"

task_config_path = "examples/configs/task_configs/wikiann_ner.json"
data_file_name = "data/wikiann_test.csv"
data_config_path = "examples/configs/dataset_configs/wikiann.json"

# task_config_path = "examples/configs/task_configs/wikiann_ner.json"
# data_file_name = "data/wikiann_test.csv"
# data_config_path = "examples/configs/dataset_configs/wikiann.json"


# config_path = os.path.join(curr_directory, "examples/config_wikiann.json")
# data_file_name = "examples/wikiann.csv"

l = LabelingAgent(task_config_path, model_config_path)
labels, df, metrics_list = l.run(
    data_file_name,
    data_config_path,
    max_items=500,
)
