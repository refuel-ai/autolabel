from functools import cached_property
from typing import Dict, List, Union

from jsonschema import validate

from .base import BaseConfig


class AutolabelConfig(BaseConfig):
    """Class to parse and store configs passed to Autolabel agent."""

    # Top-level config keys
    TASK_NAME_KEY = "task_name"
    TASK_TYPE_KEY = "task_type"
    DATASET_CONFIG_KEY = "dataset"
    MODEL_CONFIG_KEY = "model"
    EMBEDDING_CONFIG_KEY = "embedding"
    PROMPT_CONFIG_KEY = "prompt"
    DATASET_GENERATION_CONFIG_KEY = "dataset_generation"

    # Dataset config keys (config["dataset"][<key>])
    LABEL_COLUMN_KEY = "label_column"
    LABEL_SEPARATOR_KEY = "label_separator"
    EXPLANATION_COLUMN_KEY = "explanation_column"
    TEXT_COLUMN_KEY = "text_column"
    INPUT_COLUMNS_KEY = "input_columns"
    DELIMITER_KEY = "delimiter"
    DISABLE_QUOTING = "disable_quoting"

    # Model config keys (config["model"][<key>])
    PROVIDER_KEY = "provider"
    MODEL_NAME_KEY = "name"
    MODEL_PARAMS_KEY = "params"
    COMPUTE_CONFIDENCE_KEY = "compute_confidence"
    LOGIT_BIAS_KEY = "logit_bias"

    # Embedding config keys (config["embedding"][<key>])
    EMBEDDING_PROVIDER_KEY = "provider"
    EMBEDDING_MODEL_NAME_KEY = "model"

    # Prompt config keys (config["prompt"][<key>])
    TASK_GUIDELINE_KEY = "task_guidelines"
    VALID_LABELS_KEY = "labels"
    FEW_SHOT_EXAMPLE_SET_KEY = "few_shot_examples"
    FEW_SHOT_SELECTION_ALGORITHM_KEY = "few_shot_selection"
    FEW_SHOT_NUM_KEY = "few_shot_num"
    VECTOR_STORE_PARAMS_KEY = "vector_store_params"
    EXAMPLE_TEMPLATE_KEY = "example_template"
    OUTPUT_GUIDELINE_KEY = "output_guidelines"
    OUTPUT_FORMAT_KEY = "output_format"
    CHAIN_OF_THOUGHT_KEY = "chain_of_thought"
    LABEL_SELECTION_KEY = "label_selection"
    LABEL_SELECTION_COUNT_KEY = "label_selection_count"
    ATTRIBUTES_KEY = "attributes"
    TRANSFORM_KEY = "transforms"

    # Dataset generation config keys (config["dataset_generation"][<key>])
    DATASET_GENERATION_GUIDELINES_KEY = "guidelines"
    DATASET_GENERATION_NUM_ROWS_KEY = "num_rows"

    def __init__(self, config: Union[str, Dict], validate: bool = True) -> None:
        super().__init__(config, validate=validate)

    def _validate(self) -> bool:
        """Returns true if the config settings are valid"""
        from autolabel.configs.schema import schema

        validate(
            instance=self.config,
            schema=schema,
        )
        return True

    @cached_property
    def _dataset_config(self) -> Dict:
        """Returns information about the dataset being used for labeling (e.g. label_column, text_column, delimiter)"""
        return self.config.get(self.DATASET_CONFIG_KEY, {})

    @cached_property
    def _model_config(self) -> Dict:
        """Returns information about the model being used for labeling (e.g. provider name, model name, parameters)"""
        return self.config[self.MODEL_CONFIG_KEY]

    @cached_property
    def _embedding_config(self) -> Dict:
        """Returns information about the model being used for computing embeddings (e.g. provider name, model name)"""
        return self.config.get(self.EMBEDDING_CONFIG_KEY, {})

    @cached_property
    def _prompt_config(self) -> Dict:
        """Returns information about the prompt we are passing to the model (e.g. task guidelines, examples, output formatting)"""
        return self.config[self.PROMPT_CONFIG_KEY]

    @cached_property
    def _dataset_generation_config(self) -> Dict:
        """Returns information about the prompt for synthetic dataset generation"""
        return self.config.get(self.DATASET_GENERATION_CONFIG_KEY, {})

    # project and task definition config
    def task_name(self) -> str:
        return self.config[self.TASK_NAME_KEY]

    def task_type(self) -> str:
        """Returns the type of task we have configured the labeler to perform (e.g. Classification, Question Answering)"""
        return self.config[self.TASK_TYPE_KEY]

    # Dataset config
    def label_column(self) -> str:
        """Returns the name of the column containing labels for the dataset. Used for comparing accuracy of autolabel results vs ground truth"""
        return self._dataset_config.get(self.LABEL_COLUMN_KEY, None)

    def label_separator(self) -> str:
        """Returns the token used to seperate multiple labels in the dataset. Defaults to a semicolon ';'"""
        return self._dataset_config.get(self.LABEL_SEPARATOR_KEY, ";")

    def text_column(self) -> str:
        """Returns the name of the column containing text data we intend to label"""
        return self._dataset_config.get(self.TEXT_COLUMN_KEY, None)

    def input_columns(self) -> List[str]:
        """Returns the names of the input columns from the dataset that are used in the prompt"""
        return self._dataset_config.get(self.INPUT_COLUMNS_KEY, [])

    def explanation_column(self) -> str:
        """Returns the name of the column containing an explanation as to why the data is labeled a certain way"""
        return self._dataset_config.get(self.EXPLANATION_COLUMN_KEY, None)

    def delimiter(self) -> str:
        """Returns the token used to seperate cells in the dataset. Defaults to a comma ','"""
        return self._dataset_config.get(self.DELIMITER_KEY, ",")

    def disable_quoting(self) -> bool:
        """Returns true if quoting is disabled. Defaults to false"""
        return self._dataset_config.get(self.DISABLE_QUOTING, False)

    # Model config
    def provider(self) -> str:
        """Returns the name of the entity that provides the currently configured model (e.g. OpenAI, Anthropic, Refuel)"""
        return self._model_config[self.PROVIDER_KEY]

    def model_name(self) -> str:
        """Returns the name of the model being used for labeling (e.g. gpt-4, claude-v1)"""
        return self._model_config[self.MODEL_NAME_KEY]

    def model_params(self) -> Dict:
        """Returns a dict of configured settings for the model (e.g. hyperparameters)"""
        return self._model_config.get(self.MODEL_PARAMS_KEY, {})

    def confidence(self) -> bool:
        """Returns true if the model is able to return a confidence score along with its predictions"""
        return self._model_config.get(self.COMPUTE_CONFIDENCE_KEY, False)

    def logit_bias(self) -> float:
        """Returns the logit bias for the labels specified in the config"""
        return self._model_config.get(self.LOGIT_BIAS_KEY, 0.0)

    # Embedding config
    def embedding_provider(self) -> str:
        """Returns the name of the entity that provides the model used for computing embeddings"""
        return self._embedding_config.get(self.EMBEDDING_PROVIDER_KEY, self.provider())

    def embedding_model_name(self) -> str:
        """Returns the name of the model being used for computing embeddings (e.g. sentence-transformers/all-mpnet-base-v2)"""
        return self._embedding_config.get(self.EMBEDDING_MODEL_NAME_KEY, None)

    # Prompt config
    def task_guidelines(self) -> str:
        return self._prompt_config.get(self.TASK_GUIDELINE_KEY, "")

    def labels_list(self) -> List[str]:
        """Returns a list of valid labels"""
        if isinstance(self._prompt_config.get(self.VALID_LABELS_KEY, []), List):
            return self._prompt_config.get(self.VALID_LABELS_KEY, [])
        else:
            return self._prompt_config.get(self.VALID_LABELS_KEY, {}).keys()

    def label_descriptions(self) -> Dict[str, str]:
        """Returns a dict of label descriptions"""
        if isinstance(self._prompt_config.get(self.VALID_LABELS_KEY, []), List):
            return {}
        else:
            return self._prompt_config.get(self.VALID_LABELS_KEY, {})

    def few_shot_example_set(self) -> Union[str, List]:
        """Returns examples of how data should be labeled, used to guide context to the model about the task it is performing"""
        return self._prompt_config.get(self.FEW_SHOT_EXAMPLE_SET_KEY, [])

    def few_shot_algorithm(self) -> str:
        """Returns which algorithm is being used to construct the set of examples being given to the model about the labeling task"""
        return self._prompt_config.get(self.FEW_SHOT_SELECTION_ALGORITHM_KEY, None)

    def few_shot_num_examples(self) -> int:
        """Returns how many examples should be given to the model in its instruction prompt"""
        return self._prompt_config.get(self.FEW_SHOT_NUM_KEY, 0)

    def vector_store_params(self) -> Dict:
        """Returns any parameters to be passed to the vector store"""
        return self._prompt_config.get(self.VECTOR_STORE_PARAMS_KEY, {})

    def example_template(self) -> str:
        """Returns a string containing a template for how examples will be formatted in the prompt"""
        example_template = self._prompt_config.get(self.EXAMPLE_TEMPLATE_KEY, None)
        if not example_template:
            raise ValueError("An example template needs to be specified in the config.")
        return example_template

    def output_format(self) -> str:
        return self._prompt_config.get(self.OUTPUT_FORMAT_KEY, None)

    def output_guidelines(self) -> str:
        return self._prompt_config.get(self.OUTPUT_GUIDELINE_KEY, None)

    def chain_of_thought(self) -> bool:
        """Returns true if the model is able to perform chain of thought reasoning."""
        return self._prompt_config.get(self.CHAIN_OF_THOUGHT_KEY, False)

    def label_selection(self) -> bool:
        """Returns true if label selection is enabled. Label selection is the process of
        narrowing down the list of possible labels by similarity to a given input. Useful for
        classification tasks with a large number of possible classes."""
        return self._prompt_config.get(self.LABEL_SELECTION_KEY, False)

    def label_selection_count(self) -> int:
        """Returns the number of labels to select in LabelSelector"""
        return self._prompt_config.get(self.LABEL_SELECTION_COUNT_KEY, 10)

    def attributes(self) -> List[Dict]:
        """Returns a list of attributes to extract from the text."""
        return self._prompt_config.get(self.ATTRIBUTES_KEY, [])

    def transforms(self) -> List[Dict]:
        """Returns a list of transforms to apply to the data before sending to the model."""
        return self.config.get(self.TRANSFORM_KEY, [])

    def dataset_generation_guidelines(self) -> str:
        """Returns a string containing guidelines for how to generate a synthetic dataset"""
        return self._dataset_generation_config.get(
            self.DATASET_GENERATION_GUIDELINES_KEY, ""
        )

    def dataset_generation_num_rows(self) -> int:
        """Returns the number of rows to generate for the synthetic dataset"""
        return self._dataset_generation_config.get(
            self.DATASET_GENERATION_NUM_ROWS_KEY, 1
        )
