"""Config Schema"""

from typing import List

from autolabel.schema import FewShotAlgorithm, ModelProvider, TaskType


def populate_vendors() -> List[str]:
    """Generate Predefined Vendors

    this function retrives the list of vendors from ModelProvider which acts
    as the central storage for all the vendors/providers
    """
    return [enum_member.value for enum_member in ModelProvider]


def populate_task_types() -> List[str]:
    """Generate Predefined Tasktypes

    this function retrives the list of acceptable task_types from TaskType
    """
    return [enum_member.value for enum_member in TaskType]


def populate_few_shot_selection() -> List[str]:
    """Generate Predefined Few shot selections

    this function retrives the list of acceptable few_shot_selections
    """
    return [enum_member.value for enum_member in FewShotAlgorithm]


# Schema definition for Autolabel API/ calls
schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "Label Config",
    "description": "The query configuration to generate autolabels",
    "type": "object",
    "properties": {
        "task_name": {
            "type": "string",
            "description": "The task name of the labeling job",
        },
        "task_type": {
            # Dynamically populate acceptable task_types
            "enum": populate_task_types(),
            "description": "The type of auto labeling task",
        },
        "dataset": {
            "type": "object",
            "properties": {
                "label_column": {"type": ["string", "null"]},
                "input_columns": {"type": ["array", "null"]},
                "label_separator": {"type": ["string", "null"]},
                "text_column": {"type": ["string", "null"]},
                "delimiter": {"type": ["string", "null"]},
                "explanation_column": {"type": ["string", "null"]},
                "disable_quoting": {"type": ["boolean", "null"]},
            },
            "additionalProperties": True,
        },
        "transforms": {
            "type": "array",
            "items": {"type": "object"},
            "additionalProperties": True,
        },
        "model": {
            "type": "object",
            "properties": {
                "provider": {
                    # Populate Model providers dynamically
                    "enum": populate_vendors(),
                },
                "name": {"type": "string"},
                "compute_confidence": {"type": ["boolean", "null"]},
                "logit_bias": {"type": ["number", "null"]},
                "params": {"type": ["object", "null"]},
            },
            "required": ["provider", "name"],
            "additionalProperties": True,
        },
        "embedding": {
            "type": "object",
            "properties": {
                "provider": {
                    # Populate embedding model providers dynamically
                    "enum": populate_vendors(),
                },
                "model": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "prompt": {
            "type": "object",
            "properties": {
                "task_guidelines": {"type": "string"},
                "output_guidelines": {"type": "string"},
                "labels": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {
                            "type": "object"
                        },  # This is for when the labels are provided with descriptions
                    ],
                    # "uniqueItems": True
                },
                "example_template": {"type": "string"},
                "few_shot_examples": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object"}},
                        # When few_shot_examples is a string. It should point to the
                        # filepath of a csv/other files containing the few shot examples
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "few_shot_selection": {
                    "enum": populate_few_shot_selection(),
                    "type": ["string", "null"],
                },
                "few_shot_num": {"type": ["number", "null"]},
                "chain_of_thought": {"type": ["boolean", "null"]},
                "label_selection": {"type": ["boolean", "null"]},
                "label_selection_count": {"type": ["number", "null"]},
                "attributes": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object"}},
                        {"type": "null"},
                    ]
                },
                "subtasks": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object"}},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["task_guidelines"],
            "additionalProperties": True,
        },
        "dataset_generation": {
            "type": "object",
            "properties": {
                "num_rows": {"type": ["number", "null"]},
                "guidelines": {"type": ["string", "null"]},
            },
        },
    },
    "required": [
        "task_name",
        "task_type",
        "model",
        "prompt",
    ],
    "additionalProperties": True,
}
