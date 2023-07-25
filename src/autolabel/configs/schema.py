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
# 'additionalProperties' is set to False for al properties to handle typos
# for not required properties/fields.
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
                "label_separator": {"type": ["string", "null"]},
                "text_column": {"type": ["string", "null"]},
                "delimiter": {"type": ["string", "null"]},
                "explanation_column": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
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
                "params": {"type": ["object", "null"]},
            },
            "required": ["provider", "name"],
            "additionalProperties": False,
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
            "additionalProperties": False,
        },
        "prompt": {
            "type": "object",
            "properties": {
                "task_guidelines": {"type": "string"},
                "output_guidelines": {"type": "string"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"}
                    # "uniqueItems": True
                },
                "example_template": {"type": "string"},
                "few_shot_examples": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object"}},
                        # When few_shor_example is a string. It shoulf point to the
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
            },
            "required": ["task_guidelines"],
            "additionalProperties": False,
        },
    },
    "required": [
        "task_name",
        "task_type",
        "model",
        "prompt",
    ],
    "additionalProperties": False,
}
