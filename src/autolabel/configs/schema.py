"""Config Schema"""

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
            "enum": [
                "classification",
                "entity_matching",
                "named_entity_recognition",
                "question_answering",
            ],
            "description": "The type of auto labeling task",
        },
        "dataset": {
            "type": "object",
            "properties": {
                "label_column": {"type": "string"},
                "text_column": {"type": "string"},
                "delimiter": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "model": {
            "type": "object",
            "properties": {
                "provider": {
                    "enum": [
                        # The below are supported model providers
                        "huggingface_pipeline",
                        "refuel",
                        "openai",
                        "google",
                        "anthropic",
                    ],
                },
                "name": {"type": "string"},
                "compute_confidence": {"type": "boolean"},
            },
            "required": ["provider", "name"],
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
                    ]
                },
                "few_shot_selection": {
                    "enum": [
                        "fixed",
                        "semantic_similarity",
                        # Add more to it as needed
                    ],
                },
                "few_shot_num": {"type": "number"},
            },
            "required": ["task_guidelines"],
            "additionalProperties": False,
        },
    },
    "required": [
        "task_name",
        "task_type",
        "dataset",
        "model",
        "prompt",
    ],
    "additionalProperties": False,
}
