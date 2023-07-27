TODO_TEXT = "[TODO]"

TEMPLATE_TASK_NAME = f"{TODO_TEXT} Enter task name"
TEMPLATE_TASK_TYPE = f"{TODO_TEXT} Enter task type"
TEMPLATE_DELIMITER = f"{TODO_TEXT} Enter delimiter"
TEMPLATE_LABEL_SEPARATOR = f"{TODO_TEXT} Enter label separator"
TEMPLATE_LABEL_COLUMN = f"{TODO_TEXT} Enter label column name"
TEMPLATE_TASK_GUIDELINES = f"{TODO_TEXT} Enter task guidelines"
TEMPLATE_EXAMPLE_TEMPLATE = f"{TODO_TEXT} Enter example template"
TEMPLATE_FEW_SHOT_EXAMPLES = f"{TODO_TEXT} Enter few shot examples"
TEMPLATE_FEW_SHOT_SELECTION = f"{TODO_TEXT} Enter few shot selection"
TEMPLATE_FEW_SHOT_NUM = f"{TODO_TEXT} Enter few shot num"


TEMPLATE_CONFIG = {
    "task_name": TEMPLATE_TASK_NAME,
    "task_type": TEMPLATE_TASK_TYPE,
    "dataset": {
        "delimiter": TEMPLATE_DELIMITER,
        "label_column": TEMPLATE_LABEL_COLUMN,
    },
    "prompt": {
        "task_guidelines": TEMPLATE_TASK_GUIDELINES,
        "example_template": TEMPLATE_EXAMPLE_TEMPLATE,
        "few_shot_examples": TEMPLATE_FEW_SHOT_EXAMPLES,
        "few_shot_selection": TEMPLATE_FEW_SHOT_SELECTION,
        "few_shot_num": TEMPLATE_FEW_SHOT_NUM,
    },
    "model": {"provider": "openai", "name": "gpt-3.5-turbo", "params": {}},
}
