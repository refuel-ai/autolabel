from autolabel import LabelingAgent
import logging

# logging.basicConfig(level=40, force=True)

config = {
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide fair criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy and politicians are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'. \nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
        "labels": ["toxic", "not toxic"],
        "few_shot_examples": [
            {
                "example": "It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.",
                "label": "toxic",
            },
            {
                "example": "This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!",
                "label": "not toxic",
            },
            {
                "example": "This bitch is nuts. Who would read a book by a woman",
                "label": "toxic",
            },
            {
                "example": "It was a great show. Not a combo I'd of expected to be good together but it was.",
                "label": "not toxic",
            },
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 4,
        "example_template": "Input: {example}\nOutput: {label}",
    },
}

agent = LabelingAgent(config)
agent.plan("../../examples/civil_comments/test.csv", max_items=100)
agent.run("../../examples/civil_comments/test.csv", max_items=100)
