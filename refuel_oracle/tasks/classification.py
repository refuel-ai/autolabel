from langchain.prompts.prompt import PromptTemplate
from langchain.schema import LLMResult

from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation
from refuel_oracle.tasks import BaseTask

class ClassificationTask(BaseTask):

    DEFAULT_LABELING_INSTRUCTION = """Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:{labels_list}\n"""
    DEFAULT_OUTPUT_INSTRUCTION = """You will return the answer in JSON format with two keys: {\"answered\": \"can you answere this question. say YES or NO\", \"label\": \"the correct label\"}"""
    PROMPT_TEMPLATE = """
{project_instruction}
{labeling_instruction}
{output_instruction}

Some examples with their output answers are provided below:
{seed_examples}

Example:
{current_example}

Output:

"""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        
    def get_example_generation_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["example", "output"],
            template="Example: {example}\nOutput: {output}\n"
        )

    def get_prompt_template(self) -> PromptTemplate:
        # project instructions
        project_instruction = self.config.get("project_instruction", "")

        # labeling instructions
        labels_list = self.config.get("labels_list", [])
        num_labels = len(labels_list)
        labeling_instruction = self.config.get("labeling_instruction")
        if not labeling_instruction:
            labeling_instruction = self.DEFAULT_LABELING_INSTRUCTION.format(
                num_labels=num_labels,
                labels_list="\n".join(labels_list)
            )
        
        # output format instructions
        output_instruction = self.config.get(
            "output_instruction", self.DEFAULT_OUTPUT_INSTRUCTION)

        # seed examples
        # TODO

        return PromptTemplate(
            input_variables=[
                "project_instruction",
                "labeling_instruction",
                "output_instruction",
                "seed_examples",
                "current_example"
            ],
            template=self.PROMPT_TEMPLATE
        )

    def construct_prompt(self, **kwargs) -> str:
        # TODO
        pass
        
    def parse_llm_response(self, prompt: str, response: LLMResult) -> LLMAnnotation:
        # TODO
        pass
