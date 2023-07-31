import logging
from typing import Dict, Union, List
import io

import pandas as pd

from autolabel.configs import AutolabelConfig
from autolabel.models import ModelFactory


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "{guidelines}\n\n{examples}\n\n{output_guidelines}\n{column_descriptions}\n\n```csv"
)
DEFAULT_GUIDELINES = f"""You are an expert at generating synthetic data. You will generate a dataset that satisfies the following criteria:
1. The data should be diverse and cover a wide range of scenarios.
2. The data should be as realistic as possible, closely mimicking real-world data.
3. The data should not contain any sensitive or personal information.
4. The data should be generated in a csv format."""
DEFAULT_OUTPUT_GUIDELINES = "Create a CSV file with {num_rows} rows and the columns {columns}. Each record should include the following fields:"


class DatasetGenerator:
    def __init__(
        self,
        config: Union[str, Dict],
    ):
        self.config = AutolabelConfig(config)
        self.llm = ModelFactory.from_config(self.config)
        self.columns = [
            v.split("}")[0].split("{")[1]
            for v in self.config.example_template().split(" ")
            if "{" in v and "}" in v
        ]

    def _format_examples(self, examples: List[Dict]) -> str:
        if examples is None:
            return ""
        return (
            "```csv\n"
            + "\n".join(
                [self.config.delimiter().join(self.columns)]
                + [
                    self.config.delimiter().join([example[col] for col in self.columns])
                    for example in examples
                ]
            )
            + "\n```"
        )

    def _generate_column_description(self, col: str, examples: List[Dict]) -> str:
        prompt = f"Please provide a concise one sentence description for the column '{col}'. Here are some examples of items in this column: {examples}\nDescription:"
        result = self.llm.label([prompt])
        if result.errors[0] is not None:
            raise result.errors[0]
        description = result.generations[0][0].text.strip()
        return description

    def generate(
        self,
        num_rows: int,
        guidelines: str = DEFAULT_GUIDELINES,
        output_guidelines: str = DEFAULT_OUTPUT_GUIDELINES,
        provide_examples: bool = True,
        column_descriptions: Dict[str, str] = {},
    ) -> pd.DataFrame:
        """
        This method generates a synthetic dataset based on the provided guidelines and column descriptions.

        Args:
            num_rows (int): The number of rows to generate in the dataset.
            guidelines (str, optional): The guidelines for generating the dataset. Defaults to DEFAULT_GUIDELINES.
            provide_examples (bool, optional): Whether to provide examples for generating the dataset. Defaults to True.
            column_descriptions (Dict[str, str], optional): Descriptions for each column in the dataset. Defaults to an empty dictionary.

        Returns:
            pd.DataFrame: The generated synthetic dataset.
        """
        example_set = None
        if provide_examples and self.config.few_shot_example_set() is not None:
            if isinstance(self.config.few_shot_example_set(), List):
                example_set = self.config.few_shot_example_set()
            else:
                example_set = pd.read_csv(
                    self.config.few_shot_example_set(), sep=self.config.delimiter()
                )
                example_set = example_set.sample(
                    min(len(example_set), self.config.few_shot_num_examples())
                )
                example_set = example_set.to_dict(orient="records")
        else:
            example_set = None
        examples = self._format_examples(example_set)

        output_guidelines = output_guidelines.format(
            num_rows=num_rows, columns=self.columns
        )

        prompt = PROMPT_TEMPLATE.format(
            guidelines=guidelines,
            examples=examples,
            output_guidelines=output_guidelines,
            column_descriptions="\n".join(
                [
                    f"- {col}: {column_descriptions[col] if col in column_descriptions else self._generate_column_description(col, example_set)}"
                    for col in self.columns
                ]
            ),
        )

        logger.info(f"Prompt: {prompt}")

        result = self.llm.label([prompt])
        if result.errors[0] is not None:
            raise result.errors[0]

        response = result.generations[0][0].text.strip()
        if response.endswith("```"):
            response = response[:-3].strip()

        response = io.StringIO(response)
        df = pd.read_csv(response, sep=self.config.delimiter())
        self.df = df

        return df
