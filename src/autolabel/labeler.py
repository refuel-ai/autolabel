import asyncio
import io
import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.console import Console
from transformers import AutoTokenizer

from autolabel.cache import (
    BaseCache,
    SQLAlchemyConfidenceCache,
    SQLAlchemyGenerationCache,
    SQLAlchemyTransformCache,
)
from autolabel.confidence import ConfidenceCalculator
from autolabel.configs import AutolabelConfig
from autolabel.dataset import AutolabelDataset
from autolabel.few_shot import (
    DEFAULT_EMBEDDING_PROVIDER,
    PROVIDER_TO_MODEL,
    BaseExampleSelector,
    ExampleSelectorFactory,
)
from autolabel.few_shot.label_selector import LabelSelector
from autolabel.metrics import BaseMetric
from autolabel.models import BaseModel, ModelFactory
from autolabel.schema import (
    AUTO_CONFIDENCE_CHUNKING_COLUMN,
    AggregationFunction,
    LLMAnnotation,
    MetricResult,
    TaskType,
)
from autolabel.tasks import TaskFactory
from autolabel.transforms import BaseTransform, TransformFactory
from autolabel.utils import (
    gather_async_tasks_with_progress,
    get_format_variables,
    in_notebook,
    maybe_round,
    print_table,
    safe_serialize_to_string,
    track,
    track_with_stats,
)

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


COST_TABLE_STYLES = {
    "parameter": "magenta bold",
    "value": "green bold",
}
METRIC_TABLE_STYLE = "cyan bold"

MERGE_FUNCTION = {
    AggregationFunction.MAX: np.max,
    AggregationFunction.MEAN: np.mean,
}


class LabelingAgent:
    COST_KEY = "Cost in $"
    CONFIDENCE_MAX_CONTEXT_LENGTH = 3400

    def __init__(
        self,
        config: Union[AutolabelConfig, str, dict],
        cache: Optional[bool] = True,
        example_selector: Optional[BaseExampleSelector] = None,
        console_output: Optional[bool] = True,
        generation_cache: Optional[BaseCache] = SQLAlchemyGenerationCache(),
        transform_cache: Optional[BaseCache] = SQLAlchemyTransformCache(),
        confidence_cache: Optional[BaseCache] = SQLAlchemyConfidenceCache(),
        confidence_tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.generation_cache = generation_cache
        self.transform_cache = transform_cache
        self.confidence_cache = confidence_cache
        if not cache:
            logger.warning(
                f"cache parameter is deprecated and will be removed soon. Please use generation_cache, transform_cache and confidence_cache instead."
            )
            self.generation_cache = None
            self.transform_cache = None
            self.confidence_cache = None

        if self.generation_cache is not None:
            self.generation_cache.initialize()
        if self.transform_cache is not None:
            self.transform_cache.initialize()
        if self.confidence_cache is not None:
            self.confidence_cache.initialize()

        self.console = Console(quiet=not console_output)

        self.config = (
            config if isinstance(config, AutolabelConfig) else AutolabelConfig(config)
        )
        self.task = TaskFactory.from_config(self.config)
        self.llm: BaseModel = ModelFactory.from_config(
            self.config, cache=self.generation_cache
        )

        if self.config.confidence_chunk_column():
            if not confidence_tokenizer:
                self.confidence_tokenizer = AutoTokenizer.from_pretrained(
                    "google/flan-t5-xxl"
                )
            else:
                self.confidence_tokenizer = confidence_tokenizer
        score_type = "logprob_average"
        if self.config.task_type() == TaskType.ATTRIBUTE_EXTRACTION:
            score_type = "logprob_average_per_key"
        self.confidence = ConfidenceCalculator(
            score_type=score_type,
            llm=self.llm,
            cache=self.confidence_cache,
        )

        self.example_selector = example_selector

        if in_notebook():
            import nest_asyncio

            nest_asyncio.apply()

    def run(
        self,
        dataset: AutolabelDataset,
        output_name: Optional[str] = None,
        max_items: Optional[int] = None,
        start_index: int = 0,
        additional_metrics: Optional[List[BaseMetric]] = [],
        skip_eval: Optional[bool] = False,
    ) -> Tuple[pd.Series, pd.DataFrame, List[MetricResult]]:
        return asyncio.run(
            self.arun(
                dataset=dataset,
                output_name=output_name,
                max_items=max_items,
                start_index=start_index,
                additional_metrics=additional_metrics,
                skip_eval=skip_eval,
            )
        )

    async def arun(
        self,
        dataset: AutolabelDataset,
        output_name: Optional[str] = None,
        max_items: Optional[int] = None,
        start_index: int = 0,
        additional_metrics: Optional[List[BaseMetric]] = [],
        skip_eval: Optional[bool] = False,
    ) -> Tuple[pd.Series, pd.DataFrame, List[MetricResult]]:
        """Labels data in a given dataset. Output written to new CSV file.

        Args:
            dataset: path to CSV dataset to be annotated
            max_items: maximum items in dataset to be annotated
            output_name: custom name of output CSV file
            start_index: skips annotating [0, start_index)
        """

        dataset = dataset.get_slice(max_items=max_items, start_index=start_index)

        llm_labels = []

        # Get the seed examples from the dataset config
        seed_examples = self.config.few_shot_example_set()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            seed_loader = AutolabelDataset(seed_examples, self.config)
            seed_examples = seed_loader.inputs

        # Check explanations are present in data if explanation_column is passed in
        if (
            self.config.explanation_column()
            and len(seed_examples) > 0
            and self.config.explanation_column() not in list(seed_examples[0].keys())
        ):
            raise ValueError(
                f"Explanation column {self.config.explanation_column()} not found in dataset.\nMake sure that explanations were generated using labeler.generate_explanations(seed_file)."
            )

        if self.example_selector is None:
            if (
                self.config.label_selection()
                and self.config.few_shot_algorithm() != "fixed"
            ):
                # TODO: Add support for other few shot algorithms specially semantic similarity
                raise ValueError(
                    "Error: Only 'fixed' few shot example selector is supported for label selection."
                )

            self.example_selector = ExampleSelectorFactory.initialize_selector(
                self.config,
                [safe_serialize_to_string(example) for example in seed_examples],
                dataset.df.keys().tolist(),
                cache=self.generation_cache is not None,
            )

        if self.config.label_selection():
            if self.config.task_type() != TaskType.CLASSIFICATION:
                self.console.print(
                    "Warning: label_selection only supported for classification tasks!"
                )
            else:
                self.label_selector = LabelSelector(
                    config=self.config,
                    embedding_func=PROVIDER_TO_MODEL.get(
                        self.config.embedding_provider(), DEFAULT_EMBEDDING_PROVIDER
                    )(model=self.config.embedding_model_name()),
                )

        current_index = 0
        cost = 0.0
        postfix_dict = {}

        indices = range(current_index, len(dataset.inputs))
        selected_labels = self.config.labels_list()
        for current_index in track_with_stats(
            indices,
            postfix_dict,
            total=len(dataset.inputs) - current_index,
            console=self.console,
        ):
            chunk = dataset.inputs[current_index]
            examples = []

            if (
                self.config.label_selection()
                and self.config.task_type() == TaskType.CLASSIFICATION
            ):
                # get every column except the one we want to label
                toEmbed = chunk.copy()
                if self.config.label_column() and self.config.label_column() in toEmbed:
                    del toEmbed[self.config.label_column()]

                # convert this to a string
                toEmbed = json.dumps(toEmbed)

                selected_labels = self.label_selector.select_labels(toEmbed)

                if self.example_selector:
                    examples = self.example_selector.select_examples(
                        safe_serialize_to_string(chunk),
                        selected_labels=selected_labels,
                        label_column=self.config.label_column(),
                    )
                else:
                    examples = []
            else:
                if self.example_selector:
                    examples = self.example_selector.select_examples(
                        safe_serialize_to_string(chunk),
                    )

            # Construct Prompt to pass to LLM
            final_prompt = self.task.construct_prompt(
                chunk,
                examples,
                selected_labels=selected_labels,
                max_input_tokens=self.llm.DEFAULT_CONTEXT_LENGTH,
                get_num_tokens=self.llm.get_num_tokens,
            )

            response = await self.llm.label([final_prompt])
            for i, generations, error, latency in zip(
                range(len(response.generations)),
                response.generations,
                response.errors,
                response.latencies,
            ):
                input_tokens = self.llm.get_num_tokens(final_prompt)
                if error is not None:
                    annotation = LLMAnnotation(
                        successfully_labeled=False,
                        label=self.task.NULL_LABEL_TOKEN,
                        raw_response="",
                        curr_sample=pickle.dumps(chunk),
                        prompt=final_prompt,
                        confidence_score=0,
                        error=error,
                        input_tokens=input_tokens,
                        cost=0,
                        latency=0,
                    )
                else:
                    annotations = []
                    for generation in generations:
                        annotation = self.task.parse_llm_response(
                            generation, chunk, final_prompt
                        )
                        annotation.confidence_prompt = (
                            self.task.construct_confidence_prompt(chunk, examples)
                        )
                        annotation.input_tokens = input_tokens
                        annotation.output_tokens = self.llm.get_num_tokens(
                            annotation.raw_response
                        )
                        annotation.cost = sum(response.costs)
                        annotation.latency = latency

                        if self.config.confidence():
                            try:
                                annotation.confidence_score = (
                                    await self.get_confidence_score(
                                        annotation, chunk, examples
                                    )
                                )
                            except Exception as e:
                                logger.exception(
                                    f"Error calculating confidence score: {e}"
                                )
                                logger.error(
                                    f"Could not calculate confidence score for annotation: {annotation}"
                                )
                                if (
                                    self.config.task_type()
                                    == TaskType.ATTRIBUTE_EXTRACTION
                                ):
                                    annotation.confidence_score = {}
                                else:
                                    annotation.confidence_score = 0

                        annotations.append(annotation)
                    annotation = self.majority_annotation(annotations)

                llm_labels.append(annotation)

            cost += sum(response.costs)
            postfix_dict[self.COST_KEY] = f"{cost:.2f}"

            # Evaluate the task every eval_every examples
            if not skip_eval and (current_index + 1) % 100 == 0:
                if dataset.gt_labels:
                    eval_result = self.task.eval(
                        llm_labels,
                        (
                            dataset.gt_labels[: len(llm_labels)]
                            if isinstance(dataset.gt_labels, list)
                            else {
                                k: v[: len(llm_labels)]
                                for k, v in dataset.gt_labels.items()
                            }
                        ),
                        additional_metrics=additional_metrics,
                    )

                    for m in eval_result:
                        # This is a row wise metric
                        if isinstance(m.value, list):
                            continue
                        elif m.show_running:
                            postfix_dict[m.name] = (
                                f"{m.value:.4f}"
                                if isinstance(m.value, float)
                                else m.value
                            )

        eval_result = None
        table = {}

        # if true labels are provided, evaluate accuracy of predictions
        if not skip_eval and dataset.gt_labels:
            eval_result = self.task.eval(
                llm_labels,
                (
                    dataset.gt_labels[: len(llm_labels)]
                    if isinstance(dataset.gt_labels, list)
                    else {k: v[: len(llm_labels)] for k, v in dataset.gt_labels.items()}
                ),
                additional_metrics=additional_metrics,
            )
            # TODO: serialize and write to file
            for m in eval_result:
                if isinstance(m.value, list):
                    continue
                elif m.show_running:
                    table[m.name] = m.value
                else:
                    self.console.print(f"{m.name}:\n{m.value}")

        # print cost
        self.console.print(f"Actual Cost: {maybe_round(cost)}")
        print_table(table, console=self.console, default_style=METRIC_TABLE_STYLE)

        dataset.process_labels(llm_labels, eval_result)
        # Only save to csv if output_name is provided or dataset is a string
        if not output_name and isinstance(dataset, str):
            output_name = (
                dataset.rsplit(".", 1)[0] + "_labeled." + dataset.rsplit(".", 1)[1]
            )

        if output_name:
            dataset.save(output_file_name=output_name)
        return dataset

    def plan(
        self,
        dataset: AutolabelDataset,
        max_items: Optional[int] = None,
        start_index: int = 0,
    ) -> None:
        """Calculates and prints the cost of calling autolabel.run() on a given dataset

        Args:
            dataset: path to a CSV dataset
        """
        dataset = dataset.get_slice(max_items=max_items, start_index=start_index)

        if (
            self.config.confidence()
            and "REFUEL_API_KEY" not in os.environ
            and not self.llm.returns_token_probs()
        ):
            raise ValueError(
                "REFUEL_API_KEY environment variable must be set to compute confidence scores. You can request an API key at https://refuel-ai.typeform.com/llm-access."
            )

        prompt_list = []
        total_cost = 0

        # Get the seed examples from the dataset config
        seed_examples = self.config.few_shot_example_set()

        # If this dataset config is a string, read the corrresponding csv file
        if isinstance(seed_examples, str):
            seed_loader = AutolabelDataset(seed_examples, self.config)
            seed_examples = seed_loader.inputs

        # Check explanations are present in data if explanation_column is passed in
        if (
            self.config.explanation_column()
            and len(seed_examples) > 0
            and self.config.explanation_column() not in list(seed_examples[0].keys())
        ):
            raise ValueError(
                f"Explanation column {self.config.explanation_column()} not found in dataset.\nMake sure that explanations were generated using labeler.generate_explanations(seed_file)."
            )

        self.example_selector = ExampleSelectorFactory.initialize_selector(
            self.config,
            [safe_serialize_to_string(example) for example in seed_examples],
            dataset.df.keys().tolist(),
            cache=self.generation_cache is not None,
        )

        if self.config.label_selection():
            if self.config.task_type() != TaskType.CLASSIFICATION:
                self.console.print(
                    "Warning: label_selection only supported for classification tasks!"
                )
            else:
                self.label_selector = LabelSelector(
                    config=self.config,
                    embedding_func=PROVIDER_TO_MODEL.get(
                        self.config.embedding_provider(), DEFAULT_EMBEDDING_PROVIDER
                    )(model=self.config.embedding_model_name()),
                )

        input_limit = min(len(dataset.inputs), 100) if max_items is None else max_items  # type: ignore
        for input_i in track(
            dataset.inputs[:input_limit],
            description="Generating Prompts...",
            console=self.console,
        ):
            # TODO: Check if this needs to use the example selector
            if self.example_selector:
                examples = self.example_selector.select_examples(
                    safe_serialize_to_string(input_i)
                )
            else:
                examples = []
            if (
                self.config.label_selection()
                and self.config.task_type() == TaskType.CLASSIFICATION
            ):
                selected_labels = self.label_selector.select_labels(input_i["example"])
                final_prompt = self.task.construct_prompt(
                    input_i,
                    examples,
                    selected_labels=selected_labels,
                    max_input_tokens=self.llm.DEFAULT_CONTEXT_LENGTH,
                    get_num_tokens=self.llm.get_num_tokens,
                )
            else:
                final_prompt = self.task.construct_prompt(
                    input_i,
                    examples,
                    max_input_tokens=self.llm.DEFAULT_CONTEXT_LENGTH,
                    get_num_tokens=self.llm.get_num_tokens,
                )
            prompt_list.append(final_prompt)

            # Calculate the number of tokens
            curr_cost = self.llm.get_cost(prompt=final_prompt, label="")
            total_cost += curr_cost

        total_cost = total_cost * (len(dataset.inputs) / input_limit)
        table = {
            "Total Estimated Cost": f"${maybe_round(total_cost)}",
            "Number of Examples": len(dataset.inputs),
            "Average cost per example": f"${maybe_round(total_cost / len(dataset.inputs))}",
        }
        table = {"parameter": list(table.keys()), "value": list(table.values())}

        print_table(
            table, show_header=False, console=self.console, styles=COST_TABLE_STYLES
        )
        self.console.rule("Prompt Example")
        self.console.print(f"{prompt_list[0]}", markup=False)
        self.console.rule()

    async def async_run_transform(
        self, transform: BaseTransform, dataset: AutolabelDataset
    ):
        transform_outputs = [
            transform.apply(input_dict) for input_dict in dataset.inputs
        ]

        outputs = await gather_async_tasks_with_progress(
            transform_outputs,
            description=f"Running transform {transform.name()}...",
            console=self.console,
        )
        output_df = pd.DataFrame.from_records(outputs)
        final_df = pd.concat([dataset.df, output_df], axis=1)
        dataset = AutolabelDataset(final_df, self.config)
        return dataset

    def transform(self, dataset: AutolabelDataset):
        transforms = []
        for transform_dict in self.config.transforms():
            transforms.append(
                TransformFactory.from_dict(transform_dict, cache=self.transform_cache)
            )
        for transform in transforms:
            dataset = asyncio.run(self.async_run_transform(transform, dataset))

        return dataset

    async def get_confidence_score(
        self, annotation: LLMAnnotation, chunk: Dict, examples: List[Dict]
    ) -> Union[float, dict]:
        full_confidence_input = annotation.confidence_prompt + annotation.raw_response
        if (
            self.llm.returns_token_probs()
            or not self.config.confidence_chunk_column()
            or self.get_num_tokens(full_confidence_input)
            < self.CONFIDENCE_MAX_CONTEXT_LENGTH
        ):
            return await self.confidence.calculate(model_generation=annotation)
        key_to_chunk = self.config.confidence_chunk_column()
        if not key_to_chunk:
            raise ValueError(
                "confidence_chunk_column must be set in the config to use confidence_chunk_size"
            )
        if key_to_chunk == AUTO_CONFIDENCE_CHUNKING_COLUMN:
            # If the confidence_chunk_column is set to auto,
            # we choose the column with the most tokens as the chunking column.
            max_tokens = -1
            example_template_keys = get_format_variables(self.config.example_template())
            for key in example_template_keys:
                num_tokens = self.get_num_tokens(chunk[key])
                if num_tokens > max_tokens:
                    max_tokens = num_tokens
                    key_to_chunk = key

        empty_chunk = chunk.copy()
        empty_chunk[key_to_chunk] = ""
        empty_prompt = self.task.construct_confidence_prompt(empty_chunk, examples)
        num_tokens_empty_prompt = self.get_num_tokens(empty_prompt)
        num_tokens_per_chunk = (
            self.config.confidence_chunk_size() - num_tokens_empty_prompt
        )
        confidence_chunks = self.chunk_string(chunk[key_to_chunk], num_tokens_per_chunk)

        confidence_scores = []
        for confidence_chunk in confidence_chunks:
            new_chunk = chunk.copy()
            new_chunk[key_to_chunk] = confidence_chunk
            new_prompt = self.task.construct_confidence_prompt(new_chunk, examples)
            annotation_dict = annotation.dict()
            annotation_dict["confidence_prompt"] = new_prompt
            confidence_scores.append(
                await self.confidence.calculate(
                    model_generation=LLMAnnotation(**annotation_dict),
                )
            )

        merge_function = MERGE_FUNCTION[self.config.confidence_merge_function()]
        if isinstance(confidence_scores[0], dict):
            merged_confidence = {}
            for key in confidence_scores[0].keys():
                merged_confidence[key] = merge_function(
                    [conf[key] for conf in confidence_scores]
                )
            return merged_confidence
        else:
            merged_confidence = merge_function(confidence_scores)
            return merged_confidence

    def majority_annotation(
        self, annotation_list: List[LLMAnnotation]
    ) -> LLMAnnotation:
        labels = [a.label for a in annotation_list]
        counts = {}
        for ind, label in enumerate(labels):
            # Needed for named entity recognition which outputs lists instead of strings
            label = str(label)

            if label not in counts:
                counts[label] = (1, ind)
            else:
                counts[label] = (counts[label][0] + 1, counts[label][1])
        max_label = max(counts, key=lambda x: counts[x][0])
        return annotation_list[counts[max_label][1]]

    def generate_explanations(
        self,
        seed_examples: Union[str, List[Dict]],
        include_label: bool = True,
    ) -> List[Dict]:
        return asyncio.run(
            self.agenerate_explanations(
                seed_examples=seed_examples, include_label=include_label
            )
        )

    async def agenerate_explanations(
        self,
        seed_examples: Union[str, List[Dict]],
        include_label: bool = True,
    ) -> List[Dict]:
        """Use LLM to generate explanations for why examples are labeled the way that they are."""
        out_file = None
        if isinstance(seed_examples, str):
            out_file = seed_examples
            seed_loader = AutolabelDataset(seed_examples, self.config)
            seed_examples = seed_loader.inputs

        explanation_column = self.config.explanation_column()
        if not explanation_column:
            raise ValueError(
                "The explanation column needs to be specified in the dataset config."
            )

        for seed_example in track(
            seed_examples,
            description="Generating explanations",
            console=self.console,
        ):
            explanation_prompt = self.task.get_explanation_prompt(
                seed_example, include_label=include_label
            )
            if self.task.image_col is not None:
                explanation_prompt = json.dumps(
                    {
                        "text": explanation_prompt,
                        "image_url": seed_example[self.task.image_col],
                    }
                )
            explanation = await self.llm.label([explanation_prompt])
            explanation = explanation.generations[0][0].text
            seed_example[explanation_column] = str(explanation) if explanation else ""

        if out_file:
            df = pd.DataFrame.from_records(seed_examples)
            df.to_csv(out_file, index=False)

        return seed_examples

    def generate_synthetic_dataset(self) -> AutolabelDataset:
        columns = get_format_variables(self.config.example_template())
        df = pd.DataFrame(columns=columns)
        for label in track(
            self.config.labels_list(),
            description="Generating dataset",
            console=self.console,
        ):
            prompt = self.task.get_generate_dataset_prompt(label)

            result = self.llm.label([prompt])
            if result.errors[0] is not None:
                self.console.print(
                    f"Error generating rows for label {label}: {result.errors[0]}"
                )
            else:
                response = result.generations[0][0].text.strip()

                response = io.StringIO(response)
                label_df = pd.read_csv(response, sep=self.config.delimiter())
                label_df[self.config.label_column()] = label
                df = pd.concat([df, label_df], axis=0, ignore_index=True)
        return AutolabelDataset(df, self.config)

    def clear_cache(self, use_ttl: bool = True):
        """
        Clears the generation and transformation cache from autolabel.
        Args:
            use_ttl: If true, only clears the cache if the ttl has expired.
        """
        self.generation_cache.clear(use_ttl=use_ttl)
        self.transform_cache.clear(use_ttl=use_ttl)

    def get_num_tokens(self, inp: str) -> int:
        """Returns the number of tokens in the prompt"""
        return len(self.confidence_tokenizer.encode(str(inp)))

    def chunk_string(self, inp: str, chunk_size: int) -> List[str]:
        """Chunks the input string into chunks of size chunk_size"""
        tokens = self.confidence_tokenizer.encode(inp)
        chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return [self.confidence_tokenizer.decode(chunk) for chunk in chunks]
