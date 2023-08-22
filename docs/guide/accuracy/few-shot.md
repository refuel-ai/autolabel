# Few-shot Prompting

It has been shown that the specific seed examples used while constructing the prompt have an impact on the performance of the model. Seed examples are the labeled examples which are shown as demonstration to the LLM to help it understand the task better. Optimally selecting the seed examples can help boost performance and save on labeling costs by reducing the context size.

We support the following few-shot example selection techniques:

1. **Fixed** - The same set of seed examples are used for every input data point.
2. **Semantic_similarity** - Embeddings are computed for all the examples in the seed set and a vector similarity search finds the few shot examples which are closest to the input datapoint. Closer datapoints from the seed set can give the model more context on how similar examples have been labeled, helping it improve performance.
3. **Max_marginal_relevance** - Semantic similarity search is used to retrieve a set of candidate examples. Then, a diversity-driven selection strategy is used amongst these candidates to select a final subset of examples that have the most coverage of the initial pool of candidate examples.
4. **Label diversity** - This strategy focuses on ensuring that the few-shot examples selected provide coverage across all the valid output labels.
5. **Label diversity with similarity** - This strategy is a combination of (2) and (4) above - it samples a fixed number of examples per valid label, and within each label it selects the examples that are most similar to the input.

Example:

[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qgfy7odvkCNKrB58ozAF4qXzu10rRGKx#scrollTo=x0js54dB0D7J)

Consider the following labeling runs for a classification task on the banking dataset. There are a total of 1998 items to be labeled and we assume a starting labeled seedset of 200 examples. Here is the config to label this dataset in zero-shot fashion:

```py
config_zero_shot = {
    "task_name": "BankingComplaintsClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": "You are an expert at understanding bank customers support complaints and queries.\nYour job is to correctly classify the provided input example into one of the following categories.\nCategories:\n{labels}",
        "output_guidelines": "You will answer with just the the correct output label and nothing else.",
        "labels": [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            "atm_support",
            "automatic_top_up",
            "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "beneficiary_not_allowed",
            "cancel_transfer",
            "card_about_to_expire",
            "card_acceptance",
            "card_arrival",
            "card_delivery_estimate",
            "card_linking",
            "card_not_working",
            "card_payment_fee_charged",
            "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate",
            "card_swallowed",
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
            "change_pin",
            "compromised_card",
            "contactless_not_working",
            "country_support",
            "declined_card_payment",
            "declined_cash_withdrawal",
            "declined_transfer",
            "direct_debit_payment_not_recognised",
            "disposable_card_limits",
            "edit_personal_details",
            "exchange_charge",
            "exchange_rate",
            "exchange_via_app",
            "extra_charge_on_statement",
            "failed_transfer",
            "fiat_currency_support",
            "get_disposable_virtual_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "order_physical_card",
            "passcode_forgotten",
            "pending_card_payment",
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_transfer",
            "pin_blocked",
            "receiving_money",
            "Refund_not_showing_up",
            "request_refund",
            "reverted_card_payment?",
            "supported_cards_and_currencies",
            "terminate_account",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "top_up_by_cash_or_cheque",
            "top_up_failed",
            "top_up_limits",
            "top_up_reverted",
            "topping_up_by_card",
            "transaction_charged_twice",
            "transfer_fee_charged",
            "transfer_into_account",
            "transfer_not_received_by_recipient",
            "transfer_timing",
            "unable_to_verify_identity",
            "verify_my_identity",
            "verify_source_of_funds",
            "verify_top_up",
            "virtual_card_not_working",
            "visa_or_mastercard",
            "why_verify_identity",
            "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal"
        ],
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
from autolabel import LabelingAgent, AutolabelDataset

agent = LabelingAgent(config=config_zero_shot)
ds = AutolabelDataset('../examples/banking/test.csv', config = config_zero_shot)
labeled_dataset = agent.run(ds)
```

This zero-shot task execution results in an accuracy of 70.19%.

Iterating on this, we compare a fixed few-shot example selection strategy, which randomly chooses k examples from the labeled seedset and appends these same k examples to each prompt for the 1998 items to be labeled. In this case, we use k=10 seed examples per prompt. To use this selection strategy, we need to modify the config:

```py
config_fixed_few_shot = {
    "task_name": "BankingComplaintsClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        ...
        "few_shot_examples": "../examples/banking/seed.csv",
        "few_shot_selection": "fixed",
        "few_shot_num": 10,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
agent = LabelingAgent(config=config_fixed_few_shot)
ds = AutolabelDataset('../examples/banking/test.csv', config = config_fixed_few_shot)
labeled_dataset = agent.run(ds)
```

This leads to an accuracy of 73.16%, an improvement of ~3% over the zero-shot baseline.

Finally, we compare a semantic similarity example selection strategy, which computes a text embedding for each of the 200 labeled seedset examples. Then, for each of the 1998 items to be labeled, we compute a text embedding and find the k most similar examples from the labeled seedset and append those k examples to the prompt for the current example. This leads to custom examples used for each item to be labeled, with the idea being that more similar examples and their corresponding labels may assist the LLM in labeling. Here is the config change to use semantic similarity as the example selection strategy:

```py
config_semantic_similarity = {
    "task_name": "BankingComplaintsClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        ...
        "few_shot_examples": "../examples/banking/seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 10,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
agent = LabelingAgent(config=config_semantic_similarity)
ds = AutolabelDataset('../examples/banking/test.csv', config = config_semantic_similarity)
labeled_dataset = agent.run(ds)
```

With semantic similarity example selection, we obtain a 79.02% accuracy, a significant increase of ~6% over the fixed-shot strategy.

Finally, let's take a look at label diversity set of example selection techniques in action:

```py
config_label_diversity_random = {
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        ...
        "few_shot_examples": "../examples/civil_comments/seed.csv",
        "few_shot_selection": "label_diversity_random",
        "few_shot_num": 5,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
agent = LabelingAgent(config=config_label_diversity_random)
ds = AutolabelDataset('../examples/civil_comments/test.csv', config = config_label_diversity_random)
labeled_dataset = agent.run(ds, max_items=200)
```

```py
config_label_diversity_similarity = {
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        ...
        "few_shot_examples": "../examples/civil_comments/seed.csv",
        "few_shot_selection": "label_diversity_similarity",
        "few_shot_num": 5,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
agent = LabelingAgent(config=config_label_diversity_similarity)
ds = AutolabelDataset('../examples/civil_comments/test.csv', config = config_label_diversity_similarity)
labeled_dataset = agent.run(ds, max_items=200)
```

For this run on the civil comments dataset, label diversity at random achieved 80% accuracy and label diversity with semantic similarity achieved 78% accuracy. For the same subset of data, the use of regular semantic similarity example selection obtained 72% accuracy, making for a significant improvement by using label diversity. 

Label diversity example selection strategies are likely best suited for labeling tasks with a small number of unique labels, which is the case for the civil comments dataset with only 2 labels. This is because equal representation of all the possible labels may be less likely to bias the LLM towards a particular label.

By default, Autolabel uses OpenAI to compute text embeddings for few shot example selection strategies that require them (semantic similarity, max marginal relevance). However, Autolabel also supports alternative embedding model providers such as Google Vertex AI and Huggingface as outlined [here](/guide/llms/embeddings).

It is almost always advisable to use an example selection strategy over a zero-shot approach in your autolabeling workflows, but the choice of which example selection strategy to use is dependent upon the specific labeling task and dataset.
