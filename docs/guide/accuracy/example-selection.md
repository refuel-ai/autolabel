It has been shown that the specific seed examples used while constructing the prompt have an impact on the performance of the model. Seed examples are the labeled dataset examples which the model is shown to help it understand the task better. Selecting the seed example per datapoint can help boost performance. We support the following example selection techniques:

1. Fixed_few_shot - The same set of seed examples are used for every input data point.
2. Semantic_similarity - Language embeddings are computed for all the examples in the seed set and a vector similarity search finds the few shot examples which are closest to the input datapoint. The hope is that closer datapoints from the seed set will give the model more context on how similar examples have been labeled, helping it improve performance.
3. Max_marginal_relevance - Semantic similarity search is used to retrieve a set of candidate examples. Then, a diversity-driven selection strategy is used amongst these candidates to select a final subset of examples that have the most coverage of the initial pool of candidate examples.

Example:
Consider the following labeling runs for a classification task on the banking dataset. There are a total of 1998 items to be labeled and we assume a starting labeled seedset of 200 examples. Here is the config to label this dataset in zero-shot fashion:

```json
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
from autolabel import LabelingAgent

agent = LabelingAgent(config=config_zero_shot)
labels, df, metrics_list = agent.run('../data/banking_test.csv')
```

This zero-shot task execution results in an accuracy of 70.19%. 

Iterating on this, we compare a fixed few-shot example selection strategy, which randomly chooses k examples from the labeled seedset and appends these same k examples to each prompt for the 1998 items to be labeled. In this case, we use k=10 seed examples per prompt. To use this selection strategy, we need to modify the config:

```json
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
        "few_shot_examples": {"../data/banking_seed.csv"},
        "few_shot_selection": "fixed",
        "few_shot_num": 10,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
from autolabel import LabelingAgent

agent = LabelingAgent(config=config_fixed_few_shot)
labels, df, metrics_list = agent.run('../data/banking_test.csv')
```

This leads to an accuracy of 73.16%, an improvement of ~3% over the zero-shot baseline.

Finally, we compare a semantic similarity example selection strategy, which computes a text embedding for each of the 200 labeled seedset examples. Then, for each of the 1998 items to be labeled, we compute a text embedding and find the k most similar examples from the labeled seedset and append those k examples to the prompt for the current example. This leads to custom examples used for each item to be labeled, with the idea being that more similar examples and their corresponding labels may assist the LLM in labeling. Here is the config change to use semantic similarity as the example selection strategy:

```json
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
        "few_shot_examples": {"../data/banking_seed.csv"},
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 10,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

```py
from autolabel import LabelingAgent

agent = LabelingAgent(config=config_semantic_similarity)
labels, df, metrics_list = agent.run('../data/banking_test.csv')
```

With semantic similarity example selectiom, we obtain a 79.02% accuracy, a significant increase of ~6% over the fixed-shot strategy.

It is almost always advisable to use an example selection strategy over a zero-shot approach in your autolabeling workflows, but the choice of which example selection strategy to use is dependent upon the specific labeling task and dataset. In some cases, there may not be sufficient labeled data to use as a seedset for semantic similarity and so fixed few-shot may be ideal as it requires a small fixed number of labeled examples. In other cases, a semantic similarity example selection strategy may be necessary for labeling tasks that are more complex and require more similar labeled references for the LLM.