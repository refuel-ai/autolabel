from autolabel.tasks import (
    ClassificationTask,
    EntityMatchingTask,
    QuestionAnsweringTask,
)
from autolabel.configs import AutolabelConfig
from autolabel.schema import LLMAnnotation, Metric

from langchain.schema import Generation

BANKING_CONFIG = {
    "task_name": "BankingComplaintsClassification",
    "task_type": "classification",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
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
            "wrong_exchange_rate_for_cash_withdrawal",
        ],
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 10,
        "example_template": "Input: {example}\nOutput: {label}",
    },
}

WALMART_AMAZON_CONFIG = {
    "task_name": "ProductCatalogEntityMatch",
    "task_type": "entity_matching",
    "dataset": {"label_column": "label", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at identifying duplicate products from online product catalogs.\nYour job is to tell if the two given entities are duplicates or not duplicate. Your answer must be from one of the following options:\n{labels}",
        "labels": ["duplicate", "not duplicate"],
        "example_template": "Title of entity1: {Title_entity1}; category of entity1: {Category_entity1}; brand of entity1: {Brand_entity1}; model number of entity1: {ModelNo_entity1}; price of entity1: {Price_entity1}\nTitle of entity2: {Title_entity2}; category of entity2: {Category_entity2}; brand of entity2: {Brand_entity2}; model number of entity2: {ModelNo_entity2}; price of entity2: {Price_entity2}\nDuplicate or not: {label}",
        "few_shot_examples": [
            {
                "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
                "Category_entity1": "printers",
                "Brand_entity1": "lexmark",
                "ModelNo_entity1": "c782u1mg",
                "Price_entity1": "214.88",
                "Title_entity2": "lexmark 18c1428 return program print cartridge black",
                "Category_entity2": "inkjet printer ink",
                "Brand_entity2": "lexmark",
                "ModelNo_entity2": "18c1428",
                "Price_entity2": "19.97",
                "label": "not duplicate",
            },
            {
                "Title_entity1": "edge tech proshot 4gb sdhc class 6 memory card",
                "Category_entity1": "usb drives",
                "Brand_entity1": "edge tech",
                "ModelNo_entity1": "pe209780",
                "Price_entity1": "10.88",
                "Title_entity2": "4gb edge proshot sdhc memory card class6",
                "Category_entity2": "computers accessories",
                "Brand_entity2": "edge",
                "ModelNo_entity2": "nan",
                "Price_entity2": "17.83",
                "label": "duplicate",
            },
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 2,
    },
}

SCIQ_CONFIG = {
    "task_name": "ScienceQuestionAnswering",
    "task_type": "question_answering",
    "dataset": {"label_column": "answer", "delimiter": ","},
    "model": {"provider": "openai", "name": "gpt-3.5-turbo"},
    "prompt": {
        "task_guidelines": "You are an expert at answer science questions. Your job is to answer the given question, using the options provided for each question. Choose the best answer for the question from among the options provided",
        "example_template": "Question: {question}\nOptions: {options}\nAnswer: {answer}",
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 10,
    },
}


def test_classification_construct_prompt():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)
    assert task.config != None

    input = {"example": "Here is an example", "label": "label-true"}
    examples = [
        {"example": "Here is a seed example", "label": "label1"},
        {"example": "Here is another seed example", "label": "label2"},
    ]
    prompt = task.construct_prompt(input, examples)

    assert BANKING_CONFIG["prompt"]["output_guidelines"] in prompt
    assert "\n".join(BANKING_CONFIG["prompt"]["labels"]) in prompt
    assert input["example"] in prompt
    assert input["label"] not in prompt
    for example in examples:
        assert example["example"] in prompt
        assert example["label"] in prompt

    new_config = BANKING_CONFIG.copy()
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = ClassificationTask(config=new_config)
    prompt = task.construct_prompt(input, examples)
    for example in examples:
        assert example["example"] not in prompt
        assert example["label"] not in prompt


def test_classification_parse_llm_response():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)

    input = {"example": "Here is an example", "label": "label-true"}
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nlabel-true"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "label-true"
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False


def test_classification_eval():
    config = AutolabelConfig(BANKING_CONFIG)
    task = ClassificationTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="label1",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label3-wrong",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label4",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="label5-wrong",
        ),
    ]
    gt_labels = ["label1", "label2", "label3", "label4", "label5"]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.5
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4


def test_entity_matching_construct_prompt():
    config = AutolabelConfig(WALMART_AMAZON_CONFIG)
    task = EntityMatchingTask(config=config)
    assert task.config != None

    input = {
        "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
        "Category_entity1": "printers",
        "Brand_entity1": "lexmark",
        "ModelNo_entity1": "c782u1mg",
        "Price_entity1": "214.88",
        "Title_entity2": "lexmark 18c1428 return program print cartridge black",
        "Category_entity2": "inkjet printer ink",
        "Brand_entity2": "lexmark",
        "ModelNo_entity2": "18c1428",
        "Price_entity2": "19.97",
        "label": "not duplicate - 1",
    }
    examples = [
        {
            "Title_entity1": "lexmark extra high yield return pgm print cartridge - magentaplus",
            "Category_entity1": "printers",
            "Brand_entity1": "lexmark",
            "ModelNo_entity1": "c782u1mg",
            "Price_entity1": "214.88",
            "Title_entity2": "lexmark 18c1428 return program print cartridge black",
            "Category_entity2": "inkjet printer ink",
            "Brand_entity2": "lexmark",
            "ModelNo_entity2": "18c1428",
            "Price_entity2": "19.97",
            "label": "not duplicate - 2",
        },
        {
            "Title_entity1": "edge tech proshot 4gb sdhc class 6 memory card",
            "Category_entity1": "usb drives",
            "Brand_entity1": "edge tech",
            "ModelNo_entity1": "pe209780",
            "Price_entity1": "10.88",
            "Title_entity2": "4gb edge proshot sdhc memory card class6",
            "Category_entity2": "computers accessories",
            "Brand_entity2": "edge",
            "ModelNo_entity2": "nan",
            "Price_entity2": "17.83",
            "label": "duplicate - 3",
        },
    ]
    prompt = task.construct_prompt(input, examples)

    assert "\n".join(WALMART_AMAZON_CONFIG["prompt"]["labels"]) in prompt
    assert input["Title_entity1"] in prompt
    assert input["Category_entity1"] in prompt
    assert input["label"] not in prompt
    for example in examples:
        assert example["Title_entity1"] in prompt
        assert example["label"] in prompt

    new_config = WALMART_AMAZON_CONFIG.copy()
    del new_config["prompt"]["few_shot_selection"]
    new_config = AutolabelConfig(new_config)
    task = EntityMatchingTask(config=new_config)
    prompt = task.construct_prompt(input, examples)
    for example in examples:
        assert example["Title_entity1"] not in prompt


def test_entity_matching_parse_llm_response():
    config = AutolabelConfig(WALMART_AMAZON_CONFIG)
    task = EntityMatchingTask(config=config)

    input = {
        "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
        "Category_entity1": "printers",
        "Brand_entity1": "lexmark",
        "ModelNo_entity1": "c782u1mg",
        "Price_entity1": "214.88",
        "Title_entity2": "lexmark 18c1428 return program print cartridge black",
        "Category_entity2": "inkjet printer ink",
        "Brand_entity2": "lexmark",
        "ModelNo_entity2": "18c1428",
        "Price_entity2": "19.97",
        "label": "not duplicate - 1",
    }
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nnot duplicate - 1"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "not duplicate - 1"
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False


def test_entity_matching_eval():
    config = AutolabelConfig(WALMART_AMAZON_CONFIG)
    task = EntityMatchingTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="duplicate",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="not duplicate",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="not duplicate",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="duplicate",
        ),
    ]
    gt_labels = [
        "duplicate",
        "not duplicate",
        "not duplicate",
        "not duplicate",
        "not duplicate",
    ]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.75
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4


def question_answering_construct_prompt():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = QuestionAnsweringTask(config=config)
    assert task.config != None

    input = {
        "question": "What is the capital of France?",
        "options": "[Paris, London, Berlin]",
        "answer": "Paris-label",
    }
    examples = [
        {
            "question": "What is the capital of India?",
            "options": "[Delhi, Mumbai, Bangalore]",
            "answer": "Delhi-label",
        },
        {
            "question": "What is the capital of USA?",
            "options": "[New York, Washington DC, Los Angeles]",
            "answer": "Washington DC-label",
        },
    ]
    prompt = task.construct_prompt(input, examples)

    assert input["question"] in prompt
    assert input["options"] in prompt
    assert input["answer"] not in prompt
    for example in examples:
        assert example["question"] in prompt
        assert example["options"] in prompt
        assert example["answer"] in prompt


def question_answering_parse_llm_response():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = QuestionAnsweringTask(config=config)

    input = {
        "question": "What is the capital of France?",
        "options": "[Paris, London, Berlin]",
        "answer": "Paris-label",
    }
    prompt = "This is a prompt"

    label = "This is the thing we want to test.\nParis-label"
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == "Paris-label"
    assert parsed.successfully_labeled == True
    assert parsed.raw_response == label

    label = ""
    response = Generation(text=label)
    parsed = task.parse_llm_response(response, input, prompt)
    assert parsed.label == task.NULL_LABEL_TOKEN
    assert parsed.successfully_labeled == False


def test_question_answering_eval():
    config = AutolabelConfig(SCIQ_CONFIG)
    task = QuestionAnsweringTask(config=config)

    llm_labels = [
        LLMAnnotation(
            successfully_labeled=True,
            label="Delhi",
        ),
        LLMAnnotation(
            successfully_labeled=False,
            label=task.NULL_LABEL_TOKEN,
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Paris",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Bangalore",
        ),
        LLMAnnotation(
            successfully_labeled=True,
            label="Delhi",
        ),
    ]
    gt_labels = [
        "Delhi",
        "Washington DC",
        "Paris",
        "Bangalore",
        "Washington DC",
    ]
    eval = task.eval(llm_labels, gt_labels)

    for metric in eval:
        if metric.metric_type == Metric.ACCURACY:
            assert metric.value[0] == 0.75
        elif metric.metric_type == Metric.COMPLETION_RATE:
            assert metric.value[0] == 0.8
        elif metric.metric_type == Metric.SUPPORT:
            assert metric.value[0] == 4
