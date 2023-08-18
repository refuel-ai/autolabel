import json
import copy

import pandas as pd
from langchain.schema import Generation

from autolabel import LabelingAgent
from autolabel.schema import RefuelLLMResult

BANKING_CONFIG = json.load(open("tests/assets/banking/config_banking.json", "r"))
WALMART_AMAZON_CONFIG = json.load(
    open("tests/assets/walmart_amazon/config_walmart_amazon.json", "r")
)


def test_classification_generation(mocker):
    mocker.patch(
        "autolabel.models.base.BaseModel.label",
        return_value=RefuelLLMResult(
            generations=[[Generation(text="example\ntest_example")]],
            errors=[None],
            cost=[0.0],
        ),
    )
    config = copy.deepcopy(BANKING_CONFIG)
    config["dataset_generation"] = {
        "num_rows": 1,
        "guidelines": "Example guidelines",
    }
    agent = LabelingAgent(config=BANKING_CONFIG)
    ds = agent.generate_synthetic_dataset()
    label_list = agent.config.labels_list()
    assert (
        (
            ds.df
            == pd.DataFrame.from_dict(
                {"example": ["test_example"] * len(label_list), "label": label_list}
            )
        )
        .all()
        .all()
    )


def test_entity_matching_generation(mocker):
    mocker.patch(
        "autolabel.models.base.BaseModel.label",
        return_value=RefuelLLMResult(
            generations=[
                [
                    Generation(
                        text="Title_entity1,Category_entity1,Brand_entity1,ModelNo_entity1,Price_entity1,Title_entity2,Category_entity2,Brand_entity2,ModelNo_entity2,Price_entity2\ntest_title1,test_category1,test_brand1,test_modelno1,test_price1,test_title2,test_category2,test_brand2,test_modelno2,test_price2"
                    )
                ]
            ],
            errors=[None],
            cost=[0.0],
        ),
    )
    config = copy.deepcopy(WALMART_AMAZON_CONFIG)
    config["dataset_generation"] = {
        "num_rows": 1,
        "guidelines": "Example guidelines",
    }
    agent = LabelingAgent(config=config)
    ds = agent.generate_synthetic_dataset()
    label_list = agent.config.labels_list()
    assert ds.df.shape == (2, 11)
    assert (
        (
            ds.df
            == pd.DataFrame.from_dict(
                {
                    "Title_entity1": ["test_title1"] * len(label_list),
                    "Category_entity1": ["test_category1"] * len(label_list),
                    "Brand_entity1": ["test_brand1"] * len(label_list),
                    "ModelNo_entity1": ["test_modelno1"] * len(label_list),
                    "Price_entity1": ["test_price1"] * len(label_list),
                    "Title_entity2": ["test_title2"] * len(label_list),
                    "Category_entity2": ["test_category2"] * len(label_list),
                    "Brand_entity2": ["test_brand2"] * len(label_list),
                    "ModelNo_entity2": ["test_modelno2"] * len(label_list),
                    "Price_entity2": ["test_price2"] * len(label_list),
                    "label": label_list,
                }
            )
        )
        .all()
        .all()
    )
