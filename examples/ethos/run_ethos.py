import json

from autolabel import LabelingAgent

with open("config_ethos.json", "r") as f:
    config = json.load(f)

agent = LabelingAgent(config=config)

from autolabel import AutolabelDataset

ds = AutolabelDataset("test.csv", config=config)
agent.plan(ds)

ds = agent.run(ds, max_items=5)
