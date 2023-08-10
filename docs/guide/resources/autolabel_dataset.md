## Autolabel Dataset

Autolabel interacts primarily with dataset objects. These dataset objects are the input and the output for every agent function. `agent.run`, `agent.plan` and `agent.transform` all accept AutolabelDataset as an input and output an Autolabel Dataset. Use this object to talk to autolabel and run evaluations, transformations as well as understand the labels that a model outputs. We provide utility functions to help with understanding where the labeling process can be improved.

::: src.autolabel.dataset.dataset.AutolabelDataset
rendering:
show_root_heading: yes
show_root_full_path: no
