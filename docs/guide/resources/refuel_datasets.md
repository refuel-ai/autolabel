Autolabel provides some datasets out-of-the-box so you can easily get started with LLM-powered labeling. The full list of datasets is below:

| Dataset        | Task Type             |
| ---------------| ----------------------|
| banking        | Classification        |
| civil_comments | Classification        |
| ledgar         | Classification        |
| walmart_amazon | Entity Matching       |
| company        | Entity Matching       |
| squad_v2       | Question Answering    |
| sciq           | Question Answering    |
| conll2003      | Named Entity Matching |


## Downloading datasets

To download all datasets, navigate to the home directory of `autolabel` and run:
```console
% python get_data.py
Getting data for banking
Getting data for civil_comments
Getting data for ledgar
Getting data for walmart_amazon
Getting data for company
Getting data for squad_v2
Getting data for sciq
Getting data for conll2003
```

To download a specific dataset, such as `squad_v2`, run:
```console
% python get_data.py --dataset squad_v2
Getting data for squad_v2
```