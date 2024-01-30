Autolabel provides datasets out-of-the-box so you can easily get started with LLM-powered labeling. The full list of datasets is below:

| Dataset        | Task Type             |
| ---------------| ----------------------|
| banking        | Classification        |
| civil_comments | Classification        |
| ledgar         | Classification        |
| movie_reviews  | Classification        |
| walmart_amazon | Entity Matching       |
| company        | Entity Matching       |
| squad_v2       | Question Answering    |
| sciq           | Question Answering    |
| conll2003      | Named Entity Matching |


## Downloading any dataset

To download a specific dataset, such as `squad_v2`, run:
```python
from autolabel import get_data

get_data('civil_comments')
> Downloading seed example dataset to "seed.csv"...
> 100% [..............................................................................] 65757 / 65757

> Downloading test dataset to "test.csv"...
> 100% [............................................................................] 610663 / 610663
```