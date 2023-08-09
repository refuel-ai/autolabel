Autolabel supports transformation of the input data! Input datasets are available in many shapes and form(at)s. We help you ingest your data in the format that you want in a way that is most useful for the downstream LLM or labeling task that you have in mind. We have tried to make the transforms performant, configurable and the outputs formatted in a way useful for the LLM.

We support the following transforms -

1. Webpage Transform
2. PDF Transform

We expect this list to grow in the future and need the help of the community to build transforms that work the best for their data. For this, we provide an abstraction that is easy to use. Any new transform just needs to be extend the `BaseTransform` class as penciled down below.

::: src.autolabel.transforms.base.BaseTransform
rendering:
show_root_heading: yes
show_root_full_path: no

## `_apply()` `abstractmethod`
::: src.autolabel.transforms.base.BaseTransform._apply