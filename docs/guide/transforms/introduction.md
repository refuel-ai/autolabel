Autolabel supports transformation of the input data! Input datasets are available in many shapes and form(at)s. We help you ingest your data in the format that you want in a way that is most useful for the downstream LLM or labeling task that you have in mind. We have tried to make the transforms performant, configurable and the outputs formatted in a way useful for the LLM.

## Example
Here we will show you how to run an example transform. We will use the Webpage Transform to ingest national park websites and label the state that every national park belongs to. You can find a Jupyter notebook with code that you can run on your own [here](https://github.com/refuel-ai/autolabel/blob/main/examples/transforms/example_webpage_transform.ipynb)

### Changes to config

```json
{
    "task_name": "NationalPark",
    "task_type": "question_answering",
    "dataset": {
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "transforms": [{
        "name": "webpage_transform",
        "params": {
            "url_column": "url"
        },
        "output_columns": {
            "content_column": "content"
        }
    }],
    "prompt": {
        "task_guidelines": "You are an expert at understanding websites of national parks. You will be given a webpage about a national park. Answer with the US State that the national park is located in.",
        "output_guidelines": "Answer in one word the state that the national park is located in.",
        "example_template": "Content of wikipedia page: {content}\State:",
    }
}
```

Notice the `transforms` key in the config. This is where we define our transforms. Notice that this is a list meaning we can define multiple transforms here. Every element of this list is a transform. A transform is a json requiring 3 inputs -
1. `name`: This tells the agent which transform needs to be loaded. Here we are using the webpage transform.
2. `params`: This is the set of parameters that will be passed to the transform. Read the documentation of the separate transform to see what params can be passed to the transform here. Here we pass the url_column, i.e the column containing the webpages that need to be loaded.
3. `output_columns`: Each transform can define multiple outputs. In this dictionary we map the output we need, in case `content_column` to the name of the column in the output dataset in which we want to populate this.

### Running the transform
```
from autolabel import LabelingAgent, AutolabelDataset
agent = LabelingAgent(config)
ds = agent.transform(ds)
```

This runs the transformation. We will see the content in the correct column. Access this using `ds.df` in the AutolabelDataset.

### Running the labeling job
```
ds = agent.run(ds)
```

Simply run the labeling job on the transformed dataset. This will extract the state of the national park from each webpage.

<figure markdown>
  ![Transformation Labeling Run](/assets/transform_output.png){ width="600" }
  <figcaption>Output of the transformation labeling run</figcaption>
</figure>

## Custom Transforms

We support the following transforms -

1. Webpage Transform
2. PDF Transform

We expect this list to grow in the future and need the help of the community to build transforms that work the best for their data. For this, we provide an abstraction that is easy to use. Any new transform just needs to be extend the `BaseTransform` class as penciled down below.

::: src.autolabel.transforms.base
rendering:
show_root_heading: yes
show_root_full_path: no


### `_apply()` `abstractmethod`
::: src.autolabel.transforms.base.BaseTransform._apply