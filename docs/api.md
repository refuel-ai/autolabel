# AutoLabel API




## __The LabelingAgent Class__

::: autolabel.labeler.LabelingAgent.run
    rendering:
        show_root_heading: yes
        show_root_full_path: 
        heading_level: 3

::: autolabel.labeler.LabelingAgent.plan
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

Example Output of plan()

    Total Estimated Cost: $2.062
    Number of examples to label: 2997
    Average cost per example: 0.0006918


## __DatasetConfig__

The DatasetConfig class is used to parse, validate, and store information about the dataset being annotated.



::: autolabel.configs.dataset_config.DatasetConfig.get_input_columns
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: autolabel.configs.dataset_config.DatasetConfig.get_label_column
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: autolabel.configs.dataset_config.DatasetConfig.get_labels_list
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: autolabel.configs.dataset_config.DatasetConfig.get_seed_examples
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: autolabel.configs.dataset_config.DatasetConfig.get_delimiter
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

## __TaskConfig__

The TaskConfig class is used to parse, validate, and store information about the annotation task being performed (i.e. Classification, Entity Recognition, Question Answering).

::: autolabel.configs.task_config.TaskConfig.get_task_name
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: autolabel.configs.model_config.ModelConfig
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 2