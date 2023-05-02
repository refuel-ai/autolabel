# Refuel Oracle API




## __The Oracle Class__

::: refuel_oracle.oracle.Oracle.annotate
    rendering:
        show_root_heading: yes
        show_root_full_path: 
        heading_level: 3

::: refuel_oracle.oracle.Oracle.plan
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


::: refuel_oracle.dataset_config.DatasetConfig.from_json
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.get
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.get_input_columns
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.get_label_column
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.get_labels_list
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.get_seed_examples
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.get_delimiter
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.dataset_config.DatasetConfig.keys
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

## __TaskConfig__

The TaskConfig class is used to parse, validate, and store information about the annotation task being performed (i.e. Classification, Entity Recognition, Question Answering).

::: refuel_oracle.task_config.TaskConfig._validate
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

::: refuel_oracle.task_config.TaskConfig.from_json
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3


::: refuel_oracle.models.config.ModelConfig
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 2