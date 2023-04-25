# Refuel Oracle API




## __The Oracle Class__ (oracle.py)

::: refuel_oracle.oracle.Oracle.annotate
    rendering:
        show_root_heading: yes
        show_root_full_path: 
        heading_level: 3
Labels data in a given dataset. Output written to CSV file.

Parameters:

|  Name    |      Type     |   Description  | Default | 
| -------- |:-------------:| ---------:|---------:|
| dataset  | string        |     path to a CSV file containing data to be annotated |required|
| max_items |   int        | max number of entries to annotate |  optional  |
| output_name |   string   |  sets the name of the output CSV file  | optional   |
| start_index |   int   |  skips annotating [0, start_index)  | optional   |

Returns:

|  Type    |      Description     |
| -------- |:-------------:|
| None     |                      |



::: refuel_oracle.oracle.Oracle.plan
    rendering:
        show_root_heading: yes
        show_root_full_path: no
        heading_level: 3

Calculates the cost of calling oracle.annotate() on a given dataset

    Total Estimated Cost: $2.062
    Number of examples to label: 2997
    Average cost per example: 0.0006918

Parameters:

|  Name    |      Type     |   Description  | Default | 
| -------- |:-------------:| ---------:|---------:|
| dataset  | string        |     path to a CSV file containing data to be annotated |required|

Returns:

|  Type    |      Description     |
| -------- |:-------------:|
| None     |                      |


## Config
refuel_oracle/config.py


The Config class is used to parse, validate, and store the information contained in the config.json files.

Functions:
* get_provider
* get_model_name
* get_project_name
* get_task_type
* from_json
* get
* keys
* __init__
* _validate
* __getitem__

### validate
validate provider and model names, task, prompt and seed sets, etc
Parameters:
    self
Returns:
    bool : True if valid configuration, False otherwise


### get
Allow for dictionary like access of class members

Parameters:
    key : name of variable within config to fetch, required
    default_value : value to return if key is not found, optional
Returns:
    Value stored with given key


### keys
Returns a list of keys stored within config object, similar to python dictionaries

### get_provider
Returns the name of the provider (i.e. OpenAI, Anthropic, Huggingface) currently being used

Parameters: self
Return type: string

### get_model_name
Returns the name of the language model currently being used for annotation

Parameters: self
Return type: string

### get_project_name
Returns the name of the project, as defined in the json configuration file

Parameters: self
Return type: string

### get_task_type
Returns the task that oracle is currently set to perform (i.e. classification, question answering, entity detection)

Parameters: self
Return type: string

### from_json
parses a given config.json file and returns it in a new Config object

Parameters: json_file_path
Return type: Config



