# State Management

Labeling a large dataset can take some time and if you're running the task on a Jupyter notebook and your machine decides to sleep during the time, it could be really frustrating. (we've been there! :crying_cat_face:).

Therefore, we periodically save the progress of the labeling task in a SQLite database, so if the task is interrupted, you can resume it from where you left off.

## Task Run State

When a labeling task is triggered, a task run entry gets initialized inside the database. We maintain the dataset index till where the labels have been computed. After every small chunk (size 5) of data gets labeled, the dataset index gets updated and the labels are persisted.

In case the labeling process get interrupted/terminated and you trigger the task with the same parameters again, the library first checks for a previous instance of the same task.

If there was an incomplete task present, you would be prompted with details of the previous run and asked to resume the task.
If you choose to resume the previous task, it gets loaded into the memory and resumed from previous state otherwise the previous entry gets deleted.

## Deep Dive

You'd likely never have to interact with the database directly but in case you wish to look at the state of the database, you can do that using any CLI or GUI that supports SQL.
The database is saved in the same directory from where you run the LabelingAgent notebook and is named `.autolabel.db`.

We have the following tables:

- `datasets`: Stores the dataset file information
- `tasks`: Stores the labeling task attributes
- `task_runs`: Stores the current state of a labeling task run
- `annotations`: Stores the LLM annotation corresponding to the task run
- `generation_cache`: Cache for the LLM calls
