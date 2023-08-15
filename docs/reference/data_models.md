The Data Model classes are used to save the progress of AutoLabel jobs in an SQL database.

Saved data is stored in .autolabel.db

Every Data Model class implements its own "get" and "create" methods for accessing this saved data.

::: src.autolabel.data_models.annotation.AnnotationModel
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.data_models.generation_cache.GenerationCacheEntryModel
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.data_models.transform_cache.TransformCacheEntryModel
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.data_models.dataset.DatasetModel
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.data_models.task.TaskModel
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.data_models.task_run.TaskRunModel
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.database.state_manager.StateManager
rendering:
show_root_heading: yes
show_root_full_path: no

::: src.autolabel.database.engine.create_db_engine
rendering:
show_root_heading: yes
show_root_full_path: no
