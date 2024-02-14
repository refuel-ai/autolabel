from importlib import metadata

from .labeler import LabelingAgent
from .utils import get_data
from .dataset import AutolabelDataset
from .configs import AutolabelConfig
from .task_chain import TaskChainOrchestrator

try:
    __version__ = metadata.version("refuel-autolabel")
except metadata.PackageNotFoundError:
    # If the package metadata is not available.
    __version__ = ""

__app_name__ = "autolabel"
__author__ = "Refuel.ai"
