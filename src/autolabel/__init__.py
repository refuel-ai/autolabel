from importlib import metadata

from .configs import AutolabelConfig
from .dataset import AutolabelDataset
from .labeler import LabelingAgent
from .task_chain import TaskChainOrchestrator
from .utils import get_data

try:
    __version__ = metadata.version("refuel-autolabel")
except metadata.PackageNotFoundError:
    # If the package metadata is not available.
    __version__ = ""

__app_name__ = "autolabel"
__author__ = "Refuel.ai"
