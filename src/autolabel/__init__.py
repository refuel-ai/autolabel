from importlib import metadata

from .labeler import LabelingAgent
from .utils import get_data

try:
    __version__ = metadata.version("refuel-autolabel")
except metadata.PackageNotFoundError:
    # If the package metadata is not available.
    __version__ = ""
