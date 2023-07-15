from typing import Dict, Union

from .config import AutolabelConfig


class UnvalidatedAutolabelConfig(AutolabelConfig):
    """Unvalidated Autolabel Config for uses where validating is not necessary"""

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(config)

    def _validate(self) -> bool:
        return True
