from typing import Any, Dict, List, NamedTuple, Optional

from pydantic import BaseModel

class LLMAnnotation(BaseModel):
    input_row: Dict
    prompt: str
    label: str
    confidence_score: float


