from typing import Any, Dict, Optional

from pydantic import BaseModel

class LLMAnnotation(BaseModel):
    prompt: str
    successfully_labeled: bool
    label: str
    confidence_score: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None


