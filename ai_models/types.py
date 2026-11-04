"""Shared types for the AI layer."""

from dataclasses import asdict, dataclass
from typing import Optional

# Model-level status (what the ensemble policy sees)
AVAILABLE = 'AVAILABLE'            # ran and produced a score
FAILED = 'FAILED'                  # configured but failed to load/run
TIMEOUT = 'TIMEOUT'                # exceeded AI_MODEL_TIMEOUT
DISABLED = 'DISABLED'              # switched off in configuration
NOT_CONFIGURED = 'NOT_CONFIGURED'  # no model file supplied (yet)

LABEL_MANIPULATED = 'MANIPULATED'
LABEL_AUTHENTIC = 'AUTHENTIC'


@dataclass
class ModelOutput:
    """One model's output for one face. `score` is the model's raw manipulation
    score in [0, 1]; it is NOT a calibrated probability unless stated in `score_type`."""
    key: str
    name: str
    status: str
    score: Optional[float] = None
    label: Optional[str] = None
    threshold: Optional[float] = None
    inference_ms: Optional[float] = None
    message: str = ''
    version: str = ''
    weights_sha256: str = ''
    preprocessing_version: str = ''
    score_type: str = 'uncalibrated model output'
    scope: str = 'both'     # faceswap | ai_generated | both: what the model was trained to detect

    def as_dict(self) -> dict:
        return asdict(self)
