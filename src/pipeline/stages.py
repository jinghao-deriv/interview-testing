"""Pipeline stage constants and state machine enforcement."""

from enum import Enum
from typing import List


class PipelineStage(str, Enum):
    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    TICKETS_PREPROCESSED = "TICKETS_PREPROCESSED"
    CANDIDATE_ARTICLES_RETRIEVED = "CANDIDATE_ARTICLES_RETRIEVED"
    TICKETS_CLASSIFIED = "TICKETS_CLASSIFIED"
    DECISIONS_COMPUTED = "DECISIONS_COMPUTED"
    REPLIES_DRAFTED = "REPLIES_DRAFTED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.INIT,
    PipelineStage.INPUTS_LOADED,
    PipelineStage.TICKETS_PREPROCESSED,
    PipelineStage.CANDIDATE_ARTICLES_RETRIEVED,
    PipelineStage.TICKETS_CLASSIFIED,
    PipelineStage.DECISIONS_COMPUTED,
    PipelineStage.REPLIES_DRAFTED,
    PipelineStage.VALIDATION_COMPLETE,
    PipelineStage.RESULTS_FINALISED,
]

_STAGE_INDEX = {stage: i for i, stage in enumerate(STAGE_ORDER)}


class PipelineState:
    """Tracks current pipeline stage and enforces legal forward-only transitions."""

    def __init__(self) -> None:
        self.current: PipelineStage = PipelineStage.INIT

    def advance(self, expected_next: PipelineStage) -> None:
        """Advance to expected_next; raises if transition is illegal."""
        current_idx = _STAGE_INDEX[self.current]
        expected_idx = _STAGE_INDEX[expected_next]
        if expected_idx != current_idx + 1:
            raise RuntimeError(
                f"Illegal stage transition: {self.current} -> {expected_next}. "
                f"Expected {STAGE_ORDER[current_idx + 1]}."
            )
        self.current = expected_next
        print(f"[pipeline] Stage reached: {self.current.value}")

    def require(self, minimum: PipelineStage) -> None:
        """Assert that at least `minimum` stage has been reached."""
        if _STAGE_INDEX[self.current] < _STAGE_INDEX[minimum]:
            raise RuntimeError(
                f"Stage {minimum.value} not yet reached. Current: {self.current.value}"
            )
