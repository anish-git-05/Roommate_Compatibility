from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    user_id: int = Field(..., gt=0)
    roommate_id: int = Field(..., gt=0)
    feedback_score: float = Field(..., ge=0, le=100)


class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    roommate_id: int
    matching_cycle: int
    feedback_score: float
    submitted_at: datetime
    processed: bool

    model_config = {"from_attributes": True}


class BatchJobResult(BaseModel):
    processed_feedback_count: int
    compatibility_rows_updated: int
    assignments_written: int
    message: str
