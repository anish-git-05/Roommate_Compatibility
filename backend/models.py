from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    feedback_given: Mapped[list["FeedbackStaging"]] = relationship(
        "FeedbackStaging",
        back_populates="user",
        foreign_keys="FeedbackStaging.user_id",
    )
    feedback_received: Mapped[list["FeedbackStaging"]] = relationship(
        "FeedbackStaging",
        back_populates="roommate",
        foreign_keys="FeedbackStaging.roommate_id",
    )
    compatibility_scores: Mapped[list["CompatibilityScore"]] = relationship(
        "CompatibilityScore",
        back_populates="user",
        foreign_keys="CompatibilityScore.user_id",
    )
    roommate_scores: Mapped[list["CompatibilityScore"]] = relationship(
        "CompatibilityScore",
        back_populates="roommate",
        foreign_keys="CompatibilityScore.roommate_id",
    )


class FeedbackStaging(Base):
    __tablename__ = "feedback_staging"
    __table_args__ = (
        UniqueConstraint("user_id", "roommate_id", "matching_cycle", name="uq_feedback_once_per_cycle"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    roommate_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    matching_cycle: Mapped[int] = mapped_column(Integer, nullable=False, index=True, default=1)
    feedback_score: Mapped[float] = mapped_column(Float, nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id], back_populates="feedback_given")
    roommate: Mapped["User"] = relationship("User", foreign_keys=[roommate_id], back_populates="feedback_received")


class CompatibilityScore(Base):
    __tablename__ = "compatibility_scores"
    __table_args__ = (
        UniqueConstraint("user_id", "matching_cycle", name="uq_compatibility_user_cycle"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    roommate_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    matching_cycle: Mapped[int] = mapped_column(Integer, nullable=False, index=True, default=1)
    compatibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="batch_retraining", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id], back_populates="compatibility_scores")
    roommate: Mapped[User | None] = relationship(
        "User",
        foreign_keys=[roommate_id],
        back_populates="roommate_scores",
    )
