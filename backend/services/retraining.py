from __future__ import annotations

import json
import os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from backend.ml.feature_prep import create_pairwise_features
from backend.ml.model_loader import MODEL_PATH, get_model
from backend.training.data_prep import build_training_dataset, calculate_compatibility
from backend.models import CompatibilityScore, FeedbackStaging, User


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
TRAINING_DATA_FILE = os.path.join(PROJECT_ROOT, "backend", "training", "data", "students.csv")
PREDICT_DATA_FILE = os.path.join(DATA_DIR, "predict_data.csv")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "assignments.json")


def get_active_students_csv() -> str:
    if os.path.exists(PREDICT_DATA_FILE):
        return PREDICT_DATA_FILE
    return TRAINING_DATA_FILE


def load_students_dataframe() -> pd.DataFrame:
    csv_path = get_active_students_csv()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Student data file not found at {csv_path}")
    return pd.read_csv(csv_path)


def sync_users_from_dataframe(db: Session, df_students: pd.DataFrame) -> int:
    student_ids = {int(student_id) for student_id in df_students["student_id"].dropna().tolist()}
    existing_ids = {
        user_id
        for user_id in db.scalars(select(User.id).where(User.id.in_(student_ids))).all()
    }

    new_users = [User(id=student_id) for student_id in sorted(student_ids - existing_ids)]
    if new_users:
        db.add_all(new_users)
        db.flush()
    return len(new_users)


def update_ml_model(feedback_data: list[FeedbackStaging]):
    """Retrain model using synthetic data plus weighted feedback labels."""
    print(f"update_ml_model called with {len(feedback_data)} feedback rows.")

    if not feedback_data:
        return get_model()

    # 1) Build synthetic baseline training set.
    df_students_base = pd.read_csv(TRAINING_DATA_FILE)
    df_pairs_base = build_training_dataset(TRAINING_DATA_FILE)

    merged_base = (
        df_pairs_base
        .merge(df_students_base.add_suffix('_A'), on='student_id_A')
        .merge(df_students_base.add_suffix('_B'), on='student_id_B')
    )
    df_pairs_base = df_pairs_base.copy()
    df_pairs_base['compatibility_score'] = merged_base.apply(calculate_compatibility, axis=1)

    ds_base = create_pairwise_features(df_students_base, df_pairs_base, is_training=True)
    X_base = ds_base.drop(columns=['student_id_A', 'student_id_B', 'compatibility_score'])
    y_base = ds_base['compatibility_score']

    # 2) Build feedback dataset from active student CSV and blend with heuristic prior.
    df_students_active = load_students_dataframe()
    valid_ids = set(df_students_active['student_id'].astype(int).tolist())

    rows = []
    for feedback in feedback_data:
        a = int(min(feedback.user_id, feedback.roommate_id))
        b = int(max(feedback.user_id, feedback.roommate_id))
        if a in valid_ids and b in valid_ids:
            rows.append({
                'student_id_A': a,
                'student_id_B': b,
                'feedback_score': float(feedback.feedback_score),
            })

    if not rows:
        # No valid feedback rows against current active dataset.
        return get_model()

    df_feedback_pairs = pd.DataFrame(rows)
    df_feedback_pairs = (
        df_feedback_pairs
        .groupby(['student_id_A', 'student_id_B'], as_index=False)['feedback_score']
        .mean()
    )

    merged_feedback = (
        df_feedback_pairs
        .merge(df_students_active.add_suffix('_A'), on='student_id_A')
        .merge(df_students_active.add_suffix('_B'), on='student_id_B')
    )
    heuristic_score = merged_feedback.apply(calculate_compatibility, axis=1)

    feedback_lambda = 0.7
    df_feedback_pairs = df_feedback_pairs.copy()
    df_feedback_pairs['compatibility_score'] = (
        (1.0 - feedback_lambda) * heuristic_score + feedback_lambda * df_feedback_pairs['feedback_score']
    )

    ds_feedback = create_pairwise_features(df_students_active, df_feedback_pairs, is_training=True)
    X_feedback = ds_feedback.drop(columns=['student_id_A', 'student_id_B', 'compatibility_score'])
    y_feedback = ds_feedback['compatibility_score']

    # 3) Combine datasets with weighted feedback influence.
    X_train = pd.concat([X_base, X_feedback], ignore_index=True)
    y_train = pd.concat([y_base, y_feedback], ignore_index=True)

    sample_weight = np.concatenate([
        np.ones(len(X_base), dtype=float),
        np.full(len(X_feedback), 2.0, dtype=float),
    ])

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train,
        y_train,
        sample_weight,
        test_size=0.2,
        random_state=42,
    )

    # Baseline model: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_tr, y_tr, sample_weight=w_tr)
    rf_pred = rf_model.predict(X_val)
    rf_mse = mean_squared_error(y_val, rf_pred)
    rf_mae = mean_absolute_error(y_val, rf_pred)

    # Gradient boosting model: prefer XGBoost if available, else sklearn GBR.
    gb_name = "Gradient Boosting"
    try:
        from xgboost import XGBRegressor

        gb_name = "XGBoost"
        gb_model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    except ImportError:
        gb_model = GradientBoostingRegressor(random_state=42)

    gb_model.fit(X_tr, y_tr, sample_weight=w_tr)
    gb_pred = gb_model.predict(X_val)
    gb_mse = mean_squared_error(y_val, gb_pred)
    gb_mae = mean_absolute_error(y_val, gb_pred)

    print(f"Random Forest -> MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
    print(f"{gb_name} -> MSE: {gb_mse:.4f}, MAE: {gb_mae:.4f}")

    if rf_mse <= gb_mse:
        selected_name = "Random Forest"
        selected_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    else:
        selected_name = gb_name
        if gb_name == "XGBoost":
            from xgboost import XGBRegressor

            selected_model = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
        else:
            selected_model = GradientBoostingRegressor(random_state=42)

    # Fit chosen model on full training data before persisting.
    selected_model.fit(X_train, y_train, sample_weight=sample_weight)

    joblib.dump(selected_model, MODEL_PATH)

    # Refresh in-memory singleton so next prediction uses updated model.
    import backend.ml.model_loader as model_loader
    model_loader._rf_model = selected_model

    print(
        f"Feedback-weighted retraining complete with {selected_name}. "
        f"Synthetic rows={len(X_base)}, feedback rows={len(X_feedback)}"
    )
    return selected_model


def get_current_assignment_record(db: Session, student_id: int) -> CompatibilityScore | None:
    return db.scalar(
        select(CompatibilityScore)
        .where(CompatibilityScore.user_id == student_id)
        .order_by(CompatibilityScore.matching_cycle.desc())
    )


def get_feedback_for_cycle(
    db: Session,
    *,
    user_id: int,
    roommate_id: int,
    matching_cycle: int,
) -> FeedbackStaging | None:
    return db.scalar(
        select(FeedbackStaging).where(
            FeedbackStaging.user_id == user_id,
            FeedbackStaging.roommate_id == roommate_id,
            FeedbackStaging.matching_cycle == matching_cycle,
        )
    )


def run_feedback_batch_job(db: Session) -> dict:
    pending_feedback = db.scalars(
        select(FeedbackStaging)
        .where(FeedbackStaging.processed.is_(False))
        .order_by(FeedbackStaging.submitted_at.asc())
    ).all()

    if not pending_feedback:
        return {
            "processed_feedback_count": 0,
            "compatibility_rows_updated": 0,
            "assignments_written": 0,
            "message": "No new feedback found. Batch job skipped.",
        }

    # Train model only; no matching/assignment side effects in batch feedback pipeline.
    update_ml_model(pending_feedback)

    processed_at = datetime.now(timezone.utc)
    for feedback in pending_feedback:
        feedback.processed = True
        feedback.processed_at = processed_at

    db.commit()

    return {
        "processed_feedback_count": len(pending_feedback),
        "compatibility_rows_updated": 0,
        "assignments_written": 0,
        "message": "Feedback model retraining completed successfully.",
    }
