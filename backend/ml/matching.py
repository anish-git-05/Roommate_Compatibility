# backend/ml/matching.py
import pandas as pd
import networkx as nx
import itertools

from sqlalchemy import func, select
from backend.ml.feature_prep import create_pairwise_features
import os
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.models import CompatibilityScore
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data")
TRAINING_DATA_FILE = os.path.join(PROJECT_ROOT, "backend", "training", "data", "students.csv")
PREDICT_DATA_FILE = os.path.join(DATA_DIR, "predict_data.csv")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "assignments.json")


def find_optimal_roommates(df_students, model):

    print("Starting grouped roommate matching...")

    assignments = []
    total_system_score = 0
    total_pairs = 0
    unmatched_students = []

    # ---------------------------------------------------------
    # GROUP STUDENTS BY (gender, year_of_study)
    # ---------------------------------------------------------
    grouped = df_students.groupby(['gender', 'year_of_study'])

    for (gender, year), group_df in grouped:

        print(f"Processing group: Gender={gender}, Year={year}")

        student_ids = group_df['student_id'].tolist()

        # If group has less than 2 students → unmatched
        if len(student_ids) < 2:
            unmatched_students.extend(student_ids)
            continue

        # -----------------------------------------------------
        # Generate all possible pairs inside the group
        # -----------------------------------------------------
        all_pairs = list(itertools.combinations(student_ids, 2))
        df_pairs = pd.DataFrame(all_pairs, columns=['student_id_A', 'student_id_B'])

        # Feature engineering
        X_predict = create_pairwise_features(df_students, df_pairs, is_training=False)
        features_only = X_predict.drop(columns=['student_id_A', 'student_id_B'])
        print("Predicting compatibility scores...")
        scores = model.predict(features_only)

        df_pairs['predicted_score'] = scores

        # -----------------------------------------------------
        # Build graph for Maximum Weight Matching
        # -----------------------------------------------------
        G = nx.Graph()

        for _, row in df_pairs.iterrows():
            G.add_edge(
                int(row['student_id_A']),
                int(row['student_id_B']),
                weight=row['predicted_score']
            )

        matching = nx.max_weight_matching(G, maxcardinality=True)

        matched_students = set()

        # -----------------------------------------------------
        # Store matched pairs
        # -----------------------------------------------------
        for student_a, student_b in matching:

            score = G[student_a][student_b]['weight']

            assignments.append({
                "student_1": student_a,
                "student_2": student_b,
                "compatibility_score": round(score, 2)
            })

            matched_students.add(student_a)
            matched_students.add(student_b)

            total_system_score += score
            total_pairs += 1

        # -----------------------------------------------------
        # Detect unmatched students in this group
        # -----------------------------------------------------
        for s in student_ids:
            if s not in matched_students:
                unmatched_students.append(s)

    # ---------------------------------------------------------
    # FORMAT UNMATCHED STUDENTS
    # ---------------------------------------------------------
    for student in unmatched_students:
        assignments.append({
            "student_1": student,
            "student_2": None,
            "compatibility_score": None,
            "message": "No roommate assigned"
        })

    avg_score = total_system_score / total_pairs if total_pairs else 0

    print("DEBUG: Group-based matching completed")
    print(f"DEBUG: Average hostel score = {round(avg_score,2)}")

    return {
        "average_hostel_score": round(avg_score, 2),
        "total_pairs": total_pairs,
        "unmatched_students": len(unmatched_students),
        "assignments": assignments
    }



def _write_assignments_file(matching_results: dict, matching_cycle: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = dict(matching_results)
    payload["matching_cycle"] = matching_cycle
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(ASSIGNMENTS_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=4)


def _next_matching_cycle(db: Session) -> int:
    latest_cycle = db.scalar(select(func.max(CompatibilityScore.matching_cycle)))
    return int(latest_cycle or 0) + 1


def _store_assignment_rows(
    db: Session,
    *,
    matching_results: dict,
    matching_cycle: int,
    source: str,
) -> int:
    rows_written = 0
    now_utc = datetime.now(timezone.utc)

    for assignment in matching_results.get("assignments", []):
        student_1 = int(assignment["student_1"])
        student_2 = assignment.get("student_2")
        roommate_id = int(student_2) if student_2 is not None else None
        score = assignment.get("compatibility_score")
        score_value = float(score) if score is not None else None

        db.add(
            CompatibilityScore(
                user_id=student_1,
                roommate_id=roommate_id,
                matching_cycle=matching_cycle,
                compatibility_score=score_value,
                source=source,
                updated_at=now_utc,
            )
        )
        rows_written += 1

        # Persist reverse record so both users can submit feedback and query assignments.
        if roommate_id is not None:
            db.add(
                CompatibilityScore(
                    user_id=roommate_id,
                    roommate_id=student_1,
                    matching_cycle=matching_cycle,
                    compatibility_score=score_value,
                    source=source,
                    updated_at=now_utc,
                )
            )
            rows_written += 1

    return rows_written


def persist_matching_results(db: Session, matching_results: dict, *, source: str) -> dict:
    matching_cycle = _next_matching_cycle(db)
    rows_written = _store_assignment_rows(
        db,
        matching_results=matching_results,
        matching_cycle=matching_cycle,
        source=source,
    )
    _write_assignments_file(matching_results, matching_cycle)

    return {
        "matching_cycle": matching_cycle,
        "compatibility_rows_updated": rows_written,
        "assignments_written": len(matching_results.get("assignments", [])),
    }
