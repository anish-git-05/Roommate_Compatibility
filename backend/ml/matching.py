# backend/ml/matching.py
import pandas as pd
import networkx as nx
import itertools
from backend.ml.feature_prep import create_pairwise_features

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
        X_predict = create_pairwise_features(group_df, df_pairs)

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