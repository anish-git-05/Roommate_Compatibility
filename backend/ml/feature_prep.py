# backend/ml/feature_prep.py
import pandas as pd
import numpy as np

categorical_cols = [
    'gender', 'department', 'study_noise_preference', 'fan_or_cooler_preference',
    'study_habit', 'food_preference', 'exam_preparation_style', 'social_frequency',
    'relationship_status', 'career_interest', 'cult_sports', 'language'
]

numerical_cols = [
    'year_of_study', 'sleep_time', 'wake_up_time', 'alarm_usage', 
    'morning_productivity', 'night_productivity', 'cleanliness_score', 
    'room_organization_level', 'noise_tolerance', 'daily_study_hours', 
    'introvert_extrovert_score', 'smoking_drinking', 'workout', 
    'gaming', 'anime', 'room_stay_duration'
]

def create_pairwise_features(df_students, df_pairs):
    """Merges student profiles and computes absolute differences & similarities."""

    # Build explicit A/B views so downstream feature code can rely on _A/_B names.
    students_A = df_students.add_suffix('_A')
    students_B = df_students.add_suffix('_B')

    df = df_pairs.merge(students_A, on='student_id_A')
    df = df.merge(students_B, on='student_id_B')

    features = pd.DataFrame()
    features['student_id_A'] = df['student_id_A']
    features['student_id_B'] = df['student_id_B']

    # Numerical differences (Absolute difference)
    for col in numerical_cols:
        features[f'diff_{col}'] = np.abs(df[f'{col}_A'] - df[f'{col}_B'])

    # Categorical similarities (1 if same, 0 if different)
    for col in categorical_cols:
        features[f'sim_{col}'] = (df[f'{col}_A'] == df[f'{col}_B']).astype(int)

    # Add target variable if it exists (for training phase)
    if 'compatibility_score' in df.columns:
        features['compatibility_score'] = df['compatibility_score']

    return features