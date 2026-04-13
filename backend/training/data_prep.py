# training/data_prep.py
import pandas as pd
import numpy as np
import random
from itertools import combinations

def calculate_compatibility(row):
    score = 50  # Start with a neutral score

    s1, s2 = row['sleep_time_A'], row['sleep_time_B']
    sleep_diff = min(abs(s1 - s2), 24 - abs(s1 - s2))
    if sleep_diff <= 2:
        score += 10
    elif sleep_diff > 4:
        score -= 15

    w1, w2 = row['wake_up_time_A'], row['wake_up_time_B']
    wake_diff = min(abs(w1 - w2), 24 - abs(w1 - w2))
    if wake_diff <= 2:
        score += 6

    clean_diff = abs(row['cleanliness_score_A'] - row['cleanliness_score_B'])
    if clean_diff <= 2:
        score += 8
    elif clean_diff > 5:
        score -= 15

    if abs(row['room_organization_level_A'] - row['room_organization_level_B']) <= 2:
        score += 6

    if row['food_preference_A'] == row['food_preference_B']:
        score += 5
    if row['fan_or_cooler_preference_A'] == row['fan_or_cooler_preference_B']:
        score += 5
    if row['study_noise_preference_A'] == row['study_noise_preference_B']:
        score += 5

    if row['study_habit_A'] == row['study_habit_B']:
        score += 5
    if abs(row['daily_study_hours_A'] - row['daily_study_hours_B']) <= 2:
        score += 5

    intro_diff = abs(row['introvert_extrovert_score_A'] - row['introvert_extrovert_score_B'])
    if intro_diff <= 2:
        score += 6
    elif intro_diff >= 7:
        score -= 12

    if row['social_frequency_A'] == row['social_frequency_B']:
        score += 5
    if row['language_A'] == row['language_B']:
        score += 5

    if row['department_A'] == row['department_B']:
        score += 3
    if row['career_interest_A'] == row['career_interest_B']:
        score += 3
    if row['cult_sports_A'] == row['cult_sports_B']:
        score += 3
    if abs(row['room_stay_duration_A'] - row['room_stay_duration_B']) <= 3:
        score += 3

    if row['smoking_drinking_A'] != row['smoking_drinking_B']:
        score -= 30

    nt1, nt2 = row['noise_tolerance_A'], row['noise_tolerance_B']
    np1, np2 = row['study_noise_preference_A'], row['study_noise_preference_B']
    if (nt1 <= 3 and np2 == 'cafe_noise') or (nt2 <= 3 and np1 == 'cafe_noise'):
        score -= 15

    return max(-20, min(100, score))

def build_training_dataset(csv_path):
    students_df = pd.read_csv(csv_path)
    pairs_list = []
    # 1. Generate sampled pairs for training
    grouped = students_df.groupby(['gender', 'year_of_study'])
    for _, group in grouped:
        ids = group['student_id'].tolist()
        group_pairs = list(combinations(ids, 2))
        n_samples = min(3750, len(group_pairs))
        sampled_pairs = random.sample(group_pairs, n_samples)
        pairs_list.extend(sampled_pairs)
    pairs_df = pd.DataFrame(pairs_list, columns=['student_id_A', 'student_id_B'])
    # Feature engineering + target generation now happens in ml/feature_prep.py
    # so this function only prepares candidate training pairs.
    return pairs_df
    return pairs_df