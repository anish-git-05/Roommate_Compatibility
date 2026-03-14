# backend/training/train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from backend.ml.feature_prep import create_pairwise_features
from backend.training.data_prep import build_training_dataset


def train_and_evaluate(df_students, df_pairs):
    print("Engineering features for training data...")
    dataset = create_pairwise_features(df_students, df_pairs)

    X = dataset.drop(columns=['student_id_A', 'student_id_B', 'compatibility_score'])
    y = dataset['compatibility_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}\n")

    model_path = os.path.join(os.path.dirname(__file__), '../model/rf_model.pkl')
    joblib.dump(model, model_path)

    print(f"Model successfully saved to {model_path}")

    return model


# ---------------------------------------------------------
# MAIN BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    pairs_merged = build_training_dataset("backend/data/students.csv")
    # Separate students (for feature engineering) and pairs
    df_students = pd.read_csv("backend/data/students.csv")
    df_pairs = pairs_merged[['student_id_A', 'student_id_B', 'target_compatibility_score']].rename(
        columns={'target_compatibility_score': 'compatibility_score'}
    )
    
    trained_model = train_and_evaluate(df_students, df_pairs)