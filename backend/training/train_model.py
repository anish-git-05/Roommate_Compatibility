# backend/training/train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from backend.ml.feature_prep import create_pairwise_features
from backend.training.data_prep import build_training_dataset, calculate_compatibility


def train_and_evaluate(df_students, df_pairs, is_training=True):
    print("Engineering features for training data...")
    dataset = create_pairwise_features(df_students, df_pairs, is_training=True )

    # Fallback for runtime environments where feature_prep returns features without target.
    if 'compatibility_score' not in dataset.columns:
        students_A = df_students.add_suffix('_A')
        students_B = df_students.add_suffix('_B')
        merged = df_pairs.merge(students_A, on='student_id_A').merge(students_B, on='student_id_B')
        dataset['compatibility_score'] = merged.apply(calculate_compatibility, axis=1)

    X = dataset.drop(columns=['student_id_A', 'student_id_B', 'compatibility_score'])
    y = dataset['compatibility_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training candidate regressors...")

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_test, rf_predictions)

    gb_name = "Gradient Boosting"
    try:
        from xgboost import XGBRegressor
        gb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
        )
        gb_name = "XGBoost"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        gb_model = GradientBoostingRegressor(random_state=42)

    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_predictions)
    gb_mae = mean_absolute_error(y_test, gb_predictions)
    gb_rmse = np.sqrt(gb_mse)
    gb_r2 = r2_score(y_test, gb_predictions)

    if rf_mse <= gb_mse:
        model = rf_model
        selected_model_name = "Random Forest"
        selected_mae, selected_rmse, selected_r2 = rf_mae, rf_rmse, rf_r2
    else:
        model = gb_model
        selected_model_name = gb_name
        selected_mae, selected_rmse, selected_r2 = gb_mae, gb_rmse, gb_r2

    print("\n--- Model Evaluation (Validation Split) ---")
    print(f"Random Forest -> MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
    print(f"{gb_name} -> MSE: {gb_mse:.4f}, MAE: {gb_mae:.4f}, RMSE: {gb_rmse:.4f}, R²: {gb_r2:.4f}")
    print(f"Selected model: {selected_model_name} (lower MSE)")
    print("\n--- Selected Model Metrics ---")
    print(f"MAE:  {selected_mae:.2f}")
    print(f"RMSE: {selected_rmse:.2f}")
    print(f"R²:   {selected_r2:.2f}\n")

    model_path = os.path.join(os.path.dirname(__file__), '../model/rf_model.pkl')
    joblib.dump(model, model_path)

    print(f"Model successfully saved to {model_path}")

    return model

# ---------------------------------------------------------
# MAIN BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    df_pairs = build_training_dataset("backend/training/data/students.csv")
    # Separate students (for feature engineering) and pairs
    df_students = pd.read_csv("backend/training/data/students.csv")
    trained_model = train_and_evaluate(df_students, df_pairs, is_training=True)