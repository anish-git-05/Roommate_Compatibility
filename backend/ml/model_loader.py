# backend/ml/model_loader.py
import joblib
import os

# Training saves the model under backend/model/rf_model.pkl.
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/rf_model.pkl'))
TRAINING_CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../training/data/students.csv')
)

# Global variable to hold the model in memory
_rf_model = None


def _train_model_if_missing():
    """Train and persist the model once if no model file exists yet."""
    if os.path.exists(MODEL_PATH):
        return

    if not os.path.exists(TRAINING_CSV_PATH):
        raise FileNotFoundError(
            f"Training CSV not found at {TRAINING_CSV_PATH}. "
            "Cannot auto-train the model."
        )

    print("No trained model found. Training RandomForest model once from training dataset...")

    # Local imports avoid heavy/circular imports during module load.
    import pandas as pd
    from backend.training.data_prep import build_training_dataset
    from backend.training.train_model import train_and_evaluate

    df_students = pd.read_csv(TRAINING_CSV_PATH)
    df_pairs = build_training_dataset(TRAINING_CSV_PATH)
    train_and_evaluate(df_students, df_pairs)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Auto-training completed but model file was not created at {MODEL_PATH}."
        )

def get_model():
    """
    Loads the model from disk if it hasn't been loaded yet,
    otherwise returns the already loaded model from memory.
    """
    global _rf_model
    
    # If the model is already in memory, just return it instantly
    if _rf_model is not None:
        return _rf_model

    # Auto-train once if model is missing, then load.
    _train_model_if_missing()
        
    print(f"Loading RandomForest model from {MODEL_PATH} into memory...")
    _rf_model = joblib.load(MODEL_PATH)
    
    return _rf_model