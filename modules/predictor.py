"""
modules/predictor.py
XGBoost Engagement Predictor
Loads Manas's pre-trained model and provides predict_engagement() wrapper.

Usage:
    from modules.predictor import predict_engagement
    from modules.predictor import get_optimal_posting_hour
    from modules.predictor import get_best_posting_time
    from modules.predictor import get_feature_importance

Author: Manas Sandeep Rane (modularized by Abhinav Gupta)
"""

import os
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT — must be set before xgboost import
# Prevents libomp conflicts on Mac ARM
# ─────────────────────────────────────────────────────────────

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Features Manas's model was trained on — must match exactly
MODEL_FEATURES = [
    'caption_length',
    'hashtag_count',
    'new_sentiment_score',  # VADER compound score
    'has_cta',
    'platform_encoded',     # 0=Twitter, 1=LinkedIn, 2=Instagram
    'hour_posted'           # 0-23
]

# Model file paths
MODEL_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH    = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
COLUMNS_PATH  = os.path.join(MODEL_DIR, 'xgboost_columns.pkl')

# Cached model — loaded once per session
_model         = None
_train_columns = None


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model():
    """
    Loads Manas's pre-trained XGBoost model from disk.
    Caches in memory after first load.

    Returns:
        tuple: (model, train_columns)

    Raises:
        FileNotFoundError: if pkl files not found in models/
    """
    global _model, _train_columns

    # Return cached model if already loaded
    if _model is not None:
        return _model, _train_columns

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Ensure xgboost_model.pkl is in the models/ folder."
        )

    if not os.path.exists(COLUMNS_PATH):
        raise FileNotFoundError(
            f"Columns not found at {COLUMNS_PATH}. "
            "Ensure xgboost_columns.pkl is in the models/ folder."
        )

    with open(MODEL_PATH, 'rb') as f:
        _model = pickle.load(f)

    with open(COLUMNS_PATH, 'rb') as f:
        _train_columns = pickle.load(f)

    print(f"✅ XGBoost model loaded ({len(_train_columns)} features)")
    return _model, _train_columns


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_engagement(features_dict: dict) -> float:
    """
    Predicts engagement score for a post using Manas's XGBoost model.
    Called by optimize_variants() in pipeline.py.

    Args:
        features_dict (dict): must contain these keys exactly:
            - caption_length (int)     character count of post
            - hashtag_count (int)      number of hashtags
            - new_sentiment_score (float) VADER compound -1 to 1
            - has_cta (int)            0 or 1
            - platform_encoded (int)   0=Twitter 1=LinkedIn 2=Instagram
            - hour_posted (int)        0-23

    Returns:
        float: predicted engagement score
               raw number — higher = better
               typical range: 0 to ~500

    Raises:
        FileNotFoundError: if model pkl files missing
    """
    model, train_columns = load_model()

    # Extract only the 6 features the model needs
    model_features = {
        'caption_length':      features_dict['caption_length'],
        'hashtag_count':       features_dict['hashtag_count'],
        'new_sentiment_score': features_dict['new_sentiment_score'],
        'has_cta':             features_dict['has_cta'],
        'platform_encoded':    features_dict['platform_encoded'],
        'hour_posted':         features_dict['hour_posted']
    }

    # Build input DataFrame
    input_df = pd.DataFrame([model_features])

    # One-hot encode to match training schema
    input_df['hour_posted']      = input_df['hour_posted'].astype('category')
    input_df['platform_encoded'] = input_df['platform_encoded'].astype('category')

    processed = pd.get_dummies(
        input_df,
        columns=['hour_posted', 'platform_encoded'],
        drop_first=True
    )

    # Align columns exactly with training data
    processed = processed.reindex(columns=train_columns, fill_value=0)

    # Predict and inverse transform from log scale
    log_pred = model.predict(processed)[0]
    score    = float(np.expm1(log_pred))

    return max(0.0, round(score, 4))


# ─────────────────────────────────────────────────────────────
# POSTING TIME FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_optimal_posting_hour(features_dict: dict) -> tuple:
    """
    Finds the best hour to post for maximum predicted engagement.
    Loops through all 24 hours and returns the best one.

    Args:
        features_dict: same as predict_engagement() WITHOUT hour_posted

    Returns:
        tuple: (optimal_hour: int, predicted_score: float)
    """
    best_score = -1
    best_hour  = 9  # default 9am

    for hour in range(24):
        features = features_dict.copy()
        features['hour_posted'] = hour
        score = predict_engagement(features)
        if score > best_score:
            best_score = score
            best_hour  = hour

    return best_hour, round(best_score, 4)


def get_best_posting_time(platform: str) -> str:
    """
    Returns human-readable best posting time per platform.
    Based on Manas's EDA findings from Kaggle Twitter dataset.
    Used by Aditya's UI for display.

    Args:
        platform: "twitter", "linkedin", or "instagram"

    Returns:
        str: e.g. "Tuesday 9:00 AM"
    """
    best_times = {
        "twitter":   "Wednesday 9:00 AM",
        "linkedin":  "Tuesday 9:00 AM",
        "instagram": "Wednesday 11:00 AM"
    }
    return best_times.get(platform.lower(), "Tuesday 9:00 AM")


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

def get_feature_importance() -> pd.DataFrame:
    """
    Returns feature importance from trained model.
    Used by Aditya's UI to explain why a post scored well.

    Returns:
        DataFrame with columns: Feature, Importance
        Sorted by importance descending
    """
    model, train_columns = load_model()

    return pd.DataFrame({
        'Feature':    train_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing predictor.py...\n")

    test_features = {
        'caption_length':      280,
        'hashtag_count':       3,
        'new_sentiment_score': 0.75,
        'has_cta':             1,
        'platform_encoded':    1,   # LinkedIn
        'hour_posted':         9
    }

    score = predict_engagement(test_features)
    print(f"Test prediction: {score}")

    best_hour, best_score = get_optimal_posting_hour({
        k: v for k, v in test_features.items()
        if k != 'hour_posted'
    })
    print(f"Optimal hour: {best_hour}:00 (score: {best_score})")
    print(f"Best time (LinkedIn): {get_best_posting_time('linkedin')}")

    importance = get_feature_importance()
    print(f"\nTop 5 features:")
    print(importance.head())