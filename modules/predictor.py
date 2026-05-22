"""
modules/predictor.py
XGBoost Engagement Predictor
Loads Manas's pre-trained model and provides predict_engagement() wrapper.

Usage:
    from modules.predictor import predict_engagement
    from modules.predictor import get_optimal_posting_hour
    from modules.predictor import get_best_posting_time
    from modules.predictor import get_feature_importance
    from modules.predictor import extract_features_from_raw

Author: Manas Sandeep Rane (modularized by Abhinav Gupta)
"""

import os
import pickle
import numpy as np
import pandas as pd

import ssl
import nltk

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

_CTA_KEYWORDS = [
    'link', 'register', 'learn more', 'sign up', 'click here',
    'visit', 'download', 'bio', 'get started', 'click',
    'discover', 'explore', 'try', 'get', 'join', 'shop',
    'buy', 'check out', 'read', 'watch', 'follow', 'share',
    'comment below', 'dm us', 'subscribe', 'link in bio'
]

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

# Manas's model
MODEL_PATH    = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
COLUMNS_PATH  = os.path.join(MODEL_DIR, 'xgboost_columns.pkl')

# Khushee's model
MODEL_PATH_K   = os.path.join(MODEL_DIR, 'xgboost_engagement.pkl')
COLUMNS_PATH_K = os.path.join(MODEL_DIR, 'xgboost_engagement_columns.pkl')

# Cached models — loaded once per session
_model         = None
_train_columns = None
_model_k         = None
_train_columns_k = None


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model(model_version: str = 'manas'):
    """
    Loads a pre-trained XGBoost model from disk.
    Caches in memory after first load.

    Args:
        model_version (str): 'manas' (default) or 'khushee'
                             Switch to compare which model yields better results.

    Returns:
        tuple: (model, train_columns)

    Raises:
        FileNotFoundError: if pkl files not found in models/
        ValueError: if model_version is not recognised
    """
    global _model, _train_columns, _model_k, _train_columns_k

    if model_version == 'manas':
        if _model is not None:
            return _model, _train_columns
        m_path, c_path = MODEL_PATH, COLUMNS_PATH
    elif model_version == 'khushee':
        if _model_k is not None:
            return _model_k, _train_columns_k
        m_path, c_path = MODEL_PATH_K, COLUMNS_PATH_K
    else:
        raise ValueError(f"Unknown model_version '{model_version}'. Use 'manas' or 'khushee'.")

    if not os.path.exists(m_path):
        raise FileNotFoundError(f"Model not found at {m_path}.")
    if not os.path.exists(c_path):
        raise FileNotFoundError(f"Columns not found at {c_path}.")

    with open(m_path, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(c_path, 'rb') as f:
        loaded_columns = pickle.load(f)

    if model_version == 'manas':
        _model, _train_columns = loaded_model, loaded_columns
    else:
        _model_k, _train_columns_k = loaded_model, loaded_columns

    print(f"✅ XGBoost model loaded [{model_version}] ({len(loaded_columns)} features)")
    return loaded_model, loaded_columns


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def _predict_with_version(features_dict: dict, version: str) -> float:
    """Internal helper — runs prediction for a specific model version."""
    model, train_columns = load_model(version)
    model_features = {
        'caption_length':      features_dict['caption_length'],
        'hashtag_count':       features_dict['hashtag_count'],
        'new_sentiment_score': features_dict['new_sentiment_score'],
        'has_cta':             features_dict['has_cta'],
        'platform_encoded':    features_dict['platform_encoded'],
        'hour_posted':         features_dict['hour_posted']
    }
    input_df = pd.DataFrame([model_features])
    input_df['hour_posted']      = input_df['hour_posted'].astype('category')
    input_df['platform_encoded'] = input_df['platform_encoded'].astype('category')
    processed = pd.get_dummies(input_df, columns=['hour_posted', 'platform_encoded'], drop_first=True)
    processed = processed.reindex(columns=train_columns, fill_value=0)
    log_pred  = model.predict(processed)[0]
    return max(0.0, round(float(np.expm1(log_pred)), 4))


def predict_engagement(features_dict: dict, model_version: str = 'auto') -> float:
    """
    Predicts engagement score for a post using a pre-trained XGBoost model.
    Called by optimize_variants() in pipeline.py.

    Args:
        features_dict (dict): must contain these keys exactly:
            - caption_length (int)     character count of post
            - hashtag_count (int)      number of hashtags
            - new_sentiment_score (float) VADER compound -1 to 1
            - has_cta (int)            0 or 1
            - platform_encoded (int)   0=Twitter 1=LinkedIn 2=Instagram
            - hour_posted (int)        0-23
        model_version (str): 'auto' (default) — picks whichever model scores higher.
                             'manas' or 'khushee' to force a specific model.

    Returns:
        float: predicted engagement score
               raw number — higher = better
               typical range: 0 to ~500

    Raises:
        FileNotFoundError: if model pkl files missing
    """
    if model_version == 'auto':
        score_manas   = _predict_with_version(features_dict, 'manas')
        score_khushee = _predict_with_version(features_dict, 'khushee')
        return max(score_manas, score_khushee)

    return _predict_with_version(features_dict, model_version)


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def extract_features_from_raw(text: str, platform: str = 'twitter', timestamp=None) -> dict:
    """
    Extracts XGBoost model features from raw post text.
    Use this for examiner demos or when you have plain text (not a variant dict).

    For pipeline use, prefer extract_features() in pipeline.py which
    takes a structured variant dict from generate_posts().

    Args:
        text      : raw post string
        platform  : 'twitter', 'linkedin', or 'instagram'
        timestamp : datetime-like or None — if provided, hour_posted is extracted

    Returns:
        dict matching MODEL_FEATURES schema:
            caption_length, hashtag_count, new_sentiment_score,
            has_cta, platform_encoded, hour_posted
    """
    text_lower = text.lower() if text else ''

    caption_length       = len(text) if text else 0
    hashtag_count        = text.count('#') if text else 0
    new_sentiment_score  = round(_vader.polarity_scores(text)['compound'], 4) if text else 0.0
    has_cta              = int(any(kw in text_lower for kw in _CTA_KEYWORDS))

    platform_map     = {'twitter': 0, 'linkedin': 1, 'instagram': 2}
    platform_encoded = platform_map.get(platform.lower(), 0)

    if timestamp is not None:
        try:
            hour_posted = pd.Timestamp(timestamp).hour
        except Exception:
            hour_posted = 9
    else:
        hour_posted = 9

    return {
        'caption_length':      caption_length,
        'hashtag_count':       hashtag_count,
        'new_sentiment_score': new_sentiment_score,
        'has_cta':             has_cta,
        'platform_encoded':    platform_encoded,
        'hour_posted':         hour_posted,
    }


# ─────────────────────────────────────────────────────────────
# POSTING TIME FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_optimal_posting_hour(features_dict: dict, model_version: str = 'manas') -> tuple:
    """
    Finds the best hour to post for maximum predicted engagement.
    Loops through all 24 hours and returns the best one.

    Args:
        features_dict: same as predict_engagement() WITHOUT hour_posted
        model_version (str): 'manas' (default) or 'khushee'

    Returns:
        tuple: (optimal_hour: int, predicted_score: float)
    """
    best_score = -1
    best_hour  = 9  # default 9am

    for hour in range(24):
        features = features_dict.copy()
        features['hour_posted'] = hour
        score = predict_engagement(features, model_version=model_version)
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

def get_feature_importance(model_version: str = 'manas') -> pd.DataFrame:
    """
    Returns feature importance from trained model.
    Used by Aditya's UI to explain why a post scored well.

    Args:
        model_version (str): 'manas' (default) or 'khushee'

    Returns:
        DataFrame with columns: Feature, Importance
        Sorted by importance descending
    """
    model, train_columns = load_model(model_version)

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
