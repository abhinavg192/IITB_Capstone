"""
Standalone script — generates feature_importance.png from the trained pkl.
Run from repo root: python3 notebooks/khushee/generate_chart.py
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH  = os.path.join(REPO_ROOT, 'models', 'xgboost_model.pkl')
COL_PATH    = os.path.join(REPO_ROOT, 'models', 'xgboost_columns.pkl')
CHART_PATH  = os.path.join(REPO_ROOT, 'notebooks', 'feature_importance.png')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(COL_PATH, 'rb') as f:
    train_columns = pickle.load(f)

raw_importance = dict(zip(train_columns, model.feature_importances_))

aggregated = {
    'Caption Length':    raw_importance.get('caption_length', 0),
    'Hashtag Count':     raw_importance.get('hashtag_count', 0),
    'Sentiment (VADER)': raw_importance.get('new_sentiment_score', 0),
    'Has CTA':           raw_importance.get('has_cta', 0),
    'Hour Posted':       sum(v for k, v in raw_importance.items() if k.startswith('hour_posted_')),
    'Platform':          sum(v for k, v in raw_importance.items() if k.startswith('platform_encoded_')),
}

feat_items = sorted(aggregated.items(), key=lambda x: x[1])
labels     = [x[0] for x in feat_items]
values     = [x[1] for x in feat_items]

COLORS = ['#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1565C0', '#0D47A1']

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(labels, values, color=COLORS[:len(labels)], edgecolor='white', height=0.6)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha='left', fontsize=10, color='#333')

ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
ax.set_title('XGBoost Feature Importance — Engagement Predictor', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(values) * 1.18 if max(values) > 0 else 1)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
print(f'Chart saved: {CHART_PATH}')
