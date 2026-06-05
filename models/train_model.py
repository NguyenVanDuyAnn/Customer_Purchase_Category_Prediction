# =====================================================
# IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from lightgbm import LGBMClassifier

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/processed/predictions_lgbm.csv"
)

print("=" * 60)
print("NEXT PRODUCT CATEGORY PREDICTION")
print("=" * 60)

# =====================================================
# REMOVE MISSING
# =====================================================

df = df.dropna()

# =====================================================
# TARGET ENCODER
# =====================================================

category_encoder = LabelEncoder()

df['target_category'] = (
    category_encoder.fit_transform(
        df['product_category']
    )
)

# =====================================================
# FEATURES
# =====================================================

features = [

    'age',

    'total_view',
    'total_click',
    'total_cart',
    'total_wishlist',

    'avg_price',
    'avg_discount',
    'avg_purchase_value',
    'avg_rating',

    'brand_loyalty_score',

    'unique_brands',

    'click_through_rate',
    'cart_rate',

    'category_share',

    'view_to_click_ratio',
    'cart_to_click_ratio',

    'wishlist_to_view_ratio',

    'active_engagement_ratio',

    'category_commitment_score',

    'exploration_score',

    'engagement_depth_score',

    'discount_sensitivity',

    'wishlist_to_cart_ratio',

    'click_engagement_rate',

    'value_per_click',

    'price_to_value_ratio',

    'category_breadth_score',

    'rating_concentration'
]

# =====================================================
# X / y
# =====================================================

X = df[features]

y = df['target_category']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# =====================================================
# LIGHTGBM MODEL
# =====================================================

model = LGBMClassifier(

    objective='multiclass',

    n_estimators=300,

    learning_rate=0.05,

    max_depth=10,

    num_leaves=31,

    random_state=42
)

# =====================================================
# TRAIN
# =====================================================

model.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICT
# =====================================================

predictions = model.predict(
    X_test
)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy Score: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    'Feature': features,

    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features:")
print(
    importance_df.head(10)
)

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "models/next_product_category_lgbm_model.joblib"
)

joblib.dump(
    category_encoder,
    "models/category_encoder.joblib"
)

print("\nModel saved successfully!")