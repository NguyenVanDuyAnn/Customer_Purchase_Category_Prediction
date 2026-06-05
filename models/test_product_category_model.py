"""
Test script to evaluate trained product category model on test data
"""
import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import DataPreprocessor


def load_and_prepare_data(input_file, preprocessor=None):
    """Load and prepare data for evaluation"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    if preprocessor is None:
        preprocessor = DataPreprocessor()
    
    # Check if input is pre-aggregated (has CustomerID/user_id)
    df = pd.read_csv(input_file)
    
    if 'user_id' in df.columns or 'CustomerID' in df.columns:
        print(f"Detected pre-aggregated customer CSV from {input_file}")
        # Prepare features as in train script
        df_feat = pd.DataFrame()
        if 'CustomerID' in df.columns:
            df_feat['CustomerID'] = df['CustomerID']
        else:
            df_feat['CustomerID'] = df['user_id']
        
        if 'PrimaryCategory' in df.columns:
            df_feat['PrimaryCategory'] = df['PrimaryCategory']
        else:
            df_feat['PrimaryCategory'] = df.get('product_category', 'Khác')
        
        df_feat['NumTransactions'] = df.get('total_view', df.get('InvoiceCount', 0))
        df_feat['TotalQuantity'] = df.get('total_cart', df.get('Quantity', 0))
        df_feat['AvgQuantity'] = df.get('total_cart', 0) / df_feat['NumTransactions'].replace(0, 1)
        df_feat['AvgPrice'] = df.get('avg_price', df.get('avg_purchase_value', 0))
        df_feat['MinPrice'] = df.get('min_price', df_feat['AvgPrice']).fillna(df_feat['AvgPrice'])
        df_feat['MaxPrice'] = df.get('max_price', df_feat['AvgPrice']).fillna(df_feat['AvgPrice'])
        df_feat['TotalSpending'] = df.get('avg_purchase_value', 0) * df_feat['NumTransactions']
        df_feat['AvgSpending'] = df.get('avg_purchase_value', df_feat['AvgPrice']).fillna(0)
        df_feat['Country'] = df.get('dominant_location', df.get('Country', 'Unknown'))
        df_feat['YearLastPurchase'] = pd.to_datetime('today').year
        df_feat['MonthLastPurchase'] = pd.to_datetime('today').month
        df_feat['DayOfWeekPreference'] = 0
        df_feat['HourPreference'] = df.get('hour', 12)
        df_feat['NumCategories'] = df.get('user_total_categories', df.get('NumCategories', 1))
        df_feat['NumProducts'] = df.get('unique_brands', df.get('NumProducts', 1))
        
        numeric_cols = [c for c in df_feat.columns if c not in ['CustomerID', 'PrimaryCategory', 'Country']]
        df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)
        
        # Encode
        df_encoded = preprocessor.encode_features(df_feat, fit=False)
        features = df_encoded.drop('CustomerID', axis=1)
        target = features.pop('PrimaryCategory')
        
        X = features.values
        customer_ids = df_feat['CustomerID'].values
        
        # Encode target
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(target)
        
        return X, y, customer_ids, target_encoder, features.columns.tolist()
    
    else:
        # Raw transaction data
        print(f"Detected raw transaction CSV from {input_file}")
        df_processed = preprocessor.preprocess(input_file, fit=False)
        
        features = df_processed.drop('CustomerID', axis=1)
        target = features.pop('PrimaryCategory')
        
        X = features.values
        customer_ids = df_processed['CustomerID'].values
        
        # Encode target
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(target)
        
        return X, y, customer_ids, target_encoder, features.columns.tolist()


def evaluate_model(model_path, preprocessor_path, input_file, test_size=0.2):
    """Evaluate saved model"""
    print("\n" + "=" * 60)
    print("LOADING MODEL & DATA")
    print("=" * 60)
    
    # Load model and preprocessor
    model = joblib.load(model_path)
    temp_preprocessor = DataPreprocessor()
    temp_preprocessor.load_preprocessor(preprocessor_path)
    X, y, customer_ids, target_encoder, feature_names = load_and_prepare_data(input_file, preprocessor=temp_preprocessor)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Data shape: {X.shape}")
    print(f"Target classes: {target_encoder.classes_}")
    print(f"Features: {feature_names}")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Make predictions
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)
        has_proba = True
    except:
        has_proba = False
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    
    print("\nClassification Report (Test Set):")
    unique_labels = np.unique(np.concatenate([y_test, y_pred_test]))
    print(classification_report(y_test, y_pred_test, labels=unique_labels, zero_division=0))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Evaluate on train set for comparison
    print("\n" + "=" * 60)
    print("TRAIN SET EVALUATION (for overfitting check)")
    print("=" * 60)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall:    {train_recall:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Overfit Gap:    {(train_accuracy - test_accuracy):.4f}")
    
    if has_proba:
        print("\nTop 5 predictions with highest confidence:")
        confidences = y_proba.max(axis=1)
        top_idx = np.argsort(confidences)[-5:][::-1]
        for i, idx in enumerate(top_idx):
            true_label = target_encoder.classes_[y_test[idx]]
            pred_label = target_encoder.classes_[y_pred_test[idx]]
            conf = confidences[idx]
            print(f"  {i+1}. True: {true_label}, Pred: {pred_label}, Conf: {conf:.4f}")
    
    # Save confusion matrix plot
    print("\nGenerating confusion matrix visualization...")
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Test Set')
        plot_path = 'data/processed/confusion_matrix_test.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    return {
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'target_encoder': target_encoder
    }


def main():
    parser = argparse.ArgumentParser(description='Test trained product category model')
    parser.add_argument('--model', type=str, default='models/product_category_model.joblib', help='Path to saved model')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.joblib', help='Path to saved preprocessor')
    parser.add_argument('--input', type=str, default='data/raw/predictions_lgbm.csv', help='Input data CSV')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set ratio')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PRODUCT CATEGORY MODEL - TEST & EVALUATION")
    print("=" * 60)
    
    evaluate_model(args.model, args.preprocessor, args.input, test_size=args.test_size)


if __name__ == "__main__":
    main()
