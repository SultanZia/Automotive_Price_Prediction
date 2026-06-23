"""
train.py — Automotive Price Prediction Training Script

Trains a Random Forest, XGBoost, or Ensemble Voting model on cleaned
Auto Trader UK vehicle listings data to predict vehicle price.

Usage:
    python train.py --model rf --data_path cleaned_data.csv
    python train.py --model xgb --data_path cleaned_data.csv
    python train.py --model ensemble --data_path cleaned_data.csv
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
import category_encoders as ce
import xgboost as xgb


# ── Feature groups ────────────────────────────────────────────────────────────
NUMERICAL_FEATURES   = ['mileage', 'Age']
CATEGORICAL_FEATURES = ['standard_make', 'standard_model', 'standard_colour', 'body_type']
ONE_HOT_FEATURE      = ['fuel_type']
TARGET               = 'price'


# ── Custom transformer ────────────────────────────────────────────────────────
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer  = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X); return self

    def transform(self, X):
        return self.imputer.transform(X)

    def get_feature_names_out(self, input_features=None):
        return input_features


# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_preprocessor():
    numerical_pipeline = Pipeline([
        ('imputer', CustomImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('target_encoding', ce.TargetEncoder(cols=CATEGORICAL_FEATURES)),
    ])
    one_hot_pipeline = Pipeline([
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('numerical',   numerical_pipeline,   NUMERICAL_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
        ('onehot',      one_hot_pipeline,     ONE_HOT_FEATURE),
    ])


def engineer_features(df):
    """Add polynomial interaction and Age feature."""
    df = df.copy()
    if 'year_of_registration' in df.columns and 'Age' not in df.columns:
        df['Age'] = 2021 - df['year_of_registration']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_arr = poly.fit_transform(df[['mileage', 'Age']])
    poly_df  = pd.DataFrame(poly_arr, columns=['poly_mile', 'poly_Age', 'poly_mile_Age'],
                             index=df.index)
    return pd.concat([df, poly_df], axis=1)


# ── Model builders ────────────────────────────────────────────────────────────
def build_model(model_name: str):
    if model_name == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'xgb':
        return xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100,
            max_depth=5, learning_rate=0.1, random_state=42
        )
    elif model_name == 'ensemble':
        return VotingRegressor([
            ('rf',  RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb',  GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('lr',  LinearRegression()),
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: rf, xgb, ensemble")


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    print(f"\nLoading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    df = engineer_features(df)

    a = df[TARGET]
    z = df.drop(columns=[TARGET])

    X_train, X_test, Y_train, Y_test = train_test_split(z, a, test_size=0.25, random_state=0)

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, Y_train)
    X_test_proc  = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_proc,  columns=feature_names)

    print(f"\nTraining {args.model.upper()} model...")
    model = build_model(args.model)
    model.fit(X_train_df, Y_train)

    Y_pred = model.predict(X_test_df)
    r2  = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

    print(f"\n{'='*45}")
    print(f"  {args.model.upper()} Results")
    print(f"{'='*45}")
    print(f"  Test R²:  {r2:.4f}")
    print(f"  MAE:      £{mae:,.0f}")
    print(f"  RMSE:     £{rmse:,.0f}")

    cv_scores = cross_val_score(model, X_train_df, Y_train, cv=5)
    print(f"  CV R²:    {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
    print(f"{'='*45}")

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'{args.model}_model.pkl')
    prep_path  = os.path.join(args.output_dir, 'preprocessor.pkl')
    joblib.dump(model,        model_path)
    joblib.dump(preprocessor, prep_path)
    print(f"\nModel saved:        {model_path}")
    print(f"Preprocessor saved: {prep_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Train a vehicle price prediction model.')
    parser.add_argument('--model',      type=str, default='rf',
                        choices=['rf', 'xgb', 'ensemble'],
                        help='Model type: rf, xgb, or ensemble (default: rf)')
    parser.add_argument('--data_path',  type=str, default='cleaned_data.csv',
                        help='Path to cleaned_data.csv (default: cleaned_data.csv)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained model (default: ./models)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
