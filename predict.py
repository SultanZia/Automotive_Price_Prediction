"""
predict.py — Single Vehicle Price Inference

Loads a trained model and preprocessor to predict the price of
a single vehicle based on its attributes.

Usage:
    python predict.py --model_path models/rf_model.pkl \
                      --make BMW \
                      --model_name "3 Series" \
                      --age 3 \
                      --mileage 25000 \
                      --fuel_type Petrol \
                      --body_type Saloon \
                      --colour Black
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# ── Feature engineering (must match train.py) ─────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_arr = poly.fit_transform(df[['mileage', 'Age']])
    poly_df  = pd.DataFrame(poly_arr, columns=['poly_mile', 'poly_Age', 'poly_mile_Age'],
                             index=df.index)
    return pd.concat([df, poly_df], axis=1)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    prep_path = args.model_path.replace(
        os.path.basename(args.model_path), 'preprocessor.pkl'
    )
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessor not found: {prep_path}")

    model        = joblib.load(args.model_path)
    preprocessor = joblib.load(prep_path)

    input_df = pd.DataFrame([{
        'standard_make':  args.make,
        'standard_model': args.model_name,
        'standard_colour': args.colour,
        'body_type':      args.body_type,
        'fuel_type':      args.fuel_type,
        'mileage':        args.mileage,
        'Age':            args.age,
        'vehicle_condition': 'USED',
        'year_of_registration': 2021 - args.age,
    }])

    input_df = engineer_features(input_df)

    X_proc       = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    X_df         = pd.DataFrame(X_proc, columns=feature_names)

    predicted_price = float(model.predict(X_df)[0])

    print("\n" + "=" * 50)
    print("  VEHICLE PRICE PREDICTION")
    print("=" * 50)
    print(f"  Make:       {args.make}")
    print(f"  Model:      {args.model_name}")
    print(f"  Age:        {args.age} years")
    print(f"  Mileage:    {args.mileage:,} miles")
    print(f"  Fuel type:  {args.fuel_type}")
    print(f"  Body type:  {args.body_type}")
    print(f"  Colour:     {args.colour}")
    print("-" * 50)
    print(f"  Predicted price: £{predicted_price:,.0f}")
    print("=" * 50)
    print("\n  Note: Predictions are estimates based on 2020 market data.\n")

    return predicted_price


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Predict vehicle price from attributes.')
    parser.add_argument('--model_path',  type=str, required=True,
                        help='Path to saved model .pkl file')
    parser.add_argument('--make',        type=str, required=True, help='Vehicle make (e.g. BMW)')
    parser.add_argument('--model_name',  type=str, required=True, help='Vehicle model (e.g. 3 Series)')
    parser.add_argument('--age',         type=int, required=True, help='Vehicle age in years')
    parser.add_argument('--mileage',     type=int, required=True, help='Vehicle mileage')
    parser.add_argument('--fuel_type',   type=str, default='Petrol',
                        choices=['Petrol', 'Diesel', 'Electric', 'Petrol Hybrid',
                                 'Diesel Hybrid', 'Petrol Plug-in Hybrid', 'Diesel Plug-in Hybrid'])
    parser.add_argument('--body_type',   type=str, default='Saloon')
    parser.add_argument('--colour',      type=str, default='Black')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
