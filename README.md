# Automotive Price Prediction — Auto Trader UK

**Advanced Machine Learning (AML) — MSc Data Science, Manchester Metropolitan University**

End-to-end machine learning pipeline predicting UK vehicle prices from Auto Trader listings — covering preprocessing, feature selection, model training, hyperparameter tuning, ensemble methods, and SHAP interpretability.

---

## Key Results

| Model | Test R² | MAE | CV Mean R² |
|-------|---------|-----|-----------|
| Linear Regression | 0.585 | - | - |
| **Random Forest** | **0.871** | - | - |
| XGBoost | 0.775 | - | - |
| Ensemble Voting (RF + GB + LR) | 0.864 | - | - |

**Best model:** Random Forest (R² = 0.871)  
**Key price drivers (SHAP):** Vehicle make, model, age, and mileage account for the majority of price variance.

> Run the notebook to populate exact MAE values from your results.

---

## Dataset

This notebook uses `cleaned_data.csv` produced by the companion EDA notebook.  
See **[Data_Analytics](https://github.com/SultanZia/Data_Analytics)** for data cleaning methodology.

**Original source:** Auto Trader UK listings (2020 snapshot), ~400,000 records sampled from 3M+.

---

## Methodology

### Preprocessing Pipeline
- Polynomial interaction features: `mileage × Age`
- Target encoding for high-cardinality categoricals (`standard_make`, `standard_model`, `standard_colour`, `body_type`)
- One-hot encoding for `fuel_type`
- MinMaxScaler for numerical features

### Feature Selection
- RFECV (Recursive Feature Elimination with Cross-Validation) applied independently for Linear Regression and XGBoost
- PCA explored as dimensionality reduction alternative

### Models
- **Linear Regression** — baseline
- **Random Forest** (100 estimators) — best single model
- **XGBoost** with GridSearchCV hyperparameter tuning
- **Ensemble Voting** (Random Forest + Gradient Boosting + Linear Regression)

### Interpretability
- **SHAP** (SHapley Additive exPlanations) on XGBoost — beeswarm, waterfall, and scatter plots
- **Partial Dependence Plots** for categorical price drivers

---

## Repository Structure

```
Automotive_Price_Prediction/
├── vehicle_price_modelling.ipynb   ← clean ML notebook
├── train.py                        ← training script
├── predict.py                      ← single-vehicle inference
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | scikit-learn, XGBoost |
| Interpretability | SHAP |
| Encoding | category_encoders |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |

---

## Setup

```bash
git clone https://github.com/SultanZia/Automotive_Price_Prediction.git
cd Automotive_Price_Prediction
pip install -r requirements.txt
```

Place `cleaned_data.csv` in the root directory, then open the notebook.

### Train a model
```bash
python train.py --model rf --data_path cleaned_data.csv
```

### Predict a single vehicle price
```bash
python predict.py --model_path models/rf_model.pkl --make BMW --model_name "3 Series" --age 3 --mileage 25000 --fuel_type Petrol --body_type Saloon --colour Black
```

---

## Author

**Mohammed Zia Sultan**  
MSc Data Science, Manchester Metropolitan University (2023–2024)  
[github.com/SultanZia](https://github.com/SultanZia)
