# energy-consuption-prediction
GBoost regression pipeline predicting household daily electricity consumption from RECS survey data, feature engineering, learning curve analysis, and Kaggle hackathon submission

# Household Energy Consumption Prediction

A regression pipeline predicting average daily household electricity consumption 
using data from the Residential Energy Consumption Survey (RECS). Built as part 
of the PML M3 Final Hackathon 2026, iterating from a Linear Regression baseline 
to a tuned XGBoost model, reducing validation MAE from 242 to 173.

## Business Context

Accurately predicting household energy consumption enables utilities, policymakers, 
and energy companies to better allocate resources, target efficiency programs, and 
anticipate demand. This project treats it as a supervised regression problem using 
real RECS survey data.

## Modeling Pipeline

### 1. Data Loading & Inspection
Loaded training and test sets from Google Drive. Confirmed column alignment, 
data types, and verified zero missing values across both sets, no imputation required.

### 2. Preprocessing
Label-encoded categorical variables. Defined feature matrix X and target y. 
Applied 80/20 train/validation split for model evaluation before final submission.

### 3. Linear Regression — Baseline
Simple baseline to benchmark performance. Validation MAE: **~242**. 
Linear relationships alone were insufficient to capture the complexity of 
household energy drivers.

### 4. Random Forest Regressor
Ensemble model capturing nonlinear relationships. Validation MAE improved to **~192** 
— a ~21% reduction over baseline. Generated learning curve to diagnose fit.

### 5. Gradient Boosting Regressor
Sequential tree-based model with tuned hyperparameters. Produced incremental 
improvement over Random Forest. Multiple configurations tested through iterative 
trial and error.

### 6. XGBoost — Final Model
Added engineered features before fitting:
- `rooms_per_person`: room count normalized by household size
- `sqft_per_person`: square footage normalized by household size

These ratio features helped the model capture meaningful per-capita relationships 
rather than raw counts. Final validation MAE: **~173**, a ~29% improvement over 
Random Forest and ~29% total improvement from baseline.

### 7. Learning Curve Analysis
Generated learning curves at each major model stage to assess overfitting vs. 
underfitting and confirm generalization improved with additional training data.

### 8. Final Submission
Retrained XGBoost on full training dataset, predicted on held-out test set, 
submitted to Kaggle.

## Results

| Model | Validation MAE |
|-------|---------------|
| Linear Regression | ~242 |
| Random Forest | ~192 |
| Gradient Boosting | ~185 |
| XGBoost + Feature Engineering | ~173 |

## Key Takeaways
- Raw counts (rooms, square footage) are less predictive than per-capita ratios 
  that account for household size
- Sequential boosting methods outperform linear and ensemble baselines on 
  structured tabular survey data
- Learning curves are an essential diagnostic, they guided model selection 
  at every iteration rather than relying on leaderboard position alone

## Tech Stack
Python, XGBoost, scikit-learn, pandas, numpy, matplotlib, seaborn

## Presentation
See Video: https://youtu.be/Y1txK_AUau0
for the project walkthrough.
