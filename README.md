# SID Lysine Residual Prediction and Symbolic Regression

This repository contains a pipeline for predicting the residual SID Lysine intake in pigs using time-series feature engineering, machine learning (XGBoost), and symbolic regression (PySR). The goal is to model and interpret the residual lysine intake based on weekly aggregated feeding and growth data.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Model](#machine-learning-model)
- [Symbolic Regression](#symbolic-regression)
- [Balanced Formula Selection](#balanced-formula-selection)
- [Visualization](#visualization)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

---

## Project Overview

- **Objective:** Predict the residual SID Lysine (standardized ileal digestible lysine) in pigs as the difference between lysine intake and lysine requirement.
- **Data:** Daily pig feeding and growth data from multiple cohorts (`EID` groups) aggregated weekly.
- **Approach:** 
  - Aggregate daily data into weekly summaries.
  - Create lag and moving average time-series features.
  - Train and evaluate an XGBoost model for prediction.
  - Apply symbolic regression using PySR to find interpretable formulas.
  - Select balanced symbolic equations by combining loss and complexity metrics.

---

## Data Preparation

- Load daily data from multiple Excel files, clean by removing abnormal cases.
- Compute `WEEK` index from dates.
- Aggregate data to weekly level with mean values for feeding intake, weight gain, body weight, lysine intake, lysine requirement, and residual lysine.

---

## Feature Engineering

- Generate metabolic body weight as `BW^0.75`.
- Perform Augmented Dickey-Fuller (ADF) test for stationarity on key features.
- Create rolling 3-week moving averages and 1- and 2-week lags for key predictors (`ADFI`, `ADG`, `BW`, `MetabolicBW`).

---

## Machine Learning Model

- Split pigs into train/test sets by unique animal IDs.
- Train an XGBoost regressor to predict weekly residual SID Lysine.
- Perform hyperparameter tuning via grid search with cross-validation.
- Evaluate models using MSE and R² metrics.
- Test generalization on different `ShiftDay` groups (`d10`, `d14`, `d18`).

---

## Symbolic Regression

- Use PySRRegressor with custom operators to find symbolic formulas modeling the residual lysine.
- Configure PySR for deterministic and reproducible results by setting:
  - `random_state=42`
  - `deterministic=True`
  - `parallelism='serial'`
- Fit symbolic model on training data (excluding categorical feature `ShiftDay`).

---

## Balanced Formula Selection

- Select best symbolic formulas by filtering top 5 by R² score.
- Normalize loss and complexity.
- Compute combined score as a weighted sum: `0.8 * normalized_loss + 0.2 * normalized_complexity`.
- Pick formula with minimal combined score to balance accuracy and simplicity.

---

## Visualization

- Plot feature importance from XGBoost (gain-based).
- Visualize predicted vs actual residual SID Lysine with scatter plot.
- Plot residuals distribution.
- Visualize Pareto front of symbolic formulas (loss vs complexity colored by score).

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- pysr
- statsmodels
- matplotlib
- seaborn

Install via pip:

```bash
pip install pandas numpy scikit-learn xgboost pysr statsmodels matplotlib seaborn
