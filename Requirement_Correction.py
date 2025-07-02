# data_processing_and_modeling.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_data():
    # Load data
    df_daily_EID1 = pd.read_excel(r"D:\OneDrive - UGent\Ph.D_Yao\data\22RWX\2_ProcessedData\RealtimeCalculation\2025-06-01_22RWX_daily_calculations.xlsx", sheet_name='EID')
    df_daily_EID2 = pd.read_excel(r"D:\OneDrive - UGent\Ph.D_Yao\data\23MYZ01\2_ProcessedData\realtime_calculations\2025-05-28_23MYZ01_daily_calculations.xlsx", sheet_name='EID')
    df_daily_EID3 = pd.read_excel(r"D:\OneDrive - UGent\Ph.D_Yao\data\23PDW01\2_ProcessedData\RealtimeCalculation\20250601_23PDW01_daily_calculations.xlsx", sheet_name='EID')

    # Filter out abnormal cases by EID
    exit_animal1 = df_daily_EID1[df_daily_EID1['ABNORMAL_CASE'] == 1]['EID'].unique()
    exit_animal2 = df_daily_EID2[df_daily_EID2['ABNORMAL_CASE'] == 1]['EID'].unique()
    exit_animal3 = df_daily_EID3[df_daily_EID3['ABNORMAL_CASE'] == 1]['EID'].unique()

    df_daily_EID1 = df_daily_EID1[~df_daily_EID1['EID'].isin(exit_animal1)]
    df_daily_EID2 = df_daily_EID2[~df_daily_EID2['EID'].isin(exit_animal2)]
    df_daily_EID2.drop(columns=['BWCLASS'], inplace=True)
    df_daily_EID3 = df_daily_EID3[~df_daily_EID3['EID'].isin(exit_animal3)]

    # Add ShiftDay column
    df_daily_EID1['ShiftDay'] = 'd14'
    df_daily_EID2.loc[df_daily_EID2['TREAT'].isin(['HP10', 'LP10']), 'ShiftDay'] = 'd10'
    df_daily_EID2.loc[df_daily_EID2['TREAT'].isin(['HP18', 'LP18']), 'ShiftDay'] = 'd18'
    df_daily_EID3['ShiftDay'] = 'd14'

    # Concatenate dataframes
    df_daily_EID = pd.concat([df_daily_EID1, df_daily_EID2, df_daily_EID3], ignore_index=True)
    df_daily_EID.drop(columns=['ABNORMAL_CASE', 'exitBW'], inplace=True)

    return df_daily_EID


def aggregate_weekly(df_daily):
    # Aggregate by week (7-day blocks)
    df_daily['WEEK'] = (df_daily['DTMARKER_INT'] // 7) + 1

    df_weekly = df_daily.groupby(['EID', 'WEEK', 'ShiftDay']).agg({
        'VFI': 'mean',
        'predVBWgain': 'mean',
        'predVBW': 'mean',
        'SIDLysI': 'mean',
        'SIDLysRequire': 'mean',
        'SIDLysResidual': 'mean'
    }).reset_index()

    df_weekly.rename(columns={
        'VFI': 'ADFI',
        'predVBWgain': 'ADG',
        'predVBW': 'BW',
        'SIDLysI': 'ADSIDLysI',
        'SIDLysRequire': 'ADSIDLysRequire',
        'SIDLysResidual': 'ADSIDLysResidual'
    }, inplace=True)

    df_weekly['MetabolicBW'] = df_weekly['BW'] ** 0.75

    return df_weekly


def check_stationarity(series, feature_name):
    result = adfuller(series.dropna())
    print(f'ADF Statistic for {feature_name}: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] < 0.05:
        print(f'-> {feature_name} is likely stationary.\n')
    else:
        print(f'-> {feature_name} is likely non-stationary.\n')


def add_time_features(df_weekly, features):
    df_weekly = df_weekly.sort_values(['EID', 'WEEK'])
    for feat in features:
        df_weekly[f'{feat}_MA3'] = df_weekly.groupby('EID')[feat].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df_weekly[f'{feat}_lag1'] = df_weekly.groupby('EID')[feat].shift(1)
        df_weekly[f'{feat}_lag2'] = df_weekly.groupby('EID')[feat].shift(2)
    return df_weekly


def prepare_features(df_weekly):
    df_weekly['ShiftDay'] = df_weekly['ShiftDay'].astype('category')

    time_series_features = ['ADFI', 'ADG', 'BW', 'MetabolicBW']

    features = time_series_features + \
               [f'{f}_MA3' for f in time_series_features] + \
               [f'{f}_lag1' for f in time_series_features] + \
               [f'{f}_lag2' for f in time_series_features] + \
               ['ShiftDay', 'WEEK']

    target = 'ADSIDLysResidual'

    df_clean = df_weekly.dropna(subset=features + [target])

    # Split pigs into train/test sets by EID
    unique_eids = df_clean['EID'].unique()
    train_eids, test_eids = train_test_split(unique_eids, test_size=0.3, random_state=42)

    train_df = df_clean[df_clean['EID'].isin(train_eids)].sort_values(['EID', 'WEEK'])
    test_df = df_clean[df_clean['EID'].isin(test_eids)]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    return X_train, y_train, X_test, y_test


def perform_grid_search(X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=7)
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        enable_categorical=True
    )

    params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.1, 0.2]
    }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=params,
        cv=tscv,
        scoring='r2',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best params found:", grid_search.best_params_)
    print("Best CV R²:", grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Final Test Performance:\nMSE: {mse:.4f}\nR²: {r2:.4f}")

    return y_pred


def plot_feature_importance(model):
    importance_type = 'gain'  # alternatives: 'weight', 'cover'
    importance_dict = model.get_booster().get_score(importance_type=importance_type)
    importance_df = pd.DataFrame(importance_dict.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8), dpi=200)
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f'XGBoost Feature Importance ({importance_type})', fontsize=16)
    plt.xlabel('Average Gain (Importance)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_residual_analysis(y_test, y_pred):
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_df['Residual'] = results_df['Actual'] - results_df['Predicted']

    plt.figure(figsize=(7, 7), dpi=200)
    sns.scatterplot(x='Actual', y='Predicted', data=results_df, alpha=0.6)
    plt.plot([results_df.Actual.min(), results_df.Actual.max()],
             [results_df.Actual.min(), results_df.Actual.max()],
             'r--')  # Diagonal line
    plt.xlabel('Actual SID Lys Residual')
    plt.ylabel('Predicted SID Lys Residual')
    plt.title('Predicted vs Actual')
    plt.show()

    plt.figure(figsize=(8, 4), dpi=200)
    sns.histplot(results_df['Residual'], kde=True, bins=30)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.show()


def fit_symbolic_regression(X_train, y_train, features):
    symbolic_model = PySRRegressor(
        niterations=100,  # increase for better results
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sin", "cos", "sqrt"],
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        random_state=42
    )

    symbolic_model.fit(X_train.values, y_train.values)

    print(symbolic_model)

    return symbolic_model


def main():
    # Load and clean data
    df_daily = load_and_clean_data()

    # Aggregate to weekly
    df_weekly = aggregate_weekly(df_daily)

    # Check stationarity of some variables
    check_stationarity(df_weekly['ADFI'], 'ADFI')
    check_stationarity(df_weekly['ADSIDLysResidual'], 'ADSIDLysResidual')
    check_stationarity(df_weekly['ADSIDLysRequire'], 'ADSIDLysRequire')

    # Add time features (MA3 and lags)
    time_features = ['ADFI', 'ADG', 'BW', 'MetabolicBW']
    df_weekly = add_time_features(df_weekly, time_features)

    # Prepare features for modeling
    X_train, y_train, X_test, y_test = prepare_features(df_weekly)

    # Train model with grid search and cross-validation
    best_model = perform_grid_search(X_train, y_train)

    # Evaluate on test set
    y_pred = evaluate_model(best_model, X_test, y_test)

    # Plot feature importance and residuals
    plot_feature_importance(best_model)
    plot_residual_analysis(y_test, y_pred)

    # Optional: symbolic regression for interpretability
    # symbolic_model = fit_symbolic_regression(X_train, y_train, X_train.columns)


if __name__ == "__main__":
    main()
