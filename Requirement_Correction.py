import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pysr import PySRRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data():
    # Update file paths here
    df_daily_EID1 = pd.read_excel("path/to/EID1.xlsx", sheet_name='EID')
    df_daily_EID2 = pd.read_excel("path/to/EID2.xlsx", sheet_name='EID')
    df_daily_EID3 = pd.read_excel("path/to/EID3.xlsx", sheet_name='EID')

    # Remove abnormal cases
    for df in [df_daily_EID1, df_daily_EID2, df_daily_EID3]:
        exit_animals = df[df['ABNORMAL_CASE'] == 1]['EID'].unique()
        df.drop(df[df['EID'].isin(exit_animals)].index, inplace=True)

    # Add ShiftDay and combine datasets (simplified)
    df_daily_EID1['ShiftDay'] = 'd14'
    df_daily_EID2['ShiftDay'] = df_daily_EID2['TREAT'].map({
        'HP10': 'd10', 'LP10': 'd10', 'HP18': 'd18', 'LP18': 'd18'
    }).fillna('unknown')
    df_daily_EID3['ShiftDay'] = 'd14'

    df_daily = pd.concat([df_daily_EID1, df_daily_EID2, df_daily_EID3], ignore_index=True)
    df_daily['WEEK'] = ((df_daily['DTMARKER_INT']) // 7) + 1
    df_daily.drop(columns=['ABNORMAL_CASE', 'exitBW', 'BWCLASS'], errors='ignore', inplace=True)

    return df_daily

def aggregate_weekly(df_daily):
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

def add_time_series_features(df_weekly):
    time_series_features = ['ADFI', 'ADG', 'BW', 'MetabolicBW']
    df_weekly = df_weekly.sort_values(['EID', 'WEEK'])

    for feat in time_series_features:
        df_weekly[f'{feat}_MA3'] = df_weekly.groupby('EID')[feat].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df_weekly[f'{feat}_lag1'] = df_weekly.groupby('EID')[feat].shift(1)
        df_weekly[f'{feat}_lag2'] = df_weekly.groupby('EID')[feat].shift(2)

    return df_weekly

def train_xgb_model(df_weekly):
    df_weekly['ShiftDay'] = df_weekly['ShiftDay'].astype('category')

    features = ['ADFI', 'ADG', 'BW', 'MetabolicBW'] + \
               [f'{f}_MA3' for f in ['ADFI', 'ADG', 'BW', 'MetabolicBW']] + \
               [f'{f}_lag1' for f in ['ADFI', 'ADG', 'BW', 'MetabolicBW']] + \
               [f'{f}_lag2' for f in ['ADFI', 'ADG', 'BW', 'MetabolicBW']] + \
               ['ShiftDay', 'WEEK']

    target = 'ADSIDLysResidual'
    df_clean = df_weekly.dropna(subset=features + [target])

    unique_eids = df_clean['EID'].unique()
    train_eids, test_eids = train_test_split(unique_eids, test_size=0.3, random_state=42)

    train_df = df_clean[df_clean['EID'].isin(train_eids)]
    test_df = df_clean[df_clean['EID'].isin(test_eids)]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.2, random_state=42, enable_categorical=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'XGBoost Model Test Performance:\nMSE={mse:.4f}, R²={r2:.4f}')
    return model, X_train, y_train, X_test, y_test, y_pred

def symbolic_regression(X_train, y_train, variable_names):
    symbolic_model = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sin", "cos", "sqrt"],
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        complexity_of_variables=3,
        verbosity=1,
        random_state=42,
        deterministic=True,
        parallelism="serial",
        variable_names=variable_names
    )
    symbolic_model.fit(X_train.values, y_train.values)
    print("Best symbolic formula:")
    print(symbolic_model.get_best())
    return symbolic_model

def select_balanced_formula(equations_df, top_n=5, alpha=0.8, beta=0.2):
    # Drop missing values
    equations_df = equations_df.dropna(subset=['score', 'loss', 'complexity'])
    top_eqs = equations_df.sort_values(by='score', ascending=False).head(top_n).copy()

    loss_norm = (top_eqs['loss'] - top_eqs['loss'].min()) / (top_eqs['loss'].max() - top_eqs['loss'].min() + 1e-8)
    complexity_norm = (top_eqs['complexity'] - top_eqs['complexity'].min()) / (top_eqs['complexity'].max() - top_eqs['complexity'].min() + 1e-8)

    top_eqs['combined_score'] = alpha * loss_norm + beta * complexity_norm
    best = top_eqs.loc[top_eqs['combined_score'].idxmin()]

    print("\nSelected Balanced Formula (from Top 5):")
    print(f"Score: {best['score']:.4f}, Loss: {best['loss']:.4f}, Complexity: {best['complexity']}")
    print(f"Expression: {best['sympy_format']}")
    return best

def plot_feature_importance(model):
    importance_type = 'gain'
    importance = model.get_booster().get_score(importance_type=importance_type)
    importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 8), dpi=200)
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'XGBoost Feature Importance ({importance_type})')
    plt.xlabel('Average Gain (Importance)')
    plt.ylabel('Feature')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_pred):
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results_df['Residual'] = results_df['Actual'] - results_df['Predicted']

    plt.figure(figsize=(7,7), dpi=200)
    sns.scatterplot(x='Actual', y='Predicted', data=results_df, alpha=0.6)
    plt.plot([results_df.Actual.min(), results_df.Actual.max()], [results_df.Actual.min(), results_df.Actual.max()], 'r--')
    plt.xlabel('Actual SID Lys Residual')
    plt.ylabel('Predicted SID Lys Residual')
    plt.title('Predicted vs Actual')
    plt.show()

    plt.figure(figsize=(8,4), dpi=200)
    sns.histplot(results_df['Residual'], bins=30, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.show()

def plot_pareto_front(equations_df):
    plt.figure(figsize=(8, 5), dpi=200)
    sc = plt.scatter(equations_df['complexity'], equations_df['loss'], c=equations_df['score'], cmap='viridis', s=80)
    plt.colorbar(sc, label='R² Score')
    plt.xlabel('Complexity')
    plt.ylabel('Loss')
    plt.title('Pareto Front: Loss vs Complexity')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    df_daily = load_and_clean_data()
    df_weekly = aggregate_weekly(df_daily)
    df_weekly = add_time_series_features(df_weekly)

    # Prepare data for modeling
    df_weekly['ShiftDay'] = df_weekly['ShiftDay'].astype('category')
    features = ['ADFI', 'ADG', 'BW', 'MetabolicBW'] + \
               [f'{f}_MA3' for f in ['ADFI', 'ADG', 'BW', 'MetabolicBW']] + \
               [f'{f}_lag1' for f in ['ADFI', 'ADG', 'BW', 'MetabolicBW']] + \
               [f'{f}_lag2' for f in ['ADFI', 'ADG', 'BW', 'MetabolicBW']] + \
               ['ShiftDay', 'WEEK']

    target = 'ADSIDLysResidual'
    df_clean = df_weekly.dropna(subset=features + [target])

    unique_eids = df_clean['EID'].unique()
    train_eids, test_eids = train_test_split(unique_eids, test_size=0.3, random_state=42)

    train_df = df_clean[df_clean['EID'].isin(train_eids)]
    test_df = df_clean[df_clean['EID'].isin(test_eids)]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Train XGBoost model
    xgb_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.2, random_state=42, enable_categorical=True)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'XGBoost Test Performance:\nMSE: {mse:.4f}, R²: {r2:.4f}')

    # Plot feature importance
    plot_feature_importance(xgb_model)

    # Plot predicted vs actual and residuals
    plot_predictions(y_test, y_pred)

    # Symbolic regression: drop categorical features before fitting
    X_train_sym = X_train.drop(columns=['ShiftDay'], errors='ignore')

    symbolic_model = symbolic_regression(X_train_sym, y_train, variable_names=X_train_sym.columns.tolist())

    # Select balanced symbolic formula
    eq_df = symbolic_model.equations_
    best_formula = select_balanced_formula(eq_df)

    # Plot Pareto front for symbolic regression results
    plot_pareto_front(eq_df)

if __name__ == "__main__":
    main()
