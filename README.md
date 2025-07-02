Weekly Time Series Modeling with XGBoost and Symbolic Regression
================================================================

This project contains a complete pipeline for modeling weekly aggregated time series data 
using XGBoost regression and symbolic regression via PySR. The workflow involves data cleaning, 
feature engineering, stationarity testing, model training with cross-validation, hyperparameter 
tuning, evaluation, and interpretability analysis.

------------------------------------------------------------------------------

Table of Contents:
- Project Overview
- Features
- Data Description
- Installation
- Usage
- Modeling Approach
- Results and Visualization
- Symbolic Regression
- File Structure
- Notes
- Contact

------------------------------------------------------------------------------

Project Overview
----------------
The goal of this project is to build predictive models for weekly averaged animal performance 
metrics using multiple data sources. Advanced machine learning techniques and symbolic regression 
are used to create interpretable models.

Features
--------
- Load and clean multi-source Excel datasets
- Aggregate daily data into weekly blocks
- Stationarity testing via Augmented Dickey-Fuller test
- Generate lag and moving average time series features
- Train and tune XGBoost models with time series cross-validation
- Evaluate final model on a separate test set
- Visualize feature importance and residuals
- Use PySR to discover symbolic regression formulas

Data Description
----------------
Input data are daily measurements from three different sources, containing animal IDs (EID), 
weekly timestamps (WEEK), performance metrics (feed intake, body weight, etc.), and treatment groups (ShiftDay).

Data is cleaned to remove abnormal cases and aggregated to weekly granularity.

Installation
------------
1. Clone this repository:
   git clone https://github.com/yourusername/weekly-timeseries-modeling.git
   cd weekly-timeseries-modeling

2. (Optional) Create and activate a Python environment:
   python -m venv venv
   source venv/bin/activate  (Linux/macOS)
   venv\Scripts\activate     (Windows)

3. Install required packages:
   pip install -r requirements.txt

Usage
-----
1. Place your Excel data files in the `data/` folder or update file paths in main.py.

2. Run the main script:
   python main.py

3. Outputs:
   - Model training and evaluation results printed to console
   - Feature importance and residual plots displayed/saved
   - Symbolic regression formulas printed

Modeling Approach
-----------------
- Aggregate daily data into weekly averages
- Create rolling means and lag features for time series
- Use TimeSeriesSplit cross-validation for tuning
- Train XGBoost regression model with grid search hyperparameter tuning
- Evaluate using MSE and R² on test set
- Interpret model using feature importance and symbolic regression

Results and Visualization
-------------------------
- Gain-based feature importance plots
- Residual analysis plots (predicted vs actual and residual distributions)

Symbolic Regression
-------------------
- Use PySR to find interpretable mathematical formulas approximating model output
- Includes formula selection balancing loss and complexity


File Structure
--------------
.
├── data/                   # Excel data files
├── main.py                 # Full modeling pipeline script
├── requirements.txt        # Python dependencies
├── README.txt              # This file

Notes
-----
- Update Excel file paths in main.py for your environment
- Handle categorical variables (ShiftDay) correctly during training
- Lag features cause NaNs that are dropped before modeling
- Modify hyperparameter grid in main.py as needed

