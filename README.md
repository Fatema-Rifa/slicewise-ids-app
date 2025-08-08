# 5G Network Slice Intrusion Detection System (NIDD)

![App Screenshot](https://raw.githubusercontent.com/Fatema-Rifa/slicewise-ids-app/main/app.png)

A Streamlit-based application for detecting intrusions in 5G network slices (eMBB, mMTC, URLLC) using ensemble machine learning techniques.

## Features

- **Multi-slice Support**: Detects intrusions across three 5G network slices:
  - eMBB (Enhanced Mobile Broadband)
  - mMTC (Massive Machine-Type Communications)
  - URLLC (Ultra-Reliable Low-Latency Communications)
  
- **Ensemble Learning**: Combines predictions from multiple models:
  - XGBoost
  - Random Forest
  - SVM
  - MLP
  - Isolation Forest (for anomaly detection)

- **Comprehensive Metrics**: Evaluates models using 10+ performance metrics including:
  - Accuracy, Precision, Recall
  - F1, F2, MCC
  - ROC-AUC, PR-AUC
  - Brier Score, Log Loss

- **Explainable AI**: Integrated SHAP visualizations for model interpretability
- **Custom Predictions**: Interactive interface for making predictions on custom input
- **Performance Comparison**: Compare model performance across different slices

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Fatema-Rifa/slicewise-ids-app.git
   cd slicewise-ids-app
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

The application requires the following Python packages (see `requirements.txt`):

```
streamlit
plotly
scipy
seaborn
matplotlib
joblib
scikit-learn
xgboost
shap
```

## Usage

1. Prepare your dataset:
   - Place your CSV files for each network slice in the `./5G-NIDD` folder
   - Files should be named `eMBB.csv`, `mMTC.csv`, and `URLLC.csv`
   - Each file should contain network traffic data with a 'Label' column (0/1 or benign/attack)

2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Use the application:
   - Load the datasets using the sidebar
   - Train models for selected slices
   - View performance metrics and SHAP explanations
   - Make custom predictions with the interactive interface

## Data Preparation

Your dataset should include:
- Network traffic features (numeric and categorical)
- A 'Label' column indicating benign (0) or attack (1) traffic
- Consistent feature names across all slices

Example features:
- Protocol type, duration, packet sizes
- Source/destination information
- Traffic statistics (mean, sum, min, max)
- TCP/UDP specific features

## Acknowledgments

- The 5G-NIDD dataset
- Streamlit for the web application framework
- SHAP for model explainability
- scikit-learn and XGBoost for machine learning algorithms