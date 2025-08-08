import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from scipy.optimize import differential_evolution
import shap
import joblib
import json
from datetime import datetime
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for better visualization
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .st-b7 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .model-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .slice-title {
        color: #1f77b4;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .real-time-plot {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .attack {
        color: #ff4b4b;
        font-weight: bold;
    }
    .benign {
        color: #0068c9;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Add this constant at the top of your code (after imports)
PRE_OPTIMIZED_WEIGHTS = {
        'eMBB': {
            'xgb': 0.3750,
            'rf': 0.1887,
            'svm': 0.2254,
            'mlp': 0.0352,
            'IsolationForest': 0.1758
        },
        'mMTC': {
            'xgb': 0.3814,
            'rf': 0.2468,
            'svm': 0.0275,
            'mlp': 0.1578,
            'IsolationForest': 0.1866
        },
        'URLLC': {
            'xgb': 0.2631,
            'rf': 0.2744,
            'svm': 0.1056,
            'mlp': 0.0752,
            'IsolationForest': 0.2817
        }
    }

def to_python_types(obj):
    """Recursively convert numpy types in dicts/lists to native Python types."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_feature_ranges(ids):
    """Save feature ranges to disk"""
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    ranges_path = "saved_models/feature_ranges.json"
    # Convert to Python types before saving
    with open(ranges_path, 'w') as f:
        json.dump(to_python_types(ids.feature_ranges), f)

def save_models(self):
    """Save all trained models to disk"""
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    for slice_name in self.models:
        model_path = f"saved_models/{slice_name}_model.joblib"
        # Save model data including results
        model_data = {
            'model': self.models[slice_name],
            'results': self.results.get(slice_name, None)
        }
        joblib.dump(model_data, model_path)
    
    ranges_path = "saved_models/feature_ranges.json"
    with open(ranges_path, 'w') as f:
        json.dump(to_python_types(self.feature_ranges), f)
    
    st.success("All models and feature ranges saved successfully!")

def load_models(self):
    """Load all trained models from disk"""
    loaded = False
    for slice_name in self.slices:
        model_path = f"saved_models/{slice_name}_model.joblib"
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.models[slice_name] = model_data['model']
            if model_data['results']:
                self.results[slice_name] = model_data['results']
            loaded = True
    
    ranges_path = "saved_models/feature_ranges.json"
    if os.path.exists(ranges_path):
        with open(ranges_path, 'r') as f:
            self.feature_ranges = json.load(f)
        loaded = True
    
    if loaded:
        st.success("Models and feature ranges loaded successfully!")
    return loaded

def load_feature_ranges(ids):
    """Load feature ranges from disk"""
    ranges_path = "saved_models/feature_ranges.json"
    if os.path.exists(ranges_path):
        with open(ranges_path, 'r') as f:
            ids.feature_ranges = json.load(f)
        return True
    return False

class NetworkSliceIDS:
    """Comprehensive 5G Network Slice Intrusion Detection System"""
    
    def __init__(self, data_folder='./5G-NIDD'):
        self.data_folder = data_folder
        self.slices = ['eMBB', 'mMTC', 'URLLC']
        self.datasets = {slice_name: None for slice_name in self.slices}
        self.models = {}
        self.results = {}
        self.preprocessor = None
        self.all_metrics = []
        self.training_metrics_history = {slice: [] for slice in self.slices}
        self.current_slice = None
        self.feature_ranges = {}  # Store min/max for each feature for validation

    def load_datasets(self):
        """Load individual slice datasets with validation"""
        st.info("Loading network slice datasets...")
        
        if not os.path.exists(self.data_folder):
            st.error(f"Data folder not found at {self.data_folder}")
            return False

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, slice_name in enumerate(self.slices):
            progress_bar.progress((i + 1) / len(self.slices))
            status_text.text(f"Loading {slice_name} dataset...")
            
            file_path = os.path.join(self.data_folder, f"{slice_name}.csv")
            if not os.path.exists(file_path):
                st.warning(f"Warning: {slice_name}.csv not found. Skipping {slice_name}.")
                continue

            try:
                df = pd.read_csv(file_path)
                df = self._clean_data(df)
                X, y, numeric_cols, categorical_cols = self._prepare_features_labels(df, slice_name)
                self.datasets[slice_name] = {
                    'X': X,
                    'y': y,
                    'numeric_cols': numeric_cols,
                    'categorical_cols': categorical_cols,
                    'df': df  # Store original dataframe for reference
                }
                
                # Calculate feature ranges for validation
                self._calculate_feature_ranges(slice_name, X, numeric_cols, categorical_cols)
                
                st.success(f"Loaded {slice_name} dataset with {len(df)} samples")
            except Exception as e:
                st.error(f"Error loading {slice_name}: {str(e)}")
                return False
        
        status_text.text("Dataset loading complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        return True

    def _calculate_feature_ranges(self, slice_name, X, numeric_cols, categorical_cols):
        """Calculate min/max for numeric features and categories for categorical features"""
        self.feature_ranges[slice_name] = {
            'numeric': {},
            'categorical': {}
        }
        
        # Numeric features
        for col in numeric_cols:
            self.feature_ranges[slice_name]['numeric'][col] = {
                'min': X[col].min(),
                'max': X[col].max()
            }
        
        # Categorical features
        for col in categorical_cols:
            self.feature_ranges[slice_name]['categorical'][col] = {
                'categories': list(X[col].unique())
            }

    def _clean_data(self, df):
        """Enhanced data cleaning pipeline"""
        df = df.dropna(axis=1, how='all').drop_duplicates()

        if 'Label' not in df.columns:
            raise ValueError("No 'Label' column found in dataset")

        df['Label'] = df['Label'].str.strip().str.lower()
        return df

    def _prepare_features_labels(self, df, slice_name):
        """Feature engineering and preprocessing for each slice"""
        df = df.copy()

        # Identify categorical columns
        categorical_cols = ['Proto', 'State', 'Cause', 'sTos', 'dTos', 'sDSb', 'dDSb']
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        # Identify numeric columns
        exclude_cols = ['Label', 'UniqueID', 'predicted']
        numeric_cols = [col for col in df.columns
                      if col not in exclude_cols + categorical_cols
                      and pd.api.types.is_numeric_dtype(df[col])]

        # Prepare features - select only the feature columns
        X = df[numeric_cols + categorical_cols].copy()

        # Convert categorical columns to strings
        for col in categorical_cols:
            X[col] = X[col].astype(str)

        # Convert labels to binary
        attack_labels = {'attack', 'malicious', '1', 'true'}
        y = df['Label'].apply(lambda x: 1 if str(x).lower() in attack_labels else 0)

        class_counts = y.value_counts()
        st.write(f"{slice_name} class distribution - Benign: {class_counts.get(0, 0)}, Attack: {class_counts.get(1, 0)}")

        if len(class_counts) < 2:
            raise ValueError(f"Only one class present in {slice_name}: {class_counts.index[0]}")

        return X, y, numeric_cols, categorical_cols

    def train_slice_model(self, slice_name, progress_bar=None, status_text=None):
        """Train and evaluate model for a specific network slice using pre-optimized weights"""
        if slice_name not in self.datasets or self.datasets[slice_name] is None:
            st.error(f"No data available for {slice_name} slice")
            return None

        self.current_slice = slice_name

        st.markdown(f'<div class="slice-title">{slice_name} Slice Training</div>', unsafe_allow_html=True)

        X, y, numeric_cols, categorical_cols = (
            self.datasets[slice_name]['X'], 
            self.datasets[slice_name]['y'], 
            self.datasets[slice_name]['numeric_cols'], 
            self.datasets[slice_name]['categorical_cols']
        )

        # Build preprocessing pipeline
        self.preprocessor = ColumnTransformer([
            ('num', make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler()), numeric_cols),
            ('cat', make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing'),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical_cols)
        ])

        # Stratified split preserving class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42)

        # Preprocessing
        if status_text:
            status_text.text("Preprocessing data...")
        
        try:
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            if hasattr(X_train_processed, 'toarray'):
                X_train_processed = X_train_processed.toarray()
                X_test_processed = X_test_processed.toarray()

            if np.isnan(X_train_processed).any() or np.isnan(X_test_processed).any():
                raise ValueError("NaN values in processed data")

        except Exception as e:
            st.error(f"Preprocessing error: {str(e)}")
            return None

        # Initialize models with class balancing
        base_models = [
            ('xgb', XGBClassifier(
                eval_metric='logloss',
                scale_pos_weight=sum(y_train==0)/sum(y_train==1),
                random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42)),
            ('svm', make_pipeline(
                StandardScaler(with_mean=False),
                CalibratedClassifierCV(
                    SVC(probability=True, kernel='linear',
                        class_weight='balanced',
                        random_state=42)))),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(50,25),
                early_stopping=True,
                random_state=42))
        ]

        isolation_forest = IsolationForest(
            contamination=min(0.1, sum(y_train)/len(y_train)),
            random_state=42)

        # Train models with progress updates
        if status_text:
            status_text.text("Using pre-optimized weights...")

        st.markdown("### Ensemble Weights")
        best_weights = [
            PRE_OPTIMIZED_WEIGHTS[slice_name]['xgb'],
            PRE_OPTIMIZED_WEIGHTS[slice_name]['rf'],
            PRE_OPTIMIZED_WEIGHTS[slice_name]['svm'],
            PRE_OPTIMIZED_WEIGHTS[slice_name]['mlp'],
            PRE_OPTIMIZED_WEIGHTS[slice_name]['IsolationForest']
        ]

        st.success("Using pre-optimized weights:")
        st.write(f"XGBoost: {best_weights[0]:.4f}")
        st.write(f"Random Forest: {best_weights[1]:.4f}")
        st.write(f"SVM: {best_weights[2]:.4f}")
        st.write(f"MLP: {best_weights[3]:.4f}")
        st.write(f"IsolationForest: {best_weights[4]:.4f}")

        # Evaluation
        if status_text:
            status_text.text("Evaluating ensemble model...")

        try:
            # Train base models
            trained_models = []
            for name, model in base_models:
                model.fit(X_train_processed, y_train)
                trained_models.append((name, model))
            
            # Train Isolation Forest
            isolation_forest.fit(X_train_processed)

            # Calculate test predictions
            test_preds = [model.predict_proba(X_test_processed)[:,1] for _, model in trained_models]
            test_preds.append((isolation_forest.decision_function(X_test_processed) -
                            np.min(isolation_forest.decision_function(X_test_processed))) /
                            (np.max(isolation_forest.decision_function(X_test_processed)) -
                            np.min(isolation_forest.decision_function(X_test_processed))))

            y_proba = np.average(np.vstack(test_preds), axis=0, weights=best_weights)
            y_pred = (y_proba > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)

            # Calculate metrics
            metrics = {
                'Slice': slice_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'F2': fbeta_score(y_test, y_pred, beta=2),
                'MCC': matthews_corrcoef(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'PR-AUC': average_precision_score(y_test, y_proba),
                'Brier': brier_score_loss(y_test, y_proba),
                'LogLoss': log_loss(y_test, y_proba)
            }

            # Store results
            self.results[slice_name] = metrics
            self.all_metrics.append(metrics)
            self.training_metrics_history[slice_name].append(metrics)

            # Store test data with the model
            self.models[slice_name] = {
                'base_models': dict(trained_models),
                'isolation_forest': isolation_forest,
                'weights': best_weights,
                'preprocessor': self.preprocessor,
                'test_data': (X_test, y_test, y_pred, y_proba),
                'feature_names': self._get_processed_feature_names(),
                'metrics': metrics
            }

            return metrics

        except Exception as e:
            st.error(f"Error during model training/evaluation: {str(e)}")
            return None

    def save_models(self):
        """Save all trained models to disk"""
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        for slice_name in self.models:
            model_path = f"saved_models/{slice_name}_model.joblib"
            joblib.dump(self.models[slice_name], model_path)
        
        ranges_path = "saved_models/feature_ranges.json"
        with open(ranges_path, 'w') as f:
            json.dump(to_python_types(self.feature_ranges), f)
        
        st.success("All models and feature ranges saved successfully!")

    def load_models(self):
        """Load all trained models from disk"""
        loaded = False
        for slice_name in self.slices:
            model_path = f"saved_models/{slice_name}_model.joblib"
            if os.path.exists(model_path):
                self.models[slice_name] = joblib.load(model_path)
                loaded = True
        
        ranges_path = "saved_models/feature_ranges.json"
        if os.path.exists(ranges_path):
            with open(ranges_path, 'r') as f:
                self.feature_ranges = json.load(f)
            loaded = True
        
        if loaded:
            st.success("Models and feature ranges loaded successfully!")
        return loaded

    def predict_custom_input(self, slice_name, input_data):
        """Make prediction on custom input data"""
        if slice_name not in self.models:
            st.error(f"No model trained for {slice_name}")
            return None
        
        # Validate input data
        validation_result = self._validate_input_data(slice_name, input_data)
        if not validation_result['valid']:
            st.error(f"Input validation failed: {validation_result['message']}")
            return None
        
        # Preprocess the input
        try:
            model_data = self.models[slice_name]
            preprocessor = model_data['preprocessor']
            
            # Convert to DataFrame for preprocessing
            input_df = pd.DataFrame([input_data])
            
            # Preprocess
            processed_input = preprocessor.transform(input_df)
            if hasattr(processed_input, 'toarray'):
                processed_input = processed_input.toarray()
            
            # Get predictions from each model
            preds = [model.predict_proba(processed_input)[:,1] 
                    for name, model in model_data['base_models'].items()]
            
            # Get anomaly score from Isolation Forest
            iforest_score = (model_data['isolation_forest'].decision_function(processed_input) -
                        np.min(model_data['isolation_forest'].decision_function(processed_input))) / \
                        (np.max(model_data['isolation_forest'].decision_function(processed_input)) -
                        np.min(model_data['isolation_forest'].decision_function(processed_input)))
            preds.append(iforest_score)
            
            # Combine predictions using the weights stored in model_data
            combined_score = np.average(np.array(preds), axis=0, weights=model_data['weights'])
            combined_score = float(combined_score[0])  # Extract scalar
            
            # Get SHAP explanation
            explainer = shap.Explainer(model_data['base_models']['xgb'], 
                                    feature_names=model_data['feature_names'])
            shap_values = explainer(processed_input)
            
            return {
                'score': combined_score,
                'prediction': 1 if combined_score > 0.5 else 0,
                'shap_values': shap_values,
                'processed_input': processed_input
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def _validate_input_data(self, slice_name, input_data):
        """Validate custom input data against expected features and ranges"""
        if slice_name not in self.feature_ranges:
            return {'valid': False, 'message': f"No feature ranges available for {slice_name}"}
        
        # Check all required features are present
        required_features = set(self.feature_ranges[slice_name]['numeric'].keys()) | \
                        set(self.feature_ranges[slice_name]['categorical'].keys())
        
        missing_features = required_features - set(input_data.keys())
        if missing_features:
            return {'valid': False, 'message': f"Missing features: {', '.join(missing_features)}"}
        
        # Check numeric features are within expected ranges
        for feature, ranges in self.feature_ranges[slice_name]['numeric'].items():
            value = input_data[feature]
            if not (ranges['min'] <= value <= ranges['max']):
                return {'valid': False, 
                    'message': f"Value for {feature} ({value}) outside expected range [{ranges['min']}, {ranges['max']}]"}
        
        # Check categorical features have valid values
        for feature, info in self.feature_ranges[slice_name]['categorical'].items():
            value = str(input_data[feature])
            if value not in info['categories']:
                return {'valid': False, 
                    'message': f"Invalid value for {feature}: {value}. Expected one of {info['categories']}"}
        
        return {'valid': True, 'message': 'Input data is valid'}

    def generate_shap_visualizations(self, slice_name, shap_values=None, input_data=None):
        """Generate SHAP visualizations for a slice or specific input"""
        if slice_name not in self.models:
            st.error(f"No model trained for {slice_name}")
            return

        model_data = self.models[slice_name]
        
        if shap_values is None:
            # Generate global SHAP values if none provided
            X = self.datasets[slice_name]['X']
            X_sample = X.sample(min(100, len(X)))
            X_processed = model_data['preprocessor'].transform(X_sample)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            
            explainer = shap.Explainer(model_data['base_models']['xgb'], 
                                    feature_names=model_data['feature_names'])
            shap_values = explainer(X_processed)
        
        st.markdown(f"### SHAP Analysis for {slice_name}")
        
        if input_data is not None:
            # Individual prediction explanation - use columns for side-by-side layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Prediction Explanation")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], max_display=15, show=False)
                st.pyplot(fig)
                
            with col2:
                st.markdown("#### Input Feature Values")
                input_df = pd.DataFrame([input_data]).T
                input_df.columns = ['Value']
                st.dataframe(input_df)
        else:
            # Create two columns for global visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Global Feature Importance
                st.markdown("#### Global Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=20, show=False)
                st.pyplot(fig)
                
            with col2:
                # Beeswarm Plot
                st.markdown("#### Feature Impact Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.beeswarm(shap_values, max_display=20, show=False)
                st.pyplot(fig)

    def compare_slices(self):
        """Generate comprehensive comparison report"""
        if not self.models:
            st.error("No models available. Train or load models first.")
            return None

        # Try to get metrics from either results or models
        available_metrics = []
        for slice_name in self.models:
            if slice_name in self.results:
                available_metrics.append(self.results[slice_name])
            elif hasattr(self.models[slice_name], 'metrics'):
                available_metrics.append(self.models[slice_name]['metrics'])
        
        if not available_metrics:
            st.error("No metrics available for comparison")
            return None

        st.markdown("## Slice Performance Comparison")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(available_metrics).set_index('Slice')
        
        # Display metrics comparison
        st.markdown("### Metrics Comparison")
        st.dataframe(comparison_df.drop(columns=['test_data'], errors='ignore'))
        
        # Visualizations
        st.markdown("### Performance Metrics Comparison")
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
        for metric in metrics:
            fig.add_trace(go.Bar(
                x=comparison_df.index,
                y=comparison_df[metric],
                name=metric
            ))
            
        fig.update_layout(
            barmode='group',
            title='Performance Metrics Across Slices',
            xaxis_title='Network Slice',
            yaxis_title='Score',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curves Comparison
        st.markdown("### ROC Curves Comparison")
        fig = go.Figure()
        
        for slice_name, model_data in self.models.items():
            if model_data is not None:
                X_test, y_test, _, y_proba = model_data['test_data']
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{slice_name} (AUC={auc(fpr,tpr):.3f})',
                    mode='lines'
                ))
                
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return comparison_df

    def _get_processed_feature_names(self):
        """Extract feature names after preprocessing"""
        numeric_features = self.preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out()
        return list(numeric_features) + list(categorical_features)

def get_sample_input(slice_name):
    """Return sample input data for a given slice"""
    samples = {
        'eMBB': {
            'X': 1, 'Seq': 12345, 'Dur': 0.5, 'RunTime': 100, 'Mean': 1000, 
            'Sum': 5000, 'Min': 800, 'Max': 1200, 'Proto': 'tcp', 
            'sTtl': 64, 'dTtl': 64, 'sHops': 5, 'dHops': 5, 
            'TotPkts': 10, 'SrcPkts': 5, 'DstPkts': 5, 
            'TotBytes': 5000, 'SrcBytes': 2500, 'DstBytes': 2500,
            'sMeanPktSz': 500, 'dMeanPktSz': 500, 'Load': 1000,
            'SrcLoad': 500, 'DstLoad': 500, 'Loss': 0, 'SrcLoss': 0,
            'DstLoss': 0, 'pLoss': 0, 'SrcGap': 0, 'DstGap': 0,
            'Rate': 100, 'SrcRate': 50, 'DstRate': 50,
            'SrcWin': 8192, 'DstWin': 8192, 'SrcTCPBase': 12345,
            'DstTCPBase': 54321, 'TcpRtt': 0.05, 'SynAck': 0.02,
            'AckDat': 0.03, 'sTos': '0', 'dTos': '0',
            'sDSb': '0', 'dDSb': '0', 'State': 'ESTABLISHED',
            'Cause': 'Normal'
        },
        'mMTC': {
            'X': 1, 'Seq': 54321, 'Dur': 0.2, 'RunTime': 50, 'Mean': 500, 
            'Sum': 2500, 'Min': 400, 'Max': 600, 'Proto': 'udp', 
            'sTtl': 64, 'dTtl': 64, 'sHops': 3, 'dHops': 3, 
            'TotPkts': 20, 'SrcPkts': 10, 'DstPkts': 10, 
            'TotBytes': 2000, 'SrcBytes': 1000, 'DstBytes': 1000,
            'sMeanPktSz': 100, 'dMeanPktSz': 100, 'Load': 500,
            'SrcLoad': 250, 'DstLoad': 250, 'Loss': 0, 'SrcLoss': 0,
            'DstLoss': 0, 'pLoss': 0, 'SrcGap': 0, 'DstGap': 0,
            'Rate': 200, 'SrcRate': 100, 'DstRate': 100,
            'SrcWin': 0, 'DstWin': 0, 'SrcTCPBase': 0,
            'DstTCPBase': 0, 'TcpRtt': 0, 'SynAck': 0,
            'AckDat': 0, 'sTos': '0', 'dTos': '0',
            'sDSb': '0', 'dDSb': '0', 'State': 'CONNECTED',
            'Cause': 'Normal'
        },
        'URLLC': {
            'X': 1, 'Seq': 98765, 'Dur': 0.1, 'RunTime': 20, 'Mean': 200, 
            'Sum': 1000, 'Min': 150, 'Max': 250, 'Proto': 'tcp', 
            'sTtl': 64, 'dTtl': 64, 'sHops': 2, 'dHops': 2, 
            'TotPkts': 5, 'SrcPkts': 3, 'DstPkts': 2, 
            'TotBytes': 1000, 'SrcBytes': 600, 'DstBytes': 400,
            'sMeanPktSz': 200, 'dMeanPktSz': 200, 'Load': 200,
            'SrcLoad': 120, 'DstLoad': 80, 'Loss': 0, 'SrcLoss': 0,
            'DstLoss': 0, 'pLoss': 0, 'SrcGap': 0, 'DstGap': 0,
            'Rate': 50, 'SrcRate': 30, 'DstRate': 20,
            'SrcWin': 4096, 'DstWin': 4096, 'SrcTCPBase': 98765,
            'DstTCPBase': 56789, 'TcpRtt': 0.01, 'SynAck': 0.005,
            'AckDat': 0.005, 'sTos': '0', 'dTos': '0',
            'sDSb': '0', 'dDSb': '0', 'State': 'ESTABLISHED',
            'Cause': 'Normal'
        }
    }
    return samples.get(slice_name, {})

def create_input_fields(slice_name, ids):
    """Create input fields for custom prediction based on slice features"""
    if slice_name not in ids.feature_ranges:
        st.warning("Please load and train the model for this slice first")
        return None

    st.markdown(f"### Custom Prediction for {slice_name} Slice")

    # Get sample input for this slice
    sample_input = get_sample_input(slice_name)

    # Create two columns for better organization
    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        # Numeric features
        st.markdown("#### Numeric Features")
        for feature, ranges in ids.feature_ranges[slice_name]['numeric'].items():
            min_val = float(ranges['min'])
            max_val = float(ranges['max'])
            if min_val == max_val:
                # Add constant features directly to input_data
                input_data[feature] = min_val
                continue
            sample_val = float(sample_input.get(feature, (min_val + max_val) / 2))
            default_value = min(max(sample_val, min_val), max_val)
            input_data[feature] = st.number_input(
                f"{feature} ({min_val:.2f} to {max_val:.2f})",
                min_value=min_val,
                max_value=max_val,
                value=default_value,
                step=0.01
            )

    with col2:
        # Categorical features
        st.markdown("#### Categorical Features")
        for feature, info in ids.feature_ranges[slice_name]['categorical'].items():
            categories = info['categories']
            default_value = sample_input.get(feature, categories[0] if categories else 'missing')
            input_data[feature] = st.selectbox(
                f"{feature}",
                options=categories,
                index=categories.index(default_value) if default_value in categories else 0
            )

    return input_data

# Streamlit App UI
def main():
    st.set_page_config(
        page_title="5G Network Slice IDS",
        page_icon="📶",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📶 5G Network Slice Intrusion Detection System")
    st.markdown("""
    This application implements a comprehensive intrusion detection system for 5G network slices (eMBB, mMTC, URLLC) 
    using ensemble machine learning techniques.
    """)
    
    # Initialize the IDS
    if 'ids' not in st.session_state:
        st.session_state.ids = NetworkSliceIDS()
        # Try to load saved models
        st.session_state.ids.load_models()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        data_folder = st.text_input("Data Folder Path", "./5G-NIDD")
        
        selected_slices = st.multiselect(
            "Select Slices to Process",
            options=st.session_state.ids.slices,
            default=st.session_state.ids.slices
        )
        
        if st.button("Load Datasets"):
            st.session_state.ids.data_folder = data_folder
            with st.spinner("Loading datasets..."):
                if st.session_state.ids.load_datasets():
                    st.success("Datasets loaded successfully!")
                else:
                    st.error("Failed to load datasets")
        
        st.markdown("---")
        st.header("Training Options")
        train_button = st.button("Train Selected Slices")
        
        # Add save/load buttons
        if st.button("Save All Models"):
            if hasattr(st.session_state.ids, 'models') and st.session_state.ids.models:
                st.session_state.ids.save_models()
            else:
                st.warning("No models to save. Please train models first.")
        
        if st.button("Load All Models"):
            if st.session_state.ids.load_models():
                st.success("Models loaded successfully!")
            else:
                st.warning("No saved models found")
        
        st.markdown("---")
        st.header("Visualization")
        show_shap = st.checkbox("Generate SHAP Explanations", value=True)
        compare_button = st.button("Compare Slices")
        
        st.markdown("---")
        st.header("Custom Prediction")
        predict_slice = st.selectbox(
            "Select Slice for Prediction",
            options=st.session_state.ids.slices,
            index=0
        )
    
    # Main content area
    if train_button:
        if not hasattr(st.session_state.ids, 'datasets') or all(v is None for v in st.session_state.ids.datasets.values()):
            st.warning("Please load datasets first!")
        else:
            for slice_name in selected_slices:
                if st.session_state.ids.datasets[slice_name] is not None:
                    with st.expander(f"Training {slice_name} Slice", expanded=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        metrics = st.session_state.ids.train_slice_model(
                            slice_name, progress_bar, status_text)
                        
                        if show_shap and metrics:
                            st.session_state.ids.generate_shap_visualizations(slice_name)
                        
                        # Save the model after training
                        st.session_state.ids.save_models()
                        save_feature_ranges(st.session_state.ids)
                        
                        progress_bar.empty()
    
    if compare_button:
        if not st.session_state.ids.results:
            st.warning("Please train models first!")
        else:
            st.session_state.ids.compare_slices()
    
    # Display dataset information if loaded
    if hasattr(st.session_state.ids, 'datasets'):
        st.markdown("## Dataset Information")
        cols = st.columns(len(selected_slices))
        
        for idx, slice_name in enumerate(selected_slices):
            if st.session_state.ids.datasets[slice_name] is not None:
                with cols[idx]:
                    st.markdown(f"**{slice_name}**")
                    st.write(f"Samples: {len(st.session_state.ids.datasets[slice_name]['X'])}")
                    st.write(f"Features: {len(st.session_state.ids.datasets[slice_name]['numeric_cols']) + len(st.session_state.ids.datasets[slice_name]['categorical_cols'])}")
                    st.write(f"Attack ratio: {sum(st.session_state.ids.datasets[slice_name]['y'])/len(st.session_state.ids.datasets[slice_name]['y']):.2%}")

    # Add the custom prediction section
    st.markdown("---")
    st.markdown("## Make Custom Predictions")
    
    # Check if models are trained or loaded
    model_available = (
        hasattr(st.session_state.ids, 'models') and 
        predict_slice in st.session_state.ids.models and
        hasattr(st.session_state.ids, 'feature_ranges') and 
        predict_slice in st.session_state.ids.feature_ranges
    )
    
    if not model_available:
        st.warning(f"Please train or load the {predict_slice} model first")
    else:
        # Create input fields for the selected slice
        input_data = create_input_fields(predict_slice, st.session_state.ids)
        
        if input_data and st.button("Make Prediction"):
            with st.spinner("Making prediction..."):
                prediction = st.session_state.ids.predict_custom_input(predict_slice, input_data)
                
                if prediction is not None:
                    # Display prediction result
                    st.markdown("### Prediction Result")
                    
                    if prediction['prediction'] == 1:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Prediction</h4>
                            <h2 class="attack">ATTACK DETECTED</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Prediction</h4>
                            <h2 class="benign">BENIGN TRAFFIC</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show SHAP explanation
                    st.session_state.ids.generate_shap_visualizations(
                        predict_slice, 
                        prediction['shap_values'],
                        input_data
                    )

if __name__ == "__main__":
    main()