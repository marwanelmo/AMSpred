# Function for data imputation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    classification_report,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.impute import SimpleImputer

def impute_missing_values(df):
    """
    Imputes missing values in a DataFrame for specified continuous and categorical columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with potential missing values in continuous and categorical columns.
    
    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    # Define imputers for continuous and categorical columns
    continuous_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Manually selected columns for continuous and categorical data
    continuous_columns = [
        'T0_age', 'T0_BMI', 'Time_pretreat_OK', 'OK_Duration_min', 'Length_of_stay', 
        'T0_30SCST', 'T0_fatigue', 'T0_protein_perc', 'T0_kcal_perc', 'T0_CT_SMI', 
        'T0_CT_SMRA', 't0_gses_totaal_score', 'T0_participation_ability', 
        'T0_participation_satisfaction', 't0_EQ_5D_5L_beschrijvend-systeem_score', 
        'T0_pain', 'T0', 'Time_OK_posttreat'
    ]
    categorical_columns = [
        'T0_VVMI_per', 'Education', 'household', 'T0_Tumorsize', 'T0_diseaseburden_cat', 
        'T0_selfcare', 'T0_Locusofcontrol_cat', 'T0_socialsupport_cat', 'T0_coping_cat', 
        'AMEXO_8_day1', 'AMEXO_9_day2', 'AMEXO_10_day3', 'T0_sondevoeding', 'T0_protein_cat', 
        'T0_kcal_cat', 'T0_ASM_low', 'T0_anxiety_cat', 'T0_depression_cat'
    ]
    
    # Check availability of columns and filter out any missing ones
    available_continuous_columns = [col for col in continuous_columns if col in df.columns]
    missing_continuous_columns = set(continuous_columns) - set(available_continuous_columns)
    
    available_categorical_columns = [col for col in categorical_columns if col in df.columns]
    missing_categorical_columns = set(categorical_columns) - set(available_categorical_columns)
    
    # Print missing columns if any
    if missing_continuous_columns:
        print(f"Missing continuous columns not found in df: {missing_continuous_columns}")
    if missing_categorical_columns:
        print(f"Missing categorical columns not found in df: {missing_categorical_columns}")
    
    # Apply median imputation to available continuous columns with missing values
    df[available_continuous_columns] = continuous_imputer.fit_transform(df[available_continuous_columns])
    
    # Apply mode imputation to available categorical columns with missing values
    df[available_categorical_columns] = categorical_imputer.fit_transform(df[available_categorical_columns])
    
    return df


# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Try to calculate ROC AUC, handle case where it can't be computed
    try:
        roc_auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
    except ValueError:
        roc_auc = np.nan  # Set to NaN if ROC AUC can't be calculated
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC AUC': roc_auc}


# Function to load and merge data
def load_and_merge_data(file1, file2, left_merge_column="Participant Id", right_merge_column="patientnummer", sep=";"):
    """
    Loads two CSV files and merges them on specified columns from each file.
    
    Parameters:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        left_merge_column (str): Column name in file1 to merge on.
        right_merge_column (str): Column name in file2 to merge on.
        sep (str): Delimiter for the CSV files, defaulting to ';'.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Load data with specified delimiter
    df1 = pd.read_csv(file1, sep=sep)
    df2 = pd.read_csv(file2, sep=sep)
    
    # Merge data on the specified columns
    merged_df = pd.merge(df1, df2, left_on=left_merge_column, right_on=right_merge_column, how="inner")
    merged_df.drop(columns=['patientnummer'], inplace=True)
    return merged_df

# Define configuration function with updated columns to drop
def configure_outcome(df, outcome_mode="single", outcome_type="absolute", time_horizon="T6", traditional_vars=False):
    """
    Configure the outcome prediction mode and select features based on specified time horizon 
    and traditional care variables.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        outcome_mode (str): Either 'single' or 'double'.
        outcome_type (str): If 'single', specify 'absolute' or 'relative' as the outcome type.
        time_horizon (str): The time point to use as the outcome measure (e.g., 'T6', 'T5').
        traditional_vars (bool): If True, only keeps traditionally available variables in the dataset.

    Returns:
        tuple: (x, y, stratify_label) where x is the feature DataFrame, y is the target DataFrame/Series,
               and stratify_label is the column used for stratified splitting.
    """
    # Validate time horizon input
    valid_time_horizons = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    if time_horizon not in valid_time_horizons:
        raise ValueError(f"Invalid time_horizon. Choose one of {valid_time_horizons}.")

    # Step 1: Drop rows with missing values in required columns based on outcome_mode and outcome_type
    if outcome_mode == "single" and outcome_type == "absolute":
        df = df.dropna(subset=[time_horizon])
    else:
        df = df.dropna(subset=[time_horizon, 'T0'])

    # Step 2: Create the outcome measure based on the time horizon and outcome type
    if outcome_mode == "single":
        if outcome_type == "absolute":
            df['outcome'] = (df[time_horizon] > df['T0']).astype(float)
        elif outcome_type == "relative":
            df['outcome'] = (df[time_horizon] > 45).astype(float)
        else:
            raise ValueError("Invalid outcome_type. Choose 'absolute' or 'relative'.")
        y = df['outcome']
        stratify_label = y  # Use 'outcome' directly for stratification in single outcome mode

    elif outcome_mode == "double":
        df['absolute_recovery'] = (df[time_horizon] > df['T0']).astype(float)
        df['relative_recovery'] = (df[time_horizon] > 45).astype(float)
        y = df[['absolute_recovery', 'relative_recovery']]
        
        # Create a composite label for stratification
        df['composite_recovery'] = df['absolute_recovery'].astype(str) + "_" + df['relative_recovery'].astype(str)
        stratify_label = df['composite_recovery']
    else:
        raise ValueError("Invalid outcome_mode. Choose 'single' or 'double'.")

    # Step 3: Drop rows with NaN in the outcome measure(s) after calculation
    if outcome_mode == "single":
        df = df.dropna(subset=['outcome'])
    else:
        df = df.dropna(subset=['absolute_recovery', 'relative_recovery'])

    # Step 4: Define features based on traditional_vars setting
    if traditional_vars:
        # Only retain traditional variables
        traditional_columns = [
            'pat_sexe', 'T0_age', 'T0_BMI', 'Tumorlocation_strat_1', 'Tumorlocation_strat_2',
            'Tumorlocation_strat_3', 'Tumorlocation_strat_4', 'T0_Tumorsize', 'T0_ASA', 'OK_pretreatment',
            'Time_pretreat_OK', 'OK_Technique', 'OK_Duration_min', 'Complications', 'AMEXO_8_day1', 
            'AMEXO_9_day2', 'Length_of_stay', 'OK_posttreatment', 'Time_OK_posttreat', 
            'T0_sondevoeding', 'T0_eetlust'
        ]
        x = df[traditional_columns]
    else:
        # Retain all non-time-based columns except for T0
        columns_to_drop = ['T0', 'T2', 'T3', 'T4', 'T5', 'T6', 'T2_relative', 'T3_relative', 'T4_relative', 'T5_relative', 'T6_relative']
        columns_to_drop += ['outcome', 'absolute_recovery', 'relative_recovery', 'Participant Id']
        x = df.drop(columns=columns_to_drop, errors='ignore')

    return x, y, stratify_label

def forward_feature_selection_dual(model, X_train, y_train_abs, y_train_rel, X_test, y_test_abs, y_test_rel):
    best_features, initial_features, best_score = [], [], 0
    while True:
        scores = []
        for f in X_train.columns:
            if f not in initial_features:
                # Evaluate feature for absolute recovery
                model.fit(X_train[initial_features + [f]], y_train_abs)
                pred_abs = model.predict(X_test[initial_features + [f]])
                score_abs = accuracy_score(y_test_abs, pred_abs)
                
                # Evaluate feature for relative recovery
                model.fit(X_train[initial_features + [f]], y_train_rel)
                pred_rel = model.predict(X_test[initial_features + [f]])
                score_rel = accuracy_score(y_test_rel, pred_rel)

                # Combine scores (average) for both outcomes
                combined_score = (score_abs + score_rel) / 2
                scores.append((combined_score, f))
                
        if not scores:
            break
        
        # Select the best feature in this round
        current_score, best_feature = sorted(scores, reverse=True)[0]
        if current_score > best_score:
            best_score = current_score
            initial_features.append(best_feature)
            best_features.append(best_feature)
        else:
            break
    return best_features

def train_models_with_outcomes(df, outcome_mode="single", outcome_type="absolute", time_horizon="T6", traditional_vars=False, plot=None):
    """
    Configure the dataset, perform forward feature selection, and train and evaluate models for 
    single or double outcome measures using Logistic Regression, Decision Tree, and XGBoost.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        outcome_mode (str): "single" or "double" outcome configuration.
        outcome_type (str): If outcome_mode is "single", specify "absolute" or "relative" outcome.
        time_horizon (str): The time point to use as the outcome measure (e.g., "T6", "T5").
        traditional_vars (bool): If True, retain only traditionally available variables.
        plot (dict): Dictionary specifying which plots to display. Keys are "confusion_matrix", "roc_curve", "feature_importance", "decision_tree".

    Returns:
        pd.DataFrame: DataFrame containing the evaluation metrics for each model and outcome.
    """
    # Default plot settings if None provided
    if plot is None:
        plot = {"confusion_matrix": True, "roc_curve": True, "feature_importance": True, "decision_tree": False}

    # Step 1: Configure the outcome and features using the provided arguments
    x, y, _ = configure_outcome(df, outcome_mode=outcome_mode, outcome_type=outcome_type, 
                             time_horizon=time_horizon, traditional_vars=traditional_vars)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Define models for multi-label classification
    models = [
        (OneVsRestClassifier(LogisticRegression(max_iter=10000)), "Logistic Regression"),
        (OneVsRestClassifier(DecisionTreeClassifier(random_state=42, max_leaf_nodes=15, min_samples_leaf=5)), "Decision Tree"),
        (OneVsRestClassifier(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')), "XGBoost")
    ]

    # Initialize results DataFrame
    all_results = pd.DataFrame()

    # Forward Feature Selection and Evaluation
    for model, name in models:
        # Step 2: Feature selection if outcome is double, otherwise use all features
        if outcome_mode == "double":
            features = forward_feature_selection_dual(model, X_train, y_train['absolute_recovery'], 
                                                      y_train['relative_recovery'], X_test, 
                                                      y_test['absolute_recovery'], y_test['relative_recovery'])
            X_train_fs, X_test_fs = X_train[features], X_test[features]
        else:
            features = X_train.columns.tolist()  # Use all features for single outcome
            X_train_fs, X_test_fs = X_train[features], X_test[features]

        # Step 3: Train the model and make predictions
        model.fit(X_train_fs, y_train)
        y_pred = model.predict(X_test_fs)
        
        # Step 4: Calculate metrics and handle plotting based on the outcome mode
        if outcome_mode == "double":
            # Separate predictions for absolute and relative recovery
            y_pred_abs = y_pred[:, 0]
            y_pred_rel = y_pred[:, 1]

            # Calculate metrics for both outcomes
            metrics_abs = calculate_metrics(y_test['absolute_recovery'], y_pred_abs)
            metrics_rel = calculate_metrics(y_test['relative_recovery'], y_pred_rel)

            # Store results with specified ordering
            results = pd.DataFrame({
                'Model': [f'Abs outcome {name}', f'Relative outcome {name}'],
                'Accuracy': [metrics_abs['Accuracy'], metrics_rel['Accuracy']],
                'Precision': [metrics_abs['Precision'], metrics_rel['Precision']],
                'Recall': [metrics_abs['Recall'], metrics_rel['Recall']],
                'F1 Score': [metrics_abs['F1 Score'], metrics_rel['F1 Score']],
                'ROC AUC': [metrics_abs['ROC AUC'], metrics_rel['ROC AUC']]
            })
            
            # Confusion Matrix
            if plot.get("confusion_matrix", False):
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.heatmap(confusion_matrix(y_test['absolute_recovery'], y_pred_abs), annot=True, fmt='d', cmap='Blues', ax=ax[0])
                ax[0].set_title(f'{name} Absolute Recovery Confusion Matrix')
                sns.heatmap(confusion_matrix(y_test['relative_recovery'], y_pred_rel), annot=True, fmt='d', cmap='Blues', ax=ax[1])
                ax[1].set_title(f'{name} Relative Recovery Confusion Matrix')
                plt.show()
                
            # ROC Curve
            if plot.get("roc_curve", False):
                for i, label in enumerate(['absolute_recovery', 'relative_recovery']):
                    fpr, tpr, _ = roc_curve(y_test[label], y_pred[:, i])
                    plt.plot(fpr, tpr, label=f'{name} {label}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name} ROC Curve')
                plt.legend()
                plt.show()

        else:
            # Calculate metrics for the single outcome
            metrics = calculate_metrics(y_test, y_pred)
            results = pd.DataFrame({
                'Model': [f'{outcome_type.capitalize()} outcome {name}'],
                'Accuracy': [metrics['Accuracy']],
                'Precision': [metrics['Precision']],
                'Recall': [metrics['Recall']],
                'F1 Score': [metrics['F1 Score']],
                'ROC AUC': [metrics['ROC AUC']]
            })

            # Confusion Matrix
            if plot.get("confusion_matrix", False):
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                plt.title(f'{name} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()

            # ROC Curve
            if plot.get("roc_curve", False):
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                plt.plot(fpr, tpr, label=f'{name} ROC Curve')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name} ROC Curve')
                plt.legend()
                plt.show()
        
        # Feature Importance
        if plot.get("feature_importance", False):
            if name == "Logistic Regression" and hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
                feature_importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
                plt.title(f'{name} Feature Importance')
                plt.show()
            
            elif name == "Decision Tree" and hasattr(model.estimators_[0], 'feature_importances_'):
                # Access the first estimator in the OneVsRestClassifier for feature importance
                importances = model.estimators_[0].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
                plt.title(f'{name} Feature Importance')
                plt.show()

            # Plot the decision tree structure
            if plot.get("decision_tree", False) and name == "Decision Tree":
                from sklearn.tree import plot_tree
                plt.figure(figsize=(20, 10))
                plot_tree(model.estimators_[0], filled=True, feature_names=features, class_names=['0', '1'])
                plt.title(f'{name} Decision Tree Structure')
                plt.show()

        # Append to all results
        all_results = pd.concat([all_results, results])

    # Null Model - Majority Class Predictor
    class NullModel:
        def fit(self, X, y): self.mode = y.mode().iloc[0]
        def predict(self, X): return np.tile(self.mode, (len(X), 1) if y.ndim > 1 else len(X))

    null_model = NullModel()
    null_model.fit(X_train, y_train)
    y_pred_null = null_model.predict(X_test)
    
    if outcome_mode == "double":
        metrics_abs = calculate_metrics(y_test['absolute_recovery'], y_pred_null[:, 0])
        metrics_rel = calculate_metrics(y_test['relative_recovery'], y_pred_null[:, 1])
        null_results = pd.DataFrame({
            'Model': ['Null Model Abs outcome', 'Null Model Relative outcome'],
            'Accuracy': [metrics_abs['Accuracy'], metrics_rel['Accuracy']],
            'Precision': [metrics_abs['Precision'], metrics_rel['Precision']],
            'Recall': [metrics_abs['Recall'], metrics_rel['Recall']],
            'F1 Score': [metrics_abs['F1 Score'], metrics_rel['F1 Score']],
            'ROC AUC': [metrics_abs['ROC AUC'], metrics_rel['ROC AUC']]
        })
    else:
        metrics = calculate_metrics(y_test, y_pred_null)
        null_results = pd.DataFrame({
            'Model': [f'Null Model {outcome_type.capitalize()} outcome'],
            'Accuracy': [metrics['Accuracy']],
            'Precision': [metrics['Precision']],
            'Recall': [metrics['Recall']],
            'F1 Score': [metrics['F1 Score']],
            'ROC AUC': [metrics['ROC AUC']]
        })

    # Combine all results in specified order
    all_results = pd.concat([all_results, null_results]).reset_index(drop=True)

    return all_results
print("Helper functions defined successfully.")
