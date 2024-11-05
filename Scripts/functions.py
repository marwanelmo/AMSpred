import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

def process_castor_fysiek(df, export=True):
    import pandas as pd
    # Define the columns to be used in the process, specifically targeting the scores and patient number
    columns = [
        't0_Dutch-Flemish_PROMIS_bank_v1.2_-_US_v0.2_-_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score', 
        't0_Dutch-Flemish_PROMIS_Bank_v1.2_–_US_v0.2_–_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't2_Dutch-Flemish_PROMIS_bank_v1.2_-_US_v0.2_-_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't2_Dutch-Flemish_PROMIS_Bank_v1.2_–_US_v0.2_–_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't3_Dutch-Flemish_PROMIS_bank_v1.2_-_US_v0.2_-_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't3_Dutch-Flemish_PROMIS_Bank_v1.2_–_US_v0.2_–_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't4_Dutch-Flemish_PROMIS_bank_v1.2_-_US_v0.2_-_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't4_Dutch-Flemish_PROMIS_Bank_v1.2_–_US_v0.2_–_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't5_Dutch-Flemish_PROMIS_bank_v1.2_-_US_v0.2_-_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't5_Dutch-Flemish_PROMIS_Bank_v1.2_–_US_v0.2_–_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't6_Dutch-Flemish_PROMIS_bank_v1.2_-_US_v0.2_-_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        't6_Dutch-Flemish_PROMIS_Bank_v1.2_–_US_v0.2_–_Lichamelijk_functioneren_|_Nederlandse_versie_cat_score_score',
        'patientnummer'
    ]
    # Filter the dataframe to only include the specified columns
    filtered_castor = df.loc[:, columns]

    # Initialize an empty DataFrame to store the combined physical functioning scores
    combined_fysiek = pd.DataFrame()
    # Iterate through the time points, combining scores from two versions of the questionnaire per time point
    for i in range(1, 7):  
        # Select the two columns for the current time point
        df = filtered_castor.iloc[:, (i-1)*2:(i-1)*2+2]
        # Fill missing values in the first column with values from the second column
        combined_fysiek[i] = df.iloc[:, 0].fillna(df.iloc[:, 1])
    # Add the patient number column as the last column in the combined DataFrame
    combined_fysiek[7] = filtered_castor.iloc[:, -1]
    # Rename the columns of the combined DataFrame for clarity
    combined_fysiek.columns = ['T0', 'T2', 'T3', 'T4', 'T5', 'T6', 'patientnummer']

    # Export the combined DataFrame to an Excel file if the export flag is set to True
    if export:
        combined_fysiek.to_excel('export/Export Fysiek Functioneren.xlsx')
    for col in ['T2', 'T3', 'T4', 'T5', 'T6']:
        # Create new column names by appending '_relative' to the original column names
        new_col_name = f"{col}_relative"
        # Calculate the relative value by dividing the Tx column by the T0 column
        combined_fysiek[new_col_name] = combined_fysiek[col] / combined_fysiek['T0']

    # Return the combined DataFrame
    return combined_fysiek

def process_castor_baseline(df, columns):
    import pandas as pd
    vragen = columns[0].to_list()

    castor_baseline_filter = df.loc[:, vragen]

    return castor_baseline_filter

def process_castor(df, columns):

    vragen = columns[0].to_list()

    castor_filter = df.loc[:, vragen]
    #return vragen
    return castor_filter

def impute_data(data, continuous_columns, categorical_columns):
    # Impute continuous columns with median strategy
    existing_continuous_columns = [col for col in continuous_columns if col in data.columns]
    if existing_continuous_columns:
        data[existing_continuous_columns] = SimpleImputer(strategy='median').fit_transform(data[existing_continuous_columns])

    # Impute categorical columns with mode strategy
    existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
    if existing_categorical_columns:
        data[existing_categorical_columns] = SimpleImputer(strategy='most_frequent').fit_transform(data[existing_categorical_columns])
    
    return data

# Fill specific columns with 0
def fill_columns_with_zero(data, columns):
    for col in columns:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    return data

# Function to load and merge data
def load_and_merge_data(voorspellers_path, uitkomstmaat_path):
    voorspellers_df = pd.read_csv(voorspellers_path, delimiter=';')
    uitkomstmaat_df = pd.read_csv(uitkomstmaat_path, delimiter=';')
    uitkomstmaat_vars = uitkomstmaat_df[["patientnummer", "T0", "T6"]]
    
    ML_df = pd.merge(voorspellers_df, uitkomstmaat_vars, left_on='Participant Id', right_on='patientnummer', how='inner')
    ML_df.drop(columns=['Participant Id', 'patientnummer'], inplace=True)
    
    return ML_df

# Function to create derived columns
def create_derived_columns(df):
    df['T5_relative'] = df['T5'] / df['T0']
    df['T5_relativebin'] = df['T5_relative'] >= 1
    df['T5_more_than_45'] = df['T5'] > 45
    df['T5_recovered'] = (df['T5_relativebin']) & (df['T5_more_than_45'])
    df['Outcome'] = (df['T5_relativebin'].astype(int) * 2) + df['T5_more_than_45'].astype(int)
    
    return df

# Function to create and plot the contingency table heatmap
def plot_contingency_table(df, var1, var2):
    contingency_table = pd.crosstab(df[var1], df[var2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
    plt.title(f'Contingency Table Heatmap of {var1} and {var2}')
    plt.xlabel(var2)
    plt.ylabel(var1)
    plt.show()

# Plot confusion matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Recovered', 'Relative Recovery', 'Absolute Recovery','Recovered'], 
                yticklabels=['Not Recovered', 'Relative Recovery', 'Absolute Recovery','Recovered'])
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Forward feature selection function
def forward_feature_selection(model, X_train, y_train, X_test, y_test):
    best_features, initial_features, best_score = [], [], 0
    while True:
        scores = []
        for feature in X_train.columns:
            if feature not in initial_features:
                selected_features = initial_features + [feature]
                model.fit(X_train[selected_features], y_train)
                y_pred = model.predict(X_test[selected_features])
                accuracy = accuracy_score(y_test, y_pred)
                scores.append((accuracy, feature))

        if not scores: break
        
        current_score, best_feature = sorted(scores, reverse=True)[0]
        if current_score > best_score:
            best_score = current_score
            initial_features.append(best_feature)
            best_features.append(best_feature)
        else:
            break

    return best_features

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    if len(set(y_true)) == 2:  # Binary classification
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
    else:  # Multiclass classification
        specificity = np.nan  # Or handle multiclass specificity differently if needed
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1': f1}

# Feature importance plot
def plot_feature_importance(importances, features, title):
    plt.figure(figsize=(5, 3))
    sns.barplot(x=importances, y=features)
    plt.title(f'{title} Feature Importance')
    plt.show()
    
# Plot Decision Tree with labels
def plot_decision_tree_multiclass(model, features):
    # Define class labels for your three classes
    class_labels = ['Not Recovered', 'Absolute Recovery', 'Relative Recovery', 'Fully Recovered']
    
    plt.figure(figsize=(15, 10))  # Adjust figure size to make it readable
    plot_tree(
        model, 
        feature_names=features, 
        class_names=class_labels,  # Use the multiclass labels
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title("Decision Tree Visualization - Multiclass Outcome")
    plt.show()
