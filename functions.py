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