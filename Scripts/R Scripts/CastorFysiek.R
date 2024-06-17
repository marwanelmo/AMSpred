# Create a vector of package names
packages_needed <- c('dplyr', 'readxl', 'writexl')

# Loop through the packages
for (pkg in packages_needed) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(dplyr)
library(readxl)
library(writexl)

process_castor_fysiek <- function(df, export=TRUE) {
  # Define the columns to be used in the process
  columns <- c(
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
  )
  
  # Filter the dataframe to only include the specified columns
  filtered_castor <- df[, columns, drop = FALSE]
  
  # Initialize combined_fysiek with the correct number of rows and 0 columns
  combined_fysiek <- data.frame(matrix(ncol = 7, nrow = nrow(filtered_castor)))
  names(combined_fysiek) <- c('T0', 'T2', 'T3', 'T4', 'T5', 'T6', 'patientnummer')
  
  # Iterate through the time points, combining scores from two versions of the questionnaire per time point
  for (i in 1:6) {
    # Select the two columns for the current time point
    col_start <- (i - 1) * 2 + 1
    col_end <- col_start + 1
    current_df <- filtered_castor[, col_start:col_end]
    # Fill missing values in the first column with values from the second column
    combined_fysiek[, i] <- ifelse(is.na(current_df[,1]), current_df[,2], current_df[,1])
  }
  
  # Add the patient number column as the last column in the combined DataFrame
  combined_fysiek$patientnummer <- filtered_castor$patientnummer
  
  
  for (col in c('T2', 'T3', 'T4', 'T5', 'T6')) {
    # Calculate the relative value by dividing the Tx column by the T0 column
    combined_fysiek[paste0(col, '_relative')] <- combined_fysiek[[col]] / combined_fysiek[['T0']]
  }
  
  # Export the combined DataFrame to an Excel file if the export flag is set to True
  if (export) {
    write_xlsx(combined_fysiek, 'Export Fysiek Functioneren.xlsx')
  }
  
  
  # Return the combined DataFrame
  return(combined_fysiek)
}


castor_fysiek <- read_excel("C:/Users/melmora/Documents/GitHub/Oprah - AMS/Castor/PROMIS_export_fysiekfunctioneren.xlsx")
process_castor_fysiek(castor_fysiek, export=TRUE)


  