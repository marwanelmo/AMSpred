# AMSpred - Predicting recovery of physical functioning after hospitalization

This project processes sensor and study data to create clean datasets for machine learning predictions. It includes a pipeline for data preprocessing, feature selection, and model building. The end goal is to predict specific outcomes related to patient recovery.

## Folder Structure

- **/scripts**: Contains reusable Python scripts.
  - `ml_helper.py`: Contains functions that support machine learning model building.
  - `general_functions.py`: Contains general functions used across the notebooks.

## Notebooks

- **Clean Castor.ipynb**: Preprocesses data from the Castor file, creating the `Voorspellers.csv` and `uitkomstmaat.csv` files for use in machine learning.
- **Clean_Activpal.ipynb**: Similar to Clean Castor, this notebook processes data from the Activpal file to create `Voorspellers.csv` and `uitkomstmaat.csv` files.
- **delete_cols.ipynb**: Removes unnecessary and unused columns from the processed data files to prepare them for model training.
- **Final_ML.ipynb**: The final model-building notebook, utilizing cleaned and filtered data to train predictive models on specified outcomes.
- **Variabel Selectie.xlsx**: Spreadsheet used to select and track variables that are important for the prediction models.

## Getting Started

1. **Data Cleaning**: Start by running the `Clean Castor.ipynb` and `Clean_Activpal.ipynb` notebooks to preprocess the respective files and create `.Voorspellers.csv` and `.uitkomstmaat.csv` files.
2. **Column Selection**: Run `delete_cols.ipynb` to remove unnecessary columns and finalize the dataset for modeling.
3. **Feature Selection**: Use `Variabel Selectie.xlsx` to identify and select key variables.
4. **Model Building**: Open `Final_ML.ipynb` to run the machine learning models using the cleaned data files.

## Dependencies

Install required packages by running:

```bash
pip install -r requirements.txt
```

## Usage Notes

- **Scripts Folder**: The `/scripts` folder contains reusable functions. `ml_helper.py` specifically supports model building, and `general_functions.py` contains functions that assist with various preprocessing tasks across notebooks.

- **Outputs**: The data cleaning notebooks will output CSV files named `Voorspellers.csv` and `uitkomstmaat.csv`, which are then used for modeling.

## Contributing

If you’d like to contribute, please create pull requests with detailed descriptions, ensuring all code adheres to PEP8 standards.
