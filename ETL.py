import pandas as pd
import numpy as np
from datetime import datetime

# NOTE: No need to import Streamlit here. We will let the Streamlit app handle UI errors.
# The ETL function will raise errors, and the app will catch them.

def etl_long_to_wide(
    input_source, 
    output_file=None, 
    datetime_column_name='Date (CET)',
    value_column_name='Day Ahead Price',
    input_date_format='%d/%m/%Y %H:%M'
):
    """
    ETL function to transform long format time series data to wide format.
    
    Args:
        input_source (str or file-like object): Path to input CSV or an in-memory file.
        output_file (str, optional): Path to output CSV file. Defaults to None.
        datetime_column_name (str): The name of the column containing datetime stamps.
        value_column_name (str): Column name to use for values in the wide format.
        input_date_format (str): The format of the datetime string for parsing.
    
    Returns:
        pandas.DataFrame: Wide format dataframe or raises an exception on failure.
    """
    try:
        print(f"Reading data from source, skipping the first row...")
        df = pd.read_csv(input_source, skiprows=1)
        
        if df.iloc[0].astype(str).str.contains('EUR|MW|%', na=False).any():
            print("Detected units row, removing it...")
            df = df.drop(0).reset_index(drop=True)
        
        if datetime_column_name not in df.columns:
            raise ValueError(f"Datetime column '{datetime_column_name}' not found. Available columns are: {list(df.columns)}")
        if value_column_name not in df.columns:
            raise ValueError(f"Value column '{value_column_name}' not found. Available columns are: {list(df.columns)}")
            
        print(f"Processing datetime from column: '{datetime_column_name}'")
        df[datetime_column_name] = df[datetime_column_name].astype(str).str.replace(r'[\[\]]', '', regex=True)
        
        # Add a check for empty date strings after cleaning
        df = df[df[datetime_column_name].str.strip() != ''].copy()
        
        df['datetime'] = pd.to_datetime(df[datetime_column_name], format=input_date_format, errors='coerce')
        
        # Drop rows where date parsing failed
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        if len(df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(df)} rows due to invalid date formats.")

        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')

        df[value_column_name] = pd.to_numeric(df[value_column_name], errors='coerce')
        if df[value_column_name].isnull().any():
            print(f"Warning: Coerced non-numeric values to NaN in '{value_column_name}'.")

        print("Transforming to wide format using pivot_table to handle potential duplicates...")
        
        # --- THE FIX IS HERE ---
        # Replace pivot with pivot_table.
        # Use aggfunc='mean' to average any duplicate timestamp values.
        wide_df = df.pivot_table(
            index='date', 
            columns='time', 
            values=value_column_name,
            aggfunc='mean' # This resolves the "duplicate entries" error
        )
        
        # The rest of the function remains the same...
        wide_df = wide_df.reset_index()
        wide_df['date'] = pd.to_datetime(wide_df['date']).dt.strftime('%Y-%m-%d')
        
        date_col_data = wide_df['date']
        time_cols_df = wide_df.drop('date', axis=1)

        print(f"Handling missing values. Original missing: {time_cols_df.isnull().sum().sum()}")
        time_cols_df = time_cols_df.interpolate(method='linear', axis=1, limit_direction='both').fillna(0)
        print(f"Missing values after handling: {time_cols_df.isnull().sum().sum()}")

        wide_df = pd.concat([date_col_data, time_cols_df], axis=1)
        
        time_cols = [col for col in wide_df.columns if col != 'date']
        time_cols_sorted = sorted(time_cols, key=lambda x: datetime.strptime(x, '%H:%M:%S').time())
        wide_df = wide_df[['date'] + time_cols_sorted]
        
        print("Transformation complete!")
        
        if output_file:
            print(f"Saving to {output_file}...")
            wide_df.to_csv(output_file, index=False)
        
        return wide_df

    except Exception as e:
        print(f"An error occurred during the ETL process: {e}")
        raise

# The main function can be kept for standalone testing
def main():
    input_file = 'idprices-epexshort.csv'
    output_file = 'output_wide_format.csv'
    try:
        result_df = etl_long_to_wide(input_source=input_file, output_file=output_file)
        if result_df is not None:
            print("\n--- ETL Process Successful ---")
            print(result_df.head())
    except Exception as e:
        print(f"\nETL process failed with error: {e}")

if __name__ == "__main__":
    main()