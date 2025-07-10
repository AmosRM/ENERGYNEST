import pandas as pd
import numpy as np
from datetime import datetime

def etl_long_to_wide(
    input_source, 
    output_file=None, 
    datetime_column_name='Date (CET)',
    value_column_name='Day Ahead Price',
    input_date_format='%d/%m/%Y %H:%M'
):
    """
    ETL function that automatically detects if the first row of the CSV
    needs to be skipped.
    """
    
    def _transform_dataframe(df):
        """Helper function to perform the core transformation logic."""
        # This function assumes df is a clean DataFrame with correct headers.
        
        # 1. Handle potential units row (this is now more robust)
        # Check if the first row of data looks like a units row.
        # It's safer to check the actual data row after headers are set.
        if df.iloc[0].astype(str).str.contains('EUR|MW|%', na=False).any():
            print("Detected units row, removing it...")
            df = df.drop(0).reset_index(drop=True)

        # 2. Validate columns
        if datetime_column_name not in df.columns:
            raise ValueError(f"Datetime column '{datetime_column_name}' not found.")
        if value_column_name not in df.columns:
            raise ValueError(f"Value column '{value_column_name}' not found.")
            
        # 3. Process data (the rest of the logic is the same)
        print(f"Processing datetime from column: '{datetime_column_name}'")
        df[datetime_column_name] = df[datetime_column_name].astype(str).str.replace(r'[\[\]]', '', regex=True)
        df = df[df[datetime_column_name].str.strip() != ''].copy()
        df['datetime'] = pd.to_datetime(df[datetime_column_name], format=input_date_format, errors='coerce')
        
        initial_rows = len(df)
        df.dropna(subset=['datetime'], inplace=True)
        if len(df) < initial_rows:
            print(f"Warning: Dropped {initial_rows - len(df)} rows due to invalid date formats.")

        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S')

        df[value_column_name] = pd.to_numeric(df[value_column_name], errors='coerce')
        if df[value_column_name].isnull().any():
            print(f"Warning: Coerced non-numeric values to NaN in '{value_column_name}'.")

        print("Transforming to wide format using pivot_table...")
        wide_df = df.pivot_table(
            index='date', 
            columns='time', 
            values=value_column_name,
            aggfunc='mean'
        )
        
        wide_df = wide_df.reset_index()
        wide_df['date'] = pd.to_datetime(wide_df['date']).dt.strftime('%Y-%m-%d')
        
        date_col_data = wide_df['date']
        time_cols_df = wide_df.drop('date', axis=1)
        
        print(f"Handling missing values...")
        time_cols_df = time_cols_df.interpolate(method='linear', axis=1, limit_direction='both').fillna(0)
        
        wide_df = pd.concat([date_col_data, time_cols_df], axis=1)
        
        time_cols = [col for col in wide_df.columns if col != 'date']
        time_cols_sorted = sorted(time_cols, key=lambda x: datetime.strptime(x, '%H:%M:%S').time())
        wide_df = wide_df[['date'] + time_cols_sorted]
        
        return wide_df

    # --- Main Logic with Automated Skiprows ---
    try:
        # Attempt 1: Assume a clean CSV with no metadata row (skiprows=0)
        print("Attempting to read CSV with skiprows=0...")
        # The file-like object needs to be reset if it was read before
        if hasattr(input_source, 'seek'):
            input_source.seek(0)
        df = pd.read_csv(input_source)
        # We check for the expected column immediately. If it's missing,
        # it's a strong sign that the header is wrong.
        if datetime_column_name not in df.columns:
             raise ValueError("Header not found on the first line.")
        
        print("Successfully read with skiprows=0. Transforming data...")
        final_df = _transform_dataframe(df)

    except (ValueError, KeyError) as e:
        # This block catches errors like "Header not found" or other key errors
        # that suggest the header is on the second line.
        print(f"Reading with skiprows=0 failed ({e}). Retrying with skiprows=1...")
        try:
            # Attempt 2: Assume a metadata row needs to be skipped
            if hasattr(input_source, 'seek'):
                input_source.seek(0) # Reset file pointer again
            df = pd.read_csv(input_source, skiprows=1)
            
            print("Successfully read with skiprows=1. Transforming data...")
            final_df = _transform_dataframe(df)

        except Exception as e_inner:
            # If the second attempt also fails, then the file is genuinely problematic.
            print(f"Second attempt (skiprows=1) also failed: {e_inner}")
            raise ValueError(f"Could not process the CSV file with either 0 or 1 skipped rows. Please check the file format. Error: {e_inner}")

    print("Transformation complete!")
    
    if output_file:
        print(f"Saving to {output_file}...")
        final_df.to_csv(output_file, index=False)
    
    return final_df

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