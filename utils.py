import pandas as pd

def clean_csv_in_place(filepath: str):

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            print(f"Warning: '{filepath}' is empty.")
            return

        cleaned_df = df.dropna(how='any')

        if cleaned_df.empty:
            print(f"Warning: '{filepath}' is empty after cleanning.")
            cleaned_df.to_csv(filepath, index=False)
            return

        cleaned_df.to_csv(filepath, index=False)