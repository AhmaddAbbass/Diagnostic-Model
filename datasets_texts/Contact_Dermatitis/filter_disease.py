import pandas as pd
import sys
import os

def filter_disease(file_path, disease_name):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Determine if file is CSV or Excel based on extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.csv']:
        df = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file format. Use CSV or Excel.")
        return

    # Filter rows where the 'Disease name' column matches the given disease (case-insensitive)
    if 'Disease name' not in df.columns:
        print("Column 'Disease name' not found in the file.")
        return

    filtered_df = df[df['Disease name'].str.lower() == disease_name.lower()]

    # Option: Overwrite the original file. Alternatively, you can save to a new file.
    filtered_df.to_csv(file_path, index=False) if ext == '.csv' else filtered_df.to_excel(file_path, index=False)
    print(f"File has been updated. Only rows with '{disease_name}' remain in {file_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python filter_disease.py <file_path> <disease_name>")
    else:
        file_path = sys.argv[1]
        disease_name = sys.argv[2]
        filter_disease(file_path, disease_name)
