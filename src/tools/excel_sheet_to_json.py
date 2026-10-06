import pandas as pd
import json
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"

def excel_to_json(excel_file_path, json_file_path):
    try:
        df = pd.read_excel(excel_file_path, engine='openpyxl')

        # --- THE FIX ---
        # Replace the non-breaking hyphen (\u2011) with a standard hyphen (-) across the entire dataframe
        df = df.replace('\u2011', '-', regex=True)
        # ----------------

        # Convert the DataFrame to JSON format
        # Added force_ascii=False to ensure any other special characters render correctly
        json_data = df.to_json(orient='records', indent=4, force_ascii=False)

        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json_data)

        print(f"Successfully converted '{excel_file_path}' to '{json_file_path}'.")
        
    except FileNotFoundError:
        print(f"Error: The file '{excel_file_path}' could not be found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_excel_file = _DOCS_DIR / 'Circular Design Guideline Database (JR Draft)_cleaned.xlsx'
    output_json_file = _DOCS_DIR / 'Circular Design Guideline Database_cleaned.json'
    
    excel_to_json(input_excel_file, output_json_file)
