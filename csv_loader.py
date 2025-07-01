
import os
import csv
from folder_paths import input_directory

class CSVLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = input_directory
        csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
        
        return {
            "required": {
                "csv_file": (csv_files, {"default": csv_files[0] if csv_files else ""}),
                "row": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8")
    FUNCTION = "load_csv_row"
    CATEGORY = "Katalist Tools"

    def load_csv_row(self, csv_file, row):
        csv_path = os.path.join(input_directory, csv_file)
        
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file not found: {csv_file}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                
                if row >= len(rows):
                    raise ValueError(f"Row {row} does not exist. CSV has {len(rows)} rows.")
                
                selected_row = rows[row]
                
                # Pad with empty strings if row has fewer than 8 columns
                result = []
                for i in range(8):
                    if i < len(selected_row):
                        result.append(selected_row[i])
                    else:
                        result.append("")
                
                return tuple(result)
                
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")