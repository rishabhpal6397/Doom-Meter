from dotenv import load_dotenv
load_dotenv()
import os
import pandas as pd
import numpy as np
import logging
import re
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_complex_cell(value):
    """
    Clean complex cell values that contain multiple entries, parentheses, etc.
    Returns a simplified string representation.
    """
    if pd.isna(value):
        return None
    
    # Convert to string if not already
    value = str(value)
    
    # Replace multiple spaces with a single space
    value = re.sub(r'\s+', ' ', value)
    
    # Remove special characters that might cause issues
    value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    return value.strip()

def convert_excel_to_csv(input_file, output_file=None):
    """
    Convert Excel file to CSV with proper handling of complex cells
    """
    logging.info(f"Converting {input_file} to CSV")
    
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.csv'
    
    try:
        # Try reading with default engine
        df = pd.read_excel(input_file)
    except Exception as e:
        logging.error(f"Error reading Excel with default engine: {e}")
        try:
            # Try with openpyxl engine
            df = pd.read_excel(input_file, engine='openpyxl')
        except Exception as e2:
            logging.error(f"Error reading Excel with openpyxl engine: {e2}")
            raise e
    
    logging.info(f"Excel file read successfully. Shape: {df.shape}")
    
    # Clean complex cells
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(clean_complex_cell)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"Saved cleaned data to {output_file}")
    
    return output_file

def clean_csv_file(input_file, output_file=None):
    """
    Clean a CSV file with proper handling of complex cells
    """
    logging.info(f"Cleaning {input_file}")
    
    if not output_file:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding, low_memory=False)
                logging.info(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            # Try with semicolon separator
            try:
                df = pd.read_csv(input_file, sep=';', encoding='latin1', low_memory=False)
                logging.info("Successfully read CSV with semicolon separator")
            except Exception as e:
                logging.error(f"Error reading CSV with semicolon separator: {e}")
                raise Exception("Failed to read CSV with any encoding or separator")
    
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        raise
    
    logging.info(f"CSV file read successfully. Shape: {df.shape}")
    
    # Clean complex cells
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(clean_complex_cell)
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"Saved cleaned data to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert and clean Excel/CSV files for disaster data')
    parser.add_argument('input_file', help='Input Excel or CSV file')
    parser.add_argument('--output', '-o', help='Output file name (optional)')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
    
    # Check file extension
    _, ext = os.path.splitext(args.input_file)
    
    if ext.lower() in ['.xlsx', '.xls']:
        # Convert Excel to CSV
        output_file = convert_excel_to_csv(args.input_file, args.output)
    elif ext.lower() == '.csv':
        # Clean CSV
        output_file = clean_csv_file(args.input_file, args.output)
    else:
        logging.error(f"Unsupported file format: {ext}")
        return
    
    logging.info(f"Processing complete. Output file: {output_file}")

if __name__ == "__main__":
    main()
