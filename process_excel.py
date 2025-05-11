import os
import logging
from data_fetcher import fetch_emdat_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Process the Excel file and load it into MongoDB
    """
    logging.info("Starting Excel processing script")
    
    # Check if MongoDB environment variables are set
    if "MONGODB_URI" not in os.environ or "MONGODB_DB" not in os.environ:
        logging.error("MongoDB environment variables not set")
        logging.info("Setting default MongoDB environment variables")
        os.environ["MONGODB_URI"] = "mongodb://localhost:27017/"
        os.environ["MONGODB_DB"] = "disaster_management"
    
    # Process the Excel file
    result = fetch_emdat_data()
    
    if result:
        logging.info("Excel file processed and loaded into MongoDB successfully")
    else:
        logging.error("Failed to process Excel file")

if __name__ == "__main__":
    main()
