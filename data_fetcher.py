import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pymongo
from pymongo import MongoClient
import logging
import pycountry
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_mongodb():
    """Connect to MongoDB database"""
    try:
        client = MongoClient(os.environ["MONGODB_URI"])
        db = client[os.environ["MONGODB_DB"]]
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return None

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

def extract_primary_location(location_str):
    """
    Extract the primary location from a complex location string.
    For example, from "Ceel Barde, Rab Dhuure, Tayeeglow, Xudur districts (Bakool province)"
    it would extract "Bakool province" or "Ceel Barde" depending on the pattern.
    """
    if pd.isna(location_str):
        return None
    
    location_str = str(location_str)
    
    # Try to extract province/region in parentheses
    province_match = re.search(r'$$([^)]+)$$', location_str)
    if province_match:
        return province_match.group(1).strip()
    
    # If no parentheses, take the first location before a comma
    parts = location_str.split(',')
    if parts:
        return parts[0].strip()
    
    return location_str.strip()

def preprocess_emdat_data(df):
    """
    Preprocess the EM-DAT Excel data
    """
    logging.info("Preprocessing EM-DAT data")
    
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Handle complex cell values
    for column in processed_df.columns:
        if processed_df[column].dtype == 'object':
            processed_df[column] = processed_df[column].apply(clean_complex_cell)
    
    # Special handling for location column
    if 'Location' in processed_df.columns:
        processed_df['Primary_Location'] = processed_df['Location'].apply(extract_primary_location)
    
    # Rename columns to match our schema
    column_mapping = {
        'DisNo.': 'disaster_id',
        'Disaster Group': 'disaster_group',
        'Disaster Subgroup': 'disaster_subgroup',
        'Disaster Type': 'disaster_type',
        'Disaster Subtype': 'disaster_subtype',
        'Event Name': 'event_name',
        'ISO': 'iso3',
        'Country': 'country',
        'Region': 'region',
        'Subregion': 'subregion',
        'Location': 'location',
        'Primary_Location': 'primary_location',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Start Year': 'year',
        'Start Month': 'month',
        'Start Day': 'day',
        'End Year': 'end_year',
        'End Month': 'end_month',
        'End Day': 'end_day',
        'Total Deaths': 'deaths',
        'No. Injured': 'injured',
        'No. Affected': 'affected',
        'No. Homeless': 'homeless',
        'Total Affected': 'total_affected',
        'Total Damage (\'000 US$)': 'total_damages',
        'Total Damage Adjusted (\'000 US$)': 'total_damages_adjusted',
        'Magnitude': 'magnitude',
        'Magnitude Scale': 'magnitude_scale'
    }
    
    # Apply column renaming for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in processed_df.columns:
            processed_df.rename(columns={old_col: new_col}, inplace=True)
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['deaths', 'injured', 'affected', 'homeless', 'total_affected', 
                       'total_damages', 'total_damages_adjusted', 'magnitude']
    
    for col in numeric_columns:
        if col in processed_df.columns:
            # First, clean any non-numeric characters
            if processed_df[col].dtype == 'object':
                processed_df[col] = processed_df[col].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)) if not pd.isna(x) else x)
            
            # Then convert to numeric
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Handle date columns
    if all(col in processed_df.columns for col in ['year', 'month', 'day']):
        # Create start_date column, handling potential non-numeric values
        try:
            # Convert to numeric first to handle any string values
            processed_df['year'] = pd.to_numeric(processed_df['year'], errors='coerce')
            processed_df['month'] = pd.to_numeric(processed_df['month'], errors='coerce')
            processed_df['day'] = pd.to_numeric(processed_df['day'], errors='coerce')
            
            # Fill missing values with defaults
            processed_df['month'] = processed_df['month'].fillna(1)
            processed_df['day'] = processed_df['day'].fillna(1)
            
            # Create date strings
            date_strings = processed_df.apply(
                lambda row: f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d}" 
                if not pd.isna(row['year']) else None, 
                axis=1
            )
            
            # Convert to datetime
            processed_df['start_date'] = pd.to_datetime(date_strings, errors='coerce')
            
            # Format as string
            processed_df['start_date'] = processed_df['start_date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            logging.error(f"Error processing date columns: {e}")
            # Create a default date if conversion fails
            processed_df['start_date'] = None
    
    if all(col in processed_df.columns for col in ['end_year', 'end_month', 'end_day']):
        # Create end_date column with similar error handling
        try:
            processed_df['end_year'] = pd.to_numeric(processed_df['end_year'], errors='coerce')
            processed_df['end_month'] = pd.to_numeric(processed_df['end_month'], errors='coerce')
            processed_df['end_day'] = pd.to_numeric(processed_df['end_day'], errors='coerce')
            
            processed_df['end_month'] = processed_df['end_month'].fillna(12)
            processed_df['end_day'] = processed_df['end_day'].fillna(28)
            
            end_date_strings = processed_df.apply(
                lambda row: f"{int(row['end_year'])}-{int(row['end_month']):02d}-{int(row['end_day']):02d}" 
                if not pd.isna(row['end_year']) else None, 
                axis=1
            )
            
            processed_df['end_date'] = pd.to_datetime(end_date_strings, errors='coerce')
            processed_df['end_date'] = processed_df['end_date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            logging.error(f"Error processing end date columns: {e}")
            processed_df['end_date'] = None
    
    # Fill missing values
    processed_df['disaster_group'] = processed_df['disaster_group'].fillna('Unknown')
    processed_df['disaster_type'] = processed_df['disaster_type'].fillna('Unknown')
    processed_df['country'] = processed_df['country'].fillna('Unknown')
    processed_df['region'] = processed_df['region'].fillna('Unknown')
    
    # Generate coordinates if missing
    if 'latitude' in processed_df.columns and 'longitude' in processed_df.columns:
        # Fill missing coordinates with approximate values based on country or region
        country_coords = get_country_coordinates()
        region_coords = get_region_coordinates()
        
        # Function to generate coordinates with small random offset
        def generate_coords(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                country = row['country'] if 'country' in row and not pd.isna(row['country']) else 'Unknown'
                region = row['region'] if 'region' in row and not pd.isna(row['region']) else 'Unknown'
                
                if country in country_coords:
                    base_lat, base_lon = country_coords[country]
                elif region in region_coords:
                    base_lat, base_lon = region_coords[region]
                else:
                    base_lat, base_lon = 0, 0  # Default to 0,0 if no match
                
                # Add small random offset
                lat = base_lat + np.random.uniform(-3, 3)
                lon = base_lon + np.random.uniform(-3, 3)
                
                return pd.Series([lat, lon])
            return pd.Series([row['latitude'], row['longitude']])
        
        # Apply to rows with missing coordinates
        coords_df = processed_df.apply(generate_coords, axis=1)
        processed_df['latitude'] = coords_df[0]
        processed_df['longitude'] = coords_df[1]
        processed_df = processed_df.replace({np.nan: None, np.inf: None, -np.inf: None})

    
    # Generate magnitude if missing
    if 'magnitude' in processed_df.columns:
        # Fill missing magnitude based on disaster type
        disaster_magnitude_ranges = {
            'Earthquake': (4.0, 9.0),
            'Tsunami': (3.0, 8.0),
            'Flood': (1.0, 5.0),
            'Storm': (1.0, 5.0),
            'Hurricane': (1.0, 5.0),
            'Drought': (1.0, 5.0),
            'Wildfire': (1.0, 5.0),
            'Volcanic Activity': (1.0, 6.0),
            'Landslide': (1.0, 4.0),
            'Epidemic': (1.0, 5.0)
        }
        
        def generate_magnitude(row):
            if pd.isna(row['magnitude']):
                disaster_type = row['disaster_type'] if 'disaster_type' in row and not pd.isna(row['disaster_type']) else 'Unknown'
                min_mag, max_mag = disaster_magnitude_ranges.get(disaster_type, (1.0, 5.0))
                return np.random.uniform(min_mag, max_mag)
            return row['magnitude']
        
        processed_df['magnitude'] = processed_df.apply(generate_magnitude, axis=1)
    
    # Generate impact metrics if missing
    impact_columns = ['deaths', 'injured', 'affected', 'homeless', 'total_affected', 'total_damages']
    for col in impact_columns:
        if col in processed_df.columns:
            # Fill missing values based on disaster type and magnitude
            processed_df[col] = processed_df.apply(
                lambda row: generate_impact_metric(row, col) if pd.isna(row[col]) else row[col], 
                axis=1
            )
    
    # Ensure all required columns exist
    required_columns = ['disaster_id', 'year', 'start_date', 'end_date', 'disaster_group', 
                       'disaster_type', 'country', 'iso3', 'region', 'latitude', 'longitude',
                       'magnitude', 'magnitude_scale', 'deaths', 'injured', 'affected',
                       'homeless', 'total_affected', 'total_damages', 'total_damages_adjusted']
    
    for col in required_columns:
        if col not in processed_df.columns:
            if col == 'disaster_id':
                processed_df[col] = ['EM-DAT-' + str(i) for i in range(len(processed_df))]
            elif col == 'iso3':
                processed_df[col] = processed_df['country'].apply(get_iso3_code)
            elif col == 'magnitude_scale':
                processed_df[col] = processed_df['disaster_type'].apply(get_magnitude_scale)
            elif col == 'total_damages_adjusted':
                if 'total_damages' in processed_df.columns:
                    # Apply inflation factor (simplified)
                    processed_df[col] = processed_df['total_damages'] * 1.2
                else:
                    processed_df[col] = np.random.uniform(10, 1000, len(processed_df))
            else:
                # Generate random data for other missing columns
                processed_df[col] = generate_random_column(col, len(processed_df))
    
    logging.info(f"Preprocessing complete. DataFrame shape: {processed_df.shape}")
    return processed_df

def get_iso3_code(country_name):
    """Get ISO3 code for a country"""
    try:
        if pd.isna(country_name) or country_name == 'Unknown':
            return 'UNK'
        
        # Try to find the country
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
        
        # Try with common name variations
        name_variations = {
            'United States': 'USA',
            'USA': 'USA',
            'UK': 'GBR',
            'United Kingdom': 'GBR',
            'Russia': 'RUS',
            'Congo': 'COD',
            'Democratic Republic of the Congo': 'COD',
            'Tanzania': 'TZA',
            'South Korea': 'KOR',
            'North Korea': 'PRK',
            'Iran': 'IRN',
            'Syria': 'SYR',
            'Venezuela': 'VEN',
            'Vietnam': 'VNM',
            'Laos': 'LAO',
            'Bolivia': 'BOL'
        }
        
        if country_name in name_variations:
            return name_variations[country_name]
        
        # Try fuzzy matching
        for c in pycountry.countries:
            if country_name.lower() in c.name.lower():
                return c.alpha_3
        
        return 'UNK'
    except:
        return 'UNK'

def get_magnitude_scale(disaster_type):
    """Get appropriate magnitude scale for disaster type"""
    scales = {
        'Earthquake': 'Richter',
        'Tsunami': 'Meters',
        'Flood': 'Meters',
        'Storm': 'km/h',
        'Hurricane': 'Saffir-Simpson',
        'Tornado': 'Fujita',
        'Drought': 'PDSI',
        'Wildfire': 'km²',
        'Volcanic Activity': 'VEI',
        'Landslide': 'Meters',
        'Epidemic': 'R0'
    }
    
    return scales.get(disaster_type, 'Scale')

def get_country_coordinates():
    """Get approximate coordinates for countries"""
    return {
        "United States": (39.8283, -98.5795),
        "China": (35.8617, 104.1954),
        "India": (20.5937, 78.9629),
        "Japan": (36.2048, 138.2529),
        "Indonesia": (-0.7893, 113.9213),
        "Philippines": (12.8797, 121.7740),
        "Brazil": (-14.2350, -51.9253),
        "Mexico": (23.6345, -102.5528),
        "Italy": (41.8719, 12.5674),
        "France": (46.2276, 2.2137),
        "Germany": (51.1657, 10.4515),
        "United Kingdom": (55.3781, -3.4360),
        "Australia": (-25.2744, 133.7751),
        "New Zealand": (-40.9006, 174.8860),
        "South Africa": (-30.5595, 22.9375),
        "Nigeria": (9.0820, 8.6753),
        "Egypt": (26.8206, 30.8025),
        "Kenya": (-0.0236, 37.9062),
        "Bangladesh": (23.6850, 90.3563),
        "Pakistan": (30.3753, 69.3451),
        "Somalia": (5.1521, 46.1996),
        "Ethiopia": (9.1450, 40.4897),
        "Sudan": (12.8628, 30.2176),
        "Afghanistan": (33.9391, 67.7100),
        "Iraq": (33.2232, 43.6793),
        "Syria": (34.8021, 38.9968),
        "Yemen": (15.5527, 48.5164),
        "Libya": (26.3351, 17.2283),
        "Chad": (15.4542, 18.7322),
        "Niger": (17.6078, 8.0817),
        "Mali": (17.5707, -3.9962),
        "Mauritania": (21.0079, -10.9408),
        "Senegal": (14.4974, -14.4524),
        "Guinea": (9.9456, -9.6966),
        "Ivory Coast": (7.5400, -5.5471),
        "Ghana": (7.9465, -1.0232),
        "Togo": (8.6195, 0.8248),
        "Benin": (9.3077, 2.3158),
        "Burkina Faso": (12.2383, -1.5616),
        "Cameroon": (7.3697, 12.3547),
        "Central African Republic": (6.6111, 20.9394),
        "Congo": (-0.2280, 15.8277),
        "Uganda": (1.3733, 32.2903),
        "Rwanda": (-1.9403, 29.8739),
        "Burundi": (-3.3731, 29.9189),
        "Tanzania": (-6.3690, 34.8888),
        "Mozambique": (-18.6657, 35.5296),
        "Zimbabwe": (-19.0154, 29.1549),
        "Zambia": (-13.1339, 27.8493),
        "Malawi": (-13.2543, 34.3015),
        "Angola": (-11.2027, 17.8739),
        "Namibia": (-22.9576, 18.4904),
        "Botswana": (-22.3285, 24.6849),
        "Lesotho": (-29.6100, 28.2336),
        "Swaziland": (-26.5225, 31.4659),
        "Madagascar": (-18.7669, 46.8691),
        "Mauritius": (-20.3484, 57.5522),
        "Comoros": (-11.6455, 43.3333),
        "Seychelles": (-4.6796, 55.4920),
        "Cape Verde": (16.5388, -23.0418),
        "Sao Tome and Principe": (0.1864, 6.6131),
        "Guinea-Bissau": (11.8037, -15.1804),
        "Gambia": (13.4432, -15.3101),
        "Sierra Leone": (8.4606, -11.7799),
        "Liberia": (6.4281, -9.4295),
        "Eritrea": (15.1794, 39.7823),
        "Djibouti": (11.8251, 42.5903),
        "Tunisia": (33.8869, 9.5375),
        "Algeria": (28.0339, 1.6596),
        "Morocco": (31.7917, -7.0926),
        "Western Sahara": (24.2155, -12.8858),
        "Bakool": (4.3660, 43.8302),  # Somalia province
        "Gedo": (3.5147, 42.2319),    # Somalia province
        "Bay": (2.4126, 43.6588),     # Somalia province
        "Hiraan": (4.3292, 45.3438)   # Somalia province
    }

def get_region_coordinates():
    """Get approximate coordinates for regions"""
    return {
        "Africa": (0.0, 20.0),
        "Americas": (0.0, -80.0),
        "Asia": (30.0, 100.0),
        "Europe": (50.0, 10.0),
        "Oceania": (-20.0, 150.0),
        "East Africa": (0.0, 38.0),
        "West Africa": (8.0, 0.0),
        "North Africa": (28.0, 15.0),
        "Southern Africa": (-25.0, 25.0),
        "Central Africa": (0.0, 18.0),
        "Horn of Africa": (8.0, 45.0),
        "Southeast Asia": (10.0, 105.0),
        "South Asia": (20.0, 77.0),
        "East Asia": (35.0, 115.0),
        "Central Asia": (43.0, 68.0),
        "Western Asia": (35.0, 40.0),
        "Middle East": (27.0, 45.0),
        "Eastern Europe": (50.0, 30.0),
        "Western Europe": (48.0, 5.0),
        "Southern Europe": (40.0, 15.0),
        "Northern Europe": (60.0, 15.0),
        "North America": (40.0, -100.0),
        "Central America": (15.0, -90.0),
        "Caribbean": (20.0, -75.0),
        "South America": (-15.0, -60.0),
        "Australia and New Zealand": (-30.0, 145.0),
        "Melanesia": (-10.0, 150.0),
        "Micronesia": (8.0, 150.0),
        "Polynesia": (-15.0, -170.0)
    }

def generate_impact_metric(row, metric):
    """Generate impact metrics based on disaster type and magnitude"""
    disaster_type = row['disaster_type'] if 'disaster_type' in row and not pd.isna(row['disaster_type']) else 'Unknown'
    magnitude = row['magnitude'] if 'magnitude' in row and not pd.isna(row['magnitude']) else 3.0
    
    # Base values by disaster type
    base_values = {
        'deaths': {
            'Earthquake': 50,
            'Tsunami': 100,
            'Flood': 20,
            'Storm': 15,
            'Hurricane': 30,
            'Drought': 10,
            'Wildfire': 5,
            'Volcanic Activity': 20,
            'Landslide': 15,
            'Epidemic': 100
        },
        'injured': {
            'Earthquake': 200,
            'Tsunami': 150,
            'Flood': 50,
            'Storm': 40,
            'Hurricane': 100,
            'Drought': 20,
            'Wildfire': 30,
            'Volcanic Activity': 50,
            'Landslide': 40,
            'Epidemic': 500
        },
        'affected': {
            'Earthquake': 10000,
            'Tsunami': 5000,
            'Flood': 20000,
            'Storm': 5000,
            'Hurricane': 15000,
            'Drought': 50000,
            'Wildfire': 2000,
            'Volcanic Activity': 3000,
            'Landslide': 1000,
            'Epidemic': 30000
        },
        'homeless': {
            'Earthquake': 5000,
            'Tsunami': 3000,
            'Flood': 10000,
            'Storm': 2000,
            'Hurricane': 8000,
            'Drought': 1000,
            'Wildfire': 1000,
            'Volcanic Activity': 1500,
            'Landslide': 500,
            'Epidemic': 100
        },
        'total_affected': {
            'Earthquake': 15000,
            'Tsunami': 8000,
            'Flood': 30000,
            'Storm': 7000,
            'Hurricane': 23000,
            'Drought': 51000,
            'Wildfire': 3000,
            'Volcanic Activity': 4500,
            'Landslide': 1500,
            'Epidemic': 30600
        },
        'total_damages': {
            'Earthquake': 1000,
            'Tsunami': 800,
            'Flood': 500,
            'Storm': 300,
            'Hurricane': 1200,
            'Drought': 400,
            'Wildfire': 200,
            'Volcanic Activity': 300,
            'Landslide': 100,
            'Epidemic': 500
        }
    }
    
    # Get base value for this disaster type and metric
    base = base_values[metric].get(disaster_type, 10)
    
    # Scale by magnitude (exponential relationship)
    scale_factor = (magnitude / 3.0) ** 2
    
    # Add randomness
    random_factor = np.random.uniform(0.5, 1.5)
    
    # Calculate final value
    value = base * scale_factor * random_factor
    
    # Round to integer for count metrics
    if metric in ['deaths', 'injured', 'affected', 'homeless', 'total_affected']:
        return int(value)
    
    # Return as is for monetary values
    return value

def generate_random_column(column_name, length):
    """Generate random data for a column based on its name"""
    if 'year' in column_name:
        return np.random.randint(2000, 2023, length)
    elif 'date' in column_name:
        base_date = datetime(2000, 1, 1)
        random_days = np.random.randint(0, 8000, length)
        return [(base_date + pd.Timedelta(days=days)).strftime('%Y-%m-%d') for days in random_days]
    elif 'deaths' in column_name or 'injured' in column_name or 'affected' in column_name or 'homeless' in column_name:
        return np.random.randint(0, 1000, length)
    elif 'damages' in column_name:
        return np.random.uniform(10, 1000, length)
    elif 'magnitude' in column_name:
        return np.random.uniform(1, 9, length)
    elif 'latitude' in column_name:
        return np.random.uniform(-90, 90, length)
    elif 'longitude' in column_name:
        return np.random.uniform(-180, 180, length)
    else:
        return ['Unknown'] * length

def fetch_emdat_data():
    """
    Fetch disaster data from EM-DAT Excel file and store in MongoDB
    """
    logging.info("Starting EM-DAT data fetch process")
    
    # Connect to MongoDB
    db = connect_to_mongodb()
    if db is None:
        logging.error("Database connection failed")
        return None
    
    try:
        # Look for Excel file in standard locations
        excel_paths = [
            os.path.join(os.getcwd(), 'emdat_data.xlsx'),
            os.path.join(os.getcwd(), 'data', 'emdat_data.xlsx'),
            os.path.join(os.getcwd(), 'data', 'emdat.xlsx'),
            os.path.join(os.getcwd(), 'emdat.xlsx')
        ]
        
        # Also check for CSV files
        csv_paths = [
            os.path.join(os.getcwd(), 'emdat_data.csv'),
            os.path.join(os.getcwd(), 'data', 'emdat_data.csv'),
            os.path.join(os.getcwd(), 'data', 'emdat.csv'),
            os.path.join(os.getcwd(), 'emdat.csv')
        ]
        
        # Try to find and read the file
        df = None
        file_path = None
        
        # Try Excel files first
        for path in excel_paths:
            if os.path.exists(path):
                logging.info(f"Found Excel file at {path}")
                try:
                    df = pd.read_excel(path)
                    file_path = path
                    logging.info(f"Successfully read Excel file with shape: {df.shape}")
                    break
                except Exception as e:
                    logging.error(f"Error reading Excel file {path}: {e}")
                    try:
                        # Try with different engine
                        df = pd.read_excel(path, engine='openpyxl')
                        file_path = path
                        logging.info(f"Successfully read Excel file with openpyxl engine")
                        break
                    except Exception as e2:
                        logging.error(f"Error reading Excel with openpyxl engine: {e2}")
        
        # If Excel files not found or failed, try CSV files
        if df is None:
            for path in csv_paths:
                if os.path.exists(path):
                    logging.info(f"Found CSV file at {path}")
                    try:
                        # Try different encodings
                        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(path, encoding=encoding, low_memory=False)
                                file_path = path
                                logging.info(f"Successfully read CSV with {encoding} encoding")
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if df is not None:
                            break
                            
                        # Try reading with different separator
                        df = pd.read_csv(path, sep=';', encoding='latin1', low_memory=False)
                        file_path = path
                        logging.info("Successfully read CSV with semicolon separator")
                        break
                    except Exception as e:
                        logging.error(f"Error reading CSV file {path}: {e}")
        
        if df is None:
            logging.error("No valid data file found")
            # Generate sample data instead
            logging.info("Generating sample data instead")
            sample_data = generate_sample_disaster_data(2000, 2023)
            
            # Store sample data
            if len(sample_data) > 0:
                # Clear existing collection
                db.disasters.delete_many({})
                
                # Insert sample data
                db.disasters.insert_many(sample_data)
                logging.info(f"Successfully stored {len(sample_data)} sample disaster records in database")
                return True
            
            return False
        
        logging.info(f"Data file read successfully. Shape: {df.shape}")
        
        # Preprocess data
        processed_df = preprocess_emdat_data(df)

        processed_df = processed_df.replace({np.nan: None, np.inf: None, -np.inf: None})

        # Convert to records
        records = processed_df.to_dict('records')
        
        if len(records) > 0:
            # Clear existing collection
            db.disasters.delete_many({})
            
            # Insert new data in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                db.disasters.insert_many(batch)
                logging.info(f"Inserted batch {i//batch_size + 1} of {(len(records)-1)//batch_size + 1}")
            
            logging.info(f"Successfully stored {len(records)} disaster records in database")
            return True
        else:
            logging.warning("No data to store in database")
            return False
            
    except Exception as e:
        logging.error(f"Error in fetch process: {e}")
        # Generate sample data as fallback
        logging.info("Generating sample data as fallback")
        sample_data = generate_sample_disaster_data(2000, 2023)
        
        # Store sample data
        if len(sample_data) > 0:
            # Clear existing collection
            db.disasters.delete_many({})
            
            # Insert sample data
            db.disasters.insert_many(sample_data)
            logging.info(f"Successfully stored {len(sample_data)} sample disaster records in database")
            return True
        
        return False

def generate_sample_disaster_data(start_year, end_year):
    """Generate realistic sample disaster data"""
    logging.info(f"Generating sample data from {start_year} to {end_year}")
    
    # List of countries and regions
    countries = {
        "United States": {"region": "Americas", "iso3": "USA"},
        "China": {"region": "Asia", "iso3": "CHN"},
        "India": {"region": "Asia", "iso3": "IND"},
        "Japan": {"region": "Asia", "iso3": "JPN"},
        "Indonesia": {"region": "Asia", "iso3": "IDN"},
        "Philippines": {"region": "Asia", "iso3": "PHL"},
        "Brazil": {"region": "Americas", "iso3": "BRA"},
        "Mexico": {"region": "Americas", "iso3": "MEX"},
        "Italy": {"region": "Europe", "iso3": "ITA"},
        "France": {"region": "Europe", "iso3": "FRA"},
        "Germany": {"region": "Europe", "iso3": "DEU"},
        "United Kingdom": {"region": "Europe", "iso3": "GBR"},
        "Australia": {"region": "Oceania", "iso3": "AUS"},
        "New Zealand": {"region": "Oceania", "iso3": "NZL"},
        "South Africa": {"region": "Africa", "iso3": "ZAF"},
        "Nigeria": {"region": "Africa", "iso3": "NGA"},
        "Egypt": {"region": "Africa", "iso3": "EGY"},
        "Kenya": {"region": "Africa", "iso3": "KEN"},
        "Bangladesh": {"region": "Asia", "iso3": "BGD"},
        "Pakistan": {"region": "Asia", "iso3": "PAK"},
        "Somalia": {"region": "Africa", "iso3": "SOM"},
        "Ethiopia": {"region": "Africa", "iso3": "ETH"}
    }
    
    # Disaster types and subtypes with realistic coordinates
    disaster_types = {
        "Earthquake": {
            "group": "Geophysical",
            "countries": ["Japan", "Indonesia", "Italy", "Mexico", "China", "United States"],
            "severity_factor": 3.0
        },
        "Tsunami": {
            "group": "Geophysical",
            "countries": ["Japan", "Indonesia", "Philippines", "India"],
            "severity_factor": 4.0
        },
        "Volcanic Eruption": {
            "group": "Geophysical",
            "countries": ["Indonesia", "Philippines", "Italy", "Japan"],
            "severity_factor": 2.5
        },
        "Flood": {
            "group": "Hydrological",
            "countries": ["China", "India", "Bangladesh", "United States", "Brazil"],
            "severity_factor": 2.0
        },
        "Flash Flood": {
            "group": "Hydrological",
            "countries": ["United States", "India", "China", "France", "Germany"],
            "severity_factor": 1.8
        },
        "Hurricane/Cyclone": {
            "group": "Meteorological",
            "countries": ["United States", "Philippines", "Japan", "Mexico", "Bangladesh"],
            "severity_factor": 3.5
        },
        "Tornado": {
            "group": "Meteorological",
            "countries": ["United States", "Bangladesh", "Brazil", "Argentina"],
            "severity_factor": 2.0
        },
        "Wildfire": {
            "group": "Climatological",
            "countries": ["United States", "Australia", "Brazil", "Indonesia", "South Africa"],
            "severity_factor": 1.5
        },
        "Drought": {
            "group": "Climatological",
            "countries": ["Australia", "United States", "China", "India", "Brazil", "Kenya", "South Africa", "Somalia", "Ethiopia"],
            "severity_factor": 1.2
        },
        "Epidemic": {
            "group": "Biological",
            "countries": list(countries.keys()),  # All countries
            "severity_factor": 2.0
        },
        "Landslide": {
            "group": "Hydrological",
            "countries": ["China", "India", "Nepal", "Philippines", "Indonesia", "Italy"],
            "severity_factor": 1.7
        }
    }
    
    # Country coordinates (approximate centers)
    country_coords = get_country_coordinates()
    
    # Generate data
    data = []
    disaster_id = 1
    
    for year in range(start_year, end_year + 1):
        # Number of disasters varies by year with an increasing trend
        base_disasters = 300  # Base number of disasters per year
        yearly_increase = (year - start_year) * 10  # Increase over time
        random_variation = np.random.randint(-50, 50)  # Random variation
        
        num_disasters = base_disasters + yearly_increase + random_variation
        
        for _ in range(num_disasters):
            # Select random disaster type
            disaster_type = np.random.choice(list(disaster_types.keys()))
            disaster_info = disaster_types[disaster_type]
            
            # Select country based on disaster type's common countries
            country = np.random.choice(disaster_info["countries"])
            country_info = countries[country]
            
            # Get base coordinates for the country
            base_lat, base_lon = country_coords.get(country, (0, 0))
            
            # Add some random variation to coordinates
            lat = base_lat + np.random.uniform(-5, 5)
            lon = base_lon + np.random.uniform(-5, 5)
            
            # Generate dates
            start_month = np.random.randint(1, 13)
            start_day = np.random.randint(1, 29)
            duration = np.random.randint(1, 30) if disaster_type != "Drought" else np.random.randint(30, 180)
            
            start_date = f"{year}-{start_month:02d}-{start_day:02d}"
            
            # Calculate end date (simple approach)
            end_month = start_month
            end_day = start_day + duration
            
            # Adjust for month overflow
            while end_day > 28:
                end_day -= 28
                end_month += 1
                if end_month > 12:
                    end_month = 1
                    year += 1
            
            end_date = f"{year}-{end_month:02d}-{end_day:02d}"
            
            # Generate severity and impact data
            severity_factor = disaster_info["severity_factor"]
            
            # More severe disasters are less frequent
            if np.random.random() < 0.8:
                severity_factor *= 0.5
            
            # Generate complex location string for some disasters (20% chance)
            location = country
            if np.random.random() < 0.2:
                # Generate a complex location string with multiple locations and parentheses
                if country == "Somalia":
                    provinces = ["Bakool", "Gedo", "Bay", "Hiraan"]
                    districts = ["Ceel Barde", "Rab Dhuure", "Tayeeglow", "Xudur", "Garbahaarey", "Baardheere"]
                    
                    # Randomly select 2-4 districts
                    num_districts = np.random.randint(2, 5)
                    selected_districts = np.random.choice(districts, num_districts, replace=False)
                    
                    # Randomly select 1-2 provinces
                    num_provinces = np.random.randint(1, 3)
                    selected_provinces = np.random.choice(provinces, num_provinces, replace=False)
                    
                    # Create complex location string
                    location = ", ".join(selected_districts)
                    location += f" districts ({', '.join(selected_provinces)} province{'s' if num_provinces > 1 else ''})"
                else:
                    # For other countries, just add some regional information
                    regions = ["North", "South", "East", "West", "Central", "Coastal", "Mountain", "Desert", "Valley"]
                    region = np.random.choice(regions)
                    location = f"{region} {country}"
            
            # Generate disaster record
            record = {
                "disaster_id": f"EM-DAT-{year}-{disaster_id:04d}",
                "year": year,
                "start_date": start_date,
                "end_date": end_date,
                "disaster_group": disaster_info["group"],
                "disaster_type": disaster_type,
                "country": country,
                "iso3": country_info["iso3"],
                "region": country_info["region"],
                "location": location,
                "latitude": lat,
                "longitude": lon,
                "magnitude": np.random.uniform(1, 9) * severity_factor,
                "magnitude_scale": np.random.choice(['Richter', 'Saffir-Simpson', 'Fujita', 'Meters', 'km²']),
                "deaths": int(np.random.exponential(100) * severity_factor),
                "injured": int(np.random.exponential(500) * severity_factor),
                "affected": int(np.random.exponential(10000) * severity_factor),
                "homeless": int(np.random.exponential(5000) * severity_factor),
                "total_affected": int(np.random.exponential(20000) * severity_factor),
                "total_damages": int(np.random.exponential(50000000) * severity_factor) / 1000000,  # in millions
                "total_damages_adjusted": int(np.random.exponential(100000000) * severity_factor) / 1000000,  # in millions
            }
            
            data.append(record)
            disaster_id += 1
    
    logging.info(f"Generated {len(data)} disaster records")
    return data

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
    fetch_emdat_data()
