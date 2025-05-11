from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import json
import random

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DisasterPredictionModel:
    def __init__(self, data=None):
        self.model = None
        self.model_path = 'disaster_impact_model.pkl'
        self.data = data
        self.region_risk_factors = {
            'Africa': 1.2,
            'Americas': 1.0,
            'Asia': 1.3,
            'Europe': 0.8,
            'Oceania': 0.9
        }
        self.country_risk_factors = {
            # Africa
            'Nigeria': 1.1, 'South Africa': 0.9, 'Egypt': 1.0, 'Kenya': 1.2, 'Ethiopia': 1.3,
            # Americas
            'United States': 0.8, 'Canada': 0.7, 'Brazil': 1.1, 'Mexico': 1.0, 'Argentina': 0.9,
            # Asia
            'China': 1.2, 'India': 1.3, 'Japan': 1.0, 'Indonesia': 1.4, 'Pakistan': 1.3,
            # Europe
            'Germany': 0.7, 'United Kingdom': 0.8, 'France': 0.7, 'Italy': 0.9, 'Spain': 0.8,
            # Oceania
            'Australia': 0.9, 'New Zealand': 0.7, 'Papua New Guinea': 1.2, 'Fiji': 1.1, 'Solomon Islands': 1.2
        }
        self.disaster_risk_factors = {
            'Earthquake': 1.5,
            'Tsunami': 1.8,
            'Flood': 1.2,
            'Hurricane': 1.4,
            'Wildfire': 1.1,
            'Drought': 0.9,
            'Volcanic Eruption': 1.6,
            'Landslide': 1.3,
            'Extreme Weather': 1.0,
            'Epidemic': 1.7,
            'Flash Flood': 1.3,
            'Hurricane/Cyclone': 1.5,
            'Tornado': 1.2
        }
        self.population_density_factors = {
            'Low': 0.7,      # <50 people per km²
            'Medium': 1.0,   # 50-200 people per km²
            'High': 1.3,     # 200-500 people per km²
            'Very High': 1.6 # >500 people per km²
        }
        
    def prepare_data(self):
        """Prepare disaster data for machine learning model"""
        df = self.data
        
        # Select relevant features
        features = ['disaster_group', 'disaster_type', 'magnitude', 'year', 'region', 'country']
        
        # Target variables
        targets = ['deaths', 'total_damages']
        
        # Ensure all required columns exist
        for col in features + targets:
            if col not in df.columns:
                if col == 'disaster_group' and 'disaster_type' in df.columns:
                    df['disaster_group'] = df['disaster_type'].apply(self.map_disaster_type_to_group)
                else:
                    df[col] = np.random.rand(len(df))
        
        # Handle missing values
        for col in features + targets:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
        
        # Create X and y
        X = df[features]
        y = df[targets]
        
        return X, y
    
    def map_disaster_type_to_group(self, disaster_type):
        """Map disaster type to disaster group"""
        disaster_groups = {
            'Earthquake': 'Geophysical',
            'Tsunami': 'Geophysical',
            'Volcanic Eruption': 'Geophysical',
            'Volcanic Activity': 'Geophysical',
            'Mass movement (dry)': 'Geophysical',
            'Storm': 'Meteorological',
            'Hurricane': 'Meteorological',
            'Hurricane/Cyclone': 'Meteorological',
            'Tornado': 'Meteorological',
            'Extreme temperature': 'Meteorological',
            'Fog': 'Meteorological',
            'Extreme Weather': 'Meteorological',
            'Flood': 'Hydrological',
            'Flash Flood': 'Hydrological',
            'Landslide': 'Hydrological',
            'Wave action': 'Hydrological',
            'Drought': 'Climatological',
            'Wildfire': 'Climatological',
            'Glacial lake outburst': 'Climatological',
            'Epidemic': 'Biological',
            'Insect infestation': 'Biological',
            'Animal accident': 'Biological',
            'Industrial accident': 'Technological',
            'Transport accident': 'Technological',
            'Miscellaneous accident': 'Technological'
        }
        
        return disaster_groups.get(disaster_type, 'Other')
    
    def train_model(self):
        """Train the prediction model"""
        X, y = self.prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing for categorical features
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numeric_features = [col for col in X.columns if X[col].dtype != 'object']
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
        
        # Create and train model
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        self.model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        
        return self.model
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return self.model
        else:
            return self.train_model()
    
    def predict(self, input_data):
        """Make predictions using the trained model"""
        if self.model is None:
            self.load_model()
        
        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)

        
        # Make prediction
        try:
            prediction = self.model.predict(input_data)
            casualties = prediction[0][0]
            economic_damage = prediction[0][1]
        except Exception as e:
            # If model prediction fails, use a more robust approach
            print(f"Model prediction failed: {e}. Using alternative prediction method.")
            casualties, economic_damage = self.alternative_prediction(input_data)
        
        # Apply regional and country risk factors
        def get_value(df, col, default=None):
            value = df[col].iloc[0] if col in df.columns else default
            if isinstance(value, list):
                return value[0] if value else default
            return value

        region = get_value(input_data,'region')
        country = get_value(input_data,'country')
        state = get_value(input_data,'state')
        disaster_type = get_value(input_data,'disaster_type') or get_value(input_data,'disaster_group')
        population_density = get_value(input_data,'population_density', 'Medium')
        magnitude = float(get_value(input_data,'magnitude', 5.0))
        
        # Apply risk factors
        region_factor = self.region_risk_factors.get(region, 1.0)
        country_factor = self.country_risk_factors.get(country, 1.0)
        disaster_factor = self.disaster_risk_factors.get(disaster_type, 1.0)
        density_factor = self.population_density_factors.get(population_density, 1.0)
        
        # State factor (if provided)
        state_factor = 1.0
        if state:
            # Add some variation based on state
            state_factor = 0.8 + (hash(state) % 5) / 10  # Between 0.8 and 1.3
        
        # Adjust casualties and economic damage based on factors
        casualties = casualties * region_factor * country_factor * disaster_factor * density_factor * state_factor
        economic_damage = economic_damage * region_factor * country_factor * disaster_factor * density_factor * state_factor
        
        # Adjust based on magnitude (exponential relationship)
        magnitude_factor = np.exp((magnitude - 5) / 5)
        casualties = casualties * magnitude_factor
        economic_damage = economic_damage * magnitude_factor
        
        # Calculate impact score (0-100)
        # Normalize casualties and economic damage to 0-50 scale
        max_casualties = 10000  # Adjust based on your data
        max_damage = 10000  # In millions, adjust based on your data
        
        casualties_score = min(50, (casualties / max_casualties) * 50)
        damage_score = min(50, (economic_damage / max_damage) * 50)
        
        # Combine for total impact score
        impact_score = int(casualties_score + damage_score)
        
        # Calculate recovery time based on impact score and disaster type
        recovery_factors = {
            'Earthquake': 1.5,
            'Hurricane': 1.2,
            'Hurricane/Cyclone': 1.4,
            'Flood': 1.0,
            'Flash Flood': 1.1,
            'Wildfire': 0.8,
            'Tornado': 0.7,
            'Drought': 2.0,
            'Tsunami': 1.8,
            'Volcanic Eruption': 1.6,
            'Volcanic Activity': 1.6,
            'Landslide': 1.3,
            'Extreme Weather': 1.0,
            'Epidemic': 1.7
        }
        
        # Default factor if disaster type not in the dictionary
        recovery_factor = recovery_factors.get(disaster_type, 1.0)
        
        # Calculate recovery time in months
        min_recovery = max(1, int((impact_score / 20) * recovery_factor))
        max_recovery = max(2, int((impact_score / 15) * recovery_factor))
        
        # Calculate affected population based on population density
        population_base = {
            'Low': 500,
            'Medium': 5000,
            'High': 50000,
            'Very High': 200000
        }
        
        base_population = population_base.get(population_density, 5000)
        min_affected = int(base_population * (impact_score / 100) * 0.8)
        max_affected = int(base_population * (impact_score / 100) * 1.5)
        
        # Infrastructure damage percentage
        infra_damage_pct = min(95, impact_score + 10)
        
        # Sector impact breakdown
        housing_impact = min(100, int(impact_score * 1.1))
        infrastructure_impact = min(100, int(impact_score * 0.9))
        economic_impact = min(100, int(impact_score * 1.0))
        health_impact = min(100, int(impact_score * 0.8))
        environment_impact = min(100, int(impact_score * 0.7))
        
        # Prepare prediction result
        result = {
            'impact_score': int(impact_score),
            'casualties': int(casualties),
            'casualties_low': max(0, int(casualties * 0.8)),
            'casualties_high': int(casualties * 1.2),
            'economic_damage': float(economic_damage),
            'economic_damage_low': max(0, float(economic_damage * 0.8)),
            'economic_damage_high': float(economic_damage * 1.2),
            'disaster_type': str(disaster_type),
            'region': str(region),
            'country': str(country),
            'state': str(state) if state else "N/A",
            'high_casualties': bool(casualties > 100),
            'high_damage': bool(economic_damage > 500),
            'recovery_time_min': int(min_recovery),
            'recovery_time_max': int(max_recovery),
            'affected_min': int(min_affected),
            'affected_max': int(max_affected),
            'infrastructure_damage': int(infra_damage_pct),
            'sector_impact': {
                'housing': int(housing_impact),
                'infrastructure': int(infrastructure_impact),
                'economic': int(economic_impact),
                'health': int(health_impact),
                'environment': int(environment_impact)
            }
        }
        
        return result
    
    def alternative_prediction(self, input_data):
        """Alternative prediction method when model fails"""
        def get_value(df, col, default=None):
            value = df[col].iloc[0] if col in df.columns else default
            if isinstance(value, list):
                return value[0] if value else default
            return value

        # Extract input parameters
        region = get_value(input_data,'region')
        country = get_value(input_data,'country')
        state = get_value(input_data,'state')
        disaster_type = get_value(input_data,'disaster_type') or get_value(input_data,'disaster_group')
        population_density = get_value(input_data,'population_density', 'Medium')
        magnitude = float(get_value(input_data,'magnitude', 5.0))
        
        # Base casualties and damage by disaster type
        base_values = {
            'Earthquake': (500, 2000),
            'Tsunami': (1000, 5000),
            'Flood': (200, 1000),
            'Flash Flood': (150, 800),
            'Hurricane': (300, 3000),
            'Hurricane/Cyclone': (350, 3500),
            'Tornado': (100, 500),
            'Wildfire': (50, 500),
            'Drought': (100, 1000),
            'Volcanic Eruption': (200, 1000),
            'Volcanic Activity': (200, 1000),
            'Landslide': (100, 500),
            'Extreme Weather': (50, 300),
            'Epidemic': (1000, 2000)
        }
        
        # Get base values for the disaster type
        base_casualties, base_damage = base_values.get(disaster_type, (200, 1000))
        
        # Adjust based on magnitude
        magnitude_factor = (magnitude / 5.0) ** 2
        casualties = base_casualties * magnitude_factor
        economic_damage = base_damage * magnitude_factor
        
        # Adjust based on region
        region_factor = self.region_risk_factors.get(region, 1.0)
        casualties *= region_factor
        economic_damage *= region_factor
        
        # Adjust based on country
        country_factor = self.country_risk_factors.get(country, 1.0)
        casualties *= country_factor
        economic_damage *= country_factor
        
        # Adjust based on population density
        density_factor = self.population_density_factors.get(population_density, 1.0)
        casualties *= density_factor
        economic_damage *= density_factor
        
        # Add some randomness
        casualties *= random.uniform(0.8, 1.2)
        economic_damage *= random.uniform(0.8, 1.2)
        
        return casualties, economic_damage
