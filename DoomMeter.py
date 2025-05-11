from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import joblib
from pymongo import MongoClient
import pycountry
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import warnings
import io
import csv
from prediction import DisasterPredictionModel, NumpyEncoder
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['APP_NAME'] = "Doom Meter"
app.json_encoder = NumpyEncoder

# Configure login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Connect to MongoDB
def get_db():
    uri = "mongodb+srv://rishavpal309:6397661626@personal01.gjqkp9c.mongodb.net/DisasterData?authSource=admin&retryWrites=true&w=majority&tls=true&appName=Personal01"

    client = MongoClient(uri)    
    return client[os.environ["MONGODB_DB"]]

db = get_db()
disasters_collection = db.disasters
alerts_collection = db.alerts
resources_collection = db.resources
evacuation_collection = db.evacuation_plans
damage_collection = db.damage_reports
users_collection = db.users
prediction_collection = db.predictions

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.role = user_data['role']

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': user_id})
    if user_data:
        return User(user_data)
    return None

# Role-based access control
def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Helper function to fetch disaster data
def fetch_disaster_data():
    # Check if data exists in MongoDB
    if disasters_collection.count_documents({}) > 0:
        disasters_cursor = disasters_collection.find({})
        disasters_list = list(disasters_cursor)
        
        # Convert MongoDB documents to DataFrame
        df = pd.DataFrame(disasters_list)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        return df
    else:
        # If no data exists, return empty DataFrame
        return pd.DataFrame()

# Initialize resources if they don't exist
def init_resources():
    if resources_collection.count_documents({}) == 0:
        default_resources = {
            'Medical Supplies': 1000,
            'Food Packages': 500,
            'Water (liters)': 2000,
            'Shelter Kits': 200,
            'Rescue Teams': 20
        }
        resources_collection.insert_one({'resources': default_resources})

# Initialize admin user if it doesn't exist
def init_admin_user():
    if users_collection.count_documents({'role': 'admin'}) == 0:
        admin_id = str(uuid.uuid4())
        users_collection.insert_one({
            '_id': admin_id,
            'username': 'admin',
            'email': 'admin@example.com',
            'password': generate_password_hash('admin123'),
            'role': 'admin'
        })
        print("Admin user created: username=admin, password=admin123")

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_data = users_collection.find_one({'username': username})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        # Create new user
        user_id = str(uuid.uuid4())
        users_collection.insert_one({
            '_id': user_id,
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'role': 'user'  # Default role
        })
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent alerts
    recent_alerts = list(alerts_collection.find().sort('time', -1).limit(5))
    
    # Get resource summary
    resources_doc = resources_collection.find_one({})
    resources = resources_doc['resources'] if resources_doc else {}
    
    # Get recent disasters
    disaster_data = fetch_disaster_data()
    current_year = datetime.now().year
    recent_disasters = disaster_data[disaster_data['year'] >= current_year - 5].head(10).to_dict('records') if not disaster_data.empty else []
    
    # Calculate statistics for last 12 months
    last_year = datetime.now() - timedelta(days=365)
    last_year_str = last_year.strftime('%Y-%m-%d')
    
    if not disaster_data.empty and 'start_date' in disaster_data.columns:
        recent_data = disaster_data[disaster_data['start_date'] >= last_year_str]
        
        stats = {
            'total_disasters': len(recent_data),
            'total_deaths': int(recent_data['deaths'].sum()) if 'deaths' in recent_data.columns else 0,
            'total_affected': int(recent_data['affected'].sum()) if 'affected' in recent_data.columns else 0,
            'total_damages': float(recent_data['total_damages'].sum()) if 'total_damages' in recent_data.columns else 0
        }
    else:
        stats = {
            'total_disasters': 0,
            'total_deaths': 0,
            'total_affected': 0,
            'total_damages': 0
        }
    
    # Get disaster data for map
    if not disaster_data.empty:
        map_data = disaster_data[disaster_data['year'] >= current_year - 5].head(500)
        map_data = map_data.to_dict('records')
    else:
        map_data = []
    
    # Get disaster trends
    if not disaster_data.empty:
        yearly_disasters = disaster_data.groupby('year').size().reset_index(name='count')
        yearly_disasters = yearly_disasters.to_dict('records')
    else:
        yearly_disasters = []
    
    # Get disaster types distribution
    if not disaster_data.empty:
        if 'disaster_type' in disaster_data.columns:
            type_field = 'disaster_type'
        else:
            type_field = 'disaster_group'
        
        disaster_types = disaster_data[type_field].value_counts().head(5).to_dict()
    else:
        disaster_types = {}
    
    # Sample data for dashboard
    sample_recent_disasters = [
        {
            'name': 'Hurricane Maria',
            'location': 'Puerto Rico',
            'date': '2023-09-15',
            'severity': 'Severe',
            'severity_color': 'red',
            'color': 'blue',
            'icon': 'hurricane',
            'magnitude': 4.8,
            'deaths': 124,
            'economic_impact': 95.2
        },
        {
            'name': 'Wildfire',
            'location': 'California, USA',
            'date': '2023-08-22',
            'severity': 'Moderate',
            'severity_color': 'yellow',
            'color': 'orange',
            'icon': 'fire',
            'magnitude': 3.2,
            'deaths': 12,
            'economic_impact': 45.7
        },
        {
            'name': 'Earthquake',
            'location': 'Tokyo, Japan',
            'date': '2023-07-10',
            'severity': 'Severe',
            'severity_color': 'red',
            'color': 'red',
            'icon': 'globe',
            'magnitude': 6.7,
            'deaths': 87,
            'economic_impact': 120.3
        },
        {
            'name': 'Flooding',
            'location': 'Bangladesh',
            'date': '2023-06-30',
            'severity': 'Moderate',
            'severity_color': 'yellow',
            'color': 'blue',
            'icon': 'water',
            'magnitude': 2.9,
            'deaths': 45,
            'economic_impact': 32.8
        }
    ]
    
    # Sample risk assessment data
    risk_assessment = [
        {
            'name': 'Southeast Asia',
            'risk_level': 'High',
            'risk_color': 'danger',
            'primary_threat': 'Typhoons & Flooding',
            'population': '650M',
            'econ_vulnerability': 75,
            'econ_color': 'danger'
        },
        {
            'name': 'Western United States',
            'risk_level': 'High',
            'risk_color': 'danger',
            'primary_threat': 'Wildfires & Drought',
            'population': '80M',
            'econ_vulnerability': 65,
            'econ_color': 'warning'
        },
        {
            'name': 'Caribbean',
            'risk_level': 'High',
            'risk_color': 'danger',
            'primary_threat': 'Hurricanes',
            'population': '44M',
            'econ_vulnerability': 85,
            'econ_color': 'danger'
        },
        {
            'name': 'Central Europe',
            'risk_level': 'Medium',
            'risk_color': 'warning',
            'primary_threat': 'Flooding',
            'population': '190M',
            'econ_vulnerability': 45,
            'econ_color': 'warning'
        },
        {
            'name': 'East Africa',
            'risk_level': 'High',
            'risk_color': 'danger',
            'primary_threat': 'Drought & Famine',
            'population': '260M',
            'econ_vulnerability': 90,
            'econ_color': 'danger'
        }
    ]
    
    return render_template('dashboard.html', 
                          alerts=recent_alerts, 
                          resources=resources, 
                          recent_disasters=sample_recent_disasters,
                          stats=stats,
                          map_data=json.dumps(map_data, cls=NumpyEncoder),
                          yearly_disasters=json.dumps(yearly_disasters, cls=NumpyEncoder),
                          disaster_types=disaster_types,
                          current_date=datetime.now().strftime('%Y-%m-%d'),
                          risk_assessment=risk_assessment,
                          total_disasters=256,
                          affected_population="42.3M",
                          economic_damage="32.5",
                          high_risk_regions=5)

@app.route('/alerts', methods=['GET', 'POST'])
@login_required
def alerts():
    if request.method == 'POST':
        # Create new alert
        alert_type = request.form.get('alert_type')
        alert_severity = int(request.form.get('alert_severity'))
        alert_location = request.form.get('alert_location')
        alert_description = request.form.get('alert_description')
        alert_time = request.form.get('alert_time')
        
        if alert_location:
            new_alert = {
                "_id": str(uuid.uuid4()),
                "type": alert_type,
                "severity": alert_severity,
                "location": alert_location,
                "description": alert_description,
                "time": alert_time,
                "created_by": current_user.username,
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            alerts_collection.insert_one(new_alert)
            flash('Alert issued successfully!', 'success')
            return redirect(url_for('alerts'))
    
    # Get all alerts
    all_alerts = list(alerts_collection.find().sort('time', -1))
    
    return render_template('alerts.html', alerts=all_alerts, now=datetime.now().strftime('%Y-%m-%d'))

@app.route('/alerts/delete/<alert_id>')
@login_required
def delete_alert(alert_id):
    alerts_collection.delete_one({"_id": alert_id})
    flash('Alert resolved successfully!', 'success')
    return redirect(url_for('alerts'))

@app.route('/resources', methods=['GET', 'POST'])
@login_required
def resources():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update':
            # Update resources
            resource_type = request.form.get('resource_type')
            update_action = request.form.get('update_action')
            quantity = int(request.form.get('quantity'))
            
            # Get current resources
            resources_doc = resources_collection.find_one({})
            current_resources = resources_doc['resources'] if resources_doc else {}
            
            if update_action == 'add':
                current_resources[resource_type] = current_resources.get(resource_type, 0) + quantity
            else:  # remove
                if current_resources.get(resource_type, 0) >= quantity:
                    current_resources[resource_type] = current_resources.get(resource_type, 0) - quantity
                else:
                    flash(f'Not enough {resource_type} available!', 'danger')
                    return redirect(url_for('resources'))
            
            # Update in database
            resources_collection.update_one({}, {"$set": {"resources": current_resources}}, upsert=True)
            flash(f'Resources updated successfully!', 'success')
            
        elif action == 'allocate':
            # Allocate resources
            resource_type = request.form.get('allocation_resource')
            quantity = int(request.form.get('allocation_quantity'))
            location = request.form.get('allocation_location')
            priority = request.form.get('allocation_priority')
            
            # Get current resources
            resources_doc = resources_collection.find_one({})
            current_resources = resources_doc['resources'] if resources_doc else {}
            
            if current_resources.get(resource_type, 0) >= quantity:
                current_resources[resource_type] = current_resources.get(resource_type, 0) - quantity
                
                # Update in database
                resources_collection.update_one({}, {"$set": {"resources": current_resources}}, upsert=True)
                
                # Record allocation
                allocation = {
                    "_id": str(uuid.uuid4()),
                    "resource": resource_type,
                    "quantity": quantity,
                    "location": location,
                    "priority": priority,
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "allocated_by": current_user.username
                }
                
                db.resource_allocations.insert_one(allocation)
                flash(f'Resources allocated successfully!', 'success')
            else:
                flash(f'Not enough {resource_type} available!', 'danger')
    
    # Get current resources
    resources_doc = resources_collection.find_one({})
    current_resources = resources_doc['resources'] if resources_doc else {}
    
    # Get recent allocations
    recent_allocations = list(db.resource_allocations.find().sort('date', -1).limit(10))
    
    return render_template('resources.html', 
                          resources=current_resources,
                          allocations=recent_allocations)

@app.route('/evacuation', methods=['GET', 'POST'])
@login_required
def evacuation():
    if request.method == 'POST':
        # Create evacuation plan
        evac_area = request.form.get('evac_area')
        evac_population = int(request.form.get('evac_population'))
        evac_reason = request.form.get('evac_reason')
        evac_start = request.form.get('evac_start')
        evac_duration = int(request.form.get('evac_duration'))
        evac_shelter = request.form.get('evac_shelter')
        
        if evac_area and evac_shelter:
            new_plan = {
                "_id": str(uuid.uuid4()),
                "area": evac_area,
                "population": evac_population,
                "reason": evac_reason,
                "start_date": evac_start,
                "duration": evac_duration,
                "shelter": evac_shelter,
                "status": "Planned",
                "created_by": current_user.username,
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            evacuation_collection.insert_one(new_plan)
            flash('Evacuation plan created successfully!', 'success')
            return redirect(url_for('evacuation'))
    
    # Get all evacuation plans
    all_plans = list(evacuation_collection.find())
    
    return render_template('evacuation.html', plans=all_plans)

@app.route('/evacuation/update/<plan_id>', methods=['POST'])
@login_required
def update_evacuation(plan_id):
    new_status = request.form.get('new_status')
    
    evacuation_collection.update_one(
        {"_id": plan_id},
        {"$set": {"status": new_status}}
    )
    
    flash('Plan status updated successfully!', 'success')
    return redirect(url_for('evacuation'))

@app.route('/damage', methods=['GET', 'POST'])
@login_required
def damage():
    if request.method == 'POST':
        # Create damage report
        damage_location = request.form.get('damage_location')
        damage_type = request.form.get('damage_type')
        damage_date = request.form.get('damage_date')
        infrastructure_damage = int(request.form.get('infrastructure_damage'))
        casualties = int(request.form.get('casualties'))
        economic_impact = float(request.form.get('economic_impact'))
        damage_notes = request.form.get('damage_notes')
        
        if damage_location:
            new_report = {
                "_id": str(uuid.uuid4()),
                "location": damage_location,
                "disaster_type": damage_type,
                "date": damage_date,
                "infrastructure_damage": infrastructure_damage,
                "casualties": casualties,
                "economic_impact": economic_impact,
                "notes": damage_notes,
                "reported_by": current_user.username,
                "reported_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            damage_collection.insert_one(new_report)
            flash('Damage report submitted successfully!', 'success')
            return redirect(url_for('damage'))
    
    # Get all damage reports
    all_reports = list(damage_collection.find())
    
    return render_template('damage.html', reports=all_reports)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    disaster_data = fetch_disaster_data()
    prediction_model = DisasterPredictionModel(disaster_data)
    
    # Get unique disaster types and regions
    if not disaster_data.empty:
        if 'disaster_type' in disaster_data.columns:
            disaster_types = sorted(disaster_data['disaster_type'].unique())
        else:
            disaster_types = sorted(disaster_data['disaster_group'].unique())
        
        regions = sorted(disaster_data['region'].unique()) if 'region' in disaster_data.columns else []
    else:
        disaster_types = ["Earthquake", "Flood", "Hurricane", "Wildfire", "Drought", "Tsunami", "Landslide", "Volcanic Eruption"]
        regions = ["Americas", "Asia", "Europe", "Africa", "Oceania"]
    
    prediction_result = None
    
    if request.method == 'POST':
        # Get prediction inputs
        pred_disaster_type = request.form.get('pred_disaster_type')
        pred_magnitude = float(request.form.get('pred_magnitude'))
        pred_year = int(request.form.get('pred_year'))
        pred_region = request.form.get('pred_region') if regions else "Unknown"
        pred_duration = int(request.form.get('pred_duration', 1))
        pred_population_density = request.form.get('pred_population_density', 'Medium')
        
        # Prepare input features
        input_data = {
            'disaster_group' if not disaster_data.empty and 'disaster_group' in disaster_data.columns else 'disaster_type': pred_disaster_type,
            'magnitude': pred_magnitude,
            'year': pred_year,
            'population_density': pred_population_density
        }
        
        if 'region' in disaster_data.columns or not disaster_data.empty:
            input_data['region'] = pred_region
        
        # Create DataFrame with the input
        input_df = pd.DataFrame([[input_data]])
        
        # Make prediction
        try:
            prediction_result = prediction_model.predict(input_df)
            
            # Save prediction to database
            prediction_id = str(uuid.uuid4())
            prediction_data = {
                "_id": prediction_id,
                "disaster_type": pred_disaster_type,
                "magnitude": pred_magnitude,
                "year": pred_year,
                "region": pred_region,
                "population_density": pred_population_density,
                "result": prediction_result,
                "created_by": current_user.username,
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            prediction_collection.insert_one(prediction_data)
            
            # Add prediction ID to result for download
            prediction_result['prediction_id'] = prediction_id
            
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
    
    return render_template('prediction.html', 
                          disaster_types=disaster_types,
                          regions=regions,
                          prediction=prediction_result,
                          current_year=datetime.now().year)

@app.route('/download_prediction/<prediction_id>')
@login_required
def download_prediction(prediction_id):
    # Get prediction from database
    prediction_data = prediction_collection.find_one({"_id": prediction_id})
    
    if not prediction_data:
        flash('Prediction not found', 'danger')
        return redirect(url_for('prediction'))
    
    # Create CSV file
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Disaster Impact Prediction Report'])
    writer.writerow(['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow(['Generated by', current_user.username])
    writer.writerow([])
    
    # Write input parameters
    writer.writerow(['Input Parameters'])
    writer.writerow(['Disaster Type', prediction_data['disaster_type']])
    writer.writerow(['Magnitude', prediction_data['magnitude']])
    writer.writerow(['Year', prediction_data['year']])
    writer.writerow(['Region', prediction_data['region']])
    writer.writerow(['Population Density', prediction_data['population_density']])
    writer.writerow([])
    
    # Write prediction results
    result = prediction_data['result']
    writer.writerow(['Prediction Results'])
    writer.writerow(['Impact Score', result['impact_score']])
    writer.writerow(['Casualties (Estimated)', f"{result['casualties_low']} - {result['casualties_high']}"]) 
    writer.writerow(['Economic Damage ($ millions)', f"{result['economic_damage_low']} - {result['economic_damage_high']}"])
    writer.writerow(['Recovery Time (months)', f"{result['recovery_time_min']} - {result['recovery_time_max']}"])
    writer.writerow(['Population Affected', f"{result['affected_min']} - {result['affected_max']}"])
    writer.writerow(['Infrastructure Damage (%)', result['infrastructure_damage']])
    writer.writerow([])
    
    # Write sector impact
    writer.writerow(['Sector Impact'])
    writer.writerow(['Housing', f"{result['sector_impact']['housing']}%"])
    writer.writerow(['Infrastructure', f"{result['sector_impact']['infrastructure']}%"])
    writer.writerow(['Economic', f"{result['sector_impact']['economic']}%"])
    writer.writerow(['Health', f"{result['sector_impact']['health']}%"])
    writer.writerow(['Environment', f"{result['sector_impact']['environment']}%"])
    
    # Prepare response
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'disaster_prediction_{prediction_id}.csv'
    )

@app.route('/explorer')
@login_required
def explorer():
    disaster_data = fetch_disaster_data()
    
    # Get min and max years
    if not disaster_data.empty and 'year' in disaster_data.columns:
        min_year = int(disaster_data['year'].min())
        max_year = int(disaster_data['year'].max())
    else:
        min_year = 2000
        max_year = datetime.now().year
    
    # Get unique disaster types and regions
    if not disaster_data.empty:
        if 'disaster_type' in disaster_data.columns:
            disaster_types = sorted(disaster_data['disaster_type'].unique())
        else:
            disaster_types = sorted(disaster_data['disaster_group'].unique())
        
        regions = sorted(disaster_data['region'].unique()) if 'region' in disaster_data.columns else []
    else:
        disaster_types = ["Earthquake", "Flood", "Hurricane", "Wildfire", "Drought", "Tsunami", "Landslide", "Volcanic Eruption"]
        regions = ["Americas", "Asia", "Europe", "Africa", "Oceania"]
    
    return render_template('explorer.html',
                          min_year=min_year,
                          max_year=max_year,
                          disaster_types=disaster_types,
                          regions=regions)

@app.route('/api/explorer/data')
@login_required
def explorer_data():
    disaster_data = fetch_disaster_data()
    
    # Get filter parameters
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    disaster_types = request.args.getlist('disaster_types[]')
    regions = request.args.getlist('regions[]')
    
    # Apply filters
    if not disaster_data.empty:
        filtered_data = disaster_data
        
        if start_year and end_year and 'year' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]
        
        if disaster_types:
            if 'disaster_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_type'].isin(disaster_types)]
            elif 'disaster_group' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_group'].isin(disaster_types)]
        
        if regions and 'region' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['region'].isin(regions)]
        
        # Convert to records
        filtered_data = filtered_data.replace({pd.NA: None, np.nan: None})
        records = filtered_data.head(1000).to_dict('records')  # Limit to 1000 records for performance
    else:
        records = []

    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Returning {len(records)} records")
    
    return jsonify(records)

@app.route('/api/explorer/yearly-counts')
@login_required
def explorer_yearly_counts():
    disaster_data = fetch_disaster_data()
    
    # Get filter parameters
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    disaster_types = request.args.getlist('disaster_types[]')
    regions = request.args.getlist('regions[]')
    
    # Apply filters
    if not disaster_data.empty:
        filtered_data = disaster_data
        
        if start_year and end_year and 'year' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]
        
        if disaster_types:
            if 'disaster_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_type'].isin(disaster_types)]
            elif 'disaster_group' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_group'].isin(disaster_types)]
        
        if regions and 'region' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['region'].isin(regions)]
        
        # Group by year
        if 'year' in filtered_data.columns:
            yearly_counts = filtered_data.groupby('year').size().reset_index(name='count')
            result = yearly_counts.to_dict('records')
        else:
            result = []
    else:
        result = []
    
    return jsonify(result)

@app.route('/api/explorer/type-counts')
@login_required
def explorer_type_counts():
    disaster_data = fetch_disaster_data()
    
    # Get filter parameters
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    disaster_types = request.args.getlist('disaster_types[]')
    regions = request.args.getlist('regions[]')
    
    # Apply filters
    if not disaster_data.empty:
        filtered_data = disaster_data
        
        if start_year and end_year and 'year' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]
        
        if disaster_types:
            if 'disaster_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_type'].isin(disaster_types)]
            elif 'disaster_group' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_group'].isin(disaster_types)]
        
        if regions and 'region' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['region'].isin(regions)]
        
        # Group by disaster type
        if 'disaster_type' in filtered_data.columns:
            type_field = 'disaster_type'
        elif 'disaster_group' in filtered_data.columns:
            type_field = 'disaster_group'
        else:
            type_field = None
        
        if type_field:
            type_counts = filtered_data.groupby(type_field).size().reset_index(name='count')
            result = type_counts.to_dict('records')
        else:
            result = []
    else:
        result = []
    
    return jsonify(result)

@app.route('/api/explorer/impact-by-type')
@login_required
def explorer_impact_by_type():
    disaster_data = fetch_disaster_data()
    
    # Get filter parameters
    start_year = request.args.get('start_year', type=int)
    end_year = request.args.get('end_year', type=int)
    disaster_types = request.args.getlist('disaster_types[]')
    regions = request.args.getlist('regions[]')
    
    # Apply filters
    if not disaster_data.empty:
        filtered_data = disaster_data
        
        if start_year and end_year and 'year' in filtered_data.columns:
            filtered_data = filtered_data[(filtered_data['year'] >= start_year) & (filtered_data['year'] <= end_year)]
        
        if disaster_types:
            if 'disaster_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_type'].isin(disaster_types)]
            elif 'disaster_group' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['disaster_group'].isin(disaster_types)]
        
        if regions and 'region' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['region'].isin(regions)]
        
        # Group by disaster type
        if 'disaster_type' in filtered_data.columns:
            type_field = 'disaster_type'
        elif 'disaster_group' in filtered_data.columns:
            type_field = 'disaster_group'
        else:
            type_field = None
        
        if type_field and 'deaths' in filtered_data.columns and 'total_damages' in filtered_data.columns:
            deaths_by_type = filtered_data.groupby(type_field)['deaths'].sum().reset_index()
            damages_by_type = filtered_data.groupby(type_field)['total_damages'].sum().reset_index()
            
            result = {
                'deaths': deaths_by_type.to_dict('records'),
                'damages': damages_by_type.to_dict('records')
            }
        else:
            result = {'deaths': [], 'damages': []}
    else:
        result = {'deaths': [], 'damages': []}
    
    return jsonify(result)

@app.route('/admin')
@login_required
@role_required('admin')
def admin():
    # Get user statistics
    user_count = users_collection.count_documents({})
    admin_count = users_collection.count_documents({'role': 'admin'})
    
    # Get system statistics
    alerts_count = alerts_collection.count_documents({})
    disasters_count = disasters_collection.count_documents({})
    
    # Get all users for management
    all_users = list(users_collection.find())
    
    return render_template('admin.html', 
                          user_count=user_count,
                          admin_count=admin_count,
                          alerts_count=alerts_count,
                          disasters_count=disasters_count,
                          users=all_users)

@app.route('/admin/make_admin/<user_id>')
@login_required
@role_required('admin')
def make_admin(user_id):
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {"role": "admin"}}
    )
    flash('User promoted to admin successfully!', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/remove_admin/<user_id>')
@login_required
@role_required('admin')
def remove_admin(user_id):
    # Don't allow removing the last admin
    if users_collection.count_documents({'role': 'admin'}) <= 1:
        flash('Cannot remove the last admin user!', 'danger')
        return redirect(url_for('admin'))
    
    users_collection.update_one(
        {"_id": user_id},
        {"$set": {"role": "user"}}
    )
    flash('Admin privileges removed successfully!', 'success')
    return redirect(url_for('admin'))

@app.route('/admin/delete_user/<user_id>')
@login_required
@role_required('admin')
def delete_user(user_id):
    # Don't allow deleting yourself
    if user_id == current_user.id:
        flash('You cannot delete your own account!', 'danger')
        return redirect(url_for('admin'))
    
    users_collection.delete_one({"_id": user_id})
    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin'))

@app.route('/refresh-data')
@login_required
@role_required('admin')
def refresh_data():
    from data_fetcher import fetch_emdat_data
    
    try:
        result = fetch_emdat_data()
        if result:
            flash('Data refreshed successfully!', 'success')
        else:
            flash('Failed to refresh data. Check logs for details.', 'danger')
    except Exception as e:
        flash(f'Error refreshing data: {str(e)}', 'danger')
    
    return redirect(url_for('admin'))


@app.route('/project_report')
def project_report():
    """Generate and display the project report"""
    return render_template('project_report.html')


@app.route('/admin/convert-data', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def convert_data():
    """Convert and clean data files"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'data_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        file = request.files['data_file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        if file:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.getcwd(), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(data_dir, 'emdat_data_original')
            file.save(file_path)
            
            try:
                # Process based on file type
                if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                    from data_converter import convert_excel_to_csv
                    output_file = convert_excel_to_csv(file_path, os.path.join(data_dir, 'emdat_data.csv'))
                    flash(f'Excel file converted and cleaned successfully!', 'success')
                elif file.filename.endswith('.csv'):
                    from data_converter import clean_csv_file
                    output_file = clean_csv_file(file_path, os.path.join(data_dir, 'emdat_data.csv'))
                    flash(f'CSV file cleaned successfully!', 'success')
                else:
                    flash(f'Unsupported file format: {file.filename}', 'danger')
                    return redirect(url_for('admin'))
                
                # Refresh data
                return redirect(url_for('refresh_data'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(url_for('admin'))
    
    return render_template('convert_data.html')


# Initialize resources and admin user on startup
init_resources()
init_admin_user()

if __name__ == '__main__':
    app.run(debug=True)
