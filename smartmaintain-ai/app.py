from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import datetime
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Dict

app = Flask(__name__)

# IoT Sensor Data Simulation
@dataclass
class SensorReading:
    timestamp: str
    equipment_id: str
    temperature: float
    vibration: float
    humidity: float
    pressure: float
    rotation_speed: float

class IoTSensorSimulator:
    def __init__(self):
        self.equipment_list = ['Motor_A1', 'Pump_B2', 'Conveyor_C3', 'Press_D4', 'Drill_E5']
        self.sensor_data = []
        self.running = True
        
    def generate_sensor_reading(self, equipment_id: str) -> SensorReading:
        # Simulate normal vs anomalous readings
        is_anomaly = random.random() < 0.15  # 15% chance of anomaly
        
        if is_anomaly:
            # Anomalous readings indicating potential failure
            temperature = random.uniform(85, 120)  # High temperature
            vibration = random.uniform(8, 15)      # High vibration
            humidity = random.uniform(75, 95)      # High humidity
            pressure = random.uniform(45, 60)      # Low pressure
            rotation_speed = random.uniform(800, 1100)  # Irregular speed
        else:
            # Normal readings
            temperature = random.uniform(65, 80)
            vibration = random.uniform(2, 6)
            humidity = random.uniform(40, 60)
            pressure = random.uniform(80, 100)
            rotation_speed = random.uniform(1450, 1550)
            
        return SensorReading(
            timestamp=datetime.datetime.now().isoformat(),
            equipment_id=equipment_id,
            temperature=temperature,
            vibration=vibration,
            humidity=humidity,
            pressure=pressure,
            rotation_speed=rotation_speed
        )
    
    def collect_data(self):
        while self.running:
            for equipment in self.equipment_list:
                reading = self.generate_sensor_reading(equipment)
                self.sensor_data.append(reading)
                
                # Keep only last 1000 readings per equipment
                if len(self.sensor_data) > 5000:
                    self.sensor_data = self.sensor_data[-5000:]
                    
            time.sleep(2)  # Collect data every 2 seconds

# AI Predictive Maintenance Model
class PredictiveMaintenanceAI:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = ['temperature', 'vibration', 'humidity', 'pressure', 'rotation_speed']
        
    def prepare_training_data(self):
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 10000
        
        # Normal operation data (70%)
        normal_data = {
            'temperature': np.random.normal(72, 5, int(n_samples * 0.7)),
            'vibration': np.random.normal(4, 1, int(n_samples * 0.7)),
            'humidity': np.random.normal(50, 8, int(n_samples * 0.7)),
            'pressure': np.random.normal(90, 5, int(n_samples * 0.7)),
            'rotation_speed': np.random.normal(1500, 25, int(n_samples * 0.7)),
            'failure_risk': np.zeros(int(n_samples * 0.7))  # 0 = Normal
        }
        
        # High risk data (20%)
        high_risk_data = {
            'temperature': np.random.normal(95, 10, int(n_samples * 0.2)),
            'vibration': np.random.normal(10, 2, int(n_samples * 0.2)),
            'humidity': np.random.normal(80, 10, int(n_samples * 0.2)),
            'pressure': np.random.normal(60, 10, int(n_samples * 0.2)),
            'rotation_speed': np.random.normal(1200, 100, int(n_samples * 0.2)),
            'failure_risk': np.ones(int(n_samples * 0.2))  # 1 = High Risk
        }
        
        # Critical failure data (10%)
        critical_data = {
            'temperature': np.random.normal(110, 15, int(n_samples * 0.1)),
            'vibration': np.random.normal(15, 3, int(n_samples * 0.1)),
            'humidity': np.random.normal(90, 5, int(n_samples * 0.1)),
            'pressure': np.random.normal(45, 8, int(n_samples * 0.1)),
            'rotation_speed': np.random.normal(900, 150, int(n_samples * 0.1)),
            'failure_risk': np.full(int(n_samples * 0.1), 2)  # 2 = Critical
        }
        
        # Combine all data
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], high_risk_data[key], critical_data[key]])
            
        return pd.DataFrame(all_data)
    
    def train_model(self):
        # Prepare and train the model
        df = self.prepare_training_data()
        X = df[self.feature_columns]
        y = df['failure_risk']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print("AI Model trained successfully!")
        
    def predict_failure_risk(self, sensor_reading: SensorReading) -> Dict:
        if not self.is_trained:
            self.train_model()
            
        # Prepare input data
        input_data = np.array([[
            sensor_reading.temperature,
            sensor_reading.vibration,
            sensor_reading.humidity,
            sensor_reading.pressure,
            sensor_reading.rotation_speed
        ]])
        
        # Scale input
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]
        
        risk_levels = ['Normal', 'High Risk', 'Critical']
        risk_level = risk_levels[int(prediction)]
        
        return {
            'risk_level': risk_level,
            'risk_score': int(prediction),
            'confidence': float(max(probability)),
            'probabilities': {
                'normal': float(probability[0]),
                'high_risk': float(probability[1]) if len(probability) > 1 else 0.0,
                'critical': float(probability[2]) if len(probability) > 2 else 0.0
            }
        }

# Physical AI - Automated Response System
class PhysicalAIController:
    def __init__(self):
        self.maintenance_alerts = []
        self.automated_actions = []
        
    def process_prediction(self, equipment_id: str, sensor_reading: SensorReading, prediction: Dict):
        timestamp = datetime.datetime.now().isoformat()
        
        if prediction['risk_score'] >= 2:  # Critical
            action = {
                'timestamp': timestamp,
                'equipment_id': equipment_id,
                'action': 'EMERGENCY_SHUTDOWN',
                'reason': f"Critical failure risk detected (confidence: {prediction['confidence']:.2%})",
                'sensor_data': {
                    'temperature': sensor_reading.temperature,
                    'vibration': sensor_reading.vibration,
                    'humidity': sensor_reading.humidity,
                    'pressure': sensor_reading.pressure,
                    'rotation_speed': sensor_reading.rotation_speed
                }
            }
            self.automated_actions.append(action)
            
            alert = {
                'timestamp': timestamp,
                'equipment_id': equipment_id,
                'priority': 'CRITICAL',
                'message': f"EMERGENCY: {equipment_id} requires immediate shutdown and maintenance",
                'predicted_failure_time': 'Imminent (< 1 hour)',
                'recommended_action': 'Immediate shutdown and inspection'
            }
            self.maintenance_alerts.append(alert)
            
        elif prediction['risk_score'] >= 1:  # High Risk
            alert = {
                'timestamp': timestamp,
                'equipment_id': equipment_id,
                'priority': 'HIGH',
                'message': f"WARNING: {equipment_id} showing signs of potential failure",
                'predicted_failure_time': '2-24 hours',
                'recommended_action': 'Schedule maintenance within 24 hours'
            }
            self.maintenance_alerts.append(alert)
            
        # Keep only recent alerts (last 100)
        if len(self.maintenance_alerts) > 100:
            self.maintenance_alerts = self.maintenance_alerts[-100:]
        if len(self.automated_actions) > 50:
            self.automated_actions = self.automated_actions[-50:]

# Initialize components
sensor_simulator = IoTSensorSimulator()
predictive_ai = PredictiveMaintenanceAI()
physical_controller = PhysicalAIController()

# Start sensor data collection in background
sensor_thread = threading.Thread(target=sensor_simulator.collect_data, daemon=True)
sensor_thread.start()

# Flask Routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/sensor-data')
def get_sensor_data():
    # Get latest readings for each equipment
    latest_readings = {}
    for reading in reversed(sensor_simulator.sensor_data[-50:]):
        if reading.equipment_id not in latest_readings:
            latest_readings[reading.equipment_id] = {
                'timestamp': reading.timestamp,
                'temperature': reading.temperature,
                'vibration': reading.vibration,
                'humidity': reading.humidity,
                'pressure': reading.pressure,
                'rotation_speed': reading.rotation_speed
            }
    
    return jsonify(latest_readings)

@app.route('/api/predictions')
def get_predictions():
    predictions = {}
    
    # Get latest reading for each equipment and predict
    for equipment_id in sensor_simulator.equipment_list:
        equipment_readings = [r for r in sensor_simulator.sensor_data if r.equipment_id == equipment_id]
        if equipment_readings:
            latest_reading = equipment_readings[-1]
            prediction = predictive_ai.predict_failure_risk(latest_reading)
            predictions[equipment_id] = prediction
            
            # Process with Physical AI controller
            physical_controller.process_prediction(equipment_id, latest_reading, prediction)
    
    return jsonify(predictions)

@app.route('/api/alerts')
def get_alerts():
    return jsonify({
        'maintenance_alerts': physical_controller.maintenance_alerts[-20:],
        'automated_actions': physical_controller.automated_actions[-10:]
    })

@app.route('/api/equipment-status')
def get_equipment_status():
    status = {}
    
    for equipment_id in sensor_simulator.equipment_list:
        equipment_readings = [r for r in sensor_simulator.sensor_data if r.equipment_id == equipment_id]
        if equipment_readings:
            latest_reading = equipment_readings[-1]
            prediction = predictive_ai.predict_failure_risk(latest_reading)
            
            # Determine operational status
            if prediction['risk_score'] >= 2:
                operational_status = 'SHUTDOWN'
            elif prediction['risk_score'] >= 1:
                operational_status = 'MAINTENANCE_REQUIRED'
            else:
                operational_status = 'OPERATIONAL'
                
            status[equipment_id] = {
                'status': operational_status,
                'risk_level': prediction['risk_level'],
                'last_updated': latest_reading.timestamp,
                'uptime_hours': random.randint(100, 8760),  # Simulated uptime
                'efficiency': random.randint(85, 99) if operational_status == 'OPERATIONAL' else random.randint(60, 84)
            }
    
    return jsonify(status)

if __name__ == '__main__':
    print("Starting SmartMaintain AI - Predictive Maintenance System")
    print("Combining IoT Sensors + AI Algorithms + Physical AI Control")
    print("Training AI model...")
    predictive_ai.train_model()
    print("System ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)