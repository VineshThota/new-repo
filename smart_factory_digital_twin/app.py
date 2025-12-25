from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.utils
import json
import threading
import time
import random
from datetime import datetime, timedelta
import sqlite3
import os

app = Flask(__name__)

# Global variables for real-time data
current_sensor_data = {
    'temperature': 0,
    'vibration': 0,
    'pressure': 0,
    'timestamp': datetime.now()
}

maintenance_alerts = []
robot_status = {'position': [0, 0, 0], 'task': 'idle', 'battery': 100}

class IoTSensorSimulator:
    def __init__(self):
        self.running = False
        self.thread = None
        
    def start_simulation(self):
        self.running = True
        self.thread = threading.Thread(target=self._simulate_sensors)
        self.thread.daemon = True
        self.thread.start()
        
    def _simulate_sensors(self):
        while self.running:
            # Simulate realistic factory sensor data
            base_temp = 75 + random.gauss(0, 2)
            base_vibration = 0.5 + random.gauss(0, 0.1)
            base_pressure = 14.7 + random.gauss(0, 0.3)
            
            # Add anomalies occasionally
            if random.random() < 0.05:  # 5% chance of anomaly
                base_temp += random.uniform(10, 20)
                base_vibration += random.uniform(0.5, 1.0)
                
            current_sensor_data.update({
                'temperature': round(base_temp, 2),
                'vibration': round(base_vibration, 3),
                'pressure': round(base_pressure, 2),
                'timestamp': datetime.now()
            })
            
            # Store data in database
            self._store_sensor_data()
            time.sleep(2)  # Update every 2 seconds
            
    def _store_sensor_data(self):
        conn = sqlite3.connect('factory_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_data (timestamp, temperature, vibration, pressure)
            VALUES (?, ?, ?, ?)
        ''', (current_sensor_data['timestamp'], current_sensor_data['temperature'],
              current_sensor_data['vibration'], current_sensor_data['pressure']))
        
        conn.commit()
        conn.close()

class PredictiveMaintenanceAI:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_models(self):
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Normal operating conditions
        normal_temp = np.random.normal(75, 2, n_samples)
        normal_vibration = np.random.normal(0.5, 0.1, n_samples)
        normal_pressure = np.random.normal(14.7, 0.3, n_samples)
        
        # Anomalous conditions
        anomaly_temp = np.random.normal(95, 5, 100)
        anomaly_vibration = np.random.normal(1.2, 0.2, 100)
        anomaly_pressure = np.random.normal(12, 1, 100)
        
        # Combine data
        all_temp = np.concatenate([normal_temp, anomaly_temp])
        all_vibration = np.concatenate([normal_vibration, anomaly_vibration])
        all_pressure = np.concatenate([normal_pressure, anomaly_pressure])
        
        X = np.column_stack([all_temp, all_vibration, all_pressure])
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train failure predictor (time to failure in hours)
        failure_times = np.random.exponential(168, len(X))  # Average 1 week
        self.failure_predictor.fit(X_scaled, failure_times)
        
        self.is_trained = True
        
    def predict_anomaly(self, sensor_data):
        if not self.is_trained:
            self.train_models()
            
        data_point = np.array([[sensor_data['temperature'], 
                               sensor_data['vibration'], 
                               sensor_data['pressure']]])
        data_scaled = self.scaler.transform(data_point)
        
        anomaly_score = self.anomaly_detector.decision_function(data_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(data_scaled)[0] == -1
        
        time_to_failure = self.failure_predictor.predict(data_scaled)[0]
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'time_to_failure_hours': float(time_to_failure)
        }

class RobotController:
    def __init__(self):
        self.position = [0, 0, 0]
        self.task_queue = []
        self.current_task = None
        self.battery_level = 100
        
    def move_to_position(self, x, y, z):
        # Simulate robot movement
        self.position = [x, y, z]
        robot_status['position'] = self.position
        return f"Robot moved to position ({x}, {y}, {z})"
        
    def perform_maintenance_task(self, task_type, equipment_id):
        maintenance_tasks = {
            'inspection': 30,  # minutes
            'lubrication': 15,
            'calibration': 45,
            'replacement': 120
        }
        
        duration = maintenance_tasks.get(task_type, 30)
        self.current_task = f"{task_type} on equipment {equipment_id}"
        robot_status['task'] = self.current_task
        
        # Simulate task execution
        threading.Thread(target=self._execute_task, args=(duration,)).start()
        
        return f"Started {task_type} task on equipment {equipment_id}"
        
    def _execute_task(self, duration_minutes):
        # Simulate task execution time
        time.sleep(duration_minutes / 10)  # Accelerated for demo
        self.current_task = None
        robot_status['task'] = 'idle'
        robot_status['battery'] -= random.randint(5, 15)

# Initialize components
sensor_simulator = IoTSensorSimulator()
predictive_ai = PredictiveMaintenanceAI()
robot_controller = RobotController()

# Initialize database
def init_database():
    conn = sqlite3.connect('factory_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            temperature REAL,
            vibration REAL,
            pressure REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            equipment_id TEXT,
            task_type TEXT,
            robot_id TEXT,
            status TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/sensor_data')
def get_sensor_data():
    # Get AI predictions
    predictions = predictive_ai.predict_anomaly(current_sensor_data)
    
    # Check for alerts
    if predictions['is_anomaly'] or predictions['time_to_failure_hours'] < 24:
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': 'anomaly' if predictions['is_anomaly'] else 'maintenance_due',
            'message': f"Anomaly detected! Time to failure: {predictions['time_to_failure_hours']:.1f} hours",
            'severity': 'high' if predictions['is_anomaly'] else 'medium'
        }
        
        if len(maintenance_alerts) == 0 or maintenance_alerts[-1]['message'] != alert['message']:
            maintenance_alerts.append(alert)
            
            # Trigger robot maintenance if critical
            if predictions['time_to_failure_hours'] < 12:
                robot_controller.perform_maintenance_task('inspection', 'MACHINE_001')
    
    return jsonify({
        'sensor_data': current_sensor_data,
        'predictions': predictions,
        'alerts': maintenance_alerts[-5:],  # Last 5 alerts
        'robot_status': robot_status
    })

@app.route('/api/historical_data')
def get_historical_data():
    conn = sqlite3.connect('factory_data.db')
    df = pd.read_sql_query(
        "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100", 
        conn
    )
    conn.close()
    
    if df.empty:
        return jsonify({'data': []})
    
    # Create plotly charts
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['temperature'],
        mode='lines',
        name='Temperature (°F)',
        line=dict(color='red')
    ))
    
    fig_vibration = go.Figure()
    fig_vibration.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['vibration'],
        mode='lines',
        name='Vibration (g)',
        line=dict(color='blue')
    ))
    
    return jsonify({
        'temperature_chart': json.dumps(fig_temp, cls=plotly.utils.PlotlyJSONEncoder),
        'vibration_chart': json.dumps(fig_vibration, cls=plotly.utils.PlotlyJSONEncoder)
    })

@app.route('/api/robot_control', methods=['POST'])
def control_robot():
    data = request.json
    action = data.get('action')
    
    if action == 'move':
        x, y, z = data.get('x', 0), data.get('y', 0), data.get('z', 0)
        result = robot_controller.move_to_position(x, y, z)
    elif action == 'maintenance':
        task_type = data.get('task_type', 'inspection')
        equipment_id = data.get('equipment_id', 'MACHINE_001')
        result = robot_controller.perform_maintenance_task(task_type, equipment_id)
    else:
        result = "Unknown action"
    
    return jsonify({'result': result, 'robot_status': robot_status})

@app.route('/api/digital_twin')
def get_digital_twin_data():
    # Simulate digital twin data
    equipment_status = {
        'MACHINE_001': {
            'status': 'operational',
            'efficiency': 87.5,
            'temperature': current_sensor_data['temperature'],
            'vibration': current_sensor_data['vibration'],
            'last_maintenance': '2024-12-20',
            'next_maintenance': '2025-01-15'
        },
        'MACHINE_002': {
            'status': 'maintenance_required',
            'efficiency': 65.2,
            'temperature': current_sensor_data['temperature'] + 5,
            'vibration': current_sensor_data['vibration'] + 0.2,
            'last_maintenance': '2024-11-15',
            'next_maintenance': '2024-12-26'
        }
    }
    
    return jsonify(equipment_status)

if __name__ == '__main__':
    init_database()
    sensor_simulator.start_simulation()
    app.run(debug=True, host='0.0.0.0', port=5000)