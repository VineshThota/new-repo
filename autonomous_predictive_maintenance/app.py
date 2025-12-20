from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import sqlite3
import os

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'autonomous_maintenance_2025'
DATABASE = 'maintenance_system.db'

@dataclass
class IoTSensor:
    """IoT Sensor data structure"""
    sensor_id: str
    equipment_id: str
    sensor_type: str  # temperature, vibration, pressure, current
    location: str
    normal_range: tuple
    critical_threshold: float

@dataclass
class EquipmentStatus:
    """Equipment status tracking"""
    equipment_id: str
    name: str
    status: str  # operational, warning, critical, maintenance
    last_maintenance: datetime
    predicted_failure: datetime
    confidence_score: float

@dataclass
class MaintenanceRobot:
    """Autonomous maintenance robot"""
    robot_id: str
    name: str
    capabilities: List[str]
    current_task: str
    location: str
    battery_level: float
    status: str  # idle, working, charging, error

class AutonomousMaintenanceSystem:
    """Main system class integrating IoT, AI, and Physical AI"""
    
    def __init__(self):
        self.sensors = self._initialize_sensors()
        self.equipment = self._initialize_equipment()
        self.robots = self._initialize_robots()
        self.ml_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.sensor_data = []
        self.maintenance_tasks = []
        self.digital_twin_data = {}
        self._setup_database()
        self._train_models()
        self._start_sensor_simulation()
    
    def _initialize_sensors(self) -> List[IoTSensor]:
        """Initialize IoT sensors for equipment monitoring"""
        sensors = [
            IoTSensor('TEMP_001', 'MOTOR_A1', 'temperature', 'Motor Bearing', (60, 80), 95),
            IoTSensor('VIB_001', 'MOTOR_A1', 'vibration', 'Motor Housing', (0.1, 2.0), 5.0),
            IoTSensor('CURR_001', 'MOTOR_A1', 'current', 'Motor Input', (10, 15), 20),
            IoTSensor('TEMP_002', 'PUMP_B1', 'temperature', 'Pump Casing', (40, 60), 75),
            IoTSensor('PRESS_001', 'PUMP_B1', 'pressure', 'Outlet', (2.0, 3.5), 4.5),
            IoTSensor('VIB_002', 'CONVEYOR_C1', 'vibration', 'Belt Drive', (0.05, 1.0), 3.0),
            IoTSensor('TEMP_003', 'CONVEYOR_C1', 'temperature', 'Drive Motor', (50, 70), 85),
        ]
        return sensors
    
    def _initialize_equipment(self) -> List[EquipmentStatus]:
        """Initialize equipment status tracking"""
        equipment = [
            EquipmentStatus('MOTOR_A1', 'Production Motor A1', 'operational', 
                          datetime.now() - timedelta(days=45), 
                          datetime.now() + timedelta(days=15), 0.85),
            EquipmentStatus('PUMP_B1', 'Hydraulic Pump B1', 'warning', 
                          datetime.now() - timedelta(days=60), 
                          datetime.now() + timedelta(days=8), 0.92),
            EquipmentStatus('CONVEYOR_C1', 'Conveyor Belt C1', 'operational', 
                          datetime.now() - timedelta(days=30), 
                          datetime.now() + timedelta(days=25), 0.78),
        ]
        return equipment
    
    def _initialize_robots(self) -> List[MaintenanceRobot]:
        """Initialize autonomous maintenance robots"""
        robots = [
            MaintenanceRobot('ROBOT_001', 'Maintenance Bot Alpha', 
                           ['lubrication', 'inspection', 'cleaning'], 
                           'idle', 'Station_A', 95.0, 'idle'),
            MaintenanceRobot('ROBOT_002', 'Diagnostic Bot Beta', 
                           ['thermal_scan', 'vibration_analysis', 'visual_inspection'], 
                           'idle', 'Station_B', 88.0, 'idle'),
            MaintenanceRobot('ROBOT_003', 'Repair Bot Gamma', 
                           ['part_replacement', 'calibration', 'testing'], 
                           'idle', 'Station_C', 92.0, 'idle'),
        ]
        return robots
    
    def _setup_database(self):
        """Setup SQLite database for data storage"""
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Sensor data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                sensor_id TEXT,
                equipment_id TEXT,
                sensor_type TEXT,
                value REAL,
                status TEXT
            )
        ''')
        
        # Maintenance tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                equipment_id TEXT,
                robot_id TEXT,
                task_type TEXT,
                priority TEXT,
                status TEXT,
                estimated_duration INTEGER
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                equipment_id TEXT,
                predicted_failure_date DATETIME,
                confidence_score REAL,
                risk_factors TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _train_models(self):
        """Train ML models for predictive maintenance"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: temperature, vibration, current, pressure, operating_hours
        X = np.random.rand(n_samples, 5)
        X[:, 0] = X[:, 0] * 40 + 60  # Temperature (60-100)
        X[:, 1] = X[:, 1] * 5  # Vibration (0-5)
        X[:, 2] = X[:, 2] * 10 + 10  # Current (10-20)
        X[:, 3] = X[:, 3] * 3 + 1  # Pressure (1-4)
        X[:, 4] = X[:, 4] * 8760  # Operating hours (0-8760)
        
        # Create failure labels based on thresholds
        y = ((X[:, 0] > 90) | (X[:, 1] > 3.5) | (X[:, 2] > 18) | (X[:, 3] > 4)).astype(int)
        
        # Train models
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_scaled, y)
        
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(X_scaled)
    
    def _start_sensor_simulation(self):
        """Start IoT sensor data simulation in background thread"""
        def simulate_sensors():
            while True:
                for sensor in self.sensors:
                    # Simulate realistic sensor readings
                    base_value = np.mean(sensor.normal_range)
                    noise = np.random.normal(0, 0.1) * base_value
                    
                    # Add occasional anomalies
                    if random.random() < 0.05:  # 5% chance of anomaly
                        value = sensor.critical_threshold * 0.9
                    else:
                        value = base_value + noise
                    
                    # Determine status
                    if value > sensor.critical_threshold:
                        status = 'critical'
                    elif value > sensor.normal_range[1]:
                        status = 'warning'
                    else:
                        status = 'normal'
                    
                    # Store reading
                    reading = {
                        'timestamp': datetime.now(),
                        'sensor_id': sensor.sensor_id,
                        'equipment_id': sensor.equipment_id,
                        'sensor_type': sensor.sensor_type,
                        'value': round(value, 2),
                        'status': status
                    }
                    
                    self.sensor_data.append(reading)
                    
                    # Keep only last 1000 readings
                    if len(self.sensor_data) > 1000:
                        self.sensor_data = self.sensor_data[-1000:]
                    
                    # Store in database
                    self._store_sensor_reading(reading)
                    
                    # Update digital twin
                    self._update_digital_twin(sensor.equipment_id, reading)
                    
                    # Check for maintenance needs
                    if status in ['warning', 'critical']:
                        self._schedule_maintenance(sensor.equipment_id, status)
                
                time.sleep(2)  # Update every 2 seconds
        
        thread = threading.Thread(target=simulate_sensors, daemon=True)
        thread.start()
    
    def _store_sensor_reading(self, reading):
        """Store sensor reading in database"""
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_readings (timestamp, sensor_id, equipment_id, sensor_type, value, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (reading['timestamp'], reading['sensor_id'], reading['equipment_id'], 
              reading['sensor_type'], reading['value'], reading['status']))
        conn.commit()
        conn.close()
    
    def _update_digital_twin(self, equipment_id, reading):
        """Update digital twin with latest sensor data"""
        if equipment_id not in self.digital_twin_data:
            self.digital_twin_data[equipment_id] = {
                'sensors': {},
                'health_score': 100,
                'last_updated': datetime.now(),
                'operating_hours': 0
            }
        
        twin = self.digital_twin_data[equipment_id]
        twin['sensors'][reading['sensor_type']] = {
            'value': reading['value'],
            'status': reading['status'],
            'timestamp': reading['timestamp']
        }
        twin['last_updated'] = datetime.now()
        
        # Calculate health score based on sensor statuses
        sensor_scores = []
        for sensor_type, data in twin['sensors'].items():
            if data['status'] == 'normal':
                sensor_scores.append(100)
            elif data['status'] == 'warning':
                sensor_scores.append(70)
            else:  # critical
                sensor_scores.append(30)
        
        twin['health_score'] = np.mean(sensor_scores) if sensor_scores else 100
    
    def _schedule_maintenance(self, equipment_id, priority):
        """Schedule maintenance task for equipment"""
        # Find available robot
        available_robot = None
        for robot in self.robots:
            if robot.status == 'idle' and robot.battery_level > 20:
                available_robot = robot
                break
        
        if available_robot:
            task_type = 'inspection' if priority == 'warning' else 'emergency_repair'
            duration = 30 if priority == 'warning' else 60  # minutes
            
            task = {
                'timestamp': datetime.now(),
                'equipment_id': equipment_id,
                'robot_id': available_robot.robot_id,
                'task_type': task_type,
                'priority': priority,
                'status': 'scheduled',
                'estimated_duration': duration
            }
            
            self.maintenance_tasks.append(task)
            available_robot.status = 'working'
            available_robot.current_task = f"{task_type} on {equipment_id}"
            
            # Store in database
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO maintenance_tasks (timestamp, equipment_id, robot_id, task_type, priority, status, estimated_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (task['timestamp'], task['equipment_id'], task['robot_id'], 
                  task['task_type'], task['priority'], task['status'], task['estimated_duration']))
            conn.commit()
            conn.close()
    
    def predict_failure(self, equipment_id):
        """Predict equipment failure using ML models"""
        # Get recent sensor data for equipment
        recent_data = [r for r in self.sensor_data[-50:] if r['equipment_id'] == equipment_id]
        
        if len(recent_data) < 5:
            return None
        
        # Prepare features
        sensor_values = {}
        for reading in recent_data:
            sensor_values[reading['sensor_type']] = reading['value']
        
        # Create feature vector
        features = [
            sensor_values.get('temperature', 70),
            sensor_values.get('vibration', 1.0),
            sensor_values.get('current', 12),
            sensor_values.get('pressure', 2.5),
            8000  # Operating hours (simulated)
        ]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict failure probability
        failure_prob = self.ml_model.predict_proba(features_scaled)[0][1]
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        
        # Calculate days until predicted failure
        if failure_prob > 0.7:
            days_until_failure = max(1, int(10 * (1 - failure_prob)))
        else:
            days_until_failure = max(7, int(30 * (1 - failure_prob)))
        
        predicted_date = datetime.now() + timedelta(days=days_until_failure)
        
        return {
            'failure_probability': failure_prob,
            'anomaly_score': anomaly_score,
            'predicted_failure_date': predicted_date,
            'days_until_failure': days_until_failure,
            'confidence': min(0.95, failure_prob + 0.1)
        }

# Initialize the system
maintenance_system = AutonomousMaintenanceSystem()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get latest sensor readings"""
    recent_data = maintenance_system.sensor_data[-20:]
    return jsonify(recent_data)

@app.route('/api/equipment_status')
def get_equipment_status():
    """Get equipment status with predictions"""
    equipment_data = []
    
    for equipment in maintenance_system.equipment:
        prediction = maintenance_system.predict_failure(equipment.equipment_id)
        
        equipment_info = {
            'equipment_id': equipment.equipment_id,
            'name': equipment.name,
            'status': equipment.status,
            'last_maintenance': equipment.last_maintenance.isoformat(),
            'health_score': maintenance_system.digital_twin_data.get(equipment.equipment_id, {}).get('health_score', 100)
        }
        
        if prediction:
            equipment_info.update({
                'predicted_failure': prediction['predicted_failure_date'].isoformat(),
                'failure_probability': prediction['failure_probability'],
                'days_until_failure': prediction['days_until_failure'],
                'confidence': prediction['confidence']
            })
        
        equipment_data.append(equipment_info)
    
    return jsonify(equipment_data)

@app.route('/api/robots')
def get_robots():
    """Get robot status"""
    robot_data = []
    for robot in maintenance_system.robots:
        robot_data.append({
            'robot_id': robot.robot_id,
            'name': robot.name,
            'status': robot.status,
            'current_task': robot.current_task,
            'location': robot.location,
            'battery_level': robot.battery_level,
            'capabilities': robot.capabilities
        })
    return jsonify(robot_data)

@app.route('/api/maintenance_tasks')
def get_maintenance_tasks():
    """Get recent maintenance tasks"""
    recent_tasks = maintenance_system.maintenance_tasks[-10:]
    tasks_data = []
    
    for task in recent_tasks:
        tasks_data.append({
            'timestamp': task['timestamp'].isoformat(),
            'equipment_id': task['equipment_id'],
            'robot_id': task['robot_id'],
            'task_type': task['task_type'],
            'priority': task['priority'],
            'status': task['status'],
            'estimated_duration': task['estimated_duration']
        })
    
    return jsonify(tasks_data)

@app.route('/api/digital_twin/<equipment_id>')
def get_digital_twin(equipment_id):
    """Get digital twin data for specific equipment"""
    twin_data = maintenance_system.digital_twin_data.get(equipment_id, {})
    
    if twin_data:
        # Convert datetime objects to ISO format
        twin_data_copy = twin_data.copy()
        twin_data_copy['last_updated'] = twin_data['last_updated'].isoformat()
        
        for sensor_type, sensor_data in twin_data_copy.get('sensors', {}).items():
            sensor_data['timestamp'] = sensor_data['timestamp'].isoformat()
        
        return jsonify(twin_data_copy)
    else:
        return jsonify({'error': 'Equipment not found'}), 404

@app.route('/api/analytics')
def get_analytics():
    """Get system analytics and insights"""
    # Calculate system metrics
    total_sensors = len(maintenance_system.sensors)
    active_alerts = len([r for r in maintenance_system.sensor_data[-50:] if r['status'] in ['warning', 'critical']])
    pending_tasks = len([t for t in maintenance_system.maintenance_tasks if t['status'] == 'scheduled'])
    
    # Equipment health distribution
    health_scores = []
    for equipment_id in maintenance_system.digital_twin_data:
        health_scores.append(maintenance_system.digital_twin_data[equipment_id]['health_score'])
    
    avg_health = np.mean(health_scores) if health_scores else 100
    
    analytics = {
        'total_sensors': total_sensors,
        'active_alerts': active_alerts,
        'pending_tasks': pending_tasks,
        'average_health_score': round(avg_health, 1),
        'system_uptime': '99.8%',  # Simulated
        'maintenance_efficiency': '94.2%',  # Simulated
        'cost_savings': '$125,000'  # Simulated annual savings
    }
    
    return jsonify(analytics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)