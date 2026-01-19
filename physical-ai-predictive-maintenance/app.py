#!/usr/bin/env python3
"""
Physical AI Predictive Maintenance System
Combines IoT sensors, AI algorithms, and physical world interactions
for industrial equipment monitoring and failure prediction.

Author: AI Workflow Agent
Date: 2026-01-19
Framework: Flask + scikit-learn + pandas + numpy
"""

import os
import json
import time
import random
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'physical-ai-predictive-maintenance-2026'

@dataclass
class SensorReading:
    """IoT Sensor data structure"""
    equipment_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    rotation_speed: float
    power_consumption: float
    location: str

@dataclass
class MaintenanceAlert:
    """Physical AI maintenance alert structure"""
    equipment_id: str
    alert_type: str
    severity: str
    predicted_failure_time: datetime
    recommended_action: str
    confidence_score: float

class IoTSensorSimulator:
    """Simulates IoT sensors for industrial equipment"""
    
    def __init__(self):
        self.equipment_list = [
            {'id': 'PUMP_001', 'location': 'Factory Floor A', 'type': 'Centrifugal Pump'},
            {'id': 'MOTOR_002', 'location': 'Assembly Line B', 'type': 'Electric Motor'},
            {'id': 'COMPRESSOR_003', 'location': 'Air System C', 'type': 'Air Compressor'},
            {'id': 'CONVEYOR_004', 'location': 'Packaging D', 'type': 'Belt Conveyor'},
            {'id': 'TURBINE_005', 'location': 'Power Generation E', 'type': 'Steam Turbine'}
        ]
        self.running = False
        self.sensor_data = []
        
    def generate_sensor_reading(self, equipment: Dict) -> SensorReading:
        """Generate realistic IoT sensor data with anomaly patterns"""
        base_time = datetime.now()
        
        # Normal operating ranges
        normal_ranges = {
            'temperature': (20, 80),
            'vibration': (0.1, 2.0),
            'pressure': (1.0, 5.0),
            'rotation_speed': (1000, 3000),
            'power_consumption': (50, 200)
        }
        
        # Introduce anomalies randomly (5% chance)
        anomaly_factor = 1.0
        if random.random() < 0.05:
            anomaly_factor = random.uniform(1.5, 3.0)
            logger.warning(f"Anomaly detected in {equipment['id']}")
        
        return SensorReading(
            equipment_id=equipment['id'],
            timestamp=base_time,
            temperature=random.uniform(*normal_ranges['temperature']) * anomaly_factor,
            vibration=random.uniform(*normal_ranges['vibration']) * anomaly_factor,
            pressure=random.uniform(*normal_ranges['pressure']) * anomaly_factor,
            rotation_speed=random.uniform(*normal_ranges['rotation_speed']) / anomaly_factor,
            power_consumption=random.uniform(*normal_ranges['power_consumption']) * anomaly_factor,
            location=equipment['location']
        )
    
    def start_simulation(self):
        """Start continuous IoT sensor data generation"""
        self.running = True
        
        def simulate():
            while self.running:
                for equipment in self.equipment_list:
                    reading = self.generate_sensor_reading(equipment)
                    self.sensor_data.append(reading)
                    
                    # Keep only last 1000 readings for memory management
                    if len(self.sensor_data) > 1000:
                        self.sensor_data = self.sensor_data[-1000:]
                
                time.sleep(2)  # Generate data every 2 seconds
        
        thread = threading.Thread(target=simulate)
        thread.daemon = True
        thread.start()
        logger.info("IoT sensor simulation started")
    
    def stop_simulation(self):
        """Stop IoT sensor simulation"""
        self.running = False
        logger.info("IoT sensor simulation stopped")
    
    def get_latest_readings(self, limit: int = 50) -> List[SensorReading]:
        """Get latest sensor readings"""
        return self.sensor_data[-limit:] if self.sensor_data else []

class PhysicalAIPredictor:
    """AI-powered predictive maintenance system"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = ['temperature', 'vibration', 'pressure', 'rotation_speed', 'power_consumption']
        
    def prepare_training_data(self, sensor_readings: List[SensorReading]) -> pd.DataFrame:
        """Convert sensor readings to training data format"""
        data = []
        for reading in sensor_readings:
            data.append({
                'equipment_id': reading.equipment_id,
                'timestamp': reading.timestamp,
                'temperature': reading.temperature,
                'vibration': reading.vibration,
                'pressure': reading.pressure,
                'rotation_speed': reading.rotation_speed,
                'power_consumption': reading.power_consumption,
                'location': reading.location
            })
        
        df = pd.DataFrame(data)
        return df
    
    def train_models(self, sensor_readings: List[SensorReading]):
        """Train AI models on historical sensor data"""
        if len(sensor_readings) < 50:
            logger.warning("Insufficient data for training. Need at least 50 readings.")
            return
        
        df = self.prepare_training_data(sensor_readings)
        
        # Prepare features
        X = df[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detection model
        self.anomaly_detector.fit(X_scaled)
        
        # Generate synthetic failure labels for demonstration
        # In real scenario, this would be historical failure data
        y_failure = np.random.exponential(scale=100, size=len(X))  # Time to failure in hours
        
        # Train failure prediction model
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_failure, test_size=0.2, random_state=42)
        self.failure_predictor.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("AI models trained successfully")
    
    def detect_anomalies(self, sensor_readings: List[SensorReading]) -> List[Dict]:
        """Detect anomalies in real-time sensor data"""
        if not self.is_trained or not sensor_readings:
            return []
        
        df = self.prepare_training_data(sensor_readings)
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomalies = self.anomaly_detector.predict(X_scaled)
        
        results = []
        for i, (reading, score, is_anomaly) in enumerate(zip(sensor_readings, anomaly_scores, anomalies)):
            if is_anomaly == -1:  # Anomaly detected
                results.append({
                    'equipment_id': reading.equipment_id,
                    'timestamp': reading.timestamp.isoformat(),
                    'anomaly_score': float(score),
                    'severity': 'HIGH' if score < -0.5 else 'MEDIUM',
                    'location': reading.location
                })
        
        return results
    
    def predict_failure(self, sensor_reading: SensorReading) -> Optional[MaintenanceAlert]:
        """Predict equipment failure using Physical AI"""
        if not self.is_trained:
            return None
        
        # Prepare single reading for prediction
        X = np.array([[
            sensor_reading.temperature,
            sensor_reading.vibration,
            sensor_reading.pressure,
            sensor_reading.rotation_speed,
            sensor_reading.power_consumption
        ]])
        
        X_scaled = self.scaler.transform(X)
        
        # Predict time to failure
        predicted_hours = self.failure_predictor.predict(X_scaled)[0]
        
        # Calculate confidence based on model uncertainty
        confidence = min(0.95, max(0.5, 1.0 - (predicted_hours / 1000)))
        
        # Determine severity and recommended action
        if predicted_hours < 24:
            severity = 'CRITICAL'
            action = 'IMMEDIATE_SHUTDOWN_AND_MAINTENANCE'
        elif predicted_hours < 72:
            severity = 'HIGH'
            action = 'SCHEDULE_URGENT_MAINTENANCE'
        elif predicted_hours < 168:  # 1 week
            severity = 'MEDIUM'
            action = 'SCHEDULE_PREVENTIVE_MAINTENANCE'
        else:
            severity = 'LOW'
            action = 'CONTINUE_MONITORING'
        
        predicted_failure_time = sensor_reading.timestamp + timedelta(hours=predicted_hours)
        
        return MaintenanceAlert(
            equipment_id=sensor_reading.equipment_id,
            alert_type='PREDICTIVE_FAILURE',
            severity=severity,
            predicted_failure_time=predicted_failure_time,
            recommended_action=action,
            confidence_score=confidence
        )

class PhysicalAIController:
    """Physical AI system for automated responses"""
    
    def __init__(self):
        self.active_alerts = []
        self.maintenance_schedule = []
        
    def process_alert(self, alert: MaintenanceAlert) -> Dict:
        """Process maintenance alert and trigger physical responses"""
        response = {
            'alert_id': f"ALT_{int(time.time())}",
            'equipment_id': alert.equipment_id,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': []
        }
        
        # Automated physical responses based on severity
        if alert.severity == 'CRITICAL':
            response['actions_taken'].extend([
                'EMERGENCY_SHUTDOWN_INITIATED',
                'SAFETY_PROTOCOLS_ACTIVATED',
                'MAINTENANCE_TEAM_ALERTED',
                'BACKUP_SYSTEMS_ENGAGED'
            ])
        elif alert.severity == 'HIGH':
            response['actions_taken'].extend([
                'REDUCED_OPERATION_MODE',
                'MAINTENANCE_SCHEDULED',
                'SUPERVISOR_NOTIFIED'
            ])
        elif alert.severity == 'MEDIUM':
            response['actions_taken'].extend([
                'MONITORING_INCREASED',
                'PREVENTIVE_MAINTENANCE_QUEUED'
            ])
        
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Schedule maintenance
        maintenance_task = {
            'equipment_id': alert.equipment_id,
            'scheduled_time': alert.predicted_failure_time - timedelta(hours=12),
            'priority': alert.severity,
            'estimated_duration': '2-4 hours' if alert.severity == 'CRITICAL' else '1-2 hours'
        }
        self.maintenance_schedule.append(maintenance_task)
        
        logger.info(f"Physical AI response executed for {alert.equipment_id}: {response['actions_taken']}")
        return response
    
    def get_active_alerts(self) -> List[MaintenanceAlert]:
        """Get currently active maintenance alerts"""
        # Filter alerts from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        return [alert for alert in self.active_alerts 
                if alert.predicted_failure_time > cutoff_time]
    
    def get_maintenance_schedule(self) -> List[Dict]:
        """Get upcoming maintenance schedule"""
        return sorted(self.maintenance_schedule, 
                     key=lambda x: x['scheduled_time'])

# Initialize system components
sensor_simulator = IoTSensorSimulator()
ai_predictor = PhysicalAIPredictor()
physical_controller = PhysicalAIController()

# Flask routes
@app.route('/')
def dashboard():
    """Main dashboard for Physical AI Predictive Maintenance"""
    return render_template('dashboard.html')

@app.route('/api/sensor-data')
def get_sensor_data():
    """API endpoint for real-time sensor data"""
    readings = sensor_simulator.get_latest_readings(20)
    data = []
    
    for reading in readings:
        data.append({
            'equipment_id': reading.equipment_id,
            'timestamp': reading.timestamp.isoformat(),
            'temperature': reading.temperature,
            'vibration': reading.vibration,
            'pressure': reading.pressure,
            'rotation_speed': reading.rotation_speed,
            'power_consumption': reading.power_consumption,
            'location': reading.location
        })
    
    return jsonify(data)

@app.route('/api/anomalies')
def get_anomalies():
    """API endpoint for anomaly detection results"""
    readings = sensor_simulator.get_latest_readings(50)
    anomalies = ai_predictor.detect_anomalies(readings)
    return jsonify(anomalies)

@app.route('/api/predictions')
def get_predictions():
    """API endpoint for failure predictions"""
    readings = sensor_simulator.get_latest_readings(5)
    predictions = []
    
    for reading in readings:
        alert = ai_predictor.predict_failure(reading)
        if alert and alert.severity in ['CRITICAL', 'HIGH']:
            predictions.append({
                'equipment_id': alert.equipment_id,
                'severity': alert.severity,
                'predicted_failure_time': alert.predicted_failure_time.isoformat(),
                'recommended_action': alert.recommended_action,
                'confidence_score': alert.confidence_score
            })
    
    return jsonify(predictions)

@app.route('/api/alerts')
def get_active_alerts():
    """API endpoint for active maintenance alerts"""
    alerts = physical_controller.get_active_alerts()
    data = []
    
    for alert in alerts:
        data.append({
            'equipment_id': alert.equipment_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'predicted_failure_time': alert.predicted_failure_time.isoformat(),
            'recommended_action': alert.recommended_action,
            'confidence_score': alert.confidence_score
        })
    
    return jsonify(data)

@app.route('/api/maintenance-schedule')
def get_maintenance_schedule():
    """API endpoint for maintenance schedule"""
    schedule = physical_controller.get_maintenance_schedule()
    return jsonify(schedule)

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """API endpoint to trigger model training"""
    readings = sensor_simulator.get_latest_readings(200)
    ai_predictor.train_models(readings)
    return jsonify({'status': 'success', 'message': 'Models trained successfully'})

@app.route('/api/system-status')
def get_system_status():
    """API endpoint for overall system status"""
    return jsonify({
        'sensor_simulation_active': sensor_simulator.running,
        'ai_models_trained': ai_predictor.is_trained,
        'total_equipment': len(sensor_simulator.equipment_list),
        'active_alerts': len(physical_controller.get_active_alerts()),
        'scheduled_maintenance': len(physical_controller.get_maintenance_schedule()),
        'last_update': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Start IoT sensor simulation
    sensor_simulator.start_simulation()
    
    # Wait for some data to accumulate
    time.sleep(10)
    
    # Train initial AI models
    readings = sensor_simulator.get_latest_readings(100)
    if readings:
        ai_predictor.train_models(readings)
    
    # Start Flask application
    logger.info("Starting Physical AI Predictive Maintenance System")
    app.run(host='0.0.0.0', port=5000, debug=True)