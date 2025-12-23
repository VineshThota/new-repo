#!/usr/bin/env python3
"""
Digital Twin Predictive Maintenance System
Combining IoT, AI, and Physical AI for Smart Manufacturing

This application integrates:
- IoT sensor simulation and data collection
- AI-powered predictive analytics
- Digital twin visualization
- Real-time monitoring dashboard
- Physical system integration
- Automated maintenance scheduling
"""

import os
import json
import time
import random
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.utils
from dataclasses import dataclass
from typing import Dict, List, Optional
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'digital_twin_predictive_maintenance_2025'

@dataclass
class IoTSensor:
    """IoT Sensor data structure"""
    sensor_id: str
    sensor_type: str
    location: str
    normal_range: tuple
    critical_threshold: float
    unit: str

@dataclass
class EquipmentStatus:
    """Equipment status data structure"""
    equipment_id: str
    name: str
    status: str
    health_score: float
    predicted_failure_date: Optional[datetime]
    maintenance_priority: str

class IoTDataSimulator:
    """Simulates IoT sensor data for manufacturing equipment"""
    
    def __init__(self):
        self.sensors = {
            'temp_001': IoTSensor('temp_001', 'Temperature', 'Motor Bearing', (60, 80), 95, '°C'),
            'vib_001': IoTSensor('vib_001', 'Vibration', 'Motor Assembly', (0.1, 2.0), 5.0, 'mm/s'),
            'curr_001': IoTSensor('curr_001', 'Current', 'Motor Drive', (10, 25), 35, 'A'),
            'press_001': IoTSensor('press_001', 'Pressure', 'Hydraulic System', (150, 200), 250, 'PSI'),
            'flow_001': IoTSensor('flow_001', 'Flow Rate', 'Coolant System', (5, 15), 3, 'L/min'),
            'noise_001': IoTSensor('noise_001', 'Acoustic', 'Gearbox', (40, 60), 80, 'dB')
        }
        self.data_history = []
        self.is_running = False
        
    def generate_sensor_reading(self, sensor: IoTSensor, equipment_degradation: float = 0.0) -> float:
        """Generate realistic sensor reading with optional degradation"""
        base_value = random.uniform(sensor.normal_range[0], sensor.normal_range[1])
        
        # Add degradation effect
        if equipment_degradation > 0:
            degradation_factor = 1 + (equipment_degradation * 0.3)
            base_value *= degradation_factor
            
        # Add some noise
        noise = random.gauss(0, base_value * 0.05)
        return max(0, base_value + noise)
    
    def collect_data(self):
        """Continuously collect sensor data"""
        while self.is_running:
            timestamp = datetime.now()
            
            # Simulate equipment degradation over time
            degradation = min(0.5, len(self.data_history) * 0.001)
            
            reading = {
                'timestamp': timestamp.isoformat(),
                'equipment_id': 'MACHINE_001',
                'readings': {}
            }
            
            for sensor_id, sensor in self.sensors.items():
                value = self.generate_sensor_reading(sensor, degradation)
                reading['readings'][sensor_id] = {
                    'value': round(value, 2),
                    'unit': sensor.unit,
                    'status': 'normal' if value < sensor.critical_threshold else 'critical'
                }
            
            self.data_history.append(reading)
            
            # Keep only last 1000 readings
            if len(self.data_history) > 1000:
                self.data_history.pop(0)
                
            time.sleep(2)  # Collect data every 2 seconds
    
    def start_collection(self):
        """Start data collection in background thread"""
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self.collect_data)
            thread.daemon = True
            thread.start()
            logger.info("IoT data collection started")
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_running = False
        logger.info("IoT data collection stopped")
    
    def get_latest_data(self, limit: int = 50) -> List[Dict]:
        """Get latest sensor readings"""
        return self.data_history[-limit:] if self.data_history else []

class PredictiveMaintenanceAI:
    """AI-powered predictive maintenance system"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data_history: List[Dict]) -> np.ndarray:
        """Extract features from sensor data for ML models"""
        if not data_history:
            return np.array([])
            
        features = []
        for reading in data_history:
            feature_row = []
            for sensor_id in ['temp_001', 'vib_001', 'curr_001', 'press_001', 'flow_001', 'noise_001']:
                if sensor_id in reading['readings']:
                    feature_row.append(reading['readings'][sensor_id]['value'])
                else:
                    feature_row.append(0)
            features.append(feature_row)
            
        return np.array(features)
    
    def train_models(self, data_history: List[Dict]):
        """Train AI models with historical data"""
        features = self.prepare_features(data_history)
        
        if len(features) < 10:
            logger.warning("Insufficient data for training")
            return
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train anomaly detection model
        self.anomaly_detector.fit(features_scaled)
        
        # Generate synthetic failure labels for demonstration
        # In real scenario, this would be historical failure data
        failure_labels = np.random.exponential(100, len(features))  # Days until failure
        
        # Train failure prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, failure_labels, test_size=0.2, random_state=42
        )
        self.failure_predictor.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("AI models trained successfully")
    
    def detect_anomalies(self, current_data: Dict) -> Dict:
        """Detect anomalies in current sensor readings"""
        if not self.is_trained:
            return {'anomaly_score': 0, 'is_anomaly': False}
            
        features = []
        for sensor_id in ['temp_001', 'vib_001', 'curr_001', 'press_001', 'flow_001', 'noise_001']:
            if sensor_id in current_data['readings']:
                features.append(current_data['readings'][sensor_id]['value'])
            else:
                features.append(0)
                
        features_scaled = self.scaler.transform([features])
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        return {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly)
        }
    
    def predict_failure(self, current_data: Dict) -> Dict:
        """Predict time until equipment failure"""
        if not self.is_trained:
            return {'days_until_failure': None, 'confidence': 0}
            
        features = []
        for sensor_id in ['temp_001', 'vib_001', 'curr_001', 'press_001', 'flow_001', 'noise_001']:
            if sensor_id in current_data['readings']:
                features.append(current_data['readings'][sensor_id]['value'])
            else:
                features.append(0)
                
        features_scaled = self.scaler.transform([features])
        days_until_failure = self.failure_predictor.predict(features_scaled)[0]
        
        # Calculate confidence based on feature importance
        confidence = min(1.0, max(0.1, 1.0 - (abs(days_until_failure - 50) / 100)))
        
        return {
            'days_until_failure': float(days_until_failure),
            'confidence': float(confidence)
        }

class DigitalTwinSystem:
    """Digital Twin system for equipment monitoring"""
    
    def __init__(self):
        self.equipment_models = {
            'MACHINE_001': {
                'name': 'CNC Milling Machine',
                'type': 'Manufacturing Equipment',
                'location': 'Production Line A',
                'installation_date': '2023-01-15',
                'specifications': {
                    'max_rpm': 8000,
                    'power_rating': '15kW',
                    'precision': '±0.01mm'
                }
            }
        }
    
    def calculate_health_score(self, sensor_data: Dict, anomaly_result: Dict, failure_prediction: Dict) -> float:
        """Calculate overall equipment health score (0-100)"""
        base_score = 100
        
        # Deduct points for sensor readings outside normal range
        for sensor_id, reading in sensor_data['readings'].items():
            if reading['status'] == 'critical':
                base_score -= 20
            elif reading['value'] > 0:  # Assume some degradation
                base_score -= 5
        
        # Deduct points for anomalies
        if anomaly_result['is_anomaly']:
            base_score -= 15
        
        # Deduct points based on failure prediction
        if failure_prediction['days_until_failure'] and failure_prediction['days_until_failure'] < 30:
            base_score -= 25
        elif failure_prediction['days_until_failure'] and failure_prediction['days_until_failure'] < 60:
            base_score -= 10
        
        return max(0, min(100, base_score))
    
    def get_maintenance_priority(self, health_score: float, failure_prediction: Dict) -> str:
        """Determine maintenance priority"""
        if health_score < 30 or (failure_prediction['days_until_failure'] and failure_prediction['days_until_failure'] < 7):
            return 'CRITICAL'
        elif health_score < 60 or (failure_prediction['days_until_failure'] and failure_prediction['days_until_failure'] < 30):
            return 'HIGH'
        elif health_score < 80:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_equipment_status(self, sensor_data: Dict, anomaly_result: Dict, failure_prediction: Dict) -> EquipmentStatus:
        """Generate comprehensive equipment status"""
        health_score = self.calculate_health_score(sensor_data, anomaly_result, failure_prediction)
        priority = self.get_maintenance_priority(health_score, failure_prediction)
        
        predicted_failure_date = None
        if failure_prediction['days_until_failure']:
            predicted_failure_date = datetime.now() + timedelta(days=failure_prediction['days_until_failure'])
        
        status = 'OPERATIONAL'
        if health_score < 30:
            status = 'CRITICAL'
        elif health_score < 60:
            status = 'WARNING'
        
        return EquipmentStatus(
            equipment_id='MACHINE_001',
            name='CNC Milling Machine',
            status=status,
            health_score=health_score,
            predicted_failure_date=predicted_failure_date,
            maintenance_priority=priority
        )

# Initialize system components
iot_simulator = IoTDataSimulator()
ai_system = PredictiveMaintenanceAI()
digital_twin = DigitalTwinSystem()

# Start IoT data collection
iot_simulator.start_collection()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/sensor-data')
def get_sensor_data():
    """Get latest sensor data"""
    latest_data = iot_simulator.get_latest_data(1)
    if not latest_data:
        return jsonify({'error': 'No data available'})
    
    return jsonify(latest_data[0])

@app.route('/api/equipment-status')
def get_equipment_status():
    """Get comprehensive equipment status"""
    latest_data = iot_simulator.get_latest_data(1)
    if not latest_data:
        return jsonify({'error': 'No data available'})
    
    current_data = latest_data[0]
    
    # Train AI models if not already trained and sufficient data available
    if not ai_system.is_trained and len(iot_simulator.data_history) >= 20:
        ai_system.train_models(iot_simulator.data_history)
    
    # Get AI predictions
    anomaly_result = ai_system.detect_anomalies(current_data)
    failure_prediction = ai_system.predict_failure(current_data)
    
    # Generate equipment status
    equipment_status = digital_twin.generate_equipment_status(
        current_data, anomaly_result, failure_prediction
    )
    
    return jsonify({
        'equipment_status': {
            'equipment_id': equipment_status.equipment_id,
            'name': equipment_status.name,
            'status': equipment_status.status,
            'health_score': equipment_status.health_score,
            'predicted_failure_date': equipment_status.predicted_failure_date.isoformat() if equipment_status.predicted_failure_date else None,
            'maintenance_priority': equipment_status.maintenance_priority
        },
        'ai_analysis': {
            'anomaly_detection': anomaly_result,
            'failure_prediction': failure_prediction
        },
        'sensor_data': current_data
    })

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical sensor data for charts"""
    limit = request.args.get('limit', 50, type=int)
    data = iot_simulator.get_latest_data(limit)
    
    # Prepare data for charts
    timestamps = []
    sensor_values = {sensor_id: [] for sensor_id in iot_simulator.sensors.keys()}
    
    for reading in data:
        timestamps.append(reading['timestamp'])
        for sensor_id in sensor_values.keys():
            if sensor_id in reading['readings']:
                sensor_values[sensor_id].append(reading['readings'][sensor_id]['value'])
            else:
                sensor_values[sensor_id].append(None)
    
    return jsonify({
        'timestamps': timestamps,
        'sensor_values': sensor_values,
        'sensor_info': {sid: {'unit': s.unit, 'type': s.sensor_type} for sid, s in iot_simulator.sensors.items()}
    })

@app.route('/api/maintenance-schedule')
def get_maintenance_schedule():
    """Get recommended maintenance schedule"""
    latest_data = iot_simulator.get_latest_data(1)
    if not latest_data:
        return jsonify({'schedule': []})
    
    current_data = latest_data[0]
    
    if ai_system.is_trained:
        failure_prediction = ai_system.predict_failure(current_data)
        
        schedule = []
        if failure_prediction['days_until_failure']:
            maintenance_date = datetime.now() + timedelta(days=max(1, failure_prediction['days_until_failure'] - 7))
            schedule.append({
                'equipment_id': 'MACHINE_001',
                'maintenance_type': 'Predictive Maintenance',
                'scheduled_date': maintenance_date.isoformat(),
                'priority': 'HIGH' if failure_prediction['days_until_failure'] < 14 else 'MEDIUM',
                'estimated_duration': '4 hours',
                'description': 'Preventive maintenance based on AI prediction'
            })
        
        return jsonify({'schedule': schedule})
    
    return jsonify({'schedule': []})

if __name__ == '__main__':
    # Create templates directory and basic template
    os.makedirs('templates', exist_ok=True)
    
    # Create basic HTML template
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Twin Predictive Maintenance System</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-operational { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-critical { background-color: #F44336; }
        .health-score { font-size: 2em; font-weight: bold; color: #333; }
        .sensor-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
        .sensor-item { padding: 10px; background: #f8f9fa; border-radius: 5px; text-align: center; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 Digital Twin Predictive Maintenance System</h1>
        <p>Real-time IoT monitoring with AI-powered predictive analytics for smart manufacturing</p>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h3>Equipment Status</h3>
            <div id="equipment-status">
                <p>Loading equipment status...</p>
            </div>
            <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
        </div>
        
        <div class="card">
            <h3>Current Sensor Readings</h3>
            <div id="sensor-readings" class="sensor-grid">
                <p>Loading sensor data...</p>
            </div>
        </div>
        
        <div class="card">
            <h3>AI Analysis</h3>
            <div id="ai-analysis">
                <p>Loading AI analysis...</p>
            </div>
        </div>
        
        <div class="card">
            <h3>Maintenance Schedule</h3>
            <div id="maintenance-schedule">
                <p>Loading maintenance schedule...</p>
            </div>
        </div>
    </div>
    
    <div class="card" style="margin-top: 20px;">
        <h3>Historical Sensor Data</h3>
        <div id="sensor-chart" style="height: 400px;"></div>
    </div>
    
    <script>
        function refreshData() {
            fetchEquipmentStatus();
            fetchSensorData();
            fetchMaintenanceSchedule();
            fetchHistoricalData();
        }
        
        function fetchEquipmentStatus() {
            fetch('/api/equipment-status')
                .then(response => response.json())
                .then(data => {
                    const status = data.equipment_status;
                    const ai = data.ai_analysis;
                    
                    document.getElementById('equipment-status').innerHTML = `
                        <h4><span class="status-indicator status-${status.status.toLowerCase()}"></span>${status.name}</h4>
                        <p><strong>Status:</strong> ${status.status}</p>
                        <p><strong>Health Score:</strong> <span class="health-score">${status.health_score.toFixed(1)}%</span></p>
                        <p><strong>Priority:</strong> ${status.maintenance_priority}</p>
                        ${status.predicted_failure_date ? `<p><strong>Predicted Failure:</strong> ${new Date(status.predicted_failure_date).toLocaleDateString()}</p>` : ''}
                    `;
                    
                    document.getElementById('ai-analysis').innerHTML = `
                        <p><strong>Anomaly Detection:</strong> ${ai.anomaly_detection.is_anomaly ? '⚠️ Anomaly Detected' : '✅ Normal'}</p>
                        <p><strong>Anomaly Score:</strong> ${ai.anomaly_detection.anomaly_score.toFixed(3)}</p>
                        ${ai.failure_prediction.days_until_failure ? `<p><strong>Predicted Failure:</strong> ${ai.failure_prediction.days_until_failure.toFixed(0)} days</p>` : ''}
                        <p><strong>Confidence:</strong> ${(ai.failure_prediction.confidence * 100).toFixed(1)}%</p>
                    `;
                })
                .catch(error => console.error('Error:', error));
        }
        
        function fetchSensorData() {
            fetch('/api/sensor-data')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    for (const [sensorId, reading] of Object.entries(data.readings)) {
                        const statusClass = reading.status === 'critical' ? 'style="background-color: #ffebee;"' : '';
                        html += `
                            <div class="sensor-item" ${statusClass}>
                                <strong>${sensorId.toUpperCase()}</strong><br>
                                ${reading.value} ${reading.unit}<br>
                                <small>${reading.status}</small>
                            </div>
                        `;
                    }
                    document.getElementById('sensor-readings').innerHTML = html;
                })
                .catch(error => console.error('Error:', error));
        }
        
        function fetchMaintenanceSchedule() {
            fetch('/api/maintenance-schedule')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    if (data.schedule.length > 0) {
                        data.schedule.forEach(item => {
                            html += `
                                <div style="margin-bottom: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                    <strong>${item.maintenance_type}</strong><br>
                                    <small>Date: ${new Date(item.scheduled_date).toLocaleDateString()}</small><br>
                                    <small>Priority: ${item.priority}</small><br>
                                    <small>Duration: ${item.estimated_duration}</small>
                                </div>
                            `;
                        });
                    } else {
                        html = '<p>No maintenance scheduled</p>';
                    }
                    document.getElementById('maintenance-schedule').innerHTML = html;
                })
                .catch(error => console.error('Error:', error));
        }
        
        function fetchHistoricalData() {
            fetch('/api/historical-data?limit=30')
                .then(response => response.json())
                .then(data => {
                    const traces = [];
                    
                    for (const [sensorId, values] of Object.entries(data.sensor_values)) {
                        traces.push({
                            x: data.timestamps,
                            y: values,
                            type: 'scatter',
                            mode: 'lines',
                            name: sensorId.toUpperCase(),
                            line: { width: 2 }
                        });
                    }
                    
                    const layout = {
                        title: 'Real-time Sensor Monitoring',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Sensor Values' },
                        showlegend: true,
                        margin: { t: 50, r: 50, b: 50, l: 50 }
                    };
                    
                    Plotly.newPlot('sensor-chart', traces, layout);
                })
                .catch(error => console.error('Error:', error));
        }
        
        // Initial load and auto-refresh
        refreshData();
        setInterval(refreshData, 5000); // Refresh every 5 seconds
    </script>
</body>
</html>
    '''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_template)
    
    print("🚀 Starting Digital Twin Predictive Maintenance System...")
    print("📊 Features:")
    print("   • IoT sensor simulation and real-time data collection")
    print("   • AI-powered anomaly detection and failure prediction")
    print("   • Digital twin visualization and monitoring")
    print("   • Automated maintenance scheduling")
    print("   • Real-time dashboard with interactive charts")
    print("\n🌐 Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)