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
from collections import deque
import sqlite3

app = Flask(__name__)

# Global variables for real-time data processing
sensor_data_buffer = deque(maxlen=1000)
equipment_status = {
    'conveyor_belt_1': {'status': 'operational', 'health_score': 95, 'last_maintenance': '2026-01-10'},
    'robotic_arm_2': {'status': 'operational', 'health_score': 88, 'last_maintenance': '2026-01-08'},
    'cnc_machine_3': {'status': 'operational', 'health_score': 92, 'last_maintenance': '2026-01-12'},
    'assembly_station_4': {'status': 'operational', 'health_score': 85, 'last_maintenance': '2026-01-05'}
}

class PhysicalAIPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.failure_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.train_models()
    
    def generate_training_data(self, n_samples=1000):
        np.random.seed(42)
        # Features: temperature, vibration, pressure, speed, power_consumption
        normal_data = np.random.normal([75, 2.5, 50, 1800, 85], [10, 0.5, 8, 200, 15], (n_samples//2, 5))
        failure_data = np.random.normal([95, 4.2, 45, 1600, 110], [15, 1.0, 12, 300, 25], (n_samples//2, 5))
        X = np.vstack([normal_data, failure_data])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        return X, y
    
    def train_models(self):
        X, y = self.generate_training_data()
        X_scaled = self.scaler.fit_transform(X)
        self.failure_predictor.fit(X_scaled, y)
        normal_data = X_scaled[y == 0]
        self.anomaly_detector.fit(normal_data)
        self.is_trained = True
        print('Physical AI models trained successfully')
    
    def predict_failure_probability(self, sensor_data):
        if not self.is_trained:
            return 0.0
        sensor_array = np.array(sensor_data).reshape(1, -1)
        sensor_scaled = self.scaler.transform(sensor_array)
        failure_prob = self.failure_predictor.predict_proba(sensor_scaled)[0][1]
        return failure_prob
    
    def detect_anomaly(self, sensor_data):
        if not self.is_trained:
            return False, 0.0
        sensor_array = np.array(sensor_data).reshape(1, -1)
        sensor_scaled = self.scaler.transform(sensor_array)
        anomaly_score = self.anomaly_detector.decision_function(sensor_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(sensor_scaled)[0] == -1
        return is_anomaly, anomaly_score

class IoTSensorSimulator:
    def __init__(self):
        self.equipment_sensors = {
            'conveyor_belt_1': {'temp': 75, 'vibration': 2.5, 'pressure': 50, 'speed': 1800, 'power': 85},
            'robotic_arm_2': {'temp': 68, 'vibration': 1.8, 'pressure': 55, 'speed': 2200, 'power': 92},
            'cnc_machine_3': {'temp': 82, 'vibration': 3.1, 'pressure': 48, 'speed': 1600, 'power': 105},
            'assembly_station_4': {'temp': 71, 'vibration': 2.2, 'pressure': 52, 'speed': 1900, 'power': 78}
        }
    
    def simulate_sensor_reading(self, equipment_id):
        base_values = self.equipment_sensors[equipment_id]
        readings = {}
        for sensor, base_value in base_values.items():
            noise = random.uniform(-0.1, 0.1) * base_value
            if equipment_status[equipment_id]['health_score'] < 90:
                degradation_factor = (100 - equipment_status[equipment_id]['health_score']) / 100
                if sensor in ['temp', 'vibration', 'power']:
                    noise += degradation_factor * base_value * 0.2
                elif sensor in ['pressure', 'speed']:
                    noise -= degradation_factor * base_value * 0.1
            readings[sensor] = max(0, base_value + noise)
        return readings

class PhysicalActionController:
    def __init__(self):
        self.maintenance_queue = []
    
    def execute_physical_action(self, equipment_id, action_type, severity):
        timestamp = datetime.now().isoformat()
        action_log = {
            'timestamp': timestamp,
            'equipment_id': equipment_id,
            'action_type': action_type,
            'severity': severity,
            'status': 'executed'
        }
        
        if action_type == 'emergency_shutdown':
            equipment_status[equipment_id]['status'] = 'maintenance_required'
            action_log['description'] = f'Emergency shutdown initiated for {equipment_id}'
        elif action_type == 'schedule_maintenance':
            maintenance_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            self.maintenance_queue.append({
                'equipment_id': equipment_id,
                'scheduled_date': maintenance_date,
                'priority': severity
            })
            action_log['description'] = f'Maintenance scheduled for {equipment_id} on {maintenance_date}'
        elif action_type == 'adjust_parameters':
            equipment_status[equipment_id]['health_score'] = min(100, 
                equipment_status[equipment_id]['health_score'] + random.uniform(2, 5))
            action_log['description'] = f'Operating parameters adjusted for {equipment_id}'
        
        return action_log

# Initialize AI components
ai_predictor = PhysicalAIPredictor()
sensor_simulator = IoTSensorSimulator()
action_controller = PhysicalActionController()

@app.route('/')
def dashboard():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Physical AI Smart Manufacturing Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .equipment-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .equipment-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .health-score { font-size: 24px; font-weight: bold; }
        .operational { color: #28a745; }
        .warning { color: #ffc107; }
        .critical { color: #dc3545; }
        .sensor-data { margin: 10px 0; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-primary { background-color: #007bff; color: white; }
        .btn-warning { background-color: #ffc107; color: black; }
        .btn-danger { background-color: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 Physical AI Smart Manufacturing System</h1>
            <p>Real-time IoT monitoring with AI-powered predictive maintenance and automated physical responses</p>
        </div>
        
        <div class="equipment-grid" id="equipmentGrid">
            <!-- Equipment cards will be populated by JavaScript -->
        </div>
        
        <div style="margin-top: 30px; background: white; padding: 20px; border-radius: 10px;">
            <h3>🔧 Maintenance Queue</h3>
            <div id="maintenanceQueue"></div>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/equipment_status')
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('equipmentGrid');
                    grid.innerHTML = '';
                    
                    Object.entries(data).forEach(([equipmentId, status]) => {
                        const healthClass = status.health_score > 90 ? 'operational' : 
                                          status.health_score > 70 ? 'warning' : 'critical';
                        
                        const card = document.createElement('div');
                        card.className = 'equipment-card';
                        card.innerHTML = `
                            <h3>🤖 ${equipmentId.replace('_', ' ').toUpperCase()}</h3>
                            <div class="health-score ${healthClass}">Health: ${status.health_score.toFixed(1)}%</div>
                            <div>Status: <span class="${healthClass}">${status.status}</span></div>
                            <div>Last Maintenance: ${status.last_maintenance}</div>
                            <button class="btn btn-primary" onclick="predictFailure('${equipmentId}')">🔮 Predict Failure</button>
                            <button class="btn btn-warning" onclick="scheduleMaintenance('${equipmentId}')">🔧 Schedule Maintenance</button>
                            <div id="prediction-${equipmentId}" style="margin-top: 10px;"></div>
                        `;
                        grid.appendChild(card);
                    });
                });
            
            fetch('/api/maintenance_queue')
                .then(response => response.json())
                .then(data => {
                    const queue = document.getElementById('maintenanceQueue');
                    if (data.length === 0) {
                        queue.innerHTML = '<p>No maintenance scheduled</p>';
                    } else {
                        queue.innerHTML = data.map(item => 
                            `<div style="padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px;">
                                📅 ${item.equipment_id} - ${item.scheduled_date} (Priority: ${item.priority})
                            </div>`
                        ).join('');
                    }
                });
        }
        
        function predictFailure(equipmentId) {
            fetch('/api/predict_failure', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({equipment_id: equipmentId})
            })
            .then(response => response.json())
            .then(data => {
                const predictionDiv = document.getElementById(`prediction-${equipmentId}`);
                const riskLevel = data.failure_probability > 0.7 ? 'critical' : 
                                data.failure_probability > 0.4 ? 'warning' : 'operational';
                
                predictionDiv.innerHTML = `
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <strong>🎯 AI Prediction:</strong><br>
                        Failure Risk: <span class="${riskLevel}">${(data.failure_probability * 100).toFixed(1)}%</span><br>
                        Anomaly: ${data.anomaly_detected ? '⚠️ Detected' : '✅ Normal'}<br>
                        <small>${data.recommendation}</small>
                    </div>
                `;
            });
        }
        
        function scheduleMaintenance(equipmentId) {
            fetch('/api/trigger_action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({equipment_id: equipmentId, action_type: 'schedule_maintenance'})
            })
            .then(response => response.json())
            .then(data => {
                alert('✅ ' + data.description);
                updateDashboard();
            });
        }
        
        // Update dashboard every 10 seconds
        updateDashboard();
        setInterval(updateDashboard, 10000);
    </script>
</body>
</html>'''

@app.route('/api/equipment_status')
def get_equipment_status():
    return jsonify(equipment_status)

@app.route('/api/maintenance_queue')
def get_maintenance_queue():
    return jsonify(action_controller.maintenance_queue)

@app.route('/api/predict_failure', methods=['POST'])
def predict_failure():
    data = request.json
    equipment_id = data.get('equipment_id')
    
    if equipment_id not in equipment_status:
        return jsonify({'error': 'Invalid equipment ID'}), 400
    
    readings = sensor_simulator.simulate_sensor_reading(equipment_id)
    sensor_array = [readings['temp'], readings['vibration'], readings['pressure'], 
                   readings['speed'], readings['power']]
    
    failure_prob = ai_predictor.predict_failure_probability(sensor_array)
    is_anomaly, anomaly_score = ai_predictor.detect_anomaly(sensor_array)
    
    return jsonify({
        'equipment_id': equipment_id,
        'current_readings': readings,
        'failure_probability': failure_prob,
        'anomaly_detected': is_anomaly,
        'anomaly_score': anomaly_score,
        'health_score': equipment_status[equipment_id]['health_score'],
        'recommendation': get_maintenance_recommendation(failure_prob, is_anomaly)
    })

def get_maintenance_recommendation(failure_prob, is_anomaly):
    if failure_prob > 0.8:
        return 'IMMEDIATE SHUTDOWN REQUIRED - Critical failure risk detected'
    elif failure_prob > 0.6:
        return 'SCHEDULE URGENT MAINTENANCE - High failure probability'
    elif failure_prob > 0.4:
        return 'PLAN MAINTENANCE WITHIN 48 HOURS - Moderate risk'
    elif is_anomaly:
        return 'INVESTIGATE ANOMALY - Unusual sensor patterns detected'
    else:
        return 'NORMAL OPERATION - Continue monitoring'

@app.route('/api/trigger_action', methods=['POST'])
def trigger_manual_action():
    data = request.json
    equipment_id = data.get('equipment_id')
    action_type = data.get('action_type')
    
    if equipment_id not in equipment_status:
        return jsonify({'error': 'Invalid equipment ID'}), 400
    
    action_log = action_controller.execute_physical_action(
        equipment_id, action_type, 'manual')
    
    return jsonify(action_log)

if __name__ == '__main__':
    print('🏭 Physical AI Smart Manufacturing System Starting...')
    print('Features:')
    print('- Real-time IoT sensor monitoring')
    print('- AI-powered predictive maintenance')
    print('- Automated physical responses')
    print('- Anomaly detection and alerting')
    print('- Equipment health scoring')
    print('\n🌐 Access the dashboard at: http://localhost:5000')
    
    app.run(debug=True, host='0.0.0.0', port=5000)