from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import random
import threading
import time
from collections import deque

app = Flask(__name__)

# Global variables for IoT sensor data simulation
sensor_data = deque(maxlen=1000)  # Store last 1000 readings
equipment_status = {
    'motor_1': {'health': 85, 'temperature': 45, 'vibration': 2.1, 'status': 'normal'},
    'motor_2': {'health': 92, 'temperature': 42, 'vibration': 1.8, 'status': 'normal'},
    'conveyor_1': {'health': 78, 'temperature': 38, 'vibration': 3.2, 'status': 'warning'},
    'pump_1': {'health': 88, 'temperature': 50, 'vibration': 2.5, 'status': 'normal'}
}

# Machine Learning Models
scaler = StandardScaler()
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)

class IoTSensorSimulator:
    """Simulates IoT sensors for industrial equipment"""
    
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start_simulation(self):
        """Start IoT sensor data simulation"""
        self.running = True
        self.thread = threading.Thread(target=self._simulate_sensors)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_simulation(self):
        """Stop IoT sensor data simulation"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _simulate_sensors(self):
        """Simulate real-time IoT sensor data"""
        while self.running:
            timestamp = datetime.now()
            
            for equipment_id in equipment_status.keys():
                # Simulate sensor readings with some randomness
                base_temp = equipment_status[equipment_id]['temperature']
                base_vibration = equipment_status[equipment_id]['vibration']
                
                # Add noise and potential anomalies
                temperature = base_temp + random.gauss(0, 2)
                vibration = base_vibration + random.gauss(0, 0.3)
                pressure = random.uniform(10, 50)
                current = random.uniform(5, 15)
                
                # Simulate equipment degradation over time
                if random.random() < 0.05:  # 5% chance of anomaly
                    temperature += random.uniform(10, 20)
                    vibration += random.uniform(1, 3)
                
                sensor_reading = {
                    'timestamp': timestamp.isoformat(),
                    'equipment_id': equipment_id,
                    'temperature': round(temperature, 2),
                    'vibration': round(vibration, 2),
                    'pressure': round(pressure, 2),
                    'current': round(current, 2)
                }
                
                sensor_data.append(sensor_reading)
                
                # Update equipment status based on sensor readings
                self._update_equipment_health(equipment_id, sensor_reading)
            
            time.sleep(2)  # Update every 2 seconds
    
    def _update_equipment_health(self, equipment_id, reading):
        """Update equipment health based on sensor readings"""
        # Simple health calculation based on temperature and vibration
        temp_score = max(0, 100 - (reading['temperature'] - 40) * 2)
        vibration_score = max(0, 100 - (reading['vibration'] - 2) * 10)
        
        health = (temp_score + vibration_score) / 2
        equipment_status[equipment_id]['health'] = round(health, 1)
        equipment_status[equipment_id]['temperature'] = reading['temperature']
        equipment_status[equipment_id]['vibration'] = reading['vibration']
        
        # Determine status
        if health > 80:
            equipment_status[equipment_id]['status'] = 'normal'
        elif health > 60:
            equipment_status[equipment_id]['status'] = 'warning'
        else:
            equipment_status[equipment_id]['status'] = 'critical'

class PredictiveMaintenanceAI:
    """AI algorithms for predictive maintenance"""
    
    def __init__(self):
        self.is_trained = False
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate synthetic training data for ML models"""
        # Generate 1000 samples of historical sensor data
        np.random.seed(42)
        n_samples = 1000
        
        # Normal operation data
        normal_temp = np.random.normal(45, 5, n_samples // 2)
        normal_vibration = np.random.normal(2.0, 0.5, n_samples // 2)
        normal_pressure = np.random.normal(25, 5, n_samples // 2)
        normal_current = np.random.normal(10, 2, n_samples // 2)
        
        # Anomalous data (equipment issues)
        anomaly_temp = np.random.normal(65, 10, n_samples // 2)
        anomaly_vibration = np.random.normal(4.0, 1.0, n_samples // 2)
        anomaly_pressure = np.random.normal(35, 8, n_samples // 2)
        anomaly_current = np.random.normal(15, 3, n_samples // 2)
        
        # Combine data
        self.training_data = np.column_stack([
            np.concatenate([normal_temp, anomaly_temp]),
            np.concatenate([normal_vibration, anomaly_vibration]),
            np.concatenate([normal_pressure, anomaly_pressure]),
            np.concatenate([normal_current, anomaly_current])
        ])
        
        # Labels for failure prediction (days until failure)
        self.failure_labels = np.concatenate([
            np.random.uniform(30, 90, n_samples // 2),  # Normal: 30-90 days
            np.random.uniform(1, 15, n_samples // 2)    # Anomaly: 1-15 days
        ])
        
        self._train_models()
    
    def _train_models(self):
        """Train ML models for anomaly detection and failure prediction"""
        # Scale the data
        scaled_data = scaler.fit_transform(self.training_data)
        
        # Train anomaly detector
        anomaly_detector.fit(scaled_data)
        
        # Train failure predictor
        failure_predictor.fit(scaled_data, self.failure_labels)
        
        self.is_trained = True
    
    def detect_anomalies(self, sensor_readings):
        """Detect anomalies in sensor readings"""
        if not sensor_readings or not self.is_trained:
            return []
        
        # Prepare data for prediction
        data = []
        for reading in sensor_readings:
            data.append([
                reading['temperature'],
                reading['vibration'],
                reading['pressure'],
                reading['current']
            ])
        
        if not data:
            return []
        
        # Scale and predict
        scaled_data = scaler.transform(data)
        anomaly_scores = anomaly_detector.decision_function(scaled_data)
        predictions = anomaly_detector.predict(scaled_data)
        
        anomalies = []
        for i, (reading, score, pred) in enumerate(zip(sensor_readings, anomaly_scores, predictions)):
            if pred == -1:  # Anomaly detected
                anomalies.append({
                    'equipment_id': reading['equipment_id'],
                    'timestamp': reading['timestamp'],
                    'anomaly_score': float(score),
                    'severity': 'high' if score < -0.5 else 'medium'
                })
        
        return anomalies
    
    def predict_failure(self, equipment_id):
        """Predict time until failure for specific equipment"""
        if not self.is_trained:
            return None
        
        # Get recent sensor data for the equipment
        recent_data = [r for r in list(sensor_data)[-50:] if r['equipment_id'] == equipment_id]
        
        if not recent_data:
            return None
        
        # Use the most recent reading
        latest_reading = recent_data[-1]
        data = [[
            latest_reading['temperature'],
            latest_reading['vibration'],
            latest_reading['pressure'],
            latest_reading['current']
        ]]
        
        scaled_data = scaler.transform(data)
        days_until_failure = failure_predictor.predict(scaled_data)[0]
        
        return {
            'equipment_id': equipment_id,
            'days_until_failure': round(days_until_failure, 1),
            'confidence': random.uniform(0.7, 0.95),  # Simulated confidence
            'recommendation': self._get_maintenance_recommendation(days_until_failure)
        }
    
    def _get_maintenance_recommendation(self, days_until_failure):
        """Get maintenance recommendation based on predicted failure time"""
        if days_until_failure < 7:
            return "URGENT: Schedule immediate maintenance"
        elif days_until_failure < 14:
            return "HIGH: Schedule maintenance within 1 week"
        elif days_until_failure < 30:
            return "MEDIUM: Schedule maintenance within 2 weeks"
        else:
            return "LOW: Continue monitoring, maintenance not urgent"

# Initialize components
iot_simulator = IoTSensorSimulator()
predictive_ai = PredictiveMaintenanceAI()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/equipment/status')
def get_equipment_status():
    """Get current status of all equipment"""
    return jsonify(equipment_status)

@app.route('/api/sensor/data')
def get_sensor_data():
    """Get recent sensor data"""
    recent_data = list(sensor_data)[-50:]  # Last 50 readings
    return jsonify(recent_data)

@app.route('/api/anomalies')
def get_anomalies():
    """Get detected anomalies"""
    recent_data = list(sensor_data)[-20:]  # Check last 20 readings
    anomalies = predictive_ai.detect_anomalies(recent_data)
    return jsonify(anomalies)

@app.route('/api/prediction/<equipment_id>')
def get_failure_prediction(equipment_id):
    """Get failure prediction for specific equipment"""
    prediction = predictive_ai.predict_failure(equipment_id)
    if prediction:
        return jsonify(prediction)
    else:
        return jsonify({'error': 'No data available for prediction'}), 404

@app.route('/api/maintenance/schedule', methods=['POST'])
def schedule_maintenance():
    """Schedule maintenance for equipment"""
    data = request.get_json()
    equipment_id = data.get('equipment_id')
    maintenance_type = data.get('type', 'routine')
    scheduled_date = data.get('date')
    
    # In a real system, this would integrate with a maintenance management system
    response = {
        'success': True,
        'message': f'Maintenance scheduled for {equipment_id}',
        'equipment_id': equipment_id,
        'type': maintenance_type,
        'scheduled_date': scheduled_date,
        'ticket_id': f'MT-{random.randint(1000, 9999)}'
    }
    
    return jsonify(response)

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start IoT sensor simulation"""
    iot_simulator.start_simulation()
    return jsonify({'status': 'Simulation started'})

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop IoT sensor simulation"""
    iot_simulator.stop_simulation()
    return jsonify({'status': 'Simulation stopped'})

if __name__ == '__main__':
    # Start IoT simulation automatically
    iot_simulator.start_simulation()
    
    print("Smart Manufacturing Predictive Maintenance System")
    print("Features:")
    print("- Real-time IoT sensor monitoring")
    print("- AI-powered anomaly detection")
    print("- Predictive maintenance algorithms")
    print("- Equipment health tracking")
    print("- Maintenance scheduling")
    print("\nAccess the dashboard at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)