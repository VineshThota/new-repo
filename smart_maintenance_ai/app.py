from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import time
import random
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class IoTSensorSimulator:
    """Simulates IoT sensors for temperature, vibration, and pressure"""
    
    def __init__(self):
        self.sensors = {
            'temperature': {'value': 75.0, 'normal_range': (70, 80), 'unit': '°F'},
            'vibration': {'value': 2.5, 'normal_range': (1.0, 4.0), 'unit': 'mm/s'},
            'pressure': {'value': 150.0, 'normal_range': (140, 160), 'unit': 'PSI'}
        }
        self.data_history = deque(maxlen=1000)
        self.running = False
        
    def generate_sensor_data(self):
        """Generate realistic sensor data with occasional anomalies"""
        data = {}
        timestamp = datetime.now()
        
        for sensor_name, sensor_info in self.sensors.items():
            # Normal operation with small variations
            base_value = sensor_info['value']
            normal_variation = np.random.normal(0, 0.5)
            
            # Introduce anomalies 5% of the time
            if random.random() < 0.05:
                # Create anomaly
                anomaly_factor = random.choice([0.7, 1.3])  # 30% deviation
                value = base_value * anomaly_factor
            else:
                value = base_value + normal_variation
            
            # Update sensor value for next iteration
            self.sensors[sensor_name]['value'] = value
            
            data[sensor_name] = {
                'value': round(value, 2),
                'unit': sensor_info['unit'],
                'timestamp': timestamp.isoformat(),
                'normal_range': sensor_info['normal_range']
            }
        
        # Add to history
        self.data_history.append({
            'timestamp': timestamp,
            'temperature': data['temperature']['value'],
            'vibration': data['vibration']['value'],
            'pressure': data['pressure']['value']
        })
        
        return data
    
    def start_simulation(self):
        """Start continuous sensor data generation"""
        self.running = True
        
        def simulate():
            while self.running:
                self.generate_sensor_data()
                time.sleep(2)  # Generate data every 2 seconds
        
        thread = threading.Thread(target=simulate)
        thread.daemon = True
        thread.start()
    
    def stop_simulation(self):
        """Stop sensor simulation"""
        self.running = False

class PredictiveMaintenanceAI:
    """AI model for predictive maintenance using anomaly detection"""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.anomaly_threshold = -0.5
        
    def prepare_features(self, sensor_data):
        """Prepare features from sensor data for ML model"""
        if len(sensor_data) < 10:
            return None
            
        df = pd.DataFrame(sensor_data)
        
        # Calculate rolling statistics
        features = []
        for column in ['temperature', 'vibration', 'pressure']:
            if column in df.columns:
                # Current value
                features.append(df[column].iloc[-1])
                # Rolling mean (last 5 readings)
                features.append(df[column].tail(5).mean())
                # Rolling std (last 5 readings)
                features.append(df[column].tail(5).std())
                # Rate of change
                if len(df) > 1:
                    features.append(df[column].iloc[-1] - df[column].iloc[-2])
                else:
                    features.append(0)
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, historical_data):
        """Train the anomaly detection model"""
        if len(historical_data) < 50:
            logger.warning("Insufficient data for training. Need at least 50 samples.")
            return False
            
        try:
            # Prepare training features
            training_features = []
            for i in range(10, len(historical_data)):
                features = self.prepare_features(historical_data[max(0, i-10):i])
                if features is not None:
                    training_features.append(features.flatten())
            
            if len(training_features) < 10:
                return False
                
            X = np.array(training_features)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True
            
            logger.info(f"Model trained successfully with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_anomaly(self, sensor_data):
        """Predict if current sensor readings indicate an anomaly"""
        if not self.is_trained:
            return {'anomaly': False, 'confidence': 0, 'message': 'Model not trained'}
        
        try:
            features = self.prepare_features(sensor_data)
            if features is None:
                return {'anomaly': False, 'confidence': 0, 'message': 'Insufficient data'}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict anomaly
            anomaly_score = self.model.decision_function(features_scaled)[0]
            is_anomaly = anomaly_score < self.anomaly_threshold
            
            # Calculate confidence (0-100)
            confidence = min(100, max(0, abs(anomaly_score) * 100))
            
            return {
                'anomaly': bool(is_anomaly),
                'confidence': round(confidence, 2),
                'score': round(anomaly_score, 4),
                'message': 'Anomaly detected' if is_anomaly else 'Normal operation'
            }
            
        except Exception as e:
            logger.error(f"Error predicting anomaly: {e}")
            return {'anomaly': False, 'confidence': 0, 'message': f'Prediction error: {e}'}

class DigitalTwin:
    """Digital twin representation of the physical equipment"""
    
    def __init__(self):
        self.equipment_status = {
            'health_score': 100,
            'operational_hours': 0,
            'last_maintenance': datetime.now() - timedelta(days=30),
            'next_maintenance': datetime.now() + timedelta(days=60),
            'efficiency': 95.0
        }
        self.maintenance_alerts = []
    
    def update_status(self, sensor_data, anomaly_result):
        """Update digital twin status based on sensor data and AI predictions"""
        # Update operational hours
        self.equipment_status['operational_hours'] += 1/1800  # Increment by 2 seconds worth
        
        # Calculate health score based on sensor readings and anomalies
        health_impact = 0
        
        # Check sensor readings against normal ranges
        for sensor_name, data in sensor_data.items():
            if 'normal_range' in data:
                min_val, max_val = data['normal_range']
                value = data['value']
                
                if value < min_val or value > max_val:
                    health_impact += 5  # Reduce health by 5 points
        
        # Apply anomaly impact
        if anomaly_result['anomaly']:
            health_impact += anomaly_result['confidence'] / 10
        
        # Update health score (minimum 0, maximum 100)
        self.equipment_status['health_score'] = max(0, 
            min(100, self.equipment_status['health_score'] - health_impact * 0.1))
        
        # Update efficiency based on health
        self.equipment_status['efficiency'] = max(50, 
            self.equipment_status['health_score'] * 0.95)
        
        # Generate maintenance alerts
        self._check_maintenance_needs(anomaly_result)
    
    def _check_maintenance_needs(self, anomaly_result):
        """Check if maintenance is needed and generate alerts"""
        current_time = datetime.now()
        
        # Clear old alerts (older than 1 hour)
        self.maintenance_alerts = [
            alert for alert in self.maintenance_alerts 
            if current_time - alert['timestamp'] < timedelta(hours=1)
        ]
        
        # Check for critical conditions
        if self.equipment_status['health_score'] < 70:
            alert = {
                'type': 'warning',
                'message': f"Equipment health degraded to {self.equipment_status['health_score']:.1f}%",
                'timestamp': current_time,
                'priority': 'medium'
            }
            if alert not in self.maintenance_alerts:
                self.maintenance_alerts.append(alert)
        
        if anomaly_result['anomaly'] and anomaly_result['confidence'] > 70:
            alert = {
                'type': 'critical',
                'message': f"Critical anomaly detected (confidence: {anomaly_result['confidence']:.1f}%)",
                'timestamp': current_time,
                'priority': 'high'
            }
            if alert not in self.maintenance_alerts:
                self.maintenance_alerts.append(alert)
        
        # Check scheduled maintenance
        days_until_maintenance = (self.equipment_status['next_maintenance'] - current_time).days
        if days_until_maintenance <= 7:
            alert = {
                'type': 'info',
                'message': f"Scheduled maintenance due in {days_until_maintenance} days",
                'timestamp': current_time,
                'priority': 'low'
            }
            if alert not in self.maintenance_alerts:
                self.maintenance_alerts.append(alert)

# Initialize global components
sensor_simulator = IoTSensorSimulator()
ai_model = PredictiveMaintenanceAI()
digital_twin = DigitalTwin()

# Start sensor simulation
sensor_simulator.start_simulation()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/sensor-data')
def get_sensor_data():
    """Get current sensor readings"""
    try:
        current_data = sensor_simulator.generate_sensor_data()
        return jsonify({
            'status': 'success',
            'data': current_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical sensor data"""
    try:
        # Convert deque to list for JSON serialization
        history = list(sensor_simulator.data_history)
        
        # Format for chart display
        formatted_data = {
            'timestamps': [item['timestamp'].isoformat() for item in history[-50:]],
            'temperature': [item['temperature'] for item in history[-50:]],
            'vibration': [item['vibration'] for item in history[-50:]],
            'pressure': [item['pressure'] for item in history[-50:]]
        }
        
        return jsonify({
            'status': 'success',
            'data': formatted_data
        })
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/anomaly-detection')
def detect_anomaly():
    """Perform anomaly detection on current sensor data"""
    try:
        # Get recent sensor data
        recent_data = list(sensor_simulator.data_history)[-20:]
        
        if len(recent_data) < 10:
            return jsonify({
                'status': 'warning',
                'message': 'Insufficient data for anomaly detection',
                'anomaly': False
            })
        
        # Train model if not already trained
        if not ai_model.is_trained and len(sensor_simulator.data_history) >= 50:
            training_success = ai_model.train_model(list(sensor_simulator.data_history))
            if not training_success:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to train AI model'
                })
        
        # Perform anomaly detection
        anomaly_result = ai_model.predict_anomaly(recent_data)
        
        return jsonify({
            'status': 'success',
            'anomaly_detection': anomaly_result,
            'model_trained': ai_model.is_trained
        })
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/digital-twin')
def get_digital_twin_status():
    """Get digital twin status and maintenance alerts"""
    try:
        # Update digital twin with latest data
        if sensor_simulator.data_history:
            latest_sensor_data = sensor_simulator.generate_sensor_data()
            recent_data = list(sensor_simulator.data_history)[-10:]
            anomaly_result = ai_model.predict_anomaly(recent_data) if ai_model.is_trained else {
                'anomaly': False, 'confidence': 0
            }
            
            digital_twin.update_status(latest_sensor_data, anomaly_result)
        
        return jsonify({
            'status': 'success',
            'equipment_status': {
                **digital_twin.equipment_status,
                'last_maintenance': digital_twin.equipment_status['last_maintenance'].isoformat(),
                'next_maintenance': digital_twin.equipment_status['next_maintenance'].isoformat()
            },
            'maintenance_alerts': [
                {
                    **alert,
                    'timestamp': alert['timestamp'].isoformat()
                } for alert in digital_twin.maintenance_alerts
            ]
        })
        
    except Exception as e:
        logger.error(f"Error getting digital twin status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Manually trigger model training"""
    try:
        if len(sensor_simulator.data_history) < 50:
            return jsonify({
                'status': 'error',
                'message': f'Need at least 50 data points. Currently have {len(sensor_simulator.data_history)}'
            })
        
        success = ai_model.train_model(list(sensor_simulator.data_history))
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'training_samples': len(sensor_simulator.data_history)
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to train model'
            })
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting SmartMaintenance AI application...")
    logger.info("Sensor simulation started")
    logger.info("Access dashboard at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)