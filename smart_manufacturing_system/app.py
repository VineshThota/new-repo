from flask import Flask, render_template, jsonify, request
import json
import random
import time
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import threading
import queue
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables for system state
sensor_data_queue = queue.Queue()
equipment_status = {
    'machine_1': {'status': 'operational', 'health_score': 95, 'last_maintenance': '2026-01-10'},
    'machine_2': {'status': 'operational', 'health_score': 87, 'last_maintenance': '2026-01-08'},
    'robot_arm_1': {'status': 'operational', 'health_score': 92, 'safety_zone': 'clear'},
    'robot_arm_2': {'status': 'operational', 'health_score': 89, 'safety_zone': 'clear'}
}

human_robot_zones = {
    'zone_1': {'humans_present': 0, 'robot_active': True, 'safety_level': 'safe'},
    'zone_2': {'humans_present': 0, 'robot_active': True, 'safety_level': 'safe'},
    'zone_3': {'humans_present': 0, 'robot_active': False, 'safety_level': 'safe'}
}

# AI Models for predictive maintenance
class PredictiveMaintenanceAI:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.historical_data = []
        
    def add_sensor_data(self, data):
        self.historical_data.append(data)
        if len(self.historical_data) > 100:
            self.historical_data.pop(0)
            
    def train_model(self):
        if len(self.historical_data) >= 20:
            features = np.array([[d['temperature'], d['vibration'], d['pressure']] 
                               for d in self.historical_data])
            self.scaler.fit(features)
            self.anomaly_detector.fit(self.scaler.transform(features))
            self.is_trained = True
            
    def predict_anomaly(self, sensor_data):
        if not self.is_trained:
            return False, 0.5
            
        features = np.array([[sensor_data['temperature'], 
                            sensor_data['vibration'], 
                            sensor_data['pressure']]])
        scaled_features = self.scaler.transform(features)
        anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
        is_anomaly = self.anomaly_detector.predict(scaled_features)[0] == -1
        
        return is_anomaly, abs(anomaly_score)

# Physical AI Safety System
class PhysicalAISafety:
    def __init__(self):
        self.safety_protocols = {
            'human_detected': self.human_detection_protocol,
            'collision_risk': self.collision_avoidance_protocol,
            'emergency_stop': self.emergency_stop_protocol
        }
        
    def human_detection_protocol(self, zone_id, human_count):
        """Protocol when humans are detected in robot zones"""
        if human_count > 0:
            human_robot_zones[zone_id]['safety_level'] = 'caution'
            if human_count > 2:
                human_robot_zones[zone_id]['robot_active'] = False
                human_robot_zones[zone_id]['safety_level'] = 'danger'
                return {'action': 'stop_robot', 'message': f'Multiple humans detected in {zone_id}'}
            else:
                return {'action': 'reduce_speed', 'message': f'Human detected in {zone_id}, reducing robot speed'}
        else:
            human_robot_zones[zone_id]['safety_level'] = 'safe'
            human_robot_zones[zone_id]['robot_active'] = True
            return {'action': 'normal_operation', 'message': f'{zone_id} clear for normal operation'}
            
    def collision_avoidance_protocol(self, proximity_data):
        """Protocol for collision avoidance"""
        if proximity_data < 0.5:  # Less than 50cm
            return {'action': 'emergency_stop', 'message': 'Collision imminent - emergency stop activated'}
        elif proximity_data < 1.0:  # Less than 1m
            return {'action': 'slow_down', 'message': 'Object detected nearby - slowing down'}
        else:
            return {'action': 'continue', 'message': 'Path clear'}
            
    def emergency_stop_protocol(self):
        """Emergency stop protocol"""
        for zone in human_robot_zones:
            human_robot_zones[zone]['robot_active'] = False
            human_robot_zones[zone]['safety_level'] = 'emergency'
        return {'action': 'all_stop', 'message': 'Emergency stop activated for all systems'}

# IoT Sensor Simulation
class IoTSensorSimulator:
    def __init__(self):
        self.sensors = {
            'temperature': {'min': 20, 'max': 80, 'current': 45},
            'vibration': {'min': 0, 'max': 10, 'current': 2.5},
            'pressure': {'min': 1, 'max': 5, 'current': 2.8},
            'proximity': {'min': 0.1, 'max': 5.0, 'current': 2.0}
        }
        
    def generate_sensor_data(self, equipment_id):
        """Generate realistic sensor data with some randomness"""
        data = {
            'equipment_id': equipment_id,
            'timestamp': datetime.now().isoformat(),
            'temperature': self.sensors['temperature']['current'] + random.uniform(-5, 5),
            'vibration': max(0, self.sensors['vibration']['current'] + random.uniform(-1, 1)),
            'pressure': max(0, self.sensors['pressure']['current'] + random.uniform(-0.5, 0.5)),
            'proximity': max(0.1, self.sensors['proximity']['current'] + random.uniform(-1, 1))
        }
        
        # Simulate equipment degradation over time
        if random.random() < 0.1:  # 10% chance of slight degradation
            self.sensors['vibration']['current'] += 0.1
            self.sensors['temperature']['current'] += 0.5
            
        return data

# Initialize AI systems
predictive_ai = PredictiveMaintenanceAI()
safety_ai = PhysicalAISafety()
sensor_simulator = IoTSensorSimulator()

# Background thread for continuous sensor monitoring
def sensor_monitoring_thread():
    """Background thread that continuously monitors sensors"""
    while True:
        try:
            # Generate sensor data for all equipment
            for equipment_id in equipment_status.keys():
                sensor_data = sensor_simulator.generate_sensor_data(equipment_id)
                
                # Add to predictive maintenance AI
                predictive_ai.add_sensor_data(sensor_data)
                
                # Check for anomalies
                is_anomaly, anomaly_score = predictive_ai.predict_anomaly(sensor_data)
                
                if is_anomaly:
                    equipment_status[equipment_id]['health_score'] = max(0, 
                        equipment_status[equipment_id]['health_score'] - 5)
                    logging.warning(f"Anomaly detected in {equipment_id}: score {anomaly_score:.2f}")
                
                # Simulate human detection in zones
                for zone_id in human_robot_zones.keys():
                    if random.random() < 0.05:  # 5% chance of human detection
                        human_count = random.randint(0, 3)
                        human_robot_zones[zone_id]['humans_present'] = human_count
                        safety_response = safety_ai.human_detection_protocol(zone_id, human_count)
                        logging.info(f"Safety protocol: {safety_response['message']}")
                
                # Add to queue for real-time display
                if not sensor_data_queue.full():
                    sensor_data_queue.put(sensor_data)
                    
            # Retrain AI model periodically
            if len(predictive_ai.historical_data) >= 20 and not predictive_ai.is_trained:
                predictive_ai.train_model()
                logging.info("Predictive maintenance AI model trained")
                
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logging.error(f"Error in sensor monitoring: {e}")
            time.sleep(5)

# Start background monitoring
monitoring_thread = threading.Thread(target=sensor_monitoring_thread, daemon=True)
monitoring_thread.start()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/equipment_status')
def get_equipment_status():
    """Get current equipment status"""
    return jsonify(equipment_status)

@app.route('/api/safety_zones')
def get_safety_zones():
    """Get human-robot collaboration zones status"""
    return jsonify(human_robot_zones)

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get latest sensor data"""
    data = []
    while not sensor_data_queue.empty() and len(data) < 10:
        try:
            data.append(sensor_data_queue.get_nowait())
        except queue.Empty:
            break
    return jsonify(data)

@app.route('/api/maintenance_prediction/<equipment_id>')
def get_maintenance_prediction(equipment_id):
    """Get maintenance prediction for specific equipment"""
    if equipment_id in equipment_status:
        health_score = equipment_status[equipment_id]['health_score']
        
        # Calculate days until maintenance based on health score
        if health_score > 90:
            days_until_maintenance = 30
            priority = 'low'
        elif health_score > 70:
            days_until_maintenance = 14
            priority = 'medium'
        else:
            days_until_maintenance = 7
            priority = 'high'
            
        return jsonify({
            'equipment_id': equipment_id,
            'health_score': health_score,
            'days_until_maintenance': days_until_maintenance,
            'priority': priority,
            'recommended_actions': [
                'Check vibration sensors',
                'Inspect temperature regulation',
                'Verify pressure systems'
            ]
        })
    else:
        return jsonify({'error': 'Equipment not found'}), 404

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop all systems"""
    response = safety_ai.emergency_stop_protocol()
    return jsonify(response)

@app.route('/api/reset_systems', methods=['POST'])
def reset_systems():
    """Reset all systems to normal operation"""
    for zone in human_robot_zones:
        human_robot_zones[zone]['robot_active'] = True
        human_robot_zones[zone]['safety_level'] = 'safe'
        human_robot_zones[zone]['humans_present'] = 0
        
    return jsonify({'message': 'All systems reset to normal operation'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)