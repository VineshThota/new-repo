from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import json
import time
import random
from datetime import datetime
import threading
from collections import deque
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables for IoT data and AI processing
iot_data = {
    'temperature': deque(maxlen=100),
    'vibration': deque(maxlen=100),
    'pressure': deque(maxlen=100),
    'timestamps': deque(maxlen=100)
}

robot_status = {
    'position': {'x': 0, 'y': 0, 'z': 0},
    'battery': 100,
    'status': 'idle',
    'current_task': 'standby',
    'inspection_progress': 0
}

anomalies_detected = []
inspection_results = []

class IoTSensorSimulator:
    """Simulates IoT sensors for industrial monitoring"""
    
    def __init__(self):
        self.running = False
        self.thread = None
    
    def start_simulation(self):
        self.running = True
        self.thread = threading.Thread(target=self._simulate_sensors)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_simulation(self):
        self.running = False
    
    def _simulate_sensors(self):
        while self.running:
            # Simulate sensor readings with occasional anomalies
            temp = 25 + random.gauss(0, 2) + (5 if random.random() < 0.1 else 0)
            vibration = 0.5 + random.gauss(0, 0.1) + (2 if random.random() < 0.05 else 0)
            pressure = 101.3 + random.gauss(0, 0.5) + (10 if random.random() < 0.08 else 0)
            
            timestamp = datetime.now().isoformat()
            
            iot_data['temperature'].append(temp)
            iot_data['vibration'].append(vibration)
            iot_data['pressure'].append(pressure)
            iot_data['timestamps'].append(timestamp)
            
            # Check for anomalies
            self._detect_anomalies(temp, vibration, pressure, timestamp)
            
            time.sleep(1)
    
    def _detect_anomalies(self, temp, vibration, pressure, timestamp):
        """AI-based anomaly detection"""
        anomaly_detected = False
        anomaly_type = []
        
        if temp > 35:  # Temperature threshold
            anomaly_detected = True
            anomaly_type.append('High Temperature')
        
        if vibration > 1.5:  # Vibration threshold
            anomaly_detected = True
            anomaly_type.append('Excessive Vibration')
        
        if pressure > 110 or pressure < 95:  # Pressure thresholds
            anomaly_detected = True
            anomaly_type.append('Pressure Anomaly')
        
        if anomaly_detected:
            anomaly = {
                'timestamp': timestamp,
                'type': ', '.join(anomaly_type),
                'severity': 'High' if len(anomaly_type) > 1 else 'Medium',
                'temperature': temp,
                'vibration': vibration,
                'pressure': pressure
            }
            anomalies_detected.append(anomaly)
            
            # Trigger robot inspection
            self._trigger_robot_inspection(anomaly)
    
    def _trigger_robot_inspection(self, anomaly):
        """Trigger Physical AI robot for inspection"""
        global robot_status
        if robot_status['status'] == 'idle':
            robot_status['status'] = 'inspecting'
            robot_status['current_task'] = f"Investigating {anomaly['type']}"
            robot_status['inspection_progress'] = 0
            
            # Start robot inspection thread
            inspection_thread = threading.Thread(target=self._robot_inspection_task, args=(anomaly,))
            inspection_thread.daemon = True
            inspection_thread.start()
    
    def _robot_inspection_task(self, anomaly):
        """Simulate robot inspection with Physical AI"""
        global robot_status
        
        # Simulate robot movement to inspection location
        target_x = random.randint(10, 50)
        target_y = random.randint(10, 50)
        
        for progress in range(0, 101, 10):
            robot_status['inspection_progress'] = progress
            robot_status['position']['x'] = target_x * (progress / 100)
            robot_status['position']['y'] = target_y * (progress / 100)
            robot_status['battery'] -= 0.5
            time.sleep(0.5)
        
        # Simulate AI computer vision analysis
        time.sleep(2)
        
        # Generate inspection result
        result = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_type': anomaly['type'],
            'location': f"({target_x}, {target_y})",
            'ai_analysis': self._generate_ai_analysis(anomaly),
            'recommendation': self._generate_recommendation(anomaly),
            'confidence': random.uniform(0.8, 0.99)
        }
        
        inspection_results.append(result)
        
        # Reset robot status
        robot_status['status'] = 'idle'
        robot_status['current_task'] = 'standby'
        robot_status['inspection_progress'] = 100
    
    def _generate_ai_analysis(self, anomaly):
        """Generate AI-based analysis using computer vision simulation"""
        analyses = {
            'High Temperature': 'Thermal imaging detected hotspot. Possible bearing failure or electrical issue.',
            'Excessive Vibration': 'Vibration pattern analysis indicates potential mechanical imbalance or loose components.',
            'Pressure Anomaly': 'Pressure sensor data suggests possible leak or blockage in the system.'
        }
        
        for anomaly_type in analyses:
            if anomaly_type in anomaly['type']:
                return analyses[anomaly_type]
        
        return 'AI analysis completed. No specific pattern identified.'
    
    def _generate_recommendation(self, anomaly):
        """Generate maintenance recommendations"""
        recommendations = {
            'High Temperature': 'Schedule immediate cooling system inspection and bearing replacement.',
            'Excessive Vibration': 'Perform mechanical alignment check and tighten all connections.',
            'Pressure Anomaly': 'Inspect seals and connections. Check for leaks or blockages.'
        }
        
        for anomaly_type in recommendations:
            if anomaly_type in anomaly['type']:
                return recommendations[anomaly_type]
        
        return 'Continue monitoring. Schedule routine maintenance check.'

class EdgeAIProcessor:
    """Edge AI processing for real-time decision making"""
    
    def __init__(self):
        self.model_loaded = True
        self.processing_queue = deque(maxlen=50)
    
    def process_sensor_data(self, sensor_data):
        """Process sensor data using edge AI algorithms"""
        # Simulate edge AI processing
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'raw_data': sensor_data,
            'ai_prediction': self._predict_failure_probability(sensor_data),
            'edge_processing_time': random.uniform(0.1, 0.5)
        }
        
        self.processing_queue.append(processed_data)
        return processed_data
    
    def _predict_failure_probability(self, sensor_data):
        """AI model for predicting equipment failure probability"""
        # Simulate ML model prediction
        temp_factor = max(0, (sensor_data.get('temperature', 25) - 25) / 20)
        vibration_factor = max(0, (sensor_data.get('vibration', 0.5) - 0.5) / 2)
        pressure_factor = abs(sensor_data.get('pressure', 101.3) - 101.3) / 20
        
        failure_probability = min(1.0, (temp_factor + vibration_factor + pressure_factor) / 3)
        
        return {
            'probability': failure_probability,
            'risk_level': 'High' if failure_probability > 0.7 else 'Medium' if failure_probability > 0.4 else 'Low',
            'factors': {
                'temperature': temp_factor,
                'vibration': vibration_factor,
                'pressure': pressure_factor
            }
        }

# Initialize components
iot_simulator = IoTSensorSimulator()
edge_ai = EdgeAIProcessor()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get current sensor data"""
    if len(iot_data['timestamps']) > 0:
        latest_data = {
            'temperature': list(iot_data['temperature'])[-10:],
            'vibration': list(iot_data['vibration'])[-10:],
            'pressure': list(iot_data['pressure'])[-10:],
            'timestamps': list(iot_data['timestamps'])[-10:]
        }
        return jsonify(latest_data)
    return jsonify({'error': 'No data available'})

@app.route('/api/robot_status')
def get_robot_status():
    """Get current robot status"""
    return jsonify(robot_status)

@app.route('/api/anomalies')
def get_anomalies():
    """Get detected anomalies"""
    return jsonify(anomalies_detected[-10:])  # Last 10 anomalies

@app.route('/api/inspection_results')
def get_inspection_results():
    """Get inspection results"""
    return jsonify(inspection_results[-10:])  # Last 10 results

@app.route('/api/edge_ai_status')
def get_edge_ai_status():
    """Get edge AI processing status"""
    return jsonify({
        'model_loaded': edge_ai.model_loaded,
        'queue_size': len(edge_ai.processing_queue),
        'recent_predictions': list(edge_ai.processing_queue)[-5:]
    })

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

@app.route('/api/manual_inspection', methods=['POST'])
def manual_inspection():
    """Trigger manual robot inspection"""
    global robot_status
    if robot_status['status'] == 'idle':
        fake_anomaly = {
            'type': 'Manual Inspection',
            'timestamp': datetime.now().isoformat()
        }
        iot_simulator._trigger_robot_inspection(fake_anomaly)
        return jsonify({'status': 'Manual inspection started'})
    else:
        return jsonify({'status': 'Robot is busy'})

if __name__ == '__main__':
    print("Smart Industrial Inspection Robot with Edge AI")
    print("Features:")
    print("- IoT Sensor Monitoring (Temperature, Vibration, Pressure)")
    print("- AI-based Anomaly Detection")
    print("- Physical AI Robot Inspection")
    print("- Edge Computing Processing")
    print("- Real-time Dashboard")
    print("\nStarting application...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)