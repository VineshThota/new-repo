from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class IoTSensorData:
    """IoT sensor data structure"""
    sensor_id: str
    equipment_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    rotation_speed: float
    power_consumption: float
    noise_level: float

@dataclass
class MaintenanceAlert:
    """Maintenance alert structure"""
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
            "PUMP_001", "MOTOR_002", "COMPRESSOR_003", 
            "CONVEYOR_004", "ROBOT_ARM_005", "CNC_MACHINE_006"
        ]
        self.sensor_data_buffer = []
        self.is_running = False
        
    def generate_sensor_data(self, equipment_id: str) -> IoTSensorData:
        """Generate realistic sensor data with anomaly patterns"""
        base_time = datetime.now()
        
        # Normal operating ranges
        normal_ranges = {
            "temperature": (20, 80),
            "vibration": (0.1, 2.0),
            "pressure": (1.0, 5.0),
            "rotation_speed": (1000, 3000),
            "power_consumption": (50, 200),
            "noise_level": (30, 70)
        }
        
        # Introduce anomalies for predictive maintenance scenarios
        anomaly_factor = 1.0
        if random.random() < 0.15:  # 15% chance of anomaly
            anomaly_factor = random.uniform(1.5, 3.0)
            
        sensor_data = IoTSensorData(
            sensor_id=f"SENSOR_{equipment_id}",
            equipment_id=equipment_id,
            timestamp=base_time,
            temperature=random.uniform(*normal_ranges["temperature"]) * anomaly_factor,
            vibration=random.uniform(*normal_ranges["vibration"]) * anomaly_factor,
            pressure=random.uniform(*normal_ranges["pressure"]) * anomaly_factor,
            rotation_speed=random.uniform(*normal_ranges["rotation_speed"]) / anomaly_factor,
            power_consumption=random.uniform(*normal_ranges["power_consumption"]) * anomaly_factor,
            noise_level=random.uniform(*normal_ranges["noise_level"]) * anomaly_factor
        )
        
        return sensor_data
    
    def start_simulation(self):
        """Start continuous sensor data generation"""
        self.is_running = True
        
        def simulate():
            while self.is_running:
                for equipment_id in self.equipment_list:
                    sensor_data = self.generate_sensor_data(equipment_id)
                    self.sensor_data_buffer.append(sensor_data)
                    
                    # Keep buffer size manageable
                    if len(self.sensor_data_buffer) > 1000:
                        self.sensor_data_buffer = self.sensor_data_buffer[-500:]
                        
                time.sleep(2)  # Generate data every 2 seconds
                
        thread = threading.Thread(target=simulate)
        thread.daemon = True
        thread.start()
        
    def stop_simulation(self):
        """Stop sensor data generation"""
        self.is_running = False
        
    def get_latest_data(self, equipment_id: Optional[str] = None) -> List[IoTSensorData]:
        """Get latest sensor data"""
        if equipment_id:
            return [data for data in self.sensor_data_buffer if data.equipment_id == equipment_id][-10:]
        return self.sensor_data_buffer[-50:]

class PredictiveMaintenanceAI:
    """AI engine for predictive maintenance using machine learning"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'temperature', 'vibration', 'pressure', 
            'rotation_speed', 'power_consumption', 'noise_level'
        ]
        
    def prepare_training_data(self) -> tuple:
        """Generate synthetic training data for the ML models"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate normal operating data
        normal_data = np.random.normal(0, 1, (n_samples * 0.8, 6))
        normal_labels = np.zeros(int(n_samples * 0.8))
        
        # Generate failure condition data
        failure_data = np.random.normal(2, 1.5, (int(n_samples * 0.2), 6))
        failure_labels = np.ones(int(n_samples * 0.2))
        
        # Combine data
        X = np.vstack([normal_data, failure_data])
        y = np.hstack([normal_labels, failure_labels])
        
        return X, y
    
    def train_models(self):
        """Train the predictive maintenance models"""
        logger.info("Training predictive maintenance models...")
        
        X, y = self.prepare_training_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train failure predictor
        self.failure_predictor.fit(X_scaled, y)
        
        self.is_trained = True
        logger.info("Models trained successfully")
        
    def sensor_data_to_features(self, sensor_data: IoTSensorData) -> np.array:
        """Convert sensor data to feature vector"""
        return np.array([
            sensor_data.temperature,
            sensor_data.vibration,
            sensor_data.pressure,
            sensor_data.rotation_speed,
            sensor_data.power_consumption,
            sensor_data.noise_level
        ]).reshape(1, -1)
    
    def detect_anomaly(self, sensor_data: IoTSensorData) -> Dict:
        """Detect anomalies in sensor data"""
        if not self.is_trained:
            self.train_models()
            
        features = self.sensor_data_to_features(sensor_data)
        features_scaled = self.scaler.transform(features)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # Failure prediction
        failure_probability = self.failure_predictor.predict_proba(features_scaled)[0][1]
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'failure_probability': float(failure_probability),
            'risk_level': self._calculate_risk_level(failure_probability)
        }
    
    def _calculate_risk_level(self, failure_probability: float) -> str:
        """Calculate risk level based on failure probability"""
        if failure_probability > 0.8:
            return "CRITICAL"
        elif failure_probability > 0.6:
            return "HIGH"
        elif failure_probability > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

class PhysicalAIRobotController:
    """Physical AI controller for robotic maintenance assistance"""
    
    def __init__(self):
        self.robot_status = {
            "MAINTENANCE_ROBOT_01": {"status": "idle", "location": "dock", "battery": 100},
            "INSPECTION_DRONE_01": {"status": "idle", "location": "dock", "battery": 95},
            "REPAIR_ARM_01": {"status": "idle", "location": "station_A", "battery": 100}
        }
        
    def dispatch_robot(self, equipment_id: str, maintenance_type: str) -> Dict:
        """Dispatch appropriate robot for maintenance task"""
        robot_assignments = {
            "inspection": "INSPECTION_DRONE_01",
            "repair": "REPAIR_ARM_01",
            "maintenance": "MAINTENANCE_ROBOT_01"
        }
        
        robot_id = robot_assignments.get(maintenance_type, "MAINTENANCE_ROBOT_01")
        
        if self.robot_status[robot_id]["status"] == "idle":
            self.robot_status[robot_id]["status"] = "dispatched"
            self.robot_status[robot_id]["location"] = f"en_route_to_{equipment_id}"
            
            return {
                "success": True,
                "robot_id": robot_id,
                "estimated_arrival": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "task_type": maintenance_type
            }
        else:
            return {
                "success": False,
                "message": f"Robot {robot_id} is currently {self.robot_status[robot_id]['status']}",
                "alternative_robots": self._find_available_robots()
            }
    
    def _find_available_robots(self) -> List[str]:
        """Find available robots"""
        return [robot_id for robot_id, status in self.robot_status.items() 
                if status["status"] == "idle"]
    
    def get_robot_status(self) -> Dict:
        """Get current status of all robots"""
        return self.robot_status
    
    def complete_task(self, robot_id: str) -> Dict:
        """Mark robot task as completed"""
        if robot_id in self.robot_status:
            self.robot_status[robot_id]["status"] = "idle"
            self.robot_status[robot_id]["location"] = "dock"
            return {"success": True, "message": f"Task completed by {robot_id}"}
        return {"success": False, "message": "Robot not found"}

class SmartManufacturingSystem:
    """Main system orchestrating IoT, AI, and Physical AI components"""
    
    def __init__(self):
        self.iot_simulator = IoTSensorSimulator()
        self.ai_engine = PredictiveMaintenanceAI()
        self.robot_controller = PhysicalAIRobotController()
        self.maintenance_alerts = []
        
    def start_system(self):
        """Start the complete smart manufacturing system"""
        logger.info("Starting Smart Manufacturing Physical AI System")
        self.iot_simulator.start_simulation()
        self.ai_engine.train_models()
        
        # Start continuous monitoring
        self._start_monitoring()
        
    def _start_monitoring(self):
        """Start continuous monitoring and analysis"""
        def monitor():
            while True:
                latest_data = self.iot_simulator.get_latest_data()
                
                for sensor_data in latest_data[-6:]:  # Process last 6 readings
                    analysis = self.ai_engine.detect_anomaly(sensor_data)
                    
                    if analysis['failure_probability'] > 0.7:
                        alert = self._generate_maintenance_alert(sensor_data, analysis)
                        self.maintenance_alerts.append(alert)
                        
                        # Auto-dispatch robot for critical issues
                        if analysis['risk_level'] == 'CRITICAL':
                            self.robot_controller.dispatch_robot(
                                sensor_data.equipment_id, "inspection"
                            )
                            
                time.sleep(10)  # Monitor every 10 seconds
                
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()
        
    def _generate_maintenance_alert(self, sensor_data: IoTSensorData, analysis: Dict) -> MaintenanceAlert:
        """Generate maintenance alert based on analysis"""
        severity_map = {
            "CRITICAL": "Immediate action required",
            "HIGH": "Schedule maintenance within 24 hours",
            "MEDIUM": "Schedule maintenance within 1 week",
            "LOW": "Monitor closely"
        }
        
        predicted_failure_time = datetime.now() + timedelta(
            hours=24 * (1 - analysis['failure_probability'])
        )
        
        return MaintenanceAlert(
            equipment_id=sensor_data.equipment_id,
            alert_type="Predictive Maintenance",
            severity=analysis['risk_level'],
            predicted_failure_time=predicted_failure_time,
            recommended_action=severity_map[analysis['risk_level']],
            confidence_score=analysis['failure_probability']
        )
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        latest_data = self.iot_simulator.get_latest_data()
        robot_status = self.robot_controller.get_robot_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_equipment": len(self.iot_simulator.equipment_list),
            "active_sensors": len(latest_data),
            "active_alerts": len([alert for alert in self.maintenance_alerts 
                                 if alert.severity in ['CRITICAL', 'HIGH']]),
            "robot_status": robot_status,
            "system_health": "OPERATIONAL"
        }

# Initialize the system
smart_manufacturing = SmartManufacturingSystem()

# Flask Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the smart manufacturing system"""
    try:
        smart_manufacturing.start_system()
        return jsonify({"success": True, "message": "System started successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/system/status')
def get_system_status():
    """Get current system status"""
    return jsonify(smart_manufacturing.get_system_status())

@app.route('/api/sensors/data')
def get_sensor_data():
    """Get latest sensor data"""
    equipment_id = request.args.get('equipment_id')
    data = smart_manufacturing.iot_simulator.get_latest_data(equipment_id)
    
    # Convert to JSON-serializable format
    sensor_data_json = []
    for sensor_data in data:
        sensor_data_json.append({
            "sensor_id": sensor_data.sensor_id,
            "equipment_id": sensor_data.equipment_id,
            "timestamp": sensor_data.timestamp.isoformat(),
            "temperature": sensor_data.temperature,
            "vibration": sensor_data.vibration,
            "pressure": sensor_data.pressure,
            "rotation_speed": sensor_data.rotation_speed,
            "power_consumption": sensor_data.power_consumption,
            "noise_level": sensor_data.noise_level
        })
    
    return jsonify(sensor_data_json)

@app.route('/api/maintenance/alerts')
def get_maintenance_alerts():
    """Get current maintenance alerts"""
    alerts_json = []
    for alert in smart_manufacturing.maintenance_alerts[-20:]:  # Last 20 alerts
        alerts_json.append({
            "equipment_id": alert.equipment_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "predicted_failure_time": alert.predicted_failure_time.isoformat(),
            "recommended_action": alert.recommended_action,
            "confidence_score": alert.confidence_score
        })
    
    return jsonify(alerts_json)

@app.route('/api/robots/status')
def get_robot_status():
    """Get robot status"""
    return jsonify(smart_manufacturing.robot_controller.get_robot_status())

@app.route('/api/robots/dispatch', methods=['POST'])
def dispatch_robot():
    """Dispatch robot for maintenance task"""
    data = request.get_json()
    equipment_id = data.get('equipment_id')
    maintenance_type = data.get('maintenance_type', 'inspection')
    
    result = smart_manufacturing.robot_controller.dispatch_robot(equipment_id, maintenance_type)
    return jsonify(result)

@app.route('/api/analytics/equipment/<equipment_id>')
def get_equipment_analytics(equipment_id):
    """Get analytics for specific equipment"""
    sensor_data = smart_manufacturing.iot_simulator.get_latest_data(equipment_id)
    
    if not sensor_data:
        return jsonify({"error": "No data found for equipment"}), 404
    
    # Calculate analytics
    latest = sensor_data[-1]
    analysis = smart_manufacturing.ai_engine.detect_anomaly(latest)
    
    analytics = {
        "equipment_id": equipment_id,
        "current_status": "NORMAL" if not analysis['is_anomaly'] else "ANOMALY_DETECTED",
        "failure_probability": analysis['failure_probability'],
        "risk_level": analysis['risk_level'],
        "last_updated": latest.timestamp.isoformat(),
        "sensor_readings": {
            "temperature": latest.temperature,
            "vibration": latest.vibration,
            "pressure": latest.pressure,
            "rotation_speed": latest.rotation_speed,
            "power_consumption": latest.power_consumption,
            "noise_level": latest.noise_level
        }
    }
    
    return jsonify(analytics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)