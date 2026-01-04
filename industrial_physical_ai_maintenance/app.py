from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import sqlite3
import os

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'industrial_physical_ai_2026'
DATABASE = 'maintenance_system.db'

@dataclass
class IoTSensor:
    """IoT Sensor data structure"""
    sensor_id: str
    equipment_id: str
    sensor_type: str  # temperature, vibration, pressure, current, voltage
    location: str
    status: str
    last_reading: float
    timestamp: datetime
    threshold_min: float
    threshold_max: float

@dataclass
class Equipment:
    """Industrial Equipment data structure"""
    equipment_id: str
    name: str
    type: str  # pump, motor, conveyor, compressor
    location: str
    status: str  # operational, maintenance_needed, critical, offline
    last_maintenance: datetime
    next_scheduled_maintenance: datetime
    health_score: float
    sensors: List[str]

@dataclass
class MaintenanceRobot:
    """Physical AI Robot for maintenance tasks"""
    robot_id: str
    name: str
    type: str  # inspection, repair, cleaning, lubrication
    status: str  # idle, active, charging, maintenance
    current_task: Optional[str]
    location: str
    battery_level: float
    capabilities: List[str]

class IoTSensorNetwork:
    """Simulates IoT sensor network for industrial equipment"""
    
    def __init__(self):
        self.sensors = self._initialize_sensors()
        self.running = False
        self.data_buffer = []
        
    def _initialize_sensors(self) -> List[IoTSensor]:
        """Initialize IoT sensors for different equipment"""
        sensors = []
        equipment_types = ['pump', 'motor', 'conveyor', 'compressor']
        sensor_types = ['temperature', 'vibration', 'pressure', 'current', 'voltage']
        
        for i in range(20):  # 20 sensors across different equipment
            equipment_id = f"EQ_{random.choice(equipment_types).upper()}_{i//4 + 1:03d}"
            sensor = IoTSensor(
                sensor_id=f"SENSOR_{i+1:03d}",
                equipment_id=equipment_id,
                sensor_type=random.choice(sensor_types),
                location=f"Floor_{random.randint(1,3)}_Zone_{random.choice(['A','B','C'])}",
                status="active",
                last_reading=0.0,
                timestamp=datetime.now(),
                threshold_min=self._get_threshold_min(random.choice(sensor_types)),
                threshold_max=self._get_threshold_max(random.choice(sensor_types))
            )
            sensors.append(sensor)
        return sensors
    
    def _get_threshold_min(self, sensor_type: str) -> float:
        thresholds = {
            'temperature': 20.0,
            'vibration': 0.1,
            'pressure': 10.0,
            'current': 5.0,
            'voltage': 220.0
        }
        return thresholds.get(sensor_type, 0.0)
    
    def _get_threshold_max(self, sensor_type: str) -> float:
        thresholds = {
            'temperature': 80.0,
            'vibration': 5.0,
            'pressure': 100.0,
            'current': 50.0,
            'voltage': 240.0
        }
        return thresholds.get(sensor_type, 100.0)
    
    def generate_sensor_reading(self, sensor: IoTSensor) -> float:
        """Generate realistic sensor readings with anomaly simulation"""
        base_values = {
            'temperature': 45.0,
            'vibration': 1.5,
            'pressure': 55.0,
            'current': 25.0,
            'voltage': 230.0
        }
        
        base_value = base_values.get(sensor.sensor_type, 50.0)
        
        # Normal operation with small variations
        normal_reading = base_value + random.gauss(0, base_value * 0.05)
        
        # Simulate anomalies (5% chance)
        if random.random() < 0.05:
            anomaly_factor = random.choice([0.7, 1.3])  # 30% deviation
            normal_reading *= anomaly_factor
        
        return round(normal_reading, 2)
    
    def start_monitoring(self):
        """Start continuous sensor monitoring"""
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            for sensor in self.sensors:
                reading = self.generate_sensor_reading(sensor)
                sensor.last_reading = reading
                sensor.timestamp = datetime.now()
                
                # Store reading in buffer
                self.data_buffer.append({
                    'sensor_id': sensor.sensor_id,
                    'equipment_id': sensor.equipment_id,
                    'sensor_type': sensor.sensor_type,
                    'reading': reading,
                    'timestamp': sensor.timestamp.isoformat(),
                    'anomaly': reading < sensor.threshold_min or reading > sensor.threshold_max
                })
                
                # Keep buffer size manageable
                if len(self.data_buffer) > 1000:
                    self.data_buffer = self.data_buffer[-500:]
            
            time.sleep(2)  # Update every 2 seconds

class PredictiveMaintenanceAI:
    """AI system for predictive maintenance using machine learning"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_models()
    
    def _train_models(self):
        """Train AI models with synthetic historical data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: temperature, vibration, pressure, current, voltage, operating_hours
        X = np.random.normal(0, 1, (n_samples, 6))
        
        # Simulate normal and anomalous patterns
        normal_mask = np.random.random(n_samples) > 0.1
        X[~normal_mask] *= 3  # Anomalous readings are more extreme
        
        # Train anomaly detector
        self.anomaly_detector.fit(X)
        
        # Generate failure prediction targets (days until failure)
        y_failure = np.random.exponential(30, n_samples)  # Average 30 days until failure
        y_failure[~normal_mask] /= 3  # Anomalous equipment fails sooner
        
        # Train failure predictor
        X_scaled = self.scaler.fit_transform(X)
        self.failure_predictor.fit(X_scaled, y_failure)
        
        self.is_trained = True
    
    def detect_anomaly(self, sensor_data: Dict) -> Dict:
        """Detect anomalies in sensor readings"""
        if not self.is_trained:
            return {'anomaly': False, 'confidence': 0.0}
        
        # Prepare features
        features = np.array([[
            sensor_data.get('temperature', 45.0),
            sensor_data.get('vibration', 1.5),
            sensor_data.get('pressure', 55.0),
            sensor_data.get('current', 25.0),
            sensor_data.get('voltage', 230.0),
            sensor_data.get('operating_hours', 100.0)
        ]])
        
        # Detect anomaly
        anomaly_score = self.anomaly_detector.decision_function(features)[0]
        is_anomaly = self.anomaly_detector.predict(features)[0] == -1
        
        return {
            'anomaly': is_anomaly,
            'confidence': abs(anomaly_score),
            'severity': 'high' if anomaly_score < -0.5 else 'medium' if anomaly_score < -0.2 else 'low'
        }
    
    def predict_failure(self, sensor_data: Dict) -> Dict:
        """Predict time until equipment failure"""
        if not self.is_trained:
            return {'days_until_failure': 30.0, 'confidence': 0.0}
        
        # Prepare features
        features = np.array([[
            sensor_data.get('temperature', 45.0),
            sensor_data.get('vibration', 1.5),
            sensor_data.get('pressure', 55.0),
            sensor_data.get('current', 25.0),
            sensor_data.get('voltage', 230.0),
            sensor_data.get('operating_hours', 100.0)
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict failure
        days_until_failure = self.failure_predictor.predict(features_scaled)[0]
        
        # Calculate confidence based on model uncertainty
        confidence = min(1.0, max(0.0, 1.0 - (abs(days_until_failure - 30) / 100)))
        
        return {
            'days_until_failure': max(0.1, days_until_failure),
            'confidence': confidence,
            'risk_level': 'critical' if days_until_failure < 7 else 'high' if days_until_failure < 14 else 'medium' if days_until_failure < 30 else 'low'
        }

class PhysicalAIRobotSystem:
    """Physical AI system for coordinating maintenance robots"""
    
    def __init__(self):
        self.robots = self._initialize_robots()
        self.task_queue = []
        self.active_tasks = {}
    
    def _initialize_robots(self) -> List[MaintenanceRobot]:
        """Initialize maintenance robots"""
        robots = [
            MaintenanceRobot(
                robot_id="ROBOT_001",
                name="Inspector Alpha",
                type="inspection",
                status="idle",
                current_task=None,
                location="Charging_Station_A",
                battery_level=100.0,
                capabilities=["visual_inspection", "thermal_imaging", "vibration_analysis"]
            ),
            MaintenanceRobot(
                robot_id="ROBOT_002",
                name="Repair Beta",
                type="repair",
                status="idle",
                current_task=None,
                location="Workshop_B",
                battery_level=85.0,
                capabilities=["component_replacement", "tightening", "calibration"]
            ),
            MaintenanceRobot(
                robot_id="ROBOT_003",
                name="Cleaner Gamma",
                type="cleaning",
                status="active",
                current_task="routine_cleaning",
                location="Floor_2_Zone_A",
                battery_level=60.0,
                capabilities=["surface_cleaning", "debris_removal", "filter_replacement"]
            ),
            MaintenanceRobot(
                robot_id="ROBOT_004",
                name="Lubricator Delta",
                type="lubrication",
                status="idle",
                current_task=None,
                location="Maintenance_Bay_C",
                battery_level=95.0,
                capabilities=["oil_application", "grease_dispensing", "fluid_level_check"]
            )
        ]
        return robots
    
    def assign_maintenance_task(self, equipment_id: str, task_type: str, priority: str) -> Dict:
        """Assign maintenance task to appropriate robot"""
        # Find suitable robot
        suitable_robots = [r for r in self.robots if r.status == "idle" and task_type in r.capabilities]
        
        if not suitable_robots:
            # Add to queue if no robot available
            task = {
                'task_id': f"TASK_{len(self.task_queue)+1:03d}",
                'equipment_id': equipment_id,
                'task_type': task_type,
                'priority': priority,
                'created_at': datetime.now().isoformat(),
                'status': 'queued'
            }
            self.task_queue.append(task)
            return {'success': False, 'message': 'Task queued - no available robots', 'task': task}
        
        # Assign to robot with highest battery level
        selected_robot = max(suitable_robots, key=lambda r: r.battery_level)
        
        task = {
            'task_id': f"TASK_{len(self.active_tasks)+1:03d}",
            'equipment_id': equipment_id,
            'task_type': task_type,
            'priority': priority,
            'robot_id': selected_robot.robot_id,
            'created_at': datetime.now().isoformat(),
            'estimated_completion': (datetime.now() + timedelta(minutes=random.randint(15, 60))).isoformat(),
            'status': 'assigned'
        }
        
        # Update robot status
        selected_robot.status = "active"
        selected_robot.current_task = task['task_id']
        
        # Add to active tasks
        self.active_tasks[task['task_id']] = task
        
        return {'success': True, 'message': 'Task assigned successfully', 'task': task}
    
    def get_robot_status(self) -> List[Dict]:
        """Get current status of all robots"""
        return [{
            'robot_id': robot.robot_id,
            'name': robot.name,
            'type': robot.type,
            'status': robot.status,
            'current_task': robot.current_task,
            'location': robot.location,
            'battery_level': robot.battery_level,
            'capabilities': robot.capabilities
        } for robot in self.robots]

# Initialize global systems
iot_network = IoTSensorNetwork()
ai_system = PredictiveMaintenanceAI()
robot_system = PhysicalAIRobotSystem()

# Database initialization
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT,
            equipment_id TEXT,
            sensor_type TEXT,
            reading REAL,
            timestamp TEXT,
            anomaly BOOLEAN
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maintenance_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            equipment_id TEXT,
            task_type TEXT,
            priority TEXT,
            robot_id TEXT,
            status TEXT,
            created_at TEXT,
            completed_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Flask Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/sensors')
def get_sensors():
    """Get current sensor data"""
    sensor_data = []
    for sensor in iot_network.sensors:
        sensor_data.append({
            'sensor_id': sensor.sensor_id,
            'equipment_id': sensor.equipment_id,
            'sensor_type': sensor.sensor_type,
            'location': sensor.location,
            'status': sensor.status,
            'last_reading': sensor.last_reading,
            'timestamp': sensor.timestamp.isoformat(),
            'threshold_min': sensor.threshold_min,
            'threshold_max': sensor.threshold_max,
            'anomaly': sensor.last_reading < sensor.threshold_min or sensor.last_reading > sensor.threshold_max
        })
    return jsonify(sensor_data)

@app.route('/api/equipment/<equipment_id>/analysis')
def analyze_equipment(equipment_id):
    """Analyze specific equipment using AI"""
    # Get recent sensor data for equipment
    equipment_sensors = [s for s in iot_network.sensors if s.equipment_id == equipment_id]
    
    if not equipment_sensors:
        return jsonify({'error': 'Equipment not found'}), 404
    
    # Prepare sensor data for AI analysis
    sensor_data = {}
    for sensor in equipment_sensors:
        sensor_data[sensor.sensor_type] = sensor.last_reading
    
    # Add operating hours (simulated)
    sensor_data['operating_hours'] = random.randint(50, 500)
    
    # Run AI analysis
    anomaly_result = ai_system.detect_anomaly(sensor_data)
    failure_prediction = ai_system.predict_failure(sensor_data)
    
    return jsonify({
        'equipment_id': equipment_id,
        'sensor_data': sensor_data,
        'anomaly_detection': anomaly_result,
        'failure_prediction': failure_prediction,
        'analysis_timestamp': datetime.now().isoformat()
    })

@app.route('/api/maintenance/schedule', methods=['POST'])
def schedule_maintenance():
    """Schedule maintenance task"""
    data = request.get_json()
    
    equipment_id = data.get('equipment_id')
    task_type = data.get('task_type', 'inspection')
    priority = data.get('priority', 'medium')
    
    if not equipment_id:
        return jsonify({'error': 'Equipment ID required'}), 400
    
    # Assign task to robot
    result = robot_system.assign_maintenance_task(equipment_id, task_type, priority)
    
    return jsonify(result)

@app.route('/api/robots')
def get_robots():
    """Get robot status"""
    return jsonify(robot_system.get_robot_status())

@app.route('/api/tasks')
def get_tasks():
    """Get active and queued tasks"""
    return jsonify({
        'active_tasks': list(robot_system.active_tasks.values()),
        'queued_tasks': robot_system.task_queue
    })

@app.route('/api/realtime-data')
def get_realtime_data():
    """Get real-time sensor data for dashboard"""
    # Get recent data from buffer
    recent_data = iot_network.data_buffer[-50:] if iot_network.data_buffer else []
    
    return jsonify({
        'sensor_readings': recent_data,
        'timestamp': datetime.now().isoformat(),
        'total_sensors': len(iot_network.sensors),
        'active_robots': len([r for r in robot_system.robots if r.status == 'active']),
        'pending_tasks': len(robot_system.task_queue)
    })

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Start IoT monitoring
    iot_network.start_monitoring()
    
    print("🏭 Industrial Physical AI Predictive Maintenance System Starting...")
    print("🤖 Physical AI Robots: 4 units initialized")
    print("📡 IoT Sensors: 20 sensors monitoring equipment")
    print("🧠 AI Models: Anomaly detection and failure prediction ready")
    print("🌐 Web Dashboard: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)