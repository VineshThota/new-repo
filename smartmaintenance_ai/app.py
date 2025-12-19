from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2
import json
import datetime
import threading
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class SensorReading:
    """Data class for IoT sensor readings"""
    timestamp: datetime.datetime
    equipment_id: str
    temperature: float
    vibration: float
    pressure: float
    status: str = "normal"

class IoTSensorSimulator:
    """Simulates IoT sensors for equipment monitoring"""
    
    def __init__(self):
        self.equipment_list = ["Motor_001", "Pump_002", "Compressor_003", "Generator_004"]
        self.sensor_data = []
        self.running = False
        
    def generate_sensor_reading(self, equipment_id: str) -> SensorReading:
        """Generate realistic sensor readings with occasional anomalies"""
        base_temp = 75.0
        base_vibration = 2.5
        base_pressure = 14.7
        
        # Add random variations
        temp_variation = random.uniform(-5, 15)
        vibration_variation = random.uniform(-0.5, 2.0)
        pressure_variation = random.uniform(-1.0, 3.0)
        
        # Simulate equipment degradation over time
        degradation_factor = random.uniform(0.95, 1.05)
        
        temperature = (base_temp + temp_variation) * degradation_factor
        vibration = (base_vibration + vibration_variation) * degradation_factor
        pressure = (base_pressure + pressure_variation) * degradation_factor
        
        # Determine status based on thresholds
        status = "normal"
        if temperature > 95 or vibration > 4.0 or pressure > 18.0:
            status = "warning"
        if temperature > 105 or vibration > 5.0 or pressure > 20.0:
            status = "critical"
            
        return SensorReading(
            timestamp=datetime.datetime.now(),
            equipment_id=equipment_id,
            temperature=temperature,
            vibration=vibration,
            pressure=pressure,
            status=status
        )
    
    def start_monitoring(self):
        """Start continuous sensor monitoring"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                for equipment_id in self.equipment_list:
                    reading = self.generate_sensor_reading(equipment_id)
                    self.sensor_data.append(reading)
                    
                    # Keep only last 1000 readings
                    if len(self.sensor_data) > 1000:
                        self.sensor_data = self.sensor_data[-1000:]
                        
                time.sleep(2)  # Generate reading every 2 seconds
                
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop sensor monitoring"""
        self.running = False
        
    def get_latest_readings(self, limit: int = 50) -> List[SensorReading]:
        """Get latest sensor readings"""
        return self.sensor_data[-limit:] if self.sensor_data else []

class PredictiveMaintenanceAI:
    """AI system for predictive maintenance using machine learning"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['temperature', 'vibration', 'pressure', 'temp_trend', 'vib_trend']
        
    def prepare_training_data(self):
        """Generate synthetic training data for the ML model"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        temperature = np.random.normal(80, 10, n_samples)
        vibration = np.random.normal(3.0, 0.8, n_samples)
        pressure = np.random.normal(15.5, 2.0, n_samples)
        
        # Add trend features
        temp_trend = np.random.normal(0, 0.5, n_samples)
        vib_trend = np.random.normal(0, 0.2, n_samples)
        
        # Create labels based on thresholds and combinations
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            failure_score = 0
            if temperature[i] > 95: failure_score += 1
            if vibration[i] > 4.0: failure_score += 1
            if pressure[i] > 18.0: failure_score += 1
            if temp_trend[i] > 0.3: failure_score += 1
            if vib_trend[i] > 0.15: failure_score += 1
            
            # Label as failure if multiple indicators are present
            labels[i] = 1 if failure_score >= 2 else 0
            
        X = np.column_stack([temperature, vibration, pressure, temp_trend, vib_trend])
        return X, labels
        
    def train_model(self):
        """Train the predictive maintenance model"""
        X, y = self.prepare_training_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info("Predictive maintenance model trained successfully")
        
    def predict_failure(self, sensor_readings: List[SensorReading]) -> Dict[str, Any]:
        """Predict equipment failure probability"""
        if not self.is_trained:
            self.train_model()
            
        if len(sensor_readings) < 2:
            return {"probability": 0.0, "risk_level": "unknown", "recommendation": "Insufficient data"}
            
        # Extract features from latest readings
        latest = sensor_readings[-1]
        previous = sensor_readings[-2] if len(sensor_readings) > 1 else latest
        
        # Calculate trends
        temp_trend = latest.temperature - previous.temperature
        vib_trend = latest.vibration - previous.vibration
        
        # Prepare feature vector
        features = np.array([[
            latest.temperature,
            latest.vibration,
            latest.pressure,
            temp_trend,
            vib_trend
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        failure_probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Determine risk level
        if failure_probability < 0.3:
            risk_level = "low"
            recommendation = "Continue normal operation"
        elif failure_probability < 0.7:
            risk_level = "medium"
            recommendation = "Schedule preventive maintenance"
        else:
            risk_level = "high"
            recommendation = "Immediate maintenance required"
            
        return {
            "probability": float(failure_probability),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "equipment_id": latest.equipment_id
        }

class ComputerVisionInspector:
    """Computer vision system for equipment visual inspection"""
    
    def __init__(self):
        self.cascade_classifier = None
        
    def analyze_equipment_image(self, image_path: str = None) -> Dict[str, Any]:
        """Analyze equipment image for defects (simulated)"""
        # In a real implementation, this would process actual images
        # For demo purposes, we'll simulate the analysis
        
        defects_detected = random.choice([True, False])
        confidence = random.uniform(0.6, 0.95)
        
        if defects_detected:
            defect_types = random.sample(["corrosion", "wear", "misalignment", "leak"], 
                                       random.randint(1, 2))
        else:
            defect_types = []
            
        return {
            "defects_detected": defects_detected,
            "confidence": confidence,
            "defect_types": defect_types,
            "inspection_timestamp": datetime.datetime.now().isoformat()
        }
        
    def generate_inspection_report(self, equipment_id: str) -> Dict[str, Any]:
        """Generate comprehensive visual inspection report"""
        analysis = self.analyze_equipment_image()
        
        return {
            "equipment_id": equipment_id,
            "inspection_date": datetime.datetime.now().isoformat(),
            "visual_analysis": analysis,
            "overall_condition": "poor" if analysis["defects_detected"] else "good",
            "recommended_actions": [
                "Replace worn components" if "wear" in analysis["defect_types"] else None,
                "Apply anti-corrosion treatment" if "corrosion" in analysis["defect_types"] else None,
                "Realign equipment" if "misalignment" in analysis["defect_types"] else None,
                "Repair leak" if "leak" in analysis["defect_types"] else None
            ]
        }

class RoboticMaintenanceSystem:
    """Physical AI system for robotic maintenance interventions"""
    
    def __init__(self):
        self.robot_status = "idle"
        self.maintenance_queue = []
        self.current_task = None
        
    def schedule_maintenance_task(self, equipment_id: str, task_type: str, priority: str = "medium"):
        """Schedule a maintenance task for robotic execution"""
        task = {
            "id": f"task_{len(self.maintenance_queue) + 1}",
            "equipment_id": equipment_id,
            "task_type": task_type,
            "priority": priority,
            "scheduled_time": datetime.datetime.now(),
            "status": "queued"
        }
        
        self.maintenance_queue.append(task)
        
        # Sort by priority (high, medium, low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self.maintenance_queue.sort(key=lambda x: priority_order.get(x["priority"], 1))
        
        logger.info(f"Maintenance task scheduled: {task['id']} for {equipment_id}")
        
    def execute_maintenance_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate robotic maintenance task execution"""
        self.robot_status = "working"
        self.current_task = task
        
        # Simulate task execution time
        execution_time = random.uniform(30, 120)  # 30-120 seconds
        
        # Simulate task steps
        steps = [
            "Moving to equipment location",
            "Performing visual inspection",
            "Executing maintenance procedure",
            "Verifying completion",
            "Returning to base"
        ]
        
        result = {
            "task_id": task["id"],
            "equipment_id": task["equipment_id"],
            "start_time": datetime.datetime.now().isoformat(),
            "estimated_duration": execution_time,
            "steps": steps,
            "status": "in_progress"
        }
        
        # In a real system, this would control actual robotic hardware
        logger.info(f"Executing maintenance task: {task['id']}")
        
        return result
        
    def get_robot_status(self) -> Dict[str, Any]:
        """Get current robot status and queue information"""
        return {
            "status": self.robot_status,
            "current_task": self.current_task,
            "queue_length": len(self.maintenance_queue),
            "next_tasks": self.maintenance_queue[:3]  # Next 3 tasks
        }

# Initialize system components
sensor_simulator = IoTSensorSimulator()
predictive_ai = PredictiveMaintenanceAI()
vision_inspector = ComputerVisionInspector()
robotic_system = RoboticMaintenanceSystem()

# Start sensor monitoring
sensor_simulator.start_monitoring()

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/sensor-data')
def get_sensor_data():
    """API endpoint to get latest sensor readings"""
    readings = sensor_simulator.get_latest_readings(20)
    
    data = []
    for reading in readings:
        data.append({
            'timestamp': reading.timestamp.isoformat(),
            'equipment_id': reading.equipment_id,
            'temperature': reading.temperature,
            'vibration': reading.vibration,
            'pressure': reading.pressure,
            'status': reading.status
        })
    
    return jsonify(data)

@app.route('/api/predict-failure/<equipment_id>')
def predict_failure(equipment_id):
    """API endpoint for failure prediction"""
    readings = sensor_simulator.get_latest_readings()
    equipment_readings = [r for r in readings if r.equipment_id == equipment_id]
    
    if not equipment_readings:
        return jsonify({"error": "No data available for equipment"}), 404
        
    prediction = predictive_ai.predict_failure(equipment_readings)
    
    # Schedule maintenance if high risk
    if prediction["risk_level"] == "high":
        robotic_system.schedule_maintenance_task(
            equipment_id, 
            "preventive_maintenance", 
            "high"
        )
    
    return jsonify(prediction)

@app.route('/api/visual-inspection/<equipment_id>')
def visual_inspection(equipment_id):
    """API endpoint for visual inspection"""
    report = vision_inspector.generate_inspection_report(equipment_id)
    
    # Schedule maintenance if defects detected
    if report["visual_analysis"]["defects_detected"]:
        for defect in report["visual_analysis"]["defect_types"]:
            robotic_system.schedule_maintenance_task(
                equipment_id,
                f"repair_{defect}",
                "medium"
            )
    
    return jsonify(report)

@app.route('/api/robot-status')
def robot_status():
    """API endpoint for robot status"""
    return jsonify(robotic_system.get_robot_status())

@app.route('/api/schedule-maintenance', methods=['POST'])
def schedule_maintenance():
    """API endpoint to manually schedule maintenance"""
    data = request.json
    
    robotic_system.schedule_maintenance_task(
        data.get('equipment_id'),
        data.get('task_type', 'general_maintenance'),
        data.get('priority', 'medium')
    )
    
    return jsonify({"status": "success", "message": "Maintenance task scheduled"})

@app.route('/api/equipment-health')
def equipment_health():
    """API endpoint for overall equipment health summary"""
    readings = sensor_simulator.get_latest_readings()
    
    # Group by equipment
    equipment_health = {}
    for reading in readings:
        if reading.equipment_id not in equipment_health:
            equipment_health[reading.equipment_id] = []
        equipment_health[reading.equipment_id].append(reading)
    
    # Calculate health scores
    health_summary = []
    for equipment_id, equipment_readings in equipment_health.items():
        if equipment_readings:
            latest = equipment_readings[-1]
            prediction = predictive_ai.predict_failure(equipment_readings)
            
            health_summary.append({
                'equipment_id': equipment_id,
                'current_status': latest.status,
                'failure_probability': prediction['probability'],
                'risk_level': prediction['risk_level'],
                'last_reading': latest.timestamp.isoformat(),
                'temperature': latest.temperature,
                'vibration': latest.vibration,
                'pressure': latest.pressure
            })
    
    return jsonify(health_summary)

if __name__ == '__main__':
    logger.info("Starting SmartMaintenance AI System")
    app.run(debug=True, host='0.0.0.0', port=5000)