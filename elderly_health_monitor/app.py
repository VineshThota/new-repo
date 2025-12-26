#!/usr/bin/env python3
"""
Elderly Health Monitor - IoT + AI + Physical AI Application
Combines IoT sensors, machine learning algorithms, and physical world interactions
for comprehensive elderly care monitoring and predictive health analytics.

Trending Topic: IoT with AI/ML integration for personalized healthcare
Problem Addressed: Real-time health monitoring and predictive analytics for elderly care
Technology Stack: Python, Flask, scikit-learn, pandas, numpy
IoT Components: Simulated heart rate, blood pressure, temperature, motion sensors
AI Algorithms: Random Forest for health prediction, anomaly detection
Physical Interaction: Emergency alerts, medication reminders, fall detection
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
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

app = Flask(__name__)
app.secret_key = 'elderly_health_monitor_2025'

# Configuration
CONFIG = {
    'DATABASE_PATH': 'health_data.db',
    'MODEL_PATH': 'models/',
    'ALERT_EMAIL': 'caregiver@example.com',
    'EMERGENCY_THRESHOLD': 0.8,
    'SENSOR_INTERVAL': 5,  # seconds
    'PREDICTION_INTERVAL': 30  # seconds
}

@dataclass
class SensorReading:
    """IoT sensor data structure"""
    timestamp: datetime
    heart_rate: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    body_temperature: float
    motion_level: float
    sleep_quality: float
    medication_adherence: float
    patient_id: str

class IoTSensorSimulator:
    """Simulates IoT sensors for elderly health monitoring"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.is_running = False
        self.current_readings = {}
        self.baseline_vitals = {
            'heart_rate': 72,
            'bp_systolic': 120,
            'bp_diastolic': 80,
            'temperature': 98.6,
            'motion': 0.5
        }
    
    def generate_realistic_reading(self) -> SensorReading:
        """Generate realistic sensor readings with some variability"""
        # Add realistic variations and potential anomalies
        current_time = datetime.now()
        hour = current_time.hour
        
        # Circadian rhythm effects
        if 22 <= hour or hour <= 6:  # Night time
            hr_modifier = -5
            motion_modifier = -0.8
            temp_modifier = -0.5
        elif 6 < hour <= 12:  # Morning
            hr_modifier = 0
            motion_modifier = 0.3
            temp_modifier = 0
        else:  # Afternoon/Evening
            hr_modifier = 2
            motion_modifier = 0.5
            temp_modifier = 0.2
        
        # Generate readings with normal variation
        heart_rate = max(50, min(120, 
            self.baseline_vitals['heart_rate'] + hr_modifier + random.gauss(0, 5)))
        
        bp_sys = max(90, min(180, 
            self.baseline_vitals['bp_systolic'] + random.gauss(0, 8)))
        
        bp_dia = max(60, min(110, 
            self.baseline_vitals['bp_diastolic'] + random.gauss(0, 5)))
        
        temperature = max(96.0, min(102.0, 
            self.baseline_vitals['temperature'] + temp_modifier + random.gauss(0, 0.3)))
        
        motion = max(0, min(1, 
            self.baseline_vitals['motion'] + motion_modifier + random.gauss(0, 0.2)))
        
        # Sleep quality based on time and motion
        if 22 <= hour or hour <= 6:
            sleep_quality = max(0, min(1, 0.8 - motion * 0.5 + random.gauss(0, 0.1)))
        else:
            sleep_quality = 0.5  # Awake
        
        # Medication adherence (simulate missed doses occasionally)
        med_adherence = 1.0 if random.random() > 0.1 else 0.0
        
        return SensorReading(
            timestamp=current_time,
            heart_rate=heart_rate,
            blood_pressure_systolic=bp_sys,
            blood_pressure_diastolic=bp_dia,
            body_temperature=temperature,
            motion_level=motion,
            sleep_quality=sleep_quality,
            medication_adherence=med_adherence,
            patient_id=self.patient_id
        )
    
    def start_monitoring(self):
        """Start continuous sensor monitoring"""
        self.is_running = True
        
        def monitor_loop():
            while self.is_running:
                reading = self.generate_realistic_reading()
                self.current_readings = {
                    'timestamp': reading.timestamp.isoformat(),
                    'heart_rate': reading.heart_rate,
                    'blood_pressure_systolic': reading.blood_pressure_systolic,
                    'blood_pressure_diastolic': reading.blood_pressure_diastolic,
                    'body_temperature': reading.body_temperature,
                    'motion_level': reading.motion_level,
                    'sleep_quality': reading.sleep_quality,
                    'medication_adherence': reading.medication_adherence,
                    'patient_id': reading.patient_id
                }
                
                # Store in database
                health_monitor.store_sensor_data(reading)
                
                time.sleep(CONFIG['SENSOR_INTERVAL'])
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop sensor monitoring"""
        self.is_running = False

class AIHealthPredictor:
    """AI-powered health prediction and anomaly detection"""
    
    def __init__(self):
        self.health_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data for health prediction"""
        data = []
        
        for _ in range(n_samples):
            # Generate normal and abnormal health patterns
            is_emergency = random.random() < 0.1  # 10% emergency cases
            
            if is_emergency:
                # Emergency patterns
                hr = random.uniform(40, 50) if random.random() < 0.5 else random.uniform(120, 180)
                bp_sys = random.uniform(180, 220) if random.random() < 0.7 else random.uniform(70, 90)
                bp_dia = random.uniform(110, 130) if random.random() < 0.7 else random.uniform(40, 60)
                temp = random.uniform(102, 105) if random.random() < 0.6 else random.uniform(95, 97)
                motion = random.uniform(0, 0.1)  # Low motion during emergency
                sleep_qual = random.uniform(0, 0.3)
                med_adher = random.uniform(0, 0.5)  # Poor adherence
                health_status = 1  # Emergency
            else:
                # Normal patterns
                hr = random.uniform(60, 100)
                bp_sys = random.uniform(110, 140)
                bp_dia = random.uniform(70, 90)
                temp = random.uniform(97.5, 99.5)
                motion = random.uniform(0.3, 0.8)
                sleep_qual = random.uniform(0.6, 1.0)
                med_adher = random.uniform(0.8, 1.0)
                health_status = 0  # Normal
            
            data.append({
                'heart_rate': hr,
                'blood_pressure_systolic': bp_sys,
                'blood_pressure_diastolic': bp_dia,
                'body_temperature': temp,
                'motion_level': motion,
                'sleep_quality': sleep_qual,
                'medication_adherence': med_adher,
                'health_status': health_status
            })
        
        return pd.DataFrame(data)
    
    def train_models(self):
        """Train AI models for health prediction and anomaly detection"""
        print("Training AI health prediction models...")
        
        # Generate training data
        df = self.generate_training_data(2000)
        
        # Prepare features
        features = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                   'body_temperature', 'motion_level', 'sleep_quality', 'medication_adherence']
        
        X = df[features]
        y = df['health_status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train health prediction model
        self.health_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        self.health_model.fit(X_train_scaled, y_train)
        
        # Train anomaly detection model
        normal_data = X_train_scaled[y_train == 0]  # Only normal cases
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.anomaly_detector.fit(normal_data)
        
        # Evaluate models
        train_score = self.health_model.score(X_train_scaled, y_train)
        test_score = self.health_model.score(X_test_scaled, y_test)
        
        print(f"Health Prediction Model - Train Accuracy: {train_score:.3f}, Test Accuracy: {test_score:.3f}")
        
        self.is_trained = True
        
        # Save models
        os.makedirs(CONFIG['MODEL_PATH'], exist_ok=True)
        joblib.dump(self.health_model, os.path.join(CONFIG['MODEL_PATH'], 'health_model.pkl'))
        joblib.dump(self.anomaly_detector, os.path.join(CONFIG['MODEL_PATH'], 'anomaly_model.pkl'))
        joblib.dump(self.scaler, os.path.join(CONFIG['MODEL_PATH'], 'scaler.pkl'))
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.health_model = joblib.load(os.path.join(CONFIG['MODEL_PATH'], 'health_model.pkl'))
            self.anomaly_detector = joblib.load(os.path.join(CONFIG['MODEL_PATH'], 'anomaly_model.pkl'))
            self.scaler = joblib.load(os.path.join(CONFIG['MODEL_PATH'], 'scaler.pkl'))
            self.is_trained = True
            print("AI models loaded successfully")
        except FileNotFoundError:
            print("No pre-trained models found. Training new models...")
            self.train_models()
    
    def predict_health_status(self, sensor_data: Dict) -> Dict:
        """Predict health status from sensor data"""
        if not self.is_trained:
            return {'error': 'Models not trained'}
        
        # Prepare features
        features = np.array([[
            sensor_data['heart_rate'],
            sensor_data['blood_pressure_systolic'],
            sensor_data['blood_pressure_diastolic'],
            sensor_data['body_temperature'],
            sensor_data['motion_level'],
            sensor_data['sleep_quality'],
            sensor_data['medication_adherence']
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict emergency probability
        emergency_prob = self.health_model.predict_proba(features_scaled)[0][1]
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # Generate health insights
        insights = self.generate_health_insights(sensor_data, emergency_prob)
        
        return {
            'emergency_probability': float(emergency_prob),
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'health_status': 'Emergency' if emergency_prob > CONFIG['EMERGENCY_THRESHOLD'] else 'Normal',
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_health_insights(self, sensor_data: Dict, emergency_prob: float) -> List[str]:
        """Generate actionable health insights"""
        insights = []
        
        # Heart rate analysis
        hr = sensor_data['heart_rate']
        if hr > 100:
            insights.append(f"Elevated heart rate detected ({hr:.0f} bpm). Consider rest and hydration.")
        elif hr < 60:
            insights.append(f"Low heart rate detected ({hr:.0f} bpm). Monitor for symptoms.")
        
        # Blood pressure analysis
        bp_sys = sensor_data['blood_pressure_systolic']
        bp_dia = sensor_data['blood_pressure_diastolic']
        if bp_sys > 140 or bp_dia > 90:
            insights.append(f"High blood pressure detected ({bp_sys:.0f}/{bp_dia:.0f}). Consult healthcare provider.")
        
        # Temperature analysis
        temp = sensor_data['body_temperature']
        if temp > 100.4:
            insights.append(f"Fever detected ({temp:.1f}°F). Monitor symptoms and consider medical attention.")
        elif temp < 97:
            insights.append(f"Low body temperature ({temp:.1f}°F). Ensure warmth and monitor.")
        
        # Motion and activity
        motion = sensor_data['motion_level']
        if motion < 0.2:
            insights.append("Low activity level detected. Encourage gentle movement if safe.")
        
        # Sleep quality
        sleep = sensor_data['sleep_quality']
        if sleep < 0.5:
            insights.append("Poor sleep quality detected. Review sleep environment and habits.")
        
        # Medication adherence
        med_adher = sensor_data['medication_adherence']
        if med_adher < 0.8:
            insights.append("Medication adherence concern. Set up reminders or pill organizer.")
        
        # Emergency probability
        if emergency_prob > 0.6:
            insights.append("⚠️ Health concern detected. Consider contacting healthcare provider.")
        
        return insights

class HealthMonitorSystem:
    """Main health monitoring system coordinating IoT, AI, and physical interactions"""
    
    def __init__(self):
        self.sensor_simulator = IoTSensorSimulator("patient_001")
        self.ai_predictor = AIHealthPredictor()
        self.init_database()
        self.alerts_sent = []
        
    def init_database(self):
        """Initialize SQLite database for health data storage"""
        conn = sqlite3.connect(CONFIG['DATABASE_PATH'])
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                heart_rate REAL,
                blood_pressure_systolic REAL,
                blood_pressure_diastolic REAL,
                body_temperature REAL,
                motion_level REAL,
                sleep_quality REAL,
                medication_adherence REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                emergency_probability REAL,
                is_anomaly INTEGER,
                health_status TEXT,
                insights TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                alert_type TEXT,
                message TEXT,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_sensor_data(self, reading: SensorReading):
        """Store sensor reading in database"""
        conn = sqlite3.connect(CONFIG['DATABASE_PATH'])
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_readings 
            (timestamp, patient_id, heart_rate, blood_pressure_systolic, 
             blood_pressure_diastolic, body_temperature, motion_level, 
             sleep_quality, medication_adherence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reading.timestamp.isoformat(),
            reading.patient_id,
            reading.heart_rate,
            reading.blood_pressure_systolic,
            reading.blood_pressure_diastolic,
            reading.body_temperature,
            reading.motion_level,
            reading.sleep_quality,
            reading.medication_adherence
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_readings(self, hours: int = 24) -> List[Dict]:
        """Get recent sensor readings from database"""
        conn = sqlite3.connect(CONFIG['DATABASE_PATH'])
        cursor = conn.cursor()
        
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT * FROM sensor_readings 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (since_time,))
        
        columns = [desc[0] for desc in cursor.description]
        readings = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return readings
    
    def send_emergency_alert(self, prediction: Dict, sensor_data: Dict):
        """Send emergency alert to caregivers"""
        alert_message = f"""
        🚨 HEALTH EMERGENCY ALERT 🚨
        
        Patient: {sensor_data['patient_id']}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Emergency Probability: {prediction['emergency_probability']:.1%}
        Health Status: {prediction['health_status']}
        
        Current Vitals:
        - Heart Rate: {sensor_data['heart_rate']:.0f} bpm
        - Blood Pressure: {sensor_data['blood_pressure_systolic']:.0f}/{sensor_data['blood_pressure_diastolic']:.0f} mmHg
        - Temperature: {sensor_data['body_temperature']:.1f}°F
        - Motion Level: {sensor_data['motion_level']:.1%}
        
        Health Insights:
        {chr(10).join(['- ' + insight for insight in prediction['insights']])}
        
        Please check on the patient immediately.
        """
        
        # Store alert in database
        conn = sqlite3.connect(CONFIG['DATABASE_PATH'])
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, patient_id, alert_type, message, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            sensor_data['patient_id'],
            'EMERGENCY',
            alert_message,
            'HIGH'
        ))
        
        conn.commit()
        conn.close()
        
        # Add to recent alerts for display
        self.alerts_sent.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'EMERGENCY',
            'message': alert_message,
            'severity': 'HIGH'
        })
        
        print(f"Emergency alert sent: {alert_message[:100]}...")
    
    def start_system(self):
        """Start the complete health monitoring system"""
        print("Starting Elderly Health Monitor System...")
        
        # Load/train AI models
        self.ai_predictor.load_models()
        
        # Start IoT sensor monitoring
        self.sensor_simulator.start_monitoring()
        
        # Start AI prediction loop
        def prediction_loop():
            while True:
                if self.sensor_simulator.current_readings:
                    prediction = self.ai_predictor.predict_health_status(
                        self.sensor_simulator.current_readings
                    )
                    
                    # Store prediction
                    conn = sqlite3.connect(CONFIG['DATABASE_PATH'])
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO health_predictions 
                        (timestamp, patient_id, emergency_probability, is_anomaly, health_status, insights)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        self.sensor_simulator.current_readings['patient_id'],
                        prediction['emergency_probability'],
                        int(prediction['is_anomaly']),
                        prediction['health_status'],
                        json.dumps(prediction['insights'])
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    # Check for emergency
                    if prediction['emergency_probability'] > CONFIG['EMERGENCY_THRESHOLD']:
                        self.send_emergency_alert(prediction, self.sensor_simulator.current_readings)

                time.sleep(CONFIG['PREDICTION_INTERVAL'])
        
        self.prediction_thread = threading.Thread(target=prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
        print("Health monitoring system started successfully!")

# Initialize global health monitor
health_monitor = HealthMonitorSystem()

# Flask Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/current-readings')
def get_current_readings():
    """Get current sensor readings"""
    if health_monitor.sensor_simulator.current_readings:
        # Get AI prediction for current readings
        prediction = health_monitor.ai_predictor.predict_health_status(
            health_monitor.sensor_simulator.current_readings
        )
        
        return jsonify({
            'sensor_data': health_monitor.sensor_simulator.current_readings,
            'prediction': prediction
        })
    else:
        return jsonify({'error': 'No current readings available'})

@app.route('/api/historical-data')
def get_historical_data():
    """Get historical sensor data"""
    hours = request.args.get('hours', 24, type=int)
    readings = health_monitor.get_recent_readings(hours)
    return jsonify(readings)

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify(health_monitor.alerts_sent[-10:])  # Last 10 alerts

@app.route('/api/health-insights')
def get_health_insights():
    """Get AI-generated health insights"""
    if health_monitor.sensor_simulator.current_readings:
        prediction = health_monitor.ai_predictor.predict_health_status(
            health_monitor.sensor_simulator.current_readings
        )
        return jsonify({
            'insights': prediction['insights'],
            'health_status': prediction['health_status'],
            'emergency_probability': prediction['emergency_probability']
        })
    else:
        return jsonify({'error': 'No current data available'})

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start health monitoring"""
    health_monitor.start_system()
    return jsonify({'status': 'Monitoring started'})

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop health monitoring"""
    health_monitor.sensor_simulator.stop_monitoring()
    return jsonify({'status': 'Monitoring stopped'})

if __name__ == '__main__':
    # Start the health monitoring system
    health_monitor.start_system()
    
    print("\n" + "="*60)
    print("🏥 ELDERLY HEALTH MONITOR - IoT + AI + Physical AI")
    print("="*60)
    print("Features:")
    print("✓ Real-time IoT sensor monitoring (heart rate, BP, temperature)")
    print("✓ AI-powered health prediction and anomaly detection")
    print("✓ Physical world interactions (emergency alerts, reminders)")
    print("✓ Predictive analytics for elderly care")
    print("✓ Web dashboard for caregivers")
    print("="*60)
    print("Access the dashboard at: http://localhost:5000")
    print("="*60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)