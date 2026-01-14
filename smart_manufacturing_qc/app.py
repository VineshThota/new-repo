from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sqlite3
import json
import time
import threading
from datetime import datetime
import random
import base64
from io import BytesIO
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class IoTSensorManager:
    """Manages IoT sensor data collection and monitoring"""
    
    def __init__(self):
        self.sensors = {
            'temperature': {'value': 25.0, 'threshold': 30.0, 'unit': '°C'},
            'vibration': {'value': 0.5, 'threshold': 2.0, 'unit': 'mm/s'},
            'humidity': {'value': 45.0, 'threshold': 60.0, 'unit': '%'},
            'pressure': {'value': 1013.25, 'threshold': 1020.0, 'unit': 'hPa'},
            'noise_level': {'value': 35.0, 'threshold': 70.0, 'unit': 'dB'}
        }
        self.running = False
        self.data_history = []
    
    def simulate_sensor_data(self):
        """Simulate real-time IoT sensor data"""
        while self.running:
            timestamp = datetime.now().isoformat()
            sensor_data = {'timestamp': timestamp}
            
            for sensor_name, config in self.sensors.items():
                # Simulate realistic sensor variations
                base_value = config['value']
                variation = random.uniform(-0.1, 0.1) * base_value
                new_value = base_value + variation
                
                # Add occasional anomalies
                if random.random() < 0.05:  # 5% chance of anomaly
                    new_value *= random.uniform(1.2, 1.8)
                
                config['value'] = round(new_value, 2)
                sensor_data[sensor_name] = config['value']
                sensor_data[f'{sensor_name}_status'] = 'ALERT' if new_value > config['threshold'] else 'NORMAL'
            
            self.data_history.append(sensor_data)
            if len(self.data_history) > 100:  # Keep last 100 readings
                self.data_history.pop(0)
            
            time.sleep(2)  # Update every 2 seconds
    
    def start_monitoring(self):
        """Start IoT sensor monitoring"""
        self.running = True
        thread = threading.Thread(target=self.simulate_sensor_data)
        thread.daemon = True
        thread.start()
        logger.info("IoT sensor monitoring started")
    
    def stop_monitoring(self):
        """Stop IoT sensor monitoring"""
        self.running = False
        logger.info("IoT sensor monitoring stopped")
    
    def get_current_data(self):
        """Get current sensor readings"""
        return self.sensors
    
    def get_history(self):
        """Get sensor data history"""
        return self.data_history

class ComputerVisionQC:
    """Computer Vision Quality Control System"""
    
    def __init__(self):
        self.defect_model = self._create_simple_defect_model()
        self.defect_types = ['scratch', 'dent', 'discoloration', 'crack', 'missing_part']
        self.quality_threshold = 0.7
    
    def _create_simple_defect_model(self):
        """Create a simple CNN model for defect detection"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(self.defect_types), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def analyze_image(self, image_data):
        """Analyze product image for defects"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_bytes))
            image_array = np.array(image.resize((224, 224)))
            
            # Normalize image
            image_array = image_array.astype('float32') / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Simulate defect detection (since we don't have trained weights)
            defect_probability = random.uniform(0.1, 0.9)
            defect_type = random.choice(self.defect_types)
            
            # Basic image processing for additional analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Edge detection for structural analysis
            edges = cv2.Canny(cv_image, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Color analysis
            mean_color = np.mean(cv_image, axis=(0, 1))
            
            quality_score = 1.0 - defect_probability
            status = 'PASS' if quality_score >= self.quality_threshold else 'FAIL'
            
            analysis_result = {
                'quality_score': round(quality_score, 3),
                'status': status,
                'defect_type': defect_type if status == 'FAIL' else None,
                'defect_probability': round(defect_probability, 3),
                'edge_density': round(edge_density, 4),
                'mean_color': [int(c) for c in mean_color],
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {'error': str(e)}
    
    def get_quality_statistics(self, results):
        """Calculate quality statistics from analysis results"""
        if not results:
            return {}
        
        total_items = len(results)
        passed_items = sum(1 for r in results if r.get('status') == 'PASS')
        failed_items = total_items - passed_items
        
        avg_quality_score = sum(r.get('quality_score', 0) for r in results) / total_items
        
        defect_counts = {}
        for result in results:
            if result.get('defect_type'):
                defect_type = result['defect_type']
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        return {
            'total_items': total_items,
            'passed_items': passed_items,
            'failed_items': failed_items,
            'pass_rate': round((passed_items / total_items) * 100, 2),
            'avg_quality_score': round(avg_quality_score, 3),
            'defect_distribution': defect_counts
        }

class RoboticArmController:
    """Physical AI Robotic Arm Controller for automated sorting"""
    
    def __init__(self):
        self.position = {'x': 0, 'y': 0, 'z': 0}
        self.gripper_state = 'open'
        self.is_moving = False
        self.movement_history = []
        self.sort_bins = {
            'pass_bin': {'x': 100, 'y': 0, 'z': 0},
            'fail_bin': {'x': -100, 'y': 0, 'z': 0},
            'inspection_point': {'x': 0, 'y': 50, 'z': 10}
        }
    
    def move_to_position(self, target_position, duration=2.0):
        """Simulate robotic arm movement"""
        self.is_moving = True
        start_pos = self.position.copy()
        
        # Simulate gradual movement
        steps = 20
        for step in range(steps + 1):
            progress = step / steps
            
            # Calculate intermediate position
            self.position = {
                'x': start_pos['x'] + (target_position['x'] - start_pos['x']) * progress,
                'y': start_pos['y'] + (target_position['y'] - start_pos['y']) * progress,
                'z': start_pos['z'] + (target_position['z'] - start_pos['z']) * progress
            }
            
            time.sleep(duration / steps)
        
        self.is_moving = False
        
        # Log movement
        movement_log = {
            'timestamp': datetime.now().isoformat(),
            'from': start_pos,
            'to': target_position,
            'duration': duration
        }
        self.movement_history.append(movement_log)
        
        logger.info(f"Robotic arm moved to position: {target_position}")
    
    def control_gripper(self, action):
        """Control gripper (open/close)"""
        if action in ['open', 'close']:
            self.gripper_state = action
            logger.info(f"Gripper {action}ed")
            return True
        return False
    
    def sort_product(self, quality_result):
        """Automatically sort product based on quality analysis"""
        try:
            # Move to inspection point
            self.move_to_position(self.sort_bins['inspection_point'])
            
            # Close gripper to pick up product
            self.control_gripper('close')
            time.sleep(0.5)
            
            # Determine destination based on quality
            if quality_result.get('status') == 'PASS':
                destination = self.sort_bins['pass_bin']
                bin_type = 'pass_bin'
            else:
                destination = self.sort_bins['fail_bin']
                bin_type = 'fail_bin'
            
            # Move to appropriate bin
            self.move_to_position(destination)
            
            # Open gripper to release product
            self.control_gripper('open')
            time.sleep(0.5)
            
            # Return to home position
            self.move_to_position({'x': 0, 'y': 0, 'z': 0})
            
            sort_result = {
                'timestamp': datetime.now().isoformat(),
                'quality_status': quality_result.get('status'),
                'destination_bin': bin_type,
                'quality_score': quality_result.get('quality_score'),
                'defect_type': quality_result.get('defect_type')
            }
            
            return sort_result
            
        except Exception as e:
            logger.error(f"Error during sorting: {str(e)}")
            return {'error': str(e)}
    
    def get_status(self):
        """Get current robotic arm status"""
        return {
            'position': self.position,
            'gripper_state': self.gripper_state,
            'is_moving': self.is_moving,
            'movement_count': len(self.movement_history)
        }

class DatabaseManager:
    """Database manager for storing quality control data"""
    
    def __init__(self, db_path='quality_control.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Quality analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                quality_score REAL,
                status TEXT,
                defect_type TEXT,
                defect_probability REAL,
                edge_density REAL,
                mean_color_r INTEGER,
                mean_color_g INTEGER,
                mean_color_b INTEGER
            )
        ''')
        
        # IoT sensor data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                temperature REAL,
                vibration REAL,
                humidity REAL,
                pressure REAL,
                noise_level REAL
            )
        ''')
        
        # Robotic operations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS robotic_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                operation_type TEXT,
                quality_status TEXT,
                destination_bin TEXT,
                quality_score REAL,
                defect_type TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def save_quality_result(self, result):
        """Save quality analysis result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_results 
            (timestamp, quality_score, status, defect_type, defect_probability, 
             edge_density, mean_color_r, mean_color_g, mean_color_b)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['timestamp'],
            result['quality_score'],
            result['status'],
            result.get('defect_type'),
            result['defect_probability'],
            result['edge_density'],
            result['mean_color'][0],
            result['mean_color'][1],
            result['mean_color'][2]
        ))
        
        conn.commit()
        conn.close()
    
    def save_sensor_data(self, sensor_data):
        """Save IoT sensor data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_data 
            (timestamp, temperature, vibration, humidity, pressure, noise_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            sensor_data['timestamp'],
            sensor_data.get('temperature'),
            sensor_data.get('vibration'),
            sensor_data.get('humidity'),
            sensor_data.get('pressure'),
            sensor_data.get('noise_level')
        ))
        
        conn.commit()
        conn.close()
    
    def save_robotic_operation(self, operation):
        """Save robotic operation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO robotic_operations 
            (timestamp, operation_type, quality_status, destination_bin, quality_score, defect_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            operation['timestamp'],
            'sort_product',
            operation['quality_status'],
            operation['destination_bin'],
            operation.get('quality_score'),
            operation.get('defect_type')
        ))
        
        conn.commit()
        conn.close()

# Initialize system components
iot_manager = IoTSensorManager()
cv_qc = ComputerVisionQC()
robot_controller = RoboticArmController()
db_manager = DatabaseManager()

# Global variables for storing results
quality_results = []
sorting_operations = []

@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/sensor-data')
def get_sensor_data():
    """Get current IoT sensor data"""
    return jsonify(iot_manager.get_current_data())

@app.route('/api/sensor-history')
def get_sensor_history():
    """Get IoT sensor data history"""
    return jsonify(iot_manager.get_history())

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze product image for quality control"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Analyze image
        result = cv_qc.analyze_image(image_data)
        
        if 'error' not in result:
            # Save to database
            db_manager.save_quality_result(result)
            
            # Add to results list
            quality_results.append(result)
            
            # Trigger robotic sorting if enabled
            if data.get('auto_sort', False):
                sort_result = robot_controller.sort_product(result)
                if 'error' not in sort_result:
                    db_manager.save_robotic_operation(sort_result)
                    sorting_operations.append(sort_result)
                    result['sorting_operation'] = sort_result
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/robot-status')
def get_robot_status():
    """Get robotic arm status"""
    return jsonify(robot_controller.get_status())

@app.route('/api/robot-sort', methods=['POST'])
def manual_robot_sort():
    """Manually trigger robotic sorting"""
    try:
        data = request.get_json()
        quality_result = data.get('quality_result')
        
        if not quality_result:
            return jsonify({'error': 'No quality result provided'}), 400
        
        sort_result = robot_controller.sort_product(quality_result)
        
        if 'error' not in sort_result:
            db_manager.save_robotic_operation(sort_result)
            sorting_operations.append(sort_result)
        
        return jsonify(sort_result)
        
    except Exception as e:
        logger.error(f"Error in manual_robot_sort: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/quality-statistics')
def get_quality_statistics():
    """Get quality control statistics"""
    stats = cv_qc.get_quality_statistics(quality_results)
    stats['total_operations'] = len(sorting_operations)
    return jsonify(stats)

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start IoT sensor monitoring"""
    iot_manager.start_monitoring()
    return jsonify({'status': 'Monitoring started'})

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop IoT sensor monitoring"""
    iot_manager.stop_monitoring()
    return jsonify({'status': 'Monitoring stopped'})

if __name__ == '__main__':
    # Start IoT monitoring on startup
    iot_manager.start_monitoring()
    
    logger.info("Smart Manufacturing Quality Control System starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)