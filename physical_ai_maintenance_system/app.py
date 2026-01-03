import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import random
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Physical AI Industrial Maintenance System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class IoTSensorSimulator:
    """Simulates IoT sensors for industrial equipment monitoring"""
    
    def __init__(self):
        self.equipment_types = {
            'Motor': {'temp_range': (40, 80), 'vibration_range': (0.1, 2.0), 'pressure_range': (10, 50)},
            'Pump': {'temp_range': (35, 75), 'vibration_range': (0.2, 1.8), 'pressure_range': (15, 60)},
            'Compressor': {'temp_range': (50, 90), 'vibration_range': (0.3, 2.5), 'pressure_range': (20, 80)},
            'Conveyor': {'temp_range': (25, 65), 'vibration_range': (0.1, 1.5), 'pressure_range': (5, 30)}
        }
    
    def generate_sensor_data(self, equipment_id, equipment_type, anomaly_probability=0.1):
        """Generate realistic sensor data with potential anomalies"""
        base_ranges = self.equipment_types[equipment_type]
        
        # Introduce anomalies based on probability
        is_anomaly = random.random() < anomaly_probability
        
        if is_anomaly:
            # Generate anomalous readings
            temperature = random.uniform(base_ranges['temp_range'][1] + 10, base_ranges['temp_range'][1] + 30)
            vibration = random.uniform(base_ranges['vibration_range'][1] + 0.5, base_ranges['vibration_range'][1] + 2.0)
            pressure = random.uniform(base_ranges['pressure_range'][1] + 10, base_ranges['pressure_range'][1] + 20)
        else:
            # Generate normal readings with some variation
            temperature = random.uniform(*base_ranges['temp_range']) + random.gauss(0, 2)
            vibration = random.uniform(*base_ranges['vibration_range']) + random.gauss(0, 0.1)
            pressure = random.uniform(*base_ranges['pressure_range']) + random.gauss(0, 1)
        
        return {
            'equipment_id': equipment_id,
            'equipment_type': equipment_type,
            'timestamp': datetime.now(),
            'temperature': round(temperature, 2),
            'vibration': round(vibration, 3),
            'pressure': round(pressure, 2),
            'is_anomaly': is_anomaly
        }

class AIPredictor:
    """AI-powered predictive maintenance system"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train_models(self, historical_data):
        """Train AI models on historical sensor data"""
        if len(historical_data) < 10:
            return False
        
        # Prepare features
        features = ['temperature', 'vibration', 'pressure']
        X = historical_data[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Generate synthetic failure times for training (in real scenario, use actual failure data)
        failure_times = np.random.exponential(scale=100, size=len(historical_data))
        
        # Train failure predictor
        self.failure_predictor.fit(X_scaled, failure_times)
        
        self.is_trained = True
        return True
    
    def predict_anomaly(self, sensor_data):
        """Detect anomalies in real-time sensor data"""
        if not self.is_trained:
            return 0, "Model not trained"
        
        features = np.array([[sensor_data['temperature'], sensor_data['vibration'], sensor_data['pressure']]])
        features_scaled = self.scaler.transform(features)
        
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        return anomaly_score, is_anomaly
    
    def predict_failure_time(self, sensor_data):
        """Predict time until equipment failure"""
        if not self.is_trained:
            return None, "Model not trained"
        
        features = np.array([[sensor_data['temperature'], sensor_data['vibration'], sensor_data['pressure']]])
        features_scaled = self.scaler.transform(features)
        
        predicted_time = self.failure_predictor.predict(features_scaled)[0]
        
        # Convert to maintenance priority
        if predicted_time < 24:
            priority = "CRITICAL"
        elif predicted_time < 72:
            priority = "HIGH"
        elif predicted_time < 168:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        return predicted_time, priority

class PhysicalAICoordinator:
    """Physical AI system for coordinating robotic maintenance tasks"""
    
    def __init__(self):
        self.maintenance_robots = {
            'Robot-01': {'status': 'Available', 'location': 'Station-A', 'capabilities': ['inspection', 'lubrication']},
            'Robot-02': {'status': 'Available', 'location': 'Station-B', 'capabilities': ['repair', 'replacement']},
            'Robot-03': {'status': 'Available', 'location': 'Station-C', 'capabilities': ['cleaning', 'calibration']}
        }
        self.maintenance_queue = []
    
    def schedule_maintenance_task(self, equipment_id, task_type, priority, predicted_failure_time):
        """Schedule maintenance task with Physical AI coordination"""
        # Find suitable robot based on capabilities
        suitable_robots = []
        for robot_id, robot_info in self.maintenance_robots.items():
            if task_type in robot_info['capabilities'] and robot_info['status'] == 'Available':
                suitable_robots.append(robot_id)
        
        if suitable_robots:
            assigned_robot = suitable_robots[0]
            self.maintenance_robots[assigned_robot]['status'] = 'Assigned'
            
            task = {
                'task_id': f"TASK-{len(self.maintenance_queue) + 1:03d}",
                'equipment_id': equipment_id,
                'task_type': task_type,
                'priority': priority,
                'assigned_robot': assigned_robot,
                'scheduled_time': datetime.now() + timedelta(hours=1),
                'estimated_duration': random.randint(30, 120),  # minutes
                'status': 'Scheduled'
            }
            
            self.maintenance_queue.append(task)
            return task
        else:
            return None
    
    def get_robot_status(self):
        """Get current status of all maintenance robots"""
        return self.maintenance_robots
    
    def get_maintenance_queue(self):
        """Get current maintenance task queue"""
        return sorted(self.maintenance_queue, key=lambda x: x['priority'])

# Initialize system components
if 'sensor_simulator' not in st.session_state:
    st.session_state.sensor_simulator = IoTSensorSimulator()
    st.session_state.ai_predictor = AIPredictor()
    st.session_state.physical_ai = PhysicalAICoordinator()
    st.session_state.historical_data = pd.DataFrame()
    st.session_state.equipment_list = [
        {'id': 'EQ-001', 'type': 'Motor', 'location': 'Production Line A'},
        {'id': 'EQ-002', 'type': 'Pump', 'location': 'Cooling System'},
        {'id': 'EQ-003', 'type': 'Compressor', 'location': 'Air Supply'},
        {'id': 'EQ-004', 'type': 'Conveyor', 'location': 'Assembly Line'}
    ]

# Main application interface
st.title("🏭 Physical AI Industrial Maintenance System")
st.markdown("**Combining IoT Sensors + AI Prediction + Physical AI Robotics for Smart Manufacturing**")

# Sidebar controls
st.sidebar.header("System Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh Data", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
anomaly_probability = st.sidebar.slider("Anomaly Probability", 0.0, 0.5, 0.1)

if st.sidebar.button("Generate Training Data"):
    # Generate historical data for training
    training_data = []
    for _ in range(100):
        for equipment in st.session_state.equipment_list:
            data = st.session_state.sensor_simulator.generate_sensor_data(
                equipment['id'], equipment['type'], anomaly_probability=0.05
            )
            training_data.append(data)
    
    st.session_state.historical_data = pd.DataFrame(training_data)
    
    # Train AI models
    success = st.session_state.ai_predictor.train_models(st.session_state.historical_data)
    if success:
        st.sidebar.success("AI Models Trained Successfully!")
    else:
        st.sidebar.error("Training Failed - Need More Data")

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Real-Time Equipment Monitoring")
    
    # Equipment status cards
    equipment_cols = st.columns(2)
    
    for i, equipment in enumerate(st.session_state.equipment_list):
        with equipment_cols[i % 2]:
            # Generate real-time sensor data
            sensor_data = st.session_state.sensor_simulator.generate_sensor_data(
                equipment['id'], equipment['type'], anomaly_probability
            )
            
            # AI predictions
            anomaly_score, is_anomaly = st.session_state.ai_predictor.predict_anomaly(sensor_data)
            failure_time, priority = st.session_state.ai_predictor.predict_failure_time(sensor_data)
            
            # Display equipment card
            status_color = "🔴" if is_anomaly else "🟢"
            st.markdown(f"### {status_color} {equipment['id']} - {equipment['type']}")
            st.markdown(f"**Location:** {equipment['location']}")
            
            # Sensor readings
            col_temp, col_vib, col_press = st.columns(3)
            with col_temp:
                st.metric("Temperature", f"{sensor_data['temperature']}°C")
            with col_vib:
                st.metric("Vibration", f"{sensor_data['vibration']} mm/s")
            with col_press:
                st.metric("Pressure", f"{sensor_data['pressure']} bar")
            
            # AI predictions
            if failure_time:
                st.markdown(f"**Predicted Failure:** {failure_time:.1f} hours")
                st.markdown(f"**Priority:** {priority}")
                
                # Schedule maintenance if critical
                if priority in ['CRITICAL', 'HIGH'] and is_anomaly:
                    task_type = 'inspection' if priority == 'HIGH' else 'repair'
                    task = st.session_state.physical_ai.schedule_maintenance_task(
                        equipment['id'], task_type, priority, failure_time
                    )
                    if task:
                        st.success(f"Maintenance Task {task['task_id']} Scheduled")
            
            st.markdown("---")

with col2:
    st.header("Physical AI Control Center")
    
    # Robot status
    st.subheader("🤖 Maintenance Robots")
    robot_status = st.session_state.physical_ai.get_robot_status()
    
    for robot_id, robot_info in robot_status.items():
        status_emoji = "🟢" if robot_info['status'] == 'Available' else "🟡"
        st.markdown(f"{status_emoji} **{robot_id}**")
        st.markdown(f"Status: {robot_info['status']}")
        st.markdown(f"Location: {robot_info['location']}")
        st.markdown(f"Capabilities: {', '.join(robot_info['capabilities'])}")
        st.markdown("")
    
    # Maintenance queue
    st.subheader("📋 Maintenance Queue")
    maintenance_queue = st.session_state.physical_ai.get_maintenance_queue()
    
    if maintenance_queue:
        for task in maintenance_queue[-5:]:  # Show last 5 tasks
            priority_color = {
                'CRITICAL': '🔴',
                'HIGH': '🟠', 
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(task['priority'], '⚪')
            
            st.markdown(f"{priority_color} **{task['task_id']}**")
            st.markdown(f"Equipment: {task['equipment_id']}")
            st.markdown(f"Task: {task['task_type']}")
            st.markdown(f"Robot: {task['assigned_robot']}")
            st.markdown(f"Status: {task['status']}")
            st.markdown("")
    else:
        st.info("No maintenance tasks scheduled")

# Analytics dashboard
st.header("📊 Analytics Dashboard")

if not st.session_state.historical_data.empty:
    # Equipment performance trends
    fig_trends = go.Figure()
    
    for equipment in st.session_state.equipment_list:
        equipment_data = st.session_state.historical_data[
            st.session_state.historical_data['equipment_id'] == equipment['id']
        ]
        
        if not equipment_data.empty:
            fig_trends.add_trace(go.Scatter(
                x=equipment_data['timestamp'],
                y=equipment_data['temperature'],
                mode='lines',
                name=f"{equipment['id']} Temperature",
                line=dict(width=2)
            ))
    
    fig_trends.update_layout(
        title="Equipment Temperature Trends",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=400
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Anomaly detection summary
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        total_readings = len(st.session_state.historical_data)
        st.metric("Total Readings", total_readings)
    
    with col_stats2:
        anomaly_count = st.session_state.historical_data['is_anomaly'].sum()
        anomaly_rate = (anomaly_count / total_readings * 100) if total_readings > 0 else 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    with col_stats3:
        maintenance_tasks = len(st.session_state.physical_ai.get_maintenance_queue())
        st.metric("Scheduled Tasks", maintenance_tasks)

else:
    st.info("Generate training data to see analytics")

# System information
st.header("ℹ️ System Information")
st.markdown("""
**Physical AI Industrial Maintenance System Features:**

🔧 **IoT Integration:**
- Real-time sensor monitoring (temperature, vibration, pressure)
- Multi-equipment support with different sensor profiles
- Anomaly detection with configurable thresholds

🧠 **AI-Powered Predictions:**
- Machine learning-based anomaly detection
- Predictive maintenance scheduling
- Failure time estimation with priority classification

🤖 **Physical AI Coordination:**
- Automated robot task assignment
- Capability-based robot selection
- Real-time maintenance queue management

📊 **Smart Analytics:**
- Historical trend analysis
- Performance metrics tracking
- Predictive insights dashboard

**Technology Stack:** Python, Streamlit, Scikit-learn, Plotly, Pandas, NumPy
""")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()