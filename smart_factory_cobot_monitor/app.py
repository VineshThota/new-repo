import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio
import threading

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Factory Cobot Health Monitor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class CobotSensor:
    """Represents IoT sensor data from collaborative robots"""
    cobot_id: str
    timestamp: datetime
    temperature: float
    vibration_x: float
    vibration_y: float
    vibration_z: float
    motor_current: float
    joint_positions: List[float]
    force_torque: List[float]
    operational_hours: float
    error_count: int
    efficiency_score: float

class IoTSensorSimulator:
    """Simulates real-time IoT sensor data from collaborative robots"""
    
    def __init__(self):
        self.cobots = ["COBOT_001", "COBOT_002", "COBOT_003", "COBOT_004"]
        self.base_params = {
            "temperature": {"normal": (35, 45), "warning": (45, 55), "critical": (55, 70)},
            "vibration": {"normal": (0.1, 0.5), "warning": (0.5, 1.0), "critical": (1.0, 2.0)},
            "motor_current": {"normal": (2.0, 4.0), "warning": (4.0, 6.0), "critical": (6.0, 8.0)},
            "efficiency": {"normal": (85, 95), "warning": (70, 85), "critical": (50, 70)}
        }
    
    def generate_sensor_data(self, cobot_id: str, anomaly_probability: float = 0.1) -> CobotSensor:
        """Generate realistic sensor data with occasional anomalies"""
        
        # Determine if this reading should be anomalous
        is_anomaly = random.random() < anomaly_probability
        
        if is_anomaly:
            temp_range = self.base_params["temperature"]["critical"]
            vib_range = self.base_params["vibration"]["critical"]
            current_range = self.base_params["motor_current"]["critical"]
            eff_range = self.base_params["efficiency"]["critical"]
        else:
            temp_range = self.base_params["temperature"]["normal"]
            vib_range = self.base_params["vibration"]["normal"]
            current_range = self.base_params["motor_current"]["normal"]
            eff_range = self.base_params["efficiency"]["normal"]
        
        return CobotSensor(
            cobot_id=cobot_id,
            timestamp=datetime.now(),
            temperature=random.uniform(*temp_range),
            vibration_x=random.uniform(*vib_range),
            vibration_y=random.uniform(*vib_range),
            vibration_z=random.uniform(*vib_range),
            motor_current=random.uniform(*current_range),
            joint_positions=[random.uniform(-180, 180) for _ in range(6)],
            force_torque=[random.uniform(-10, 10) for _ in range(6)],
            operational_hours=random.uniform(1000, 8000),
            error_count=random.randint(0, 5) if is_anomaly else random.randint(0, 1),
            efficiency_score=random.uniform(*eff_range)
        )

class PredictiveMaintenanceAI:
    """AI engine for predictive maintenance analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.is_trained = False
        self.feature_columns = [
            'temperature', 'vibration_magnitude', 'motor_current', 
            'operational_hours', 'error_count', 'efficiency_score'
        ]
    
    def prepare_features(self, sensor_data: List[CobotSensor]) -> pd.DataFrame:
        """Convert sensor data to feature matrix for AI analysis"""
        features = []
        
        for data in sensor_data:
            vibration_magnitude = np.sqrt(
                data.vibration_x**2 + data.vibration_y**2 + data.vibration_z**2
            )
            
            features.append({
                'cobot_id': data.cobot_id,
                'timestamp': data.timestamp,
                'temperature': data.temperature,
                'vibration_magnitude': vibration_magnitude,
                'motor_current': data.motor_current,
                'operational_hours': data.operational_hours,
                'error_count': data.error_count,
                'efficiency_score': data.efficiency_score
            })
        
        return pd.DataFrame(features)
    
    def train_model(self, historical_data: pd.DataFrame):
        """Train the anomaly detection model"""
        if len(historical_data) < 10:
            return False
        
        features = historical_data[self.feature_columns]
        features_scaled = self.scaler.fit_transform(features)
        self.anomaly_detector.fit(features_scaled)
        self.is_trained = True
        return True
    
    def predict_anomalies(self, current_data: pd.DataFrame) -> Dict:
        """Predict anomalies and maintenance needs"""
        if not self.is_trained or len(current_data) == 0:
            return {"predictions": [], "risk_scores": [], "maintenance_recommendations": []}
        
        features = current_data[self.feature_columns]
        features_scaled = self.scaler.transform(features)
        
        # Anomaly detection (-1 = anomaly, 1 = normal)
        anomaly_predictions = self.anomaly_detector.predict(features_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
        
        # Convert to risk scores (0-100)
        risk_scores = ((1 - anomaly_scores) * 50).clip(0, 100)
        
        # Generate maintenance recommendations
        recommendations = []
        for i, (pred, score, row) in enumerate(zip(anomaly_predictions, risk_scores, current_data.itertuples())):
            cobot_id = row.cobot_id
            
            if pred == -1 or score > 70:
                if row.temperature > 50:
                    recommendations.append(f"{cobot_id}: URGENT - Check cooling system")
                elif row.vibration_magnitude > 1.0:
                    recommendations.append(f"{cobot_id}: WARNING - Inspect mechanical components")
                elif row.motor_current > 5.5:
                    recommendations.append(f"{cobot_id}: CAUTION - Motor overload detected")
                elif row.efficiency_score < 75:
                    recommendations.append(f"{cobot_id}: MAINTENANCE - Performance degradation")
                else:
                    recommendations.append(f"{cobot_id}: INSPECTION - Anomaly detected")
            elif score > 50:
                recommendations.append(f"{cobot_id}: MONITOR - Elevated risk levels")
            else:
                recommendations.append(f"{cobot_id}: NORMAL - Operating within parameters")
        
        return {
            "predictions": anomaly_predictions.tolist(),
            "risk_scores": risk_scores.tolist(),
            "maintenance_recommendations": recommendations
        }

class PhysicalActionController:
    """Handles physical world interactions and automated responses"""
    
    def __init__(self):
        self.maintenance_queue = []
        self.alert_history = []
    
    def schedule_maintenance(self, cobot_id: str, priority: str, description: str):
        """Schedule physical maintenance actions"""
        maintenance_task = {
            "cobot_id": cobot_id,
            "priority": priority,
            "description": description,
            "scheduled_time": datetime.now() + timedelta(hours=1 if priority == "URGENT" else 24),
            "status": "SCHEDULED",
            "created_at": datetime.now()
        }
        self.maintenance_queue.append(maintenance_task)
        return maintenance_task
    
    def trigger_physical_alert(self, cobot_id: str, alert_type: str, message: str):
        """Simulate physical alerts (lights, sounds, notifications)"""
        alert = {
            "cobot_id": cobot_id,
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "acknowledged": False
        }
        self.alert_history.append(alert)
        
        # Simulate physical actions
        if alert_type == "CRITICAL":
            # Would trigger red warning lights and stop cobot
            pass
        elif alert_type == "WARNING":
            # Would trigger yellow warning lights
            pass
        
        return alert
    
    def get_maintenance_queue(self) -> List[Dict]:
        """Get current maintenance queue"""
        return sorted(self.maintenance_queue, key=lambda x: x["scheduled_time"])
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert["timestamp"] > cutoff_time]

class SmartFactoryMonitor:
    """Main application class integrating IoT, AI, and Physical AI"""
    
    def __init__(self):
        self.iot_simulator = IoTSensorSimulator()
        self.ai_engine = PredictiveMaintenanceAI()
        self.physical_controller = PhysicalActionController()
        self.sensor_history = []
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for data storage"""
        self.conn = sqlite3.connect('cobot_data.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cobot_id TEXT,
                timestamp TEXT,
                temperature REAL,
                vibration_magnitude REAL,
                motor_current REAL,
                operational_hours REAL,
                error_count INTEGER,
                efficiency_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cobot_id TEXT,
                action_type TEXT,
                description TEXT,
                timestamp TEXT,
                priority TEXT
            )
        ''')
        
        self.conn.commit()
    
    def collect_sensor_data(self) -> List[CobotSensor]:
        """Collect real-time sensor data from all cobots"""
        current_readings = []
        for cobot_id in self.iot_simulator.cobots:
            sensor_data = self.iot_simulator.generate_sensor_data(cobot_id)
            current_readings.append(sensor_data)
            self.sensor_history.append(sensor_data)
        
        # Keep only last 1000 readings for performance
        if len(self.sensor_history) > 1000:
            self.sensor_history = self.sensor_history[-1000:]
        
        return current_readings
    
    def process_ai_analysis(self, sensor_data: List[CobotSensor]) -> Dict:
        """Process sensor data through AI engine"""
        # Prepare data for AI analysis
        df_current = self.ai_engine.prepare_features(sensor_data)
        df_historical = self.ai_engine.prepare_features(self.sensor_history)
        
        # Train model if not already trained
        if not self.ai_engine.is_trained and len(df_historical) >= 10:
            self.ai_engine.train_model(df_historical)
        
        # Get predictions
        predictions = self.ai_engine.predict_anomalies(df_current)
        
        return predictions, df_current
    
    def execute_physical_actions(self, predictions: Dict, sensor_df: pd.DataFrame):
        """Execute physical world actions based on AI predictions"""
        for i, (cobot_id, risk_score, recommendation) in enumerate(
            zip(sensor_df['cobot_id'], predictions['risk_scores'], predictions['maintenance_recommendations'])
        ):
            
            if risk_score > 80:
                # Critical - immediate action required
                self.physical_controller.trigger_physical_alert(
                    cobot_id, "CRITICAL", f"Critical anomaly detected - Risk: {risk_score:.1f}%"
                )
                self.physical_controller.schedule_maintenance(
                    cobot_id, "URGENT", recommendation
                )
            elif risk_score > 60:
                # Warning - schedule maintenance
                self.physical_controller.trigger_physical_alert(
                    cobot_id, "WARNING", f"Elevated risk detected - Risk: {risk_score:.1f}%"
                )
                self.physical_controller.schedule_maintenance(
                    cobot_id, "HIGH", recommendation
                )
            elif risk_score > 40:
                # Monitor - log for tracking
                self.physical_controller.schedule_maintenance(
                    cobot_id, "MEDIUM", recommendation
                )

# Initialize the main application
if 'monitor' not in st.session_state:
    st.session_state.monitor = SmartFactoryMonitor()

# Streamlit UI
st.title("🤖 Smart Factory Collaborative Robot Health Monitor")
st.markdown("""
**Real-time IoT monitoring + AI predictive maintenance + Physical world automation**

This application demonstrates the integration of:
- **IoT Sensors**: Real-time monitoring of collaborative robots
- **AI Algorithms**: Predictive maintenance using machine learning
- **Physical AI**: Automated maintenance scheduling and alert systems
""")

# Sidebar controls
st.sidebar.header("🔧 Control Panel")
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)
anomaly_rate = st.sidebar.slider("Anomaly simulation rate", 0.0, 0.5, 0.1)

if st.sidebar.button("🔄 Manual Refresh") or auto_refresh:
    # Collect new sensor data
    current_sensors = st.session_state.monitor.collect_sensor_data()
    
    # Process through AI
    predictions, sensor_df = st.session_state.monitor.process_ai_analysis(current_sensors)
    
    # Execute physical actions
    st.session_state.monitor.execute_physical_actions(predictions, sensor_df)
    
    # Store in session state for display
    st.session_state.current_data = sensor_df
    st.session_state.predictions = predictions

# Main dashboard
if 'current_data' in st.session_state:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Cobots", len(st.session_state.current_data))
    
    with col2:
        avg_risk = np.mean(st.session_state.predictions['risk_scores'])
        st.metric("Average Risk Score", f"{avg_risk:.1f}%", 
                 delta=f"{avg_risk-50:.1f}%" if avg_risk > 50 else None)
    
    with col3:
        anomalies = sum(1 for pred in st.session_state.predictions['predictions'] if pred == -1)
        st.metric("Anomalies Detected", anomalies)
    
    with col4:
        maintenance_tasks = len(st.session_state.monitor.physical_controller.get_maintenance_queue())
        st.metric("Pending Maintenance", maintenance_tasks)
    
    # Real-time sensor data visualization
    st.subheader("📊 Real-time Sensor Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "Vibration", "Motor Current", "Efficiency"])
    
    with tab1:
        fig_temp = px.bar(st.session_state.current_data, x='cobot_id', y='temperature',
                         title="Cobot Temperature Readings",
                         color='temperature',
                         color_continuous_scale='RdYlBu_r')
        fig_temp.add_hline(y=50, line_dash="dash", line_color="red", 
                          annotation_text="Critical Threshold")
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with tab2:
        fig_vib = px.bar(st.session_state.current_data, x='cobot_id', y='vibration_magnitude',
                        title="Cobot Vibration Levels",
                        color='vibration_magnitude',
                        color_continuous_scale='Viridis')
        fig_vib.add_hline(y=1.0, line_dash="dash", line_color="red",
                         annotation_text="Warning Threshold")
        st.plotly_chart(fig_vib, use_container_width=True)
    
    with tab3:
        fig_current = px.bar(st.session_state.current_data, x='cobot_id', y='motor_current',
                           title="Motor Current Draw",
                           color='motor_current',
                           color_continuous_scale='Plasma')
        fig_current.add_hline(y=5.5, line_dash="dash", line_color="red",
                            annotation_text="Overload Threshold")
        st.plotly_chart(fig_current, use_container_width=True)
    
    with tab4:
        fig_eff = px.bar(st.session_state.current_data, x='cobot_id', y='efficiency_score',
                        title="Operational Efficiency",
                        color='efficiency_score',
                        color_continuous_scale='RdYlGn')
        fig_eff.add_hline(y=75, line_dash="dash", line_color="orange",
                         annotation_text="Performance Threshold")
        st.plotly_chart(fig_eff, use_container_width=True)
    
    # AI Predictions and Risk Assessment
    st.subheader("🧠 AI Predictive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk scores visualization
        risk_df = pd.DataFrame({
            'Cobot': st.session_state.current_data['cobot_id'],
            'Risk Score': st.session_state.predictions['risk_scores']
        })
        
        fig_risk = px.bar(risk_df, x='Cobot', y='Risk Score',
                         title="AI Risk Assessment",
                         color='Risk Score',
                         color_continuous_scale='RdYlGn_r')
        fig_risk.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="Critical Risk")
        fig_risk.add_hline(y=60, line_dash="dash", line_color="orange",
                          annotation_text="High Risk")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Maintenance recommendations
        st.write("**🔧 AI Maintenance Recommendations:**")
        for i, rec in enumerate(st.session_state.predictions['maintenance_recommendations']):
            risk_score = st.session_state.predictions['risk_scores'][i]
            
            if risk_score > 80:
                st.error(f"🚨 {rec}")
            elif risk_score > 60:
                st.warning(f"⚠️ {rec}")
            elif risk_score > 40:
                st.info(f"ℹ️ {rec}")
            else:
                st.success(f"✅ {rec}")
    
    # Physical Actions and Automation
    st.subheader("🔧 Physical World Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📋 Maintenance Queue:**")
        maintenance_queue = st.session_state.monitor.physical_controller.get_maintenance_queue()
        
        if maintenance_queue:
            for task in maintenance_queue[-10:]:  # Show last 10 tasks
                priority_color = {
                    "URGENT": "🔴",
                    "HIGH": "🟠", 
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }.get(task['priority'], "⚪")
                
                st.write(f"{priority_color} **{task['cobot_id']}** - {task['description']}")
                st.caption(f"Scheduled: {task['scheduled_time'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No maintenance tasks scheduled")
    
    with col2:
        st.write("**🚨 Recent Alerts:**")
        recent_alerts = st.session_state.monitor.physical_controller.get_recent_alerts()
        
        if recent_alerts:
            for alert in recent_alerts[-10:]:  # Show last 10 alerts
                alert_icon = {
                    "CRITICAL": "🚨",
                    "WARNING": "⚠️",
                    "INFO": "ℹ️"
                }.get(alert['alert_type'], "📢")
                
                st.write(f"{alert_icon} **{alert['cobot_id']}** - {alert['message']}")
                st.caption(f"{alert['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("No recent alerts")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
**Technology Stack:**
- **IoT Simulation**: Real-time sensor data generation
- **AI Engine**: Scikit-learn Isolation Forest for anomaly detection
- **Physical AI**: Automated maintenance scheduling and alert systems
- **Frontend**: Streamlit with real-time visualization
- **Data Storage**: SQLite for historical data

*This application demonstrates the convergence of IoT, AI, and Physical AI technologies for smart factory automation.*
""")