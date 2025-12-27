import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import json
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Industrial Predictive Maintenance System",
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
            'Generator': {'temp_range': (45, 85), 'vibration_range': (0.1, 1.5), 'pressure_range': (12, 45)}
        }
    
    def generate_sensor_data(self, equipment_type, num_readings=100, anomaly_probability=0.05):
        """Generate simulated IoT sensor data"""
        if equipment_type not in self.equipment_types:
            raise ValueError(f"Unknown equipment type: {equipment_type}")
        
        ranges = self.equipment_types[equipment_type]
        data = []
        
        for i in range(num_readings):
            timestamp = datetime.now() - timedelta(hours=num_readings-i)
            
            # Normal operation with some noise
            temp = np.random.normal(
                (ranges['temp_range'][0] + ranges['temp_range'][1]) / 2, 5
            )
            vibration = np.random.normal(
                (ranges['vibration_range'][0] + ranges['vibration_range'][1]) / 2, 0.2
            )
            pressure = np.random.normal(
                (ranges['pressure_range'][0] + ranges['pressure_range'][1]) / 2, 3
            )
            
            # Introduce anomalies
            if np.random.random() < anomaly_probability:
                temp *= np.random.uniform(1.2, 1.5)  # Temperature spike
                vibration *= np.random.uniform(1.3, 2.0)  # Vibration increase
                pressure *= np.random.uniform(0.7, 0.9)  # Pressure drop
            
            data.append({
                'timestamp': timestamp,
                'equipment_id': f"{equipment_type}_001",
                'temperature': max(0, temp),
                'vibration': max(0, vibration),
                'pressure': max(0, pressure),
                'equipment_type': equipment_type
            })
        
        return pd.DataFrame(data)

class PredictiveMaintenanceAI:
    """AI-powered predictive maintenance system"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        features = df[['temperature', 'vibration', 'pressure']].copy()
        
        # Add rolling statistics
        features['temp_rolling_mean'] = df['temperature'].rolling(window=5).mean()
        features['vibration_rolling_std'] = df['vibration'].rolling(window=5).std()
        features['pressure_trend'] = df['pressure'].diff()
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        return features
    
    def train_models(self, df):
        """Train anomaly detection and failure prediction models"""
        features = self.prepare_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train anomaly detector
        self.anomaly_detector.fit(features_scaled)
        
        # Create synthetic failure labels for demonstration
        # In real scenario, this would be historical failure data
        failure_risk = np.random.beta(2, 8, len(df))  # Most equipment is healthy
        
        # Train failure predictor
        self.failure_predictor.fit(features_scaled, failure_risk)
        
        self.is_trained = True
        return True
    
    def predict_anomalies(self, df):
        """Detect anomalies in sensor data"""
        if not self.is_trained:
            return np.zeros(len(df))
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # -1 for anomalies, 1 for normal
        anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
        anomalies = self.anomaly_detector.predict(features_scaled)
        
        return anomaly_scores, anomalies
    
    def predict_failure_risk(self, df):
        """Predict failure risk for equipment"""
        if not self.is_trained:
            return np.zeros(len(df))
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        failure_risk = self.failure_predictor.predict(features_scaled)
        return failure_risk

class PhysicalAIRobotics:
    """Physical AI system for robotic maintenance recommendations"""
    
    def __init__(self):
        self.maintenance_actions = {
            'high_temperature': {
                'action': 'Cooling System Check',
                'robot_task': 'Deploy thermal imaging robot',
                'human_collaboration': 'Technician to verify cooling system',
                'priority': 'High',
                'estimated_time': '2 hours'
            },
            'high_vibration': {
                'action': 'Bearing Inspection',
                'robot_task': 'Vibration analysis robot deployment',
                'human_collaboration': 'Mechanical engineer for bearing replacement',
                'priority': 'Medium',
                'estimated_time': '3 hours'
            },
            'low_pressure': {
                'action': 'Pressure System Check',
                'robot_task': 'Automated pressure testing',
                'human_collaboration': 'Hydraulic specialist consultation',
                'priority': 'Medium',
                'estimated_time': '1.5 hours'
            },
            'anomaly_detected': {
                'action': 'Comprehensive Inspection',
                'robot_task': 'Multi-sensor diagnostic robot',
                'human_collaboration': 'Senior technician assessment',
                'priority': 'High',
                'estimated_time': '4 hours'
            }
        }
    
    def analyze_sensor_data(self, latest_data, anomaly_scores, failure_risk):
        """Analyze sensor data and generate maintenance recommendations"""
        recommendations = []
        
        # Check for specific issues
        temp_threshold = 75
        vibration_threshold = 1.8
        pressure_threshold = 20
        
        if latest_data['temperature'] > temp_threshold:
            recommendations.append(self.maintenance_actions['high_temperature'])
        
        if latest_data['vibration'] > vibration_threshold:
            recommendations.append(self.maintenance_actions['high_vibration'])
        
        if latest_data['pressure'] < pressure_threshold:
            recommendations.append(self.maintenance_actions['low_pressure'])
        
        # Check for anomalies
        if len(anomaly_scores) > 0 and anomaly_scores[-1] < -0.1:
            recommendations.append(self.maintenance_actions['anomaly_detected'])
        
        return recommendations
    
    def generate_robot_instructions(self, recommendations):
        """Generate specific instructions for robotic systems"""
        robot_instructions = []
        
        for rec in recommendations:
            instruction = {
                'task_id': f"TASK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'robot_type': rec['robot_task'],
                'priority': rec['priority'],
                'estimated_duration': rec['estimated_time'],
                'human_support_needed': rec['human_collaboration'],
                'safety_protocols': ['Lockout/Tagout', 'PPE Required', 'Area Isolation']
            }
            robot_instructions.append(instruction)
        
        return robot_instructions

class DatabaseManager:
    """Manage SQLite database for historical data"""
    
    def __init__(self, db_path='maintenance_system.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sensor data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equipment_id TEXT,
                temperature REAL,
                vibration REAL,
                pressure REAL,
                equipment_type TEXT,
                anomaly_score REAL,
                failure_risk REAL
            )
        ''')
        
        # Maintenance actions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equipment_id TEXT,
                action_type TEXT,
                robot_task TEXT,
                human_collaboration TEXT,
                priority TEXT,
                status TEXT,
                completion_time TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_sensor_data(self, df, anomaly_scores=None, failure_risks=None):
        """Save sensor data to database"""
        conn = sqlite3.connect(self.db_path)
        
        for idx, row in df.iterrows():
            anomaly_score = anomaly_scores[idx] if anomaly_scores is not None else 0
            failure_risk = failure_risks[idx] if failure_risks is not None else 0
            
            conn.execute('''
                INSERT INTO sensor_data 
                (timestamp, equipment_id, temperature, vibration, pressure, 
                 equipment_type, anomaly_score, failure_risk)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['timestamp'].isoformat(),
                row['equipment_id'],
                row['temperature'],
                row['vibration'],
                row['pressure'],
                row['equipment_type'],
                anomaly_score,
                failure_risk
            ))
        
        conn.commit()
        conn.close()
    
    def get_recent_data(self, equipment_id, hours=24):
        """Get recent sensor data for equipment"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM sensor_data 
            WHERE equipment_id = ? 
            AND datetime(timestamp) > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours)
        
        df = pd.read_sql_query(query, conn, params=(equipment_id,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

def main():
    """Main Streamlit application"""
    st.title("🏭 Smart Industrial Predictive Maintenance System")
    st.markdown("""
    **Combining IoT Sensors + AI Analytics + Physical AI Robotics**
    
    This system integrates:
    - 📡 **IoT Sensors**: Real-time equipment monitoring
    - 🤖 **AI Analytics**: Predictive maintenance algorithms
    - 🦾 **Physical AI**: Robotic maintenance recommendations
    """)
    
    # Initialize components
    if 'sensor_sim' not in st.session_state:
        st.session_state.sensor_sim = IoTSensorSimulator()
        st.session_state.ai_system = PredictiveMaintenanceAI()
        st.session_state.robot_system = PhysicalAIRobotics()
        st.session_state.db_manager = DatabaseManager()
    
    # Sidebar controls
    st.sidebar.header("🔧 System Controls")
    
    equipment_type = st.sidebar.selectbox(
        "Select Equipment Type",
        list(st.session_state.sensor_sim.equipment_types.keys())
    )
    
    num_readings = st.sidebar.slider("Number of Sensor Readings", 50, 500, 100)
    anomaly_prob = st.sidebar.slider("Anomaly Probability", 0.0, 0.2, 0.05)
    
    if st.sidebar.button("🔄 Generate New Data"):
        # Generate sensor data
        with st.spinner("Generating IoT sensor data..."):
            sensor_data = st.session_state.sensor_sim.generate_sensor_data(
                equipment_type, num_readings, anomaly_prob
            )
        
        # Train AI models
        with st.spinner("Training AI models..."):
            st.session_state.ai_system.train_models(sensor_data)
        
        # Get predictions
        anomaly_scores, anomalies = st.session_state.ai_system.predict_anomalies(sensor_data)
        failure_risks = st.session_state.ai_system.predict_failure_risk(sensor_data)
        
        # Save to database
        st.session_state.db_manager.save_sensor_data(sensor_data, anomaly_scores, failure_risks)
        
        # Store in session state
        st.session_state.sensor_data = sensor_data
        st.session_state.anomaly_scores = anomaly_scores
        st.session_state.failure_risks = failure_risks
        st.session_state.anomalies = anomalies
        
        st.sidebar.success("✅ Data generated and models trained!")
    
    # Main dashboard
    if 'sensor_data' in st.session_state:
        df = st.session_state.sensor_data
        anomaly_scores = st.session_state.anomaly_scores
        failure_risks = st.session_state.failure_risks
        anomalies = st.session_state.anomalies
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        latest_data = df.iloc[-1]
        
        with col1:
            st.metric(
                "🌡️ Temperature (°C)",
                f"{latest_data['temperature']:.1f}",
                f"{latest_data['temperature'] - df.iloc[-2]['temperature']:.1f}"
            )
        
        with col2:
            st.metric(
                "📳 Vibration (mm/s)",
                f"{latest_data['vibration']:.2f}",
                f"{latest_data['vibration'] - df.iloc[-2]['vibration']:.2f}"
            )
        
        with col3:
            st.metric(
                "💨 Pressure (bar)",
                f"{latest_data['pressure']:.1f}",
                f"{latest_data['pressure'] - df.iloc[-2]['pressure']:.1f}"
            )
        
        with col4:
            risk_level = "🔴 High" if failure_risks[-1] > 0.7 else "🟡 Medium" if failure_risks[-1] > 0.3 else "🟢 Low"
            st.metric(
                "⚠️ Failure Risk",
                risk_level,
                f"{failure_risks[-1]:.3f}"
            )
        
        # Sensor data visualization
        st.subheader("📊 Real-time Sensor Data")
        
        tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🔍 Anomaly Detection", "🎯 Failure Prediction"])
        
        with tab1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['temperature'],
                mode='lines',
                name='Temperature (°C)',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['vibration'] * 20,  # Scale for visibility
                mode='lines',
                name='Vibration (×20)',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['pressure'],
                mode='lines',
                name='Pressure (bar)',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="IoT Sensor Readings Over Time",
                xaxis_title="Time",
                yaxis_title="Sensor Values",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Anomaly detection visualization
            anomaly_df = df.copy()
            anomaly_df['anomaly_score'] = anomaly_scores
            anomaly_df['is_anomaly'] = anomalies == -1
            
            fig = px.scatter(
                anomaly_df,
                x='timestamp',
                y='anomaly_score',
                color='is_anomaly',
                title="Anomaly Detection Results",
                color_discrete_map={True: 'red', False: 'blue'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            num_anomalies = sum(anomalies == -1)
            st.info(f"🔍 Detected {num_anomalies} anomalies out of {len(df)} readings ({num_anomalies/len(df)*100:.1f}%)")
        
        with tab3:
            # Failure prediction visualization
            failure_df = df.copy()
            failure_df['failure_risk'] = failure_risks
            
            fig = px.line(
                failure_df,
                x='timestamp',
                y='failure_risk',
                title="Equipment Failure Risk Prediction",
                color_discrete_sequence=['orange']
            )
            
            # Add risk threshold lines
            fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig.add_hline(y=0.3, line_dash="dash", line_color="yellow", annotation_text="Medium Risk")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Physical AI Recommendations
        st.subheader("🤖 Physical AI Maintenance Recommendations")
        
        recommendations = st.session_state.robot_system.analyze_sensor_data(
            latest_data, anomaly_scores, failure_risks[-1]
        )
        
        if recommendations:
            st.warning(f"⚠️ {len(recommendations)} maintenance action(s) recommended!")
            
            for i, rec in enumerate(recommendations):
                with st.expander(f"🔧 {rec['action']} - Priority: {rec['priority']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🤖 Robot Task:**")
                        st.write(rec['robot_task'])
                        st.write("**⏱️ Estimated Time:**")
                        st.write(rec['estimated_time'])
                    
                    with col2:
                        st.write("**👥 Human Collaboration:**")
                        st.write(rec['human_collaboration'])
                        st.write("**🎯 Priority Level:**")
                        st.write(rec['priority'])
            
            # Generate robot instructions
            robot_instructions = st.session_state.robot_system.generate_robot_instructions(recommendations)
            
            st.subheader("🦾 Robot Task Instructions")
            
            for instruction in robot_instructions:
                st.json(instruction)
        
        else:
            st.success("✅ No immediate maintenance actions required. Equipment operating normally.")
        
        # Equipment Health Dashboard
        st.subheader("📋 Equipment Health Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Health score calculation
            temp_score = max(0, 100 - (latest_data['temperature'] - 50) * 2)
            vibration_score = max(0, 100 - latest_data['vibration'] * 30)
            pressure_score = max(0, 100 - abs(latest_data['pressure'] - 35) * 2)
            
            overall_health = (temp_score + vibration_score + pressure_score) / 3
            
            st.metric("🏥 Overall Health Score", f"{overall_health:.1f}%")
            
            if overall_health > 80:
                st.success("Equipment in excellent condition")
            elif overall_health > 60:
                st.warning("Equipment needs attention")
            else:
                st.error("Equipment requires immediate maintenance")
        
        with col2:
            st.write("**📊 Component Health:**")
            st.progress(temp_score/100, "Temperature System")
            st.progress(vibration_score/100, "Mechanical Components")
            st.progress(pressure_score/100, "Pressure System")
        
        with col3:
            st.write("**🔮 Predictive Insights:**")
            
            if failure_risks[-1] > 0.7:
                st.error("High failure risk detected")
                st.write("Recommended: Immediate inspection")
            elif failure_risks[-1] > 0.3:
                st.warning("Moderate failure risk")
                st.write("Recommended: Schedule maintenance")
            else:
                st.success("Low failure risk")
                st.write("Continue normal operation")
    
    else:
        st.info("👆 Click 'Generate New Data' in the sidebar to start monitoring equipment!")
        
        # Show system capabilities
        st.subheader("🚀 System Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📡 IoT Integration**
            - Multi-sensor monitoring
            - Real-time data collection
            - Edge computing support
            - Industrial protocols
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Analytics**
            - Anomaly detection
            - Failure prediction
            - Pattern recognition
            - Adaptive learning
            """)
        
        with col3:
            st.markdown("""
            **🦾 Physical AI**
            - Robotic diagnostics
            - Human-robot collaboration
            - Automated maintenance
            - Safety protocols
            """)

if __name__ == "__main__":
    main()