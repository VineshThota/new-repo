import streamlit as st
import pandas as pd
import numpy as np
import cv2
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional
import sqlite3
import os

# Configuration and Data Models
@dataclass
class IoTSensor:
    sensor_id: str
    sensor_type: str
    location: str
    status: str
    last_reading: float
    timestamp: datetime
    
@dataclass
class RobotStatus:
    robot_id: str
    position: tuple
    task: str
    status: str
    battery_level: float
    collaboration_mode: bool
    human_proximity: float
    
@dataclass
class QualityCheck:
    product_id: str
    defect_detected: bool
    confidence_score: float
    defect_type: str
    timestamp: datetime

class SmartFactoryAI:
    def __init__(self):
        self.sensors = {}
        self.robots = {}
        self.quality_checks = []
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.scaler = StandardScaler()
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for storing factory data"""
        self.conn = sqlite3.connect('factory_data.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_id TEXT,
                sensor_type TEXT,
                location TEXT,
                reading REAL,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS robot_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                robot_id TEXT,
                position_x REAL,
                position_y REAL,
                task TEXT,
                status TEXT,
                battery_level REAL,
                collaboration_mode BOOLEAN,
                human_proximity REAL,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                defect_detected BOOLEAN,
                confidence_score REAL,
                defect_type TEXT,
                timestamp DATETIME
            )
        ''')
        
        self.conn.commit()
    
    def simulate_iot_sensors(self):
        """Simulate IoT sensor data for demo purposes"""
        sensor_types = ['temperature', 'vibration', 'pressure', 'humidity', 'noise']
        locations = ['Assembly Line 1', 'Assembly Line 2', 'Quality Control', 'Packaging', 'Storage']
        
        for i in range(10):
            sensor_id = f"sensor_{i+1:03d}"
            sensor_type = np.random.choice(sensor_types)
            location = np.random.choice(locations)
            
            # Generate realistic sensor readings based on type
            if sensor_type == 'temperature':
                reading = np.random.normal(25, 3)  # 25°C ± 3°C
            elif sensor_type == 'vibration':
                reading = np.random.exponential(2)  # Exponential distribution for vibration
            elif sensor_type == 'pressure':
                reading = np.random.normal(101.3, 0.5)  # Atmospheric pressure
            elif sensor_type == 'humidity':
                reading = np.random.uniform(40, 60)  # 40-60% humidity
            else:  # noise
                reading = np.random.normal(50, 10)  # 50dB ± 10dB
            
            status = 'normal' if np.random.random() > 0.1 else 'alert'
            
            sensor = IoTSensor(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                location=location,
                status=status,
                last_reading=reading,
                timestamp=datetime.now()
            )
            
            self.sensors[sensor_id] = sensor
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO sensor_data (sensor_id, sensor_type, location, reading, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (sensor_id, sensor_type, location, reading, datetime.now()))
            self.conn.commit()
    
    def simulate_robot_status(self):
        """Simulate robot status data"""
        tasks = ['welding', 'assembly', 'painting', 'inspection', 'material_handling']
        statuses = ['active', 'idle', 'maintenance', 'charging']
        
        for i in range(5):
            robot_id = f"robot_{i+1:02d}"
            position = (np.random.uniform(0, 100), np.random.uniform(0, 100))
            task = np.random.choice(tasks)
            status = np.random.choice(statuses)
            battery_level = np.random.uniform(20, 100)
            collaboration_mode = np.random.choice([True, False])
            human_proximity = np.random.uniform(0, 10)  # meters
            
            robot = RobotStatus(
                robot_id=robot_id,
                position=position,
                task=task,
                status=status,
                battery_level=battery_level,
                collaboration_mode=collaboration_mode,
                human_proximity=human_proximity
            )
            
            self.robots[robot_id] = robot
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO robot_status (robot_id, position_x, position_y, task, status, 
                                        battery_level, collaboration_mode, human_proximity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (robot_id, position[0], position[1], task, status, battery_level, 
                  collaboration_mode, human_proximity, datetime.now()))
            self.conn.commit()
    
    def computer_vision_quality_check(self, image_path=None):
        """Simulate computer vision quality control"""
        # In a real implementation, this would process actual images
        # For demo, we'll simulate quality check results
        
        defect_types = ['scratch', 'dent', 'color_variation', 'dimension_error', 'none']
        
        for i in range(20):
            product_id = f"product_{i+1:05d}"
            defect_detected = np.random.choice([True, False], p=[0.15, 0.85])  # 15% defect rate
            confidence_score = np.random.uniform(0.7, 0.99)
            defect_type = np.random.choice(defect_types[:-1]) if defect_detected else 'none'
            
            quality_check = QualityCheck(
                product_id=product_id,
                defect_detected=defect_detected,
                confidence_score=confidence_score,
                defect_type=defect_type,
                timestamp=datetime.now()
            )
            
            self.quality_checks.append(quality_check)
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO quality_checks (product_id, defect_detected, confidence_score, defect_type, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (product_id, defect_detected, confidence_score, defect_type, datetime.now()))
            self.conn.commit()
    
    def predictive_maintenance(self):
        """Analyze sensor data for predictive maintenance"""
        if not self.sensors:
            return {}
        
        # Collect sensor readings for analysis
        sensor_data = []
        for sensor in self.sensors.values():
            sensor_data.append([
                hash(sensor.sensor_type) % 100,  # Encode sensor type
                sensor.last_reading,
                hash(sensor.location) % 100  # Encode location
            ])
        
        if len(sensor_data) < 2:
            return {}
        
        # Normalize data and detect anomalies
        sensor_data_scaled = self.scaler.fit_transform(sensor_data)
        anomalies = self.anomaly_detector.fit_predict(sensor_data_scaled)
        
        # Create maintenance recommendations
        maintenance_alerts = {}
        for i, (sensor_id, sensor) in enumerate(self.sensors.items()):
            if anomalies[i] == -1:  # Anomaly detected
                maintenance_alerts[sensor_id] = {
                    'priority': 'high' if sensor.status == 'alert' else 'medium',
                    'recommendation': f'Check {sensor.sensor_type} sensor at {sensor.location}',
                    'last_reading': sensor.last_reading,
                    'timestamp': sensor.timestamp
                }
        
        return maintenance_alerts
    
    def human_robot_collaboration_safety(self):
        """Monitor human-robot collaboration safety"""
        safety_alerts = []
        
        for robot in self.robots.values():
            if robot.collaboration_mode and robot.human_proximity < 2.0:  # Less than 2 meters
                if robot.status == 'active':
                    safety_alerts.append({
                        'robot_id': robot.robot_id,
                        'alert_type': 'proximity_warning',
                        'message': f'Human detected within 2m of active robot {robot.robot_id}',
                        'proximity': robot.human_proximity,
                        'recommended_action': 'Reduce robot speed or pause operation'
                    })
            
            if robot.battery_level < 25:
                safety_alerts.append({
                    'robot_id': robot.robot_id,
                    'alert_type': 'low_battery',
                    'message': f'Robot {robot.robot_id} battery level critical: {robot.battery_level:.1f}%',
                    'recommended_action': 'Schedule charging cycle'
                })
        
        return safety_alerts

# Initialize the Smart Factory AI system
factory_ai = SmartFactoryAI()

# Streamlit Dashboard
def main():
    st.set_page_config(
        page_title="SmartFactory Collaborative AI",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏭 SmartFactory Collaborative AI Dashboard")
    st.markdown("""
    **Physical AI System for Smart Manufacturing**
    
    This dashboard demonstrates the integration of IoT sensors, computer vision AI, and human-robot collaboration 
    for next-generation smart manufacturing environments.
    """)
    
    # Sidebar controls
    st.sidebar.header("System Controls")
    
    if st.sidebar.button("🔄 Refresh Sensor Data"):
        factory_ai.simulate_iot_sensors()
        st.sidebar.success("Sensor data updated!")
    
    if st.sidebar.button("🤖 Update Robot Status"):
        factory_ai.simulate_robot_status()
        st.sidebar.success("Robot status updated!")
    
    if st.sidebar.button("👁️ Run Quality Check"):
        factory_ai.computer_vision_quality_check()
        st.sidebar.success("Quality check completed!")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🌡️ IoT Sensors", 
        "🤖 Robots", 
        "🔍 Quality Control", 
        "⚠️ Maintenance"
    ])
    
    with tab1:
        st.header("Factory Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Sensors", 
                len([s for s in factory_ai.sensors.values() if s.status == 'normal']),
                delta=len(factory_ai.sensors) - len([s for s in factory_ai.sensors.values() if s.status == 'normal'])
            )
        
        with col2:
            active_robots = len([r for r in factory_ai.robots.values() if r.status == 'active'])
            st.metric("Active Robots", active_robots, delta=active_robots - len(factory_ai.robots))
        
        with col3:
            if factory_ai.quality_checks:
                defect_rate = len([q for q in factory_ai.quality_checks if q.defect_detected]) / len(factory_ai.quality_checks) * 100
                st.metric("Defect Rate", f"{defect_rate:.1f}%", delta=f"{defect_rate - 15:.1f}%")
            else:
                st.metric("Defect Rate", "0.0%")
        
        with col4:
            maintenance_alerts = factory_ai.predictive_maintenance()
            st.metric("Maintenance Alerts", len(maintenance_alerts), delta=len(maintenance_alerts))
        
        # Factory layout visualization
        st.subheader("Factory Layout")
        
        if factory_ai.robots:
            # Create factory layout plot
            fig = go.Figure()
            
            # Add robots
            robot_x = [r.position[0] for r in factory_ai.robots.values()]
            robot_y = [r.position[1] for r in factory_ai.robots.values()]
            robot_colors = ['red' if r.status == 'maintenance' else 'green' if r.status == 'active' else 'orange' 
                           for r in factory_ai.robots.values()]
            robot_text = [f"{r.robot_id}<br>Task: {r.task}<br>Battery: {r.battery_level:.1f}%" 
                         for r in factory_ai.robots.values()]
            
            fig.add_trace(go.Scatter(
                x=robot_x, y=robot_y,
                mode='markers+text',
                marker=dict(size=15, color=robot_colors),
                text=[r.robot_id for r in factory_ai.robots.values()],
                textposition="middle center",
                hovertext=robot_text,
                name="Robots"
            ))
            
            fig.update_layout(
                title="Real-time Factory Floor Layout",
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("IoT Sensor Network")
        
        if factory_ai.sensors:
            # Sensor status overview
            sensor_df = pd.DataFrame([
                {
                    'Sensor ID': s.sensor_id,
                    'Type': s.sensor_type,
                    'Location': s.location,
                    'Status': s.status,
                    'Last Reading': f"{s.last_reading:.2f}",
                    'Timestamp': s.timestamp.strftime('%H:%M:%S')
                }
                for s in factory_ai.sensors.values()
            ])
            
            st.dataframe(sensor_df, use_container_width=True)
            
            # Sensor readings visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Sensor type distribution
                sensor_types = [s.sensor_type for s in factory_ai.sensors.values()]
                type_counts = pd.Series(sensor_types).value_counts()
                
                fig_pie = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Sensor Type Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Sensor status distribution
                sensor_statuses = [s.status for s in factory_ai.sensors.values()]
                status_counts = pd.Series(sensor_statuses).value_counts()
                
                fig_bar = px.bar(
                    x=status_counts.index,
                    y=status_counts.values,
                    title="Sensor Status Overview",
                    color=status_counts.index,
                    color_discrete_map={'normal': 'green', 'alert': 'red'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No sensor data available. Click 'Refresh Sensor Data' to generate sample data.")
    
    with tab3:
        st.header("Robot Fleet Management")
        
        if factory_ai.robots:
            # Robot status table
            robot_df = pd.DataFrame([
                {
                    'Robot ID': r.robot_id,
                    'Position': f"({r.position[0]:.1f}, {r.position[1]:.1f})",
                    'Task': r.task,
                    'Status': r.status,
                    'Battery': f"{r.battery_level:.1f}%",
                    'Collaboration Mode': "✅" if r.collaboration_mode else "❌",
                    'Human Proximity': f"{r.human_proximity:.1f}m"
                }
                for r in factory_ai.robots.values()
            ])
            
            st.dataframe(robot_df, use_container_width=True)
            
            # Safety monitoring
            st.subheader("Human-Robot Collaboration Safety")
            safety_alerts = factory_ai.human_robot_collaboration_safety()
            
            if safety_alerts:
                for alert in safety_alerts:
                    if alert['alert_type'] == 'proximity_warning':
                        st.warning(f"⚠️ {alert['message']} - {alert['recommended_action']}")
                    elif alert['alert_type'] == 'low_battery':
                        st.error(f"🔋 {alert['message']} - {alert['recommended_action']}")
            else:
                st.success("✅ All robots operating safely")
            
            # Robot performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Battery levels
                battery_data = [(r.robot_id, r.battery_level) for r in factory_ai.robots.values()]
                battery_df = pd.DataFrame(battery_data, columns=['Robot', 'Battery Level'])
                
                fig_battery = px.bar(
                    battery_df,
                    x='Robot',
                    y='Battery Level',
                    title="Robot Battery Levels",
                    color='Battery Level',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_battery, use_container_width=True)
            
            with col2:
                # Task distribution
                tasks = [r.task for r in factory_ai.robots.values()]
                task_counts = pd.Series(tasks).value_counts()
                
                fig_tasks = px.pie(
                    values=task_counts.values,
                    names=task_counts.index,
                    title="Current Task Distribution"
                )
                st.plotly_chart(fig_tasks, use_container_width=True)
        else:
            st.info("No robot data available. Click 'Update Robot Status' to generate sample data.")
    
    with tab4:
        st.header("AI-Powered Quality Control")
        
        if factory_ai.quality_checks:
            # Quality metrics
            total_checks = len(factory_ai.quality_checks)
            defects_found = len([q for q in factory_ai.quality_checks if q.defect_detected])
            defect_rate = (defects_found / total_checks) * 100 if total_checks > 0 else 0
            avg_confidence = np.mean([q.confidence_score for q in factory_ai.quality_checks])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Inspections", total_checks)
            with col2:
                st.metric("Defects Detected", defects_found, delta=f"{defect_rate:.1f}%")
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            # Quality check results table
            quality_df = pd.DataFrame([
                {
                    'Product ID': q.product_id,
                    'Defect Detected': "❌" if q.defect_detected else "✅",
                    'Defect Type': q.defect_type,
                    'Confidence': f"{q.confidence_score:.2f}",
                    'Timestamp': q.timestamp.strftime('%H:%M:%S')
                }
                for q in factory_ai.quality_checks[-20:]  # Show last 20 checks
            ])
            
            st.dataframe(quality_df, use_container_width=True)
            
            # Defect analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Defect types distribution
                defect_types = [q.defect_type for q in factory_ai.quality_checks if q.defect_detected]
                if defect_types:
                    defect_counts = pd.Series(defect_types).value_counts()
                    
                    fig_defects = px.bar(
                        x=defect_counts.index,
                        y=defect_counts.values,
                        title="Defect Types Distribution",
                        color=defect_counts.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_defects, use_container_width=True)
                else:
                    st.info("No defects detected in current batch")
            
            with col2:
                # Confidence score distribution
                confidence_scores = [q.confidence_score for q in factory_ai.quality_checks]
                
                fig_confidence = px.histogram(
                    x=confidence_scores,
                    nbins=20,
                    title="AI Confidence Score Distribution",
                    labels={'x': 'Confidence Score', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
        else:
            st.info("No quality check data available. Click 'Run Quality Check' to generate sample data.")
    
    with tab5:
        st.header("Predictive Maintenance")
        
        maintenance_alerts = factory_ai.predictive_maintenance()
        
        if maintenance_alerts:
            st.subheader("🚨 Maintenance Alerts")
            
            for sensor_id, alert in maintenance_alerts.items():
                priority_color = "🔴" if alert['priority'] == 'high' else "🟡"
                st.warning(
                    f"{priority_color} **{alert['priority'].upper()} PRIORITY** - {alert['recommendation']}\n\n"
                    f"Last Reading: {alert['last_reading']:.2f} | "
                    f"Time: {alert['timestamp'].strftime('%H:%M:%S')}"
                )
        else:
            st.success("✅ All systems operating normally - No maintenance required")
        
        # Maintenance schedule
        st.subheader("📅 Maintenance Schedule")
        
        # Generate sample maintenance schedule
        maintenance_schedule = [
            {'Equipment': 'Assembly Line 1', 'Next Maintenance': 'Tomorrow', 'Type': 'Routine', 'Priority': 'Medium'},
            {'Equipment': 'Robot 01', 'Next Maintenance': 'In 3 days', 'Type': 'Battery Check', 'Priority': 'Low'},
            {'Equipment': 'Quality Control Camera', 'Next Maintenance': 'Next week', 'Type': 'Calibration', 'Priority': 'High'},
            {'Equipment': 'Conveyor Belt 2', 'Next Maintenance': 'In 2 weeks', 'Type': 'Belt Replacement', 'Priority': 'Medium'}
        ]
        
        schedule_df = pd.DataFrame(maintenance_schedule)
        st.dataframe(schedule_df, use_container_width=True)
        
        # Maintenance cost analysis
        st.subheader("💰 Maintenance Cost Analysis")
        
        # Generate sample cost data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        preventive_costs = [15000, 12000, 18000, 14000, 16000, 13000]
        reactive_costs = [25000, 30000, 20000, 35000, 22000, 28000]
        
        fig_costs = go.Figure()
        fig_costs.add_trace(go.Bar(name='Preventive', x=months, y=preventive_costs))
        fig_costs.add_trace(go.Bar(name='Reactive', x=months, y=reactive_costs))
        
        fig_costs.update_layout(
            title='Maintenance Costs: Preventive vs Reactive',
            xaxis_title='Month',
            yaxis_title='Cost ($)',
            barmode='group'
        )
        
        st.plotly_chart(fig_costs, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **SmartFactory Collaborative AI** - Powered by Physical AI, IoT Integration, and Human-Robot Collaboration
    
    🔧 **Technologies Used:** Python, Streamlit, OpenCV, Scikit-learn, Plotly, SQLite  
    🏭 **Features:** Real-time monitoring, Predictive maintenance, Quality control, Safety management  
    🤖 **AI Capabilities:** Computer vision, Anomaly detection, Predictive analytics, Edge computing
    """)

if __name__ == "__main__":
    main()