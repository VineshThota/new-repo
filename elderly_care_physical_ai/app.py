import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import queue

# Configuration and Data Models
@dataclass
class SensorReading:
    timestamp: datetime
    sensor_type: str
    value: float
    location: str
    patient_id: str

@dataclass
class HealthAlert:
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    patient_id: str
    sensor_data: Dict

class IoTSensorSimulator:
    """Simulates various IoT sensors for elderly care monitoring"""
    
    def __init__(self):
        self.sensors = {
            'heart_rate': {'min': 60, 'max': 100, 'normal': 75},
            'blood_pressure_systolic': {'min': 90, 'max': 140, 'normal': 120},
            'blood_pressure_diastolic': {'min': 60, 'max': 90, 'normal': 80},
            'temperature': {'min': 36.1, 'max': 37.2, 'normal': 36.8},
            'oxygen_saturation': {'min': 95, 'max': 100, 'normal': 98},
            'fall_detection': {'min': 0, 'max': 1, 'normal': 0},
            'room_temperature': {'min': 18, 'max': 26, 'normal': 22},
            'humidity': {'min': 30, 'max': 70, 'normal': 45},
            'motion_detection': {'min': 0, 'max': 10, 'normal': 3},
            'sleep_quality': {'min': 1, 'max': 10, 'normal': 7}
        }
    
    def generate_reading(self, sensor_type: str, patient_id: str = "patient_001") -> SensorReading:
        """Generate a realistic sensor reading with occasional anomalies"""
        sensor_config = self.sensors[sensor_type]
        
        # 95% normal readings, 5% anomalies
        if random.random() < 0.95:
            # Normal reading with small variation
            base_value = sensor_config['normal']
            variation = (sensor_config['max'] - sensor_config['min']) * 0.1
            value = base_value + random.uniform(-variation, variation)
        else:
            # Anomalous reading
            if random.random() < 0.5:
                value = sensor_config['min'] - random.uniform(0, sensor_config['min'] * 0.2)
            else:
                value = sensor_config['max'] + random.uniform(0, sensor_config['max'] * 0.2)
        
        # Ensure value is within reasonable bounds
        value = max(0, value)
        
        return SensorReading(
            timestamp=datetime.now(),
            sensor_type=sensor_type,
            value=round(value, 2),
            location="home",
            patient_id=patient_id
        )

class AIHealthAnalyzer:
    """AI-powered health analytics and anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.health_patterns = {}
        self.alert_thresholds = {
            'heart_rate': {'low': 50, 'high': 120},
            'blood_pressure_systolic': {'low': 80, 'high': 160},
            'blood_pressure_diastolic': {'low': 50, 'high': 100},
            'temperature': {'low': 35.5, 'high': 38.5},
            'oxygen_saturation': {'low': 90, 'high': 100},
            'fall_detection': {'low': 0, 'high': 0.5}
        }
    
    def analyze_health_data(self, readings: List[SensorReading]) -> Dict:
        """Analyze health data and detect anomalies using AI"""
        if not readings:
            return {'status': 'no_data', 'alerts': []}
        
        # Convert readings to DataFrame
        data = []
        for reading in readings:
            data.append({
                'timestamp': reading.timestamp,
                'sensor_type': reading.sensor_type,
                'value': reading.value,
                'patient_id': reading.patient_id
            })
        
        df = pd.DataFrame(data)
        alerts = []
        
        # Check for immediate health alerts
        for sensor_type in self.alert_thresholds:
            sensor_data = df[df['sensor_type'] == sensor_type]
            if not sensor_data.empty:
                latest_value = sensor_data.iloc[-1]['value']
                thresholds = self.alert_thresholds[sensor_type]
                
                if latest_value < thresholds['low']:
                    alerts.append(HealthAlert(
                        timestamp=datetime.now(),
                        alert_type=f"{sensor_type}_low",
                        severity="high" if sensor_type in ['heart_rate', 'oxygen_saturation'] else "medium",
                        message=f"{sensor_type.replace('_', ' ').title()} is critically low: {latest_value}",
                        patient_id=readings[0].patient_id,
                        sensor_data={'value': latest_value, 'threshold': thresholds['low']}
                    ))
                elif latest_value > thresholds['high']:
                    alerts.append(HealthAlert(
                        timestamp=datetime.now(),
                        alert_type=f"{sensor_type}_high",
                        severity="high" if sensor_type in ['heart_rate', 'blood_pressure_systolic'] else "medium",
                        message=f"{sensor_type.replace('_', ' ').title()} is critically high: {latest_value}",
                        patient_id=readings[0].patient_id,
                        sensor_data={'value': latest_value, 'threshold': thresholds['high']}
                    ))
        
        # AI-based pattern analysis
        health_score = self._calculate_health_score(df)
        trend_analysis = self._analyze_trends(df)
        
        return {
            'status': 'analyzed',
            'alerts': alerts,
            'health_score': health_score,
            'trends': trend_analysis,
            'recommendations': self._generate_recommendations(alerts, health_score)
        }
    
    def _calculate_health_score(self, df: pd.DataFrame) -> float:
        """Calculate overall health score using AI algorithms"""
        if df.empty:
            return 50.0
        
        scores = []
        for sensor_type in ['heart_rate', 'blood_pressure_systolic', 'oxygen_saturation', 'temperature']:
            sensor_data = df[df['sensor_type'] == sensor_type]
            if not sensor_data.empty:
                latest_value = sensor_data.iloc[-1]['value']
                if sensor_type in self.alert_thresholds:
                    thresholds = self.alert_thresholds[sensor_type]
                    # Normalize score between 0-100
                    if thresholds['low'] <= latest_value <= thresholds['high']:
                        scores.append(85 + random.uniform(-10, 15))  # Good health
                    else:
                        scores.append(40 + random.uniform(-20, 20))  # Poor health
        
        return round(np.mean(scores) if scores else 50.0, 1)
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze health trends over time"""
        trends = {}
        for sensor_type in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor_type].sort_values('timestamp')
            if len(sensor_data) >= 2:
                values = sensor_data['value'].values
                if len(values) >= 3:
                    # Simple trend calculation
                    recent_avg = np.mean(values[-3:])
                    older_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]
                    trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                    trends[sensor_type] = trend
        return trends
    
    def _generate_recommendations(self, alerts: List[HealthAlert], health_score: float) -> List[str]:
        """Generate AI-powered health recommendations"""
        recommendations = []
        
        if health_score < 60:
            recommendations.append("🏥 Consider scheduling a medical consultation")
            recommendations.append("📞 Contact healthcare provider for immediate assessment")
        
        for alert in alerts:
            if "heart_rate" in alert.alert_type:
                if "high" in alert.alert_type:
                    recommendations.append("💓 Practice deep breathing exercises and avoid strenuous activity")
                else:
                    recommendations.append("💓 Light physical activity may help improve heart rate")
            
            elif "blood_pressure" in alert.alert_type:
                recommendations.append("🩺 Monitor blood pressure closely and reduce sodium intake")
            
            elif "oxygen_saturation" in alert.alert_type:
                recommendations.append("🫁 Ensure proper ventilation and consider oxygen therapy consultation")
        
        if not recommendations:
            recommendations.append("✅ Health metrics are within normal ranges")
            recommendations.append("🚶‍♂️ Continue regular physical activity and healthy lifestyle")
        
        return recommendations

class PhysicalAIAssistant:
    """Physical AI system for emergency response and assistance coordination"""
    
    def __init__(self):
        self.emergency_contacts = [
            {"name": "Emergency Services", "phone": "911", "type": "emergency"},
            {"name": "Family Doctor", "phone": "+1-555-0123", "type": "medical"},
            {"name": "Family Member", "phone": "+1-555-0456", "type": "family"},
            {"name": "Caregiver", "phone": "+1-555-0789", "type": "caregiver"}
        ]
        self.smart_devices = {
            "lights": {"status": "auto", "brightness": 75},
            "thermostat": {"temperature": 22, "mode": "auto"},
            "door_locks": {"status": "locked", "auto_unlock": True},
            "security_camera": {"status": "active", "recording": True},
            "medication_dispenser": {"status": "ready", "next_dose": "14:00"}
        }
    
    def handle_emergency_alert(self, alert: HealthAlert) -> Dict:
        """Handle emergency situations with physical AI response"""
        response_actions = []
        
        if alert.severity == "high":
            # Immediate emergency response
            response_actions.extend([
                "🚨 Emergency protocol activated",
                "📞 Contacting emergency services",
                "💡 Turning on all lights for emergency access",
                "🔓 Unlocking front door for emergency responders",
                "📹 Activating emergency recording mode",
                "📱 Sending alerts to all emergency contacts"
            ])
            
            # Simulate smart home automation
            self.smart_devices["lights"]["brightness"] = 100
            self.smart_devices["door_locks"]["status"] = "unlocked"
            self.smart_devices["security_camera"]["recording"] = True
        
        elif alert.severity == "medium":
            response_actions.extend([
                "⚠️ Health monitoring alert activated",
                "📞 Notifying primary caregiver",
                "💊 Checking medication schedule",
                "🌡️ Adjusting room environment for comfort"
            ])
        
        # Log the emergency response
        emergency_log = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "actions_taken": response_actions,
            "contacts_notified": [contact["name"] for contact in self.emergency_contacts if alert.severity == "high" or contact["type"] in ["medical", "caregiver"]]
        }
        
        return {
            "status": "emergency_handled",
            "actions": response_actions,
            "log": emergency_log
        }
    
    def coordinate_assistance(self, health_score: float, recommendations: List[str]) -> Dict:
        """Coordinate physical assistance based on health analysis"""
        assistance_actions = []
        
        if health_score < 70:
            assistance_actions.extend([
                "🤖 Activating assistance robot for mobility support",
                "📋 Preparing health summary for medical consultation",
                "💊 Verifying medication adherence schedule",
                "🏠 Optimizing home environment for safety"
            ])
        
        # Smart home optimization
        if "blood_pressure" in str(recommendations):
            assistance_actions.append("🌡️ Adjusting room temperature to reduce stress")
            self.smart_devices["thermostat"]["temperature"] = 20
        
        if "heart_rate" in str(recommendations):
            assistance_actions.append("💡 Dimming lights to promote relaxation")
            self.smart_devices["lights"]["brightness"] = 30
        
        return {
            "status": "assistance_coordinated",
            "actions": assistance_actions,
            "smart_devices": self.smart_devices
        }

class ElderlyCareDashboard:
    """Main dashboard for elderly care monitoring system"""
    
    def __init__(self):
        self.sensor_simulator = IoTSensorSimulator()
        self.ai_analyzer = AIHealthAnalyzer()
        self.physical_ai = PhysicalAIAssistant()
        self.sensor_data = []
        self.alerts_history = []
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Physical AI Elderly Care Monitor",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏥 Physical AI Elderly Care Monitoring System")
        st.markdown("### Combining IoT Sensors + AI Analytics + Physical Assistance")
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            patient_id = st.selectbox("Select Patient", ["patient_001", "patient_002", "patient_003"])
            
            if st.button("🔄 Generate New Sensor Data"):
                self._generate_sensor_readings(patient_id)
            
            if st.button("🚨 Simulate Emergency"):
                self._simulate_emergency(patient_id)
            
            st.markdown("---")
            st.header("📊 System Status")
            st.success("✅ IoT Sensors: Active")
            st.success("✅ AI Analytics: Running")
            st.success("✅ Physical AI: Ready")
            st.info(f"📡 Connected Sensors: {len(self.sensor_simulator.sensors)}")
        
        # Main dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._display_health_metrics(patient_id)
            self._display_sensor_charts()
        
        with col2:
            self._display_ai_analysis()
            self._display_physical_ai_status()
        
        # Bottom section
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            self._display_alerts_history()
        
        with col4:
            self._display_smart_home_controls()
    
    def _generate_sensor_readings(self, patient_id: str):
        """Generate new sensor readings"""
        new_readings = []
        for sensor_type in self.sensor_simulator.sensors.keys():
            reading = self.sensor_simulator.generate_reading(sensor_type, patient_id)
            new_readings.append(reading)
        
        self.sensor_data.extend(new_readings)
        
        # Keep only last 100 readings
        if len(self.sensor_data) > 100:
            self.sensor_data = self.sensor_data[-100:]
        
        # Analyze the data
        analysis = self.ai_analyzer.analyze_health_data(new_readings)
        
        # Handle any alerts
        if analysis['alerts']:
            for alert in analysis['alerts']:
                self.alerts_history.append(alert)
                emergency_response = self.physical_ai.handle_emergency_alert(alert)
        
        st.success(f"✅ Generated {len(new_readings)} new sensor readings")
    
    def _simulate_emergency(self, patient_id: str):
        """Simulate an emergency situation"""
        emergency_alert = HealthAlert(
            timestamp=datetime.now(),
            alert_type="fall_detected",
            severity="high",
            message="Fall detected! Immediate assistance required.",
            patient_id=patient_id,
            sensor_data={"fall_confidence": 0.95, "impact_force": 8.2}
        )
        
        self.alerts_history.append(emergency_alert)
        emergency_response = self.physical_ai.handle_emergency_alert(emergency_alert)
        
        st.error("🚨 EMERGENCY SIMULATED: Fall Detected!")
        st.json(emergency_response)
    
    def _display_health_metrics(self, patient_id: str):
        """Display current health metrics"""
        st.header("📊 Real-Time Health Metrics")
        
        if not self.sensor_data:
            st.info("Click 'Generate New Sensor Data' to start monitoring")
            return
        
        # Get latest readings for each sensor type
        latest_readings = {}
        for reading in reversed(self.sensor_data):
            if reading.sensor_type not in latest_readings:
                latest_readings[reading.sensor_type] = reading
        
        # Display key health metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'heart_rate' in latest_readings:
                hr = latest_readings['heart_rate'].value
                st.metric("💓 Heart Rate", f"{hr} BPM", 
                         delta=f"{hr-75:.1f}" if hr != 75 else None)
        
        with col2:
            if 'oxygen_saturation' in latest_readings:
                o2 = latest_readings['oxygen_saturation'].value
                st.metric("🫁 Oxygen Sat", f"{o2:.1f}%", 
                         delta=f"{o2-98:.1f}" if o2 != 98 else None)
        
        with col3:
            if 'temperature' in latest_readings:
                temp = latest_readings['temperature'].value
                st.metric("🌡️ Temperature", f"{temp:.1f}°C", 
                         delta=f"{temp-36.8:.1f}" if temp != 36.8 else None)
        
        with col4:
            if 'blood_pressure_systolic' in latest_readings:
                bp = latest_readings['blood_pressure_systolic'].value
                st.metric("🩺 Blood Pressure", f"{bp:.0f} mmHg", 
                         delta=f"{bp-120:.0f}" if bp != 120 else None)
    
    def _display_sensor_charts(self):
        """Display sensor data charts"""
        st.header("📈 Sensor Data Trends")
        
        if not self.sensor_data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': reading.timestamp,
                'sensor_type': reading.sensor_type,
                'value': reading.value
            } for reading in self.sensor_data
        ])
        
        # Create charts for key health metrics
        health_sensors = ['heart_rate', 'blood_pressure_systolic', 'oxygen_saturation', 'temperature']
        
        for sensor in health_sensors:
            sensor_df = df[df['sensor_type'] == sensor]
            if not sensor_df.empty:
                fig = px.line(sensor_df, x='timestamp', y='value', 
                             title=f"{sensor.replace('_', ' ').title()} Over Time")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_ai_analysis(self):
        """Display AI analysis results"""
        st.header("🤖 AI Health Analysis")
        
        if not self.sensor_data:
            st.info("No data available for analysis")
            return
        
        # Get recent readings for analysis
        recent_readings = self.sensor_data[-20:] if len(self.sensor_data) >= 20 else self.sensor_data
        analysis = self.ai_analyzer.analyze_health_data(recent_readings)
        
        # Health Score
        health_score = analysis.get('health_score', 50)
        st.metric("🏥 Overall Health Score", f"{health_score}/100")
        
        # Health score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Health Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        if 'recommendations' in analysis:
            st.subheader("💡 AI Recommendations")
            for rec in analysis['recommendations']:
                st.write(f"• {rec}")
    
    def _display_physical_ai_status(self):
        """Display Physical AI system status"""
        st.header("🤖 Physical AI Assistant")
        
        # Emergency contacts
        st.subheader("📞 Emergency Contacts")
        for contact in self.physical_ai.emergency_contacts:
            st.write(f"• {contact['name']}: {contact['phone']} ({contact['type']})")
        
        # Assistance coordination
        if self.sensor_data:
            recent_readings = self.sensor_data[-10:]
            analysis = self.ai_analyzer.analyze_health_data(recent_readings)
            health_score = analysis.get('health_score', 50)
            recommendations = analysis.get('recommendations', [])
            
            assistance = self.physical_ai.coordinate_assistance(health_score, recommendations)
            
            if assistance['actions']:
                st.subheader("🔧 Active Assistance")
                for action in assistance['actions']:
                    st.write(f"• {action}")
    
    def _display_alerts_history(self):
        """Display alerts history"""
        st.header("🚨 Recent Alerts")
        
        if not self.alerts_history:
            st.info("No alerts recorded")
            return
        
        # Show last 5 alerts
        recent_alerts = self.alerts_history[-5:]
        
        for alert in reversed(recent_alerts):
            severity_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(alert.severity, "⚪")
            
            st.write(f"{severity_color} **{alert.timestamp.strftime('%H:%M:%S')}** - {alert.message}")
    
    def _display_smart_home_controls(self):
        """Display smart home device controls"""
        st.header("🏠 Smart Home Integration")
        
        for device, config in self.physical_ai.smart_devices.items():
            with st.expander(f"{device.replace('_', ' ').title()}"):
                st.json(config)

# Main application
def main():
    dashboard = ElderlyCareDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()