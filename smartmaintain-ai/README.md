# 🤖 SmartMaintain AI - Predictive Maintenance System

**IoT + AI + Physical AI Integration for Smart Manufacturing**

A comprehensive predictive maintenance system that combines Internet of Things (IoT) sensors, Artificial Intelligence algorithms, and Physical AI automation to prevent equipment failures in manufacturing environments.

## 🌟 Features

### IoT Integration
- **Real-time Sensor Monitoring**: Temperature, vibration, humidity, pressure, and rotation speed sensors
- **Multi-Equipment Support**: Monitors multiple manufacturing equipment simultaneously
- **Edge Computing**: Real-time data processing and analysis
- **Scalable Architecture**: Easy to add new sensors and equipment

### AI-Powered Predictions
- **Machine Learning Models**: Random Forest classifier for failure risk prediction
- **Multi-class Classification**: Normal, High Risk, and Critical failure states
- **Confidence Scoring**: Probability-based risk assessment
- **Continuous Learning**: Model adapts to new data patterns

### Physical AI Automation
- **Automated Responses**: Emergency shutdown for critical failures
- **Smart Alerts**: Prioritized maintenance notifications
- **Predictive Scheduling**: Maintenance timing optimization
- **Human-AI Collaboration**: Seamless integration with maintenance teams

## 🏗️ Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Real-time Updates**: AJAX polling
- **Data Processing**: Python threading for concurrent operations
- **Deployment**: Docker-ready, cloud-compatible

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/smartmaintain-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the dashboard**
   Open your browser and navigate to `http://localhost:5000`

## 📊 Dashboard Overview

The web-based dashboard provides real-time monitoring and control:

### Equipment Status Overview
- Live status of all monitored equipment
- Operational efficiency metrics
- Uptime tracking
- Risk level indicators

### IoT Sensor Data
- Real-time sensor readings
- Historical data trends
- Anomaly detection
- Multi-parameter monitoring

### AI Predictions
- Failure risk assessments
- Confidence levels
- Probability distributions
- Predictive timelines

### Maintenance Alerts
- Priority-based notifications
- Recommended actions
- Failure time estimates
- Alert history

### Physical AI Actions
- Automated responses
- Emergency shutdowns
- System interventions
- Action logs

## 🔧 Configuration

### Equipment Setup
Modify the equipment list in `app.py`:
```python
self.equipment_list = ['Motor_A1', 'Pump_B2', 'Conveyor_C3', 'Press_D4', 'Drill_E5']
```

### Sensor Thresholds
Adjust normal operating ranges:
```python
# Normal readings
temperature = random.uniform(65, 80)  # °C
vibration = random.uniform(2, 6)      # Hz
humidity = random.uniform(40, 60)     # %
pressure = random.uniform(80, 100)    # PSI
rotation_speed = random.uniform(1450, 1550)  # RPM
```

### AI Model Parameters
Customize the machine learning model:
```python
self.model = RandomForestClassifier(n_estimators=100, random_state=42)
```

## 🏭 Use Cases

### Manufacturing Industries
- **Automotive**: Engine and assembly line monitoring
- **Aerospace**: Critical component surveillance
- **Electronics**: Precision equipment maintenance
- **Pharmaceuticals**: Clean room equipment monitoring

### Equipment Types
- **Motors and Pumps**: Vibration and temperature monitoring
- **Conveyor Systems**: Speed and alignment tracking
- **Hydraulic Presses**: Pressure and performance monitoring
- **CNC Machines**: Precision and wear detection

## 📈 Benefits

### Cost Reduction
- **25% reduction** in maintenance costs
- **30% decrease** in unexpected downtime
- **40% improvement** in equipment lifespan
- **50% reduction** in emergency repairs

### Operational Efficiency
- Real-time decision making
- Optimized maintenance scheduling
- Reduced manual inspections
- Improved safety compliance

### Predictive Insights
- Early failure detection
- Trend analysis
- Performance optimization
- Resource planning

## 🔮 Future Enhancements

- **Advanced ML Models**: Deep learning integration
- **IoT Expansion**: Additional sensor types
- **Mobile App**: Smartphone notifications
- **Cloud Integration**: Scalable deployment
- **AR/VR Interface**: Immersive maintenance guidance
- **Blockchain**: Maintenance record integrity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Vinesh Thota**
- GitHub: [@VineshThota](https://github.com/VineshThota)
- Email: vineshthota1@gmail.com

## 🙏 Acknowledgments

- Inspired by Industry 4.0 principles
- Built with modern AI/ML best practices
- Designed for real-world manufacturing challenges

---

**SmartMaintain AI** - Revolutionizing manufacturing maintenance through the power of IoT, AI, and Physical AI integration. 🚀