# Smart Manufacturing Physical AI System

🏭 **A comprehensive Python-based Physical AI application that combines IoT sensors, AI predictive maintenance, and robotic automation for smart manufacturing environments.**

## 🌟 Overview

This application addresses the critical problem of **industrial equipment downtime and inefficient maintenance scheduling** by integrating three cutting-edge technology domains:

- **IoT (Internet of Things)**: Real-time sensor monitoring of industrial equipment
- **AI (Artificial Intelligence)**: Machine learning-powered predictive maintenance algorithms
- **Physical AI**: Robotic systems for automated inspection and maintenance tasks

## 🚀 Key Features

### IoT Sensor Integration
- **Real-time monitoring** of 6 types of industrial equipment
- **Multi-parameter sensing**: Temperature, vibration, pressure, rotation speed, power consumption, and noise levels
- **Continuous data collection** with anomaly detection
- **Scalable sensor network** architecture

### AI-Powered Predictive Maintenance
- **Machine Learning Models**: Random Forest Classifier and Isolation Forest for anomaly detection
- **Failure Prediction**: Probabilistic failure forecasting with confidence scores
- **Risk Assessment**: Automated risk level categorization (LOW, MEDIUM, HIGH, CRITICAL)
- **Maintenance Scheduling**: Intelligent maintenance recommendations based on AI analysis

### Physical AI Robotics
- **Autonomous Robot Dispatch**: Automatic deployment of maintenance robots for critical issues
- **Multi-Robot Fleet**: Maintenance robots, inspection drones, and repair arms
- **Real-time Status Tracking**: Battery levels, location, and task status monitoring
- **Intelligent Task Assignment**: Optimal robot selection based on maintenance type

### Real-time Dashboard
- **Interactive Web Interface**: Modern, responsive dashboard built with HTML5/CSS3/JavaScript
- **Live Data Visualization**: Real-time charts and metrics
- **Alert Management**: Visual alerts with severity indicators
- **Equipment Monitoring**: Individual equipment status and analytics
- **System Control**: Start/stop system operations and emergency controls

## 🛠️ Technology Stack

### Backend (Python)
- **Flask**: Web framework for API and dashboard
- **scikit-learn**: Machine learning algorithms
- **NumPy & Pandas**: Data processing and analysis
- **Threading**: Concurrent sensor simulation and monitoring

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Real-time data updates and interactivity
- **CSS Grid & Flexbox**: Responsive layout system

### AI/ML Components
- **Random Forest Classifier**: Failure prediction model
- **Isolation Forest**: Anomaly detection algorithm
- **StandardScaler**: Feature normalization
- **Real-time Inference**: Live prediction on streaming sensor data

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/smart_manufacturing_physical_ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the dashboard**:
   Open your web browser and navigate to `http://localhost:5000`

## 🎯 Usage Guide

### Starting the System
1. Open the dashboard in your web browser
2. Click the **"🚀 Start System"** button
3. The system will initialize IoT sensors, train AI models, and start monitoring

### Monitoring Equipment
- **System Overview**: View total equipment count, active sensors, and alerts
- **IoT Sensors**: Real-time sensor readings from industrial equipment
- **AI Predictions**: Machine learning analysis and failure predictions
- **Maintenance Alerts**: Automated alerts with recommended actions
- **Robot Status**: Physical AI robot fleet status and battery levels

### Equipment Analytics
- Click on individual equipment cards to view detailed analytics
- Monitor specific equipment performance and health metrics
- View historical trends and prediction confidence scores

## 🔌 API Documentation

### System Control
- `POST /api/system/start` - Start the smart manufacturing system
- `GET /api/system/status` - Get current system status and metrics

### Sensor Data
- `GET /api/sensors/data` - Get latest sensor readings
- `GET /api/sensors/data?equipment_id=PUMP_001` - Get data for specific equipment

### Maintenance & Alerts
- `GET /api/maintenance/alerts` - Get current maintenance alerts
- `GET /api/analytics/equipment/<equipment_id>` - Get equipment analytics

### Robot Control
- `GET /api/robots/status` - Get robot fleet status
- `POST /api/robots/dispatch` - Dispatch robot for maintenance task

### Example API Response
```json
{
  "equipment_id": "PUMP_001",
  "current_status": "NORMAL",
  "failure_probability": 0.23,
  "risk_level": "LOW",
  "sensor_readings": {
    "temperature": 45.2,
    "vibration": 1.1,
    "pressure": 2.8,
    "rotation_speed": 1850,
    "power_consumption": 125.4,
    "noise_level": 52.1
  }
}
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   IoT Sensors   │───▶│   AI Engine     │───▶│ Physical AI     │
│                 │    │                  │    │ Robots          │
│ • Temperature   │    │ • Anomaly        │    │                 │
│ • Vibration     │    │   Detection      │    │ • Maintenance   │
│ • Pressure      │    │ • Failure        │    │   Robot         │
│ • RPM           │    │   Prediction     │    │ • Inspection    │
│ • Power         │    │ • Risk           │    │   Drone         │
│ • Noise         │    │   Assessment     │    │ • Repair Arm    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    │                     │
                    │ • Real-time         │
                    │   Monitoring        │
                    │ • Alert Management  │
                    │ • System Control    │
                    │ • Analytics         │
                    └─────────────────────┘
```

## 🎯 Use Cases

### Manufacturing Industries
- **Automotive**: Engine and assembly line monitoring
- **Aerospace**: Critical component health tracking
- **Electronics**: PCB manufacturing equipment monitoring
- **Pharmaceuticals**: Clean room equipment maintenance

### Industrial Applications
- **Oil & Gas**: Pipeline and refinery equipment monitoring
- **Power Generation**: Turbine and generator health tracking
- **Mining**: Heavy machinery predictive maintenance
- **Chemical Processing**: Reactor and pump monitoring

## 🔮 Future Enhancements

- **Edge Computing**: Deploy AI models on edge devices for faster response
- **Digital Twin**: Create virtual replicas of physical equipment
- **Advanced Robotics**: Integration with more sophisticated robotic systems
- **Cloud Integration**: AWS/Azure cloud deployment for scalability
- **Mobile App**: iOS/Android companion app for remote monitoring
- **Blockchain**: Immutable maintenance records and audit trails

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Trending Topic**: Physical AI and AIoT integration in smart manufacturing
- **Problem Addressed**: Industrial equipment downtime and maintenance inefficiency
- **Technology Integration**: IoT + AI + Physical AI for comprehensive manufacturing automation

## 📞 Support

For support, email vineshthota1@gmail.com or create an issue in the GitHub repository.

---

**Built with ❤️ using Python, Flask, scikit-learn, and modern web technologies**