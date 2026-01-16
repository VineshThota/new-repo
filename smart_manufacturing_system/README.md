# Smart Manufacturing Safety & Predictive Maintenance System

## 🏭 Project Overview

A comprehensive Python-based application that combines **IoT sensors**, **AI algorithms**, and **Physical AI safety protocols** to create an intelligent manufacturing environment. This system addresses the trending LinkedIn topic of "Edge AI in Smart Manufacturing" by providing real-time monitoring, predictive maintenance, and human-robot collaboration safety.

## 🚀 Key Features

### IoT Integration
- **Real-time Sensor Monitoring**: Temperature, vibration, pressure, and proximity sensors
- **Edge Computing Simulation**: Local data processing for reduced latency
- **Equipment Health Tracking**: Continuous monitoring of manufacturing equipment
- **Data Visualization**: Live charts and dashboards for sensor data

### AI-Powered Analytics
- **Predictive Maintenance**: Machine learning algorithms to predict equipment failures
- **Anomaly Detection**: Isolation Forest algorithm for identifying unusual patterns
- **Health Score Calculation**: AI-driven equipment health assessment
- **Maintenance Scheduling**: Intelligent scheduling based on equipment condition

### Physical AI Safety System
- **Human Detection**: Computer vision-based human presence detection in robot zones
- **Collision Avoidance**: Real-time proximity monitoring and response
- **Emergency Protocols**: Automated safety responses and emergency stops
- **Zone Management**: Multi-zone safety monitoring for human-robot collaboration

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: scikit-learn, NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Chart.js for real-time data visualization
- **IoT Simulation**: Python threading for sensor data generation
- **Deployment**: Gunicorn WSGI server

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/smart_manufacturing_system
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the dashboard**:
   Open your browser and navigate to `http://localhost:5000`

## 🎯 Usage Guide

### Dashboard Overview
The main dashboard provides four key sections:

1. **Equipment Status**: Real-time health monitoring of manufacturing equipment
2. **Safety Zones**: Human-robot collaboration zone monitoring
3. **Sensor Data**: Live visualization of IoT sensor readings
4. **Predictive Maintenance**: AI-powered maintenance predictions

### Key Operations

#### Emergency Stop
- Click the red "EMERGENCY STOP" button to halt all robotic operations
- Activates safety protocols across all zones
- Requires manual reset to resume operations

#### System Reset
- Use "Reset All Systems" to return to normal operation
- Clears all safety alerts and resumes robotic activities
- Resets human detection counters

#### Data Refresh
- "Refresh Data" button manually updates all dashboard components
- Automatic updates occur every 3 seconds
- Real-time sensor data continuously streams

## 🔌 API Documentation

### Equipment Status
```
GET /api/equipment_status
Returns: JSON object with equipment health scores and status
```

### Safety Zones
```
GET /api/safety_zones
Returns: JSON object with human-robot zone information
```

### Sensor Data
```
GET /api/sensor_data
Returns: Array of latest sensor readings
```

### Maintenance Prediction
```
GET /api/maintenance_prediction/<equipment_id>
Returns: Predictive maintenance information for specific equipment
```

### Emergency Controls
```
POST /api/emergency_stop
Activates emergency stop protocol

POST /api/reset_systems
Resets all systems to normal operation
```

## 🏗️ System Architecture

### IoT Layer
- **Sensor Simulation**: Generates realistic sensor data with controlled randomness
- **Data Collection**: Continuous monitoring of temperature, vibration, pressure, proximity
- **Edge Processing**: Local data processing to reduce cloud dependency

### AI Processing Layer
- **Predictive Maintenance AI**: Uses Isolation Forest for anomaly detection
- **Health Score Calculation**: Combines multiple sensor inputs for equipment assessment
- **Pattern Recognition**: Identifies degradation patterns in equipment performance

### Physical AI Safety Layer
- **Human Detection Protocol**: Monitors human presence in robotic work zones
- **Collision Avoidance**: Real-time proximity analysis and response
- **Safety State Management**: Manages safety levels across multiple zones

### Web Interface Layer
- **Real-time Dashboard**: Live updates using AJAX and WebSocket-like polling
- **Responsive Design**: Mobile-friendly interface with Bootstrap
- **Interactive Controls**: Emergency stops, system resets, and manual overrides

## 📊 Key Metrics

- **Equipment Health Scores**: 0-100% scale with color-coded indicators
- **Safety Levels**: Safe (Green), Caution (Yellow), Danger (Red), Emergency (Flashing Red)
- **Maintenance Predictions**: Days until maintenance with priority levels
- **Sensor Thresholds**: Configurable limits for temperature, vibration, and pressure

## 🔒 Safety Features

1. **Multi-Zone Monitoring**: Independent safety zones with individual protocols
2. **Graduated Response**: Different actions based on threat level
3. **Emergency Override**: Manual emergency stop overrides all automated systems
4. **Fail-Safe Design**: System defaults to safe state on any error

## 🚀 Deployment Options

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🔮 Future Enhancements

- **Real IoT Integration**: Connect to actual industrial sensors
- **Computer Vision**: Implement actual camera-based human detection
- **Machine Learning Expansion**: Add more sophisticated AI models
- **Mobile App**: Develop companion mobile application
- **Cloud Integration**: Add cloud-based analytics and storage
- **Blockchain**: Implement blockchain for maintenance records

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vinesh Thota**
- GitHub: [@VineshThota](https://github.com/VineshThota)
- Email: vineshthota1@gmail.com

## 🙏 Acknowledgments

- Inspired by trending LinkedIn discussions on Edge AI in Manufacturing
- Built to address real-world IoT, AI, and Physical AI integration challenges
- Designed for the future of smart manufacturing and Industry 4.0

## 📈 Project Stats

- **Lines of Code**: 500+ Python, 400+ HTML/CSS/JS
- **AI Models**: Isolation Forest for anomaly detection
- **IoT Sensors**: 4 types (temperature, vibration, pressure, proximity)
- **Safety Zones**: 3 independent monitoring zones
- **Update Frequency**: Real-time (2-3 second intervals)

---

**Built with ❤️ for the future of smart manufacturing**