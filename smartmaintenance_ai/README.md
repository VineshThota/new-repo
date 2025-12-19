# SmartMaintenance AI

## IoT + AI + Physical AI Predictive Maintenance System

![SmartMaintenance AI](https://img.shields.io/badge/SmartMaintenance-AI-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**SmartMaintenance AI** is a comprehensive predictive maintenance system that combines **IoT sensors**, **AI algorithms**, and **Physical AI robotics** to revolutionize industrial equipment maintenance. This Python-based application provides real-time monitoring, failure prediction, and automated maintenance interventions.

## 🚀 Features

### IoT Integration
- **Real-time Sensor Monitoring**: Temperature, vibration, and pressure sensors
- **Edge Computing**: Local data processing for immediate responses
- **Multi-Equipment Support**: Monitor multiple industrial assets simultaneously
- **Anomaly Detection**: Automatic identification of unusual sensor patterns

### AI-Powered Analytics
- **Predictive Maintenance**: Machine learning models for failure prediction
- **Computer Vision**: Visual inspection using OpenCV for defect detection
- **Risk Assessment**: Multi-level risk categorization (Low, Medium, High)
- **Trend Analysis**: Historical data analysis for pattern recognition

### Physical AI Robotics
- **Automated Maintenance**: Robotic intervention for maintenance tasks
- **Task Scheduling**: Priority-based maintenance queue management
- **Human-Robot Collaboration**: Safe interaction protocols
- **Maintenance Execution**: Simulated robotic maintenance procedures

### Web Dashboard
- **Real-time Visualization**: Interactive charts and graphs
- **Equipment Health Cards**: Status overview for all monitored equipment
- **Alert System**: Immediate notifications for critical conditions
- **Mobile Responsive**: Access from any device

## 🛠 Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API and dashboard
- **scikit-learn**: Machine learning algorithms
- **OpenCV**: Computer vision processing
- **NumPy & Pandas**: Data processing and analysis

### Frontend
- **HTML5 & CSS3**: Modern web standards
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive data visualization
- **JavaScript ES6**: Dynamic user interactions

### IoT & Sensors
- **Simulated IoT Sensors**: Temperature, vibration, pressure
- **Real-time Data Streaming**: Continuous sensor monitoring
- **MQTT Support**: Optional IoT communication protocol
- **Edge Computing**: Local data processing capabilities

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/VineshThota/new-repo.git
cd new-repo/smartmaintenance_ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Dashboard
Open your web browser and navigate to: `http://localhost:5000`

## 🎯 Usage Guide

### Dashboard Overview
1. **Equipment Health Cards**: View real-time status of all monitored equipment
2. **Sensor Charts**: Monitor temperature, vibration, and pressure trends
3. **Failure Predictions**: AI-powered risk assessment and recommendations
4. **Robot Status**: Current robotic system status and maintenance queue
5. **Quick Actions**: Manual maintenance scheduling and inspections

### API Endpoints

#### Sensor Data
```http
GET /api/sensor-data
```
Returns latest sensor readings from all equipment.

#### Equipment Health
```http
GET /api/equipment-health
```
Provides comprehensive health summary for all monitored equipment.

#### Failure Prediction
```http
GET /api/predict-failure/{equipment_id}
```
Generates AI-powered failure prediction for specific equipment.

#### Visual Inspection
```http
GET /api/visual-inspection/{equipment_id}
```
Performs computer vision-based equipment inspection.

#### Robot Status
```http
GET /api/robot-status
```
Returns current robotic system status and maintenance queue.

#### Schedule Maintenance
```http
POST /api/schedule-maintenance
```
Schedules new maintenance task for robotic execution.

**Request Body:**
```json
{
  "equipment_id": "Motor_001",
  "task_type": "preventive_maintenance",
  "priority": "high"
}
```

## 🤖 AI Algorithms

### Predictive Maintenance Model
- **Algorithm**: Random Forest Classifier
- **Features**: Temperature, vibration, pressure, trend analysis
- **Training**: Synthetic data generation with realistic failure patterns
- **Accuracy**: Optimized for industrial equipment failure prediction

### Computer Vision System
- **Technology**: OpenCV-based image processing
- **Detection**: Corrosion, wear, misalignment, leaks
- **Confidence Scoring**: Reliability assessment for detected defects
- **Integration**: Automated defect reporting and maintenance scheduling

## 🔌 IoT Sensor Integration

### Supported Sensors
1. **Temperature Sensors**: Thermal monitoring for overheating detection
2. **Vibration Sensors**: Mechanical health assessment
3. **Pressure Sensors**: Hydraulic and pneumatic system monitoring

### Data Processing
- **Real-time Streaming**: 2-second interval sensor readings
- **Anomaly Detection**: Threshold-based alert system
- **Data Storage**: In-memory storage with configurable retention
- **Trend Analysis**: Historical pattern recognition

## 🦾 Physical AI Robotics

### Robotic Capabilities
- **Autonomous Navigation**: Movement to equipment locations
- **Visual Inspection**: Camera-based defect detection
- **Maintenance Execution**: Automated repair procedures
- **Safety Protocols**: Human-robot collaboration safety

### Task Management
- **Priority Queue**: High, medium, low priority scheduling
- **Task Types**: Preventive, corrective, emergency maintenance
- **Execution Tracking**: Real-time task progress monitoring
- **Completion Verification**: Automated quality checks

## 🚀 Deployment

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

## 📊 Monitoring Equipment

The system currently monitors four types of industrial equipment:
- **Motor_001**: Electric motor with temperature and vibration monitoring
- **Pump_002**: Hydraulic pump with pressure and temperature sensors
- **Compressor_003**: Air compressor with comprehensive sensor suite
- **Generator_004**: Power generator with multi-parameter monitoring

## 🔧 Configuration

### Sensor Thresholds
```python
# Temperature thresholds (°F)
WARNING_TEMP = 95
CRITICAL_TEMP = 105

# Vibration thresholds (mm/s)
WARNING_VIBRATION = 4.0
CRITICAL_VIBRATION = 5.0

# Pressure thresholds (PSI)
WARNING_PRESSURE = 18.0
CRITICAL_PRESSURE = 20.0
```

### AI Model Parameters
```python
# Random Forest Configuration
N_ESTIMATORS = 100
RANDOM_STATE = 42
FEATURES = ['temperature', 'vibration', 'pressure', 'temp_trend', 'vib_trend']
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IoT Community**: For sensor integration best practices
- **scikit-learn**: For machine learning algorithms
- **OpenCV**: For computer vision capabilities
- **Flask Community**: For web framework support
- **Bootstrap**: For responsive UI components

## 📞 Support

For support, email vineshthota1@gmail.com or create an issue in the GitHub repository.

## 🔮 Future Enhancements

- [ ] Real hardware sensor integration
- [ ] Advanced AI models (Deep Learning)
- [ ] Cloud deployment support
- [ ] Mobile application
- [ ] Integration with existing CMMS systems
- [ ] Advanced robotics control
- [ ] Multi-site monitoring
- [ ] Predictive analytics dashboard

---

**Built with ❤️ by Vinesh Thota**

*Combining IoT, AI, and Physical AI for the future of industrial maintenance*