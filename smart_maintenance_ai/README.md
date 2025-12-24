# SmartMaintenance AI 🤖⚙️

**Edge AI Predictive Maintenance System with IoT Sensors & Digital Twins**

A comprehensive Python-based application that combines IoT sensors, artificial intelligence, and digital twin technology to provide predictive maintenance capabilities for industrial equipment.

## 🌟 Features

### IoT Integration
- **Real-time Sensor Monitoring**: Temperature, vibration, and pressure sensors
- **Edge Computing**: Local data processing for reduced latency
- **Sensor Simulation**: Built-in IoT sensor simulator for testing and development
- **Data Collection**: Continuous sensor data logging with historical tracking

### AI & Machine Learning
- **Anomaly Detection**: Advanced ML algorithms using Isolation Forest
- **Predictive Analytics**: Equipment failure prediction based on sensor patterns
- **Real-time Processing**: Edge AI for immediate anomaly detection
- **Model Training**: Automatic and manual model training capabilities

### Digital Twin Technology
- **Virtual Equipment Representation**: Real-time digital replica of physical assets
- **Health Scoring**: Dynamic equipment health assessment
- **Maintenance Scheduling**: Automated maintenance alerts and scheduling
- **Performance Monitoring**: Efficiency tracking and optimization

### Web Dashboard
- **Real-time Visualization**: Live sensor data with interactive charts
- **Responsive Design**: Modern Bootstrap-based UI
- **Alert System**: Comprehensive maintenance alert management
- **Control Panel**: System controls and configuration options

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API and dashboard
- **scikit-learn**: Machine learning algorithms
- **pandas & numpy**: Data processing and analysis
- **threading**: Concurrent sensor simulation

### Frontend
- **HTML5/CSS3/JavaScript**: Modern web technologies
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive data visualization
- **Gauge.js**: Real-time gauge displays
- **Font Awesome**: Icon library

### AI/ML Components
- **Isolation Forest**: Anomaly detection algorithm
- **StandardScaler**: Feature normalization
- **Rolling Statistics**: Time-series feature engineering
- **Real-time Inference**: Edge AI processing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/smart_maintenance_ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access Dashboard**
   Open your browser and navigate to: `http://localhost:5000`

## 📊 Usage Guide

### Dashboard Overview

#### Sensor Monitoring
- **Temperature Sensor**: Monitors equipment temperature (70-80°F normal range)
- **Vibration Sensor**: Tracks mechanical vibrations (1.0-4.0 mm/s normal range)
- **Pressure Sensor**: Measures system pressure (140-160 PSI normal range)

#### AI Anomaly Detection
- **Real-time Analysis**: Continuous monitoring for anomalies
- **Confidence Scoring**: Percentage-based anomaly confidence
- **Model Training**: Manual and automatic model training
- **Visual Indicators**: Color-coded status indicators

#### Digital Twin Status
- **Health Score**: Overall equipment health percentage
- **Operational Hours**: Total runtime tracking
- **Efficiency**: Current operational efficiency
- **Maintenance Schedule**: Last and next maintenance dates

#### Maintenance Alerts
- **Priority Levels**: High, medium, and low priority alerts
- **Alert Types**: Critical, warning, and informational
- **Timestamp Tracking**: Alert generation time
- **Auto-clearing**: Automatic alert management

### System Controls
- **Start/Pause Monitoring**: Control data collection
- **Train Model**: Manually trigger AI model training
- **Export Data**: Data export functionality (coming soon)

## 🔧 API Documentation

### Endpoints

#### GET `/api/sensor-data`
Retrieve current sensor readings
```json
{
  "status": "success",
  "data": {
    "temperature": {
      "value": 75.2,
      "unit": "°F",
      "timestamp": "2025-12-22T10:30:00",
      "normal_range": [70, 80]
    },
    "vibration": {
      "value": 2.1,
      "unit": "mm/s",
      "timestamp": "2025-12-22T10:30:00",
      "normal_range": [1.0, 4.0]
    },
    "pressure": {
      "value": 152.5,
      "unit": "PSI",
      "timestamp": "2025-12-22T10:30:00",
      "normal_range": [140, 160]
    }
  }
}
```

#### GET `/api/historical-data`
Retrieve historical sensor data for charts
```json
{
  "status": "success",
  "data": {
    "timestamps": ["10:25:00", "10:26:00", "10:27:00"],
    "temperature": [75.1, 75.2, 75.0],
    "vibration": [2.0, 2.1, 1.9],
    "pressure": [151.0, 152.5, 150.8]
  }
}
```

#### GET `/api/anomaly-detection`
Get anomaly detection results
```json
{
  "status": "success",
  "anomaly_detection": {
    "anomaly": false,
    "confidence": 15.2,
    "score": -0.152,
    "message": "Normal operation"
  },
  "model_trained": true
}
```

#### GET `/api/digital-twin`
Retrieve digital twin status and alerts
```json
{
  "status": "success",
  "equipment_status": {
    "health_score": 95.5,
    "operational_hours": 1250.5,
    "efficiency": 92.0,
    "last_maintenance": "2025-11-22T00:00:00",
    "next_maintenance": "2026-01-22T00:00:00"
  },
  "maintenance_alerts": []
}
```

#### POST `/api/train-model`
Manually trigger model training
```json
{
  "status": "success",
  "message": "Model trained successfully",
  "training_samples": 150
}
```

## 🧠 AI Algorithms

### Anomaly Detection
- **Algorithm**: Isolation Forest
- **Features**: Rolling statistics, rate of change, current values
- **Training**: Minimum 50 samples required
- **Threshold**: Configurable anomaly threshold (-0.5 default)

### Feature Engineering
- **Current Values**: Real-time sensor readings
- **Rolling Mean**: 5-sample moving average
- **Rolling Standard Deviation**: 5-sample variability
- **Rate of Change**: First-order difference

### Model Performance
- **Contamination Rate**: 10% (configurable)
- **Update Frequency**: Real-time inference
- **Retraining**: Automatic when sufficient data available

## 🏭 IoT Sensor Details

### Temperature Sensor
- **Type**: Thermal monitoring
- **Range**: 0-200°F
- **Normal Operation**: 70-80°F
- **Accuracy**: ±0.5°F
- **Update Rate**: 2 seconds

### Vibration Sensor
- **Type**: Accelerometer-based
- **Range**: 0-10 mm/s
- **Normal Operation**: 1.0-4.0 mm/s
- **Accuracy**: ±0.1 mm/s
- **Update Rate**: 2 seconds

### Pressure Sensor
- **Type**: Piezoelectric
- **Range**: 0-300 PSI
- **Normal Operation**: 140-160 PSI
- **Accuracy**: ±1 PSI
- **Update Rate**: 2 seconds

## 🔄 Digital Twin Functionality

### Health Scoring Algorithm
```python
# Health impact calculation
health_impact = 0

# Sensor range violations
for sensor in sensors:
    if value outside normal_range:
        health_impact += 5

# Anomaly impact
if anomaly_detected:
    health_impact += confidence_score / 10

# Update health score
health_score = max(0, min(100, current_health - health_impact * 0.1))
```

### Maintenance Alert Logic
- **Critical**: Anomaly confidence > 70%
- **Warning**: Health score < 70%
- **Info**: Scheduled maintenance within 7 days
- **Auto-clear**: Alerts older than 1 hour

## 🚀 Deployment

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

#### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export PORT=5000
```

## 📈 Performance Metrics

### System Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB for data and logs
- **Network**: 100Mbps for real-time updates

### Scalability
- **Concurrent Users**: 50+ simultaneous dashboard users
- **Data Throughput**: 1000+ sensor readings per minute
- **Model Training**: Sub-second inference, minutes for training

## 🔒 Security Features

- **Input Validation**: All API inputs validated
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed application logging
- **CORS**: Configurable cross-origin resource sharing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn**: Machine learning algorithms
- **Flask**: Web framework
- **Bootstrap**: UI framework
- **Chart.js**: Data visualization
- **Font Awesome**: Icons

## 📞 Support

For support, email vineshthota1@gmail.com or create an issue in the GitHub repository.

## 🔮 Future Enhancements

- [ ] Multi-equipment support
- [ ] Advanced ML models (LSTM, Transformer)
- [ ] Mobile application
- [ ] Cloud deployment templates
- [ ] Integration with industrial IoT platforms
- [ ] Advanced analytics and reporting
- [ ] User authentication and authorization
- [ ] Data export and backup features

---

**Built with ❤️ using Python, Flask, and cutting-edge AI technologies**