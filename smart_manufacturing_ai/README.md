# 🏭 Physical AI Smart Manufacturing System

## Overview

A cutting-edge **Physical AI** application that combines **IoT sensors**, **AI algorithms**, and **physical world interactions** for predictive maintenance in smart manufacturing environments. This system represents the convergence of three critical technologies:

- 🌐 **IoT (Internet of Things)**: Real-time sensor monitoring and data collection
- 🤖 **AI (Artificial Intelligence)**: Machine learning-powered predictive analytics and anomaly detection
- ⚙️ **Physical AI**: Automated physical responses and equipment control based on AI predictions

## 🚀 Key Features

### IoT Integration
- **Real-time sensor monitoring** for temperature, vibration, pressure, speed, and power consumption
- **Multi-equipment tracking** (conveyor belts, robotic arms, CNC machines, assembly stations)
- **Continuous data collection** with 5-second intervals
- **Equipment health scoring** based on sensor patterns

### AI-Powered Analytics
- **Random Forest Classifier** for failure probability prediction
- **Isolation Forest** for anomaly detection in sensor data
- **Real-time risk assessment** with automated severity classification
- **Predictive maintenance recommendations** based on AI analysis

### Physical AI Automation
- **Automated emergency shutdowns** when critical failure risk is detected
- **Dynamic maintenance scheduling** based on AI predictions
- **Parameter adjustment** for equipment optimization
- **Physical response execution** with real-time logging

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: scikit-learn (Random Forest, Isolation Forest)
- **Data Processing**: NumPy, Pandas
- **Database**: SQLite for sensor data storage
- **Frontend**: HTML5, CSS3, JavaScript (embedded in Flask)
- **Real-time Processing**: Threading for continuous monitoring

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   IoT Sensors   │───▶│   AI Predictor   │───▶│  Physical Actions   │
│                 │    │                  │    │                     │
│ • Temperature   │    │ • Failure Pred. │    │ • Emergency Stop    │
│ • Vibration     │    │ • Anomaly Det.   │    │ • Maintenance Sched │
│ • Pressure      │    │ • Risk Analysis  │    │ • Parameter Adjust  │
│ • Speed         │    │ • Health Scoring │    │ • Alert Generation  │
│ • Power         │    │                  │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
           │                       │                        │
           └───────────────────────┼────────────────────────┘
                                   ▼
                        ┌──────────────────┐
                        │  Web Dashboard   │
                        │                  │
                        │ • Real-time View │
                        │ • Equipment Status│
                        │ • Maintenance Q   │
                        │ • Manual Controls │
                        └──────────────────┘
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/smart_manufacturing_ai
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
   Open your browser and navigate to `http://localhost:5000`

## 🎯 Usage

### Dashboard Features

1. **Equipment Monitoring**:
   - View real-time health scores for all equipment
   - Monitor operational status and last maintenance dates
   - Visual indicators for equipment condition (green/yellow/red)

2. **AI Predictions**:
   - Click "🔮 Predict Failure" to get AI-powered risk assessment
   - View failure probability percentages
   - Receive automated maintenance recommendations

3. **Maintenance Management**:
   - Schedule maintenance with "🔧 Schedule Maintenance" button
   - View maintenance queue with priority levels
   - Track scheduled maintenance dates

4. **Real-time Monitoring**:
   - Dashboard auto-updates every 10 seconds
   - Continuous background sensor monitoring
   - Automated alerts for critical conditions

### API Endpoints

- `GET /api/equipment_status` - Get current equipment status
- `GET /api/maintenance_queue` - Get scheduled maintenance
- `POST /api/predict_failure` - Get AI failure prediction
- `POST /api/trigger_action` - Execute physical actions

## 🧠 AI Models

### Failure Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: Temperature, Vibration, Pressure, Speed, Power Consumption
- **Output**: Failure probability (0-1 scale)
- **Training**: Synthetic data with normal and failure patterns

### Anomaly Detection Model
- **Algorithm**: Isolation Forest
- **Purpose**: Detect unusual sensor patterns
- **Contamination Rate**: 10% (configurable)
- **Output**: Anomaly score and binary classification

## 🔄 Physical AI Integration

### Automated Response System

1. **Critical Risk (>80% failure probability)**:
   - Immediate emergency shutdown
   - Equipment status changed to "maintenance_required"
   - Critical alert generation

2. **High Risk (60-80% failure probability)**:
   - Urgent maintenance scheduling
   - High priority queue placement
   - Preventive action recommendations

3. **Moderate Risk (40-60% failure probability)**:
   - Maintenance planning within 48 hours
   - Continued monitoring with increased frequency

4. **Anomaly Detection**:
   - Parameter adjustment for optimization
   - Investigation recommendations
   - Continuous pattern analysis

## 📈 Benefits

- **Reduced Downtime**: Predictive maintenance prevents unexpected failures
- **Cost Savings**: Optimized maintenance scheduling reduces operational costs
- **Safety Enhancement**: Automated emergency responses protect equipment and personnel
- **Efficiency Improvement**: AI-driven optimization increases overall equipment effectiveness
- **Data-Driven Decisions**: Real-time insights enable informed maintenance strategies

## 🔮 Future Enhancements

- Integration with real IoT hardware (Arduino, Raspberry Pi)
- Advanced ML models (LSTM for time series prediction)
- Mobile app for remote monitoring
- Integration with enterprise ERP systems
- Computer vision for visual equipment inspection
- Edge computing deployment for reduced latency

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Vinesh Thota**
- Email: vineshthota1@gmail.com
- GitHub: [@VineshThota](https://github.com/VineshThota)

---

*This Physical AI system demonstrates the powerful convergence of IoT, AI, and physical automation technologies for next-generation smart manufacturing.*