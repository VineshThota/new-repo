# 🏭 Smart Manufacturing Quality Control System

## Overview

A comprehensive Python-based application that integrates **IoT sensors**, **AI computer vision**, and **Physical AI robotics** for automated quality control in smart manufacturing environments. This system addresses the trending need for intelligent automation in manufacturing by combining real-time sensor monitoring, advanced defect detection, and robotic sorting capabilities.

## 🌟 Key Features

### IoT Integration
- **Real-time sensor monitoring** (temperature, vibration, humidity, pressure, noise)
- **Threshold-based alerting** system
- **Historical data tracking** and analysis
- **Edge computing** simulation for local processing

### AI Computer Vision
- **Automated defect detection** using CNN models
- **Image analysis** with OpenCV for structural assessment
- **Quality scoring** algorithm with configurable thresholds
- **Multiple defect type classification** (scratch, dent, discoloration, crack, missing parts)

### Physical AI Robotics
- **Robotic arm simulation** with 3D positioning
- **Automated product sorting** based on quality analysis
- **Gripper control** and movement tracking
- **Real-time status monitoring** and operation logging

### Web Dashboard
- **Interactive real-time dashboard** with live updates
- **Drag-and-drop image upload** for quality analysis
- **Visual sensor monitoring** with status indicators
- **Quality statistics** and performance metrics
- **System logs** and operation history

## 🚀 Technology Stack

- **Backend**: Flask (Python web framework)
- **Computer Vision**: OpenCV, TensorFlow/Keras
- **Database**: SQLite for data persistence
- **Frontend**: HTML5, CSS3, JavaScript (ES6)
- **IoT Simulation**: Python threading for real-time data
- **Image Processing**: PIL (Python Imaging Library)
- **Data Analysis**: NumPy for numerical computations

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/smart_manufacturing_qc
   ```

2. **Create a virtual environment** (recommended):
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
   Open your web browser and navigate to `http://localhost:5000`

## 🎯 Usage

### IoT Sensor Monitoring
1. Click **"Start Monitoring"** to begin real-time sensor data collection
2. Monitor temperature, vibration, humidity, pressure, and noise levels
3. Receive alerts when sensor values exceed configured thresholds
4. View historical data trends and patterns

### Quality Control Analysis
1. **Upload product images** by clicking the upload area or dragging files
2. Enable **"automatic robotic sorting"** if desired
3. Click **"Analyze Quality"** to process the image
4. Review quality scores, defect detection results, and recommendations
5. Monitor automated sorting operations if enabled

### Robotic Operations
1. Monitor real-time robotic arm position and status
2. Track gripper state (open/closed) and movement operations
3. View sorting statistics and operation history
4. Observe automated product handling based on quality analysis

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Sensors   │    │  Computer Vision │    │ Physical AI Bot │
│                 │    │                 │    │                 │
│ • Temperature   │    │ • CNN Model     │    │ • Robotic Arm   │
│ • Vibration     │────│ • OpenCV        │────│ • Gripper       │
│ • Humidity      │    │ • Defect Det.   │    │ • Positioning   │
│ • Pressure      │    │ • Quality Score │    │ • Sorting Logic │
│ • Noise Level   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Flask Backend  │
                    │                 │
                    │ • REST APIs     │
                    │ • Data Storage  │
                    │ • Real-time     │
                    │ • Threading     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Web Dashboard   │
                    │                 │
                    │ • Real-time UI  │
                    │ • Image Upload  │
                    │ • Statistics    │
                    │ • System Logs   │
                    └─────────────────┘
```

## 🔧 Configuration

### Sensor Thresholds
Modify sensor alert thresholds in `app.py`:
```python
self.sensors = {
    'temperature': {'value': 25.0, 'threshold': 30.0, 'unit': '°C'},
    'vibration': {'value': 0.5, 'threshold': 2.0, 'unit': 'mm/s'},
    # ... other sensors
}
```

### Quality Control Parameters
Adjust quality scoring in the `ComputerVisionQC` class:
```python
self.quality_threshold = 0.7  # Pass/fail threshold
self.defect_types = ['scratch', 'dent', 'discoloration', 'crack', 'missing_part']
```

### Robotic Positioning
Customize robotic arm positions:
```python
self.sort_bins = {
    'pass_bin': {'x': 100, 'y': 0, 'z': 0},
    'fail_bin': {'x': -100, 'y': 0, 'z': 0},
    'inspection_point': {'x': 0, 'y': 50, 'z': 10}
}
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|--------------|
| `/` | GET | Main dashboard |
| `/api/sensor-data` | GET | Current sensor readings |
| `/api/sensor-history` | GET | Historical sensor data |
| `/api/analyze-image` | POST | Analyze product image |
| `/api/robot-status` | GET | Robotic arm status |
| `/api/robot-sort` | POST | Manual sorting trigger |
| `/api/quality-statistics` | GET | Quality control stats |
| `/api/start-monitoring` | POST | Start IoT monitoring |
| `/api/stop-monitoring` | POST | Stop IoT monitoring |

## 🗄️ Database Schema

### Quality Results
```sql
CREATE TABLE quality_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    quality_score REAL,
    status TEXT,
    defect_type TEXT,
    defect_probability REAL,
    edge_density REAL,
    mean_color_r INTEGER,
    mean_color_g INTEGER,
    mean_color_b INTEGER
);
```

### Sensor Data
```sql
CREATE TABLE sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    temperature REAL,
    vibration REAL,
    humidity REAL,
    pressure REAL,
    noise_level REAL
);
```

### Robotic Operations
```sql
CREATE TABLE robotic_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    operation_type TEXT,
    quality_status TEXT,
    destination_bin TEXT,
    quality_score REAL,
    defect_type TEXT
);
```

## 🔬 Technical Implementation

### IoT Sensor Simulation
- **Multi-threaded** real-time data generation
- **Realistic variations** with occasional anomalies
- **Configurable thresholds** for alert generation
- **Data persistence** with SQLite storage

### Computer Vision Pipeline
1. **Image preprocessing** (resize, normalize)
2. **CNN-based defect detection** (TensorFlow/Keras)
3. **Edge detection** analysis (OpenCV Canny)
4. **Color analysis** for surface inspection
5. **Quality scoring** algorithm

### Physical AI Integration
- **3D coordinate system** for robotic positioning
- **Smooth movement simulation** with interpolation
- **Gripper state management** (open/close)
- **Automated decision making** based on quality results

## 🚀 Deployment Options

### Local Development
```bash
python app.py
```

### Production with Gunicorn
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

## 🔍 Monitoring and Logging

- **Real-time system logs** in the web dashboard
- **Structured logging** with different severity levels
- **Operation tracking** for all robotic movements
- **Quality metrics** and performance analytics

## 🛡️ Security Considerations

- **Input validation** for image uploads
- **File type restrictions** (JPG, PNG only)
- **SQL injection prevention** with parameterized queries
- **Error handling** and graceful degradation

## 🔮 Future Enhancements

- **Machine learning model training** with real manufacturing data
- **Advanced predictive analytics** for maintenance scheduling
- **Integration with industrial IoT protocols** (MQTT, OPC-UA)
- **Multi-camera support** for comprehensive inspection
- **Cloud deployment** with scalable architecture
- **Mobile app** for remote monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vinesh Thota**
- GitHub: [@VineshThota](https://github.com/VineshThota)
- Email: vineshthota1@gmail.com

## 🙏 Acknowledgments

- **OpenCV** community for computer vision tools
- **TensorFlow** team for machine learning framework
- **Flask** developers for the web framework
- **Manufacturing industry** for inspiration and use cases

---

**Built with ❤️ for the future of smart manufacturing**