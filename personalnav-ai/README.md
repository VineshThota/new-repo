# 🤖 PersonalNav AI

**AI-Powered Indoor Navigation with Personalization**

PersonalNav AI is an intelligent indoor navigation system that combines trending AI-driven personalization with advanced indoor positioning technology. The system learns from user behavior and preferences to provide customized navigation routes for autonomous robots in GPS-denied environments.

## 🌟 Features

### 🎯 AI-Driven Personalization
- **User Preference Learning**: Adapts to individual navigation preferences over time
- **Behavioral Pattern Recognition**: Uses machine learning to identify user movement patterns
- **Customizable Priority Weights**: Adjust safety, efficiency, and comfort priorities
- **Speed Adaptation**: Personalizes navigation speed based on user preferences

### 🗺️ Advanced Indoor Navigation
- **GPS-Free Positioning**: Uses WiFi and Beacon triangulation for accurate indoor positioning
- **A* Pathfinding Algorithm**: Optimized pathfinding with personalization factors
- **Dynamic Obstacle Avoidance**: Real-time obstacle detection and route adjustment
- **Multi-level Safety Considerations**: Configurable obstacle avoidance levels

### 🔧 Interactive Web Interface
- **Visual Navigation Map**: Interactive canvas for setting waypoints and obstacles
- **Real-time Position Tracking**: Live indoor positioning simulation
- **Preference Configuration**: Easy-to-use sliders for personalization settings
- **Navigation Statistics**: Distance, time, and accuracy metrics

### 📊 Data Management
- **SQLite Database**: Stores navigation history and user preferences
- **Machine Learning Models**: Persistent model storage for continuous learning
- **Historical Analysis**: Track navigation patterns over time

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/personalnav-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## 🎮 Usage Guide

### Setting Up Navigation

1. **Set Start Point**: Click "Set Start Point" and click on the map
2. **Set End Point**: Click "Set End Point" and click on the map
3. **Add Obstacles**: Click "Add Obstacles" and click to place obstacles
4. **Generate Path**: Click "Generate Path" to create a personalized route

### Personalizing Your Experience

1. **User ID**: Enter a unique identifier for personalized learning
2. **Speed Preference**: Adjust your preferred navigation speed (0.5x - 3.0x)
3. **Priority Weights**:
   - **Safety**: How much to prioritize obstacle avoidance
   - **Efficiency**: How much to prioritize shortest path
   - **Comfort**: How much to prioritize smooth turns
4. **Avoidance Level**: Set obstacle avoidance sensitivity (Low to High)

### Indoor Positioning Simulation

1. **WiFi Signals**: Input signal strengths from 4 access points (0-100)
2. **Beacon Signals**: Input signal strengths from 4 beacons (0-100)
3. **Update Position**: Calculate estimated position based on signal triangulation

## 🏗️ Architecture

### Backend Components

- **Flask Web Framework**: RESTful API and web interface
- **PersonalNavAI Class**: Core navigation and AI logic
- **SQLite Database**: Data persistence layer
- **Scikit-learn**: Machine learning for personalization

### Frontend Components

- **HTML5 Canvas**: Interactive navigation visualization
- **Responsive CSS**: Modern, mobile-friendly interface
- **JavaScript**: Real-time user interactions and API calls

### API Endpoints

- `POST /api/position` - Get indoor position from signals
- `POST /api/navigate` - Generate personalized navigation route
- `GET/POST /api/preferences` - Manage user preferences
- `POST /api/learn` - Submit navigation data for learning

## 🧠 AI & Machine Learning

### Personalization Algorithm

1. **Data Collection**: Stores navigation history with user preferences
2. **Feature Extraction**: Analyzes distance, obstacles, speed, and duration
3. **Pattern Recognition**: Uses K-means clustering to identify user patterns
4. **Route Optimization**: Applies learned preferences to A* pathfinding

### Indoor Positioning

1. **Signal Processing**: Converts WiFi/Beacon signals to distance estimates
2. **Triangulation**: Weighted positioning based on multiple signal sources
3. **Accuracy Estimation**: Provides positioning confidence levels

## 📈 Trending Technology Integration

This project combines several trending technologies:

- **AI-Driven Personalization**: Currently trending on LinkedIn as a key business differentiator
- **Indoor Navigation**: Critical for autonomous robots and IoT applications
- **Machine Learning**: Continuous improvement through user behavior analysis
- **Real-time Processing**: Instant route calculation and position updates

## 🔧 Technical Specifications

### Dependencies

- **Flask 2.3.3**: Web framework
- **NumPy 1.24.3**: Numerical computations
- **Scikit-learn 1.3.0**: Machine learning algorithms
- **Gunicorn 21.2.0**: Production WSGI server

### Database Schema

- **navigation_history**: Stores completed navigation routes
- **user_preferences**: Stores personalization settings
- **indoor_map**: Stores environmental mapping data

### Performance

- **Real-time Processing**: Sub-second route calculation
- **Scalable Architecture**: Supports multiple concurrent users
- **Efficient Algorithms**: Optimized A* implementation with personalization

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by trending LinkedIn discussions on AI-driven personalization
- Built to address real-world indoor navigation challenges
- Combines cutting-edge AI with practical robotics applications

## 📞 Contact

**Vinesh Thota**
- Email: vineshthota1@gmail.com
- GitHub: [@VineshThota](https://github.com/VineshThota)
- LinkedIn: [Connect with me](https://linkedin.com/in/vineshthota)

---

*PersonalNav AI - Where AI-driven personalization meets autonomous navigation* 🤖🗺️