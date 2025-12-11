import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import NavigationMap from './components/NavigationMap';
import AIAgent from './components/AIAgent';
import VoiceGuidance from './components/VoiceGuidance';
import SensorPanel from './components/SensorPanel';
import EmergencyNavigation from './components/EmergencyNavigation';
import PathPlanner from './components/PathPlanner';

function App() {
  const [userLocation, setUserLocation] = useState({ x: 0, y: 0, floor: 1 });
  const [destination, setDestination] = useState(null);
  const [currentPath, setCurrentPath] = useState([]);
  const [navigationActive, setNavigationActive] = useState(false);
  const [emergencyMode, setEmergencyMode] = useState(false);
  const [sensorData, setSensorData] = useState({
    accelerometer: { x: 0, y: 0, z: 0 },
    gyroscope: { x: 0, y: 0, z: 0 },
    magnetometer: { x: 0, y: 0, z: 0 },
    wifi: [],
    bluetooth: [],
    camera: null
  });
  const [aiAgentStatus, setAiAgentStatus] = useState('idle');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [buildingData, setBuildingData] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [navigationHistory, setNavigationHistory] = useState([]);

  const aiAgentRef = useRef(null);
  const sensorManagerRef = useRef(null);

  useEffect(() => {
    // Initialize AI Agent
    initializeAIAgent();
    
    // Start sensor monitoring
    startSensorMonitoring();
    
    // Load building data
    loadBuildingData();
    
    // Setup real-time updates
    setupRealTimeUpdates();

    return () => {
      // Cleanup
      if (sensorManagerRef.current) {
        sensorManagerRef.current.stop();
      }
    };
  }, []);

  const initializeAIAgent = async () => {
    try {
      setAiAgentStatus('initializing');
      // Initialize AI models and agents
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate initialization
      setAiAgentStatus('ready');
    } catch (error) {
      console.error('AI Agent initialization failed:', error);
      setAiAgentStatus('error');
    }
  };

  const startSensorMonitoring = () => {
    // Simulate sensor data updates
    const interval = setInterval(() => {
      setSensorData(prevData => ({
        ...prevData,
        accelerometer: {
          x: (Math.random() - 0.5) * 2,
          y: (Math.random() - 0.5) * 2,
          z: 9.8 + (Math.random() - 0.5) * 0.5
        },
        gyroscope: {
          x: (Math.random() - 0.5) * 0.1,
          y: (Math.random() - 0.5) * 0.1,
          z: (Math.random() - 0.5) * 0.1
        },
        wifi: generateWiFiData(),
        bluetooth: generateBluetoothData()
      }));
    }, 100);

    sensorManagerRef.current = { stop: () => clearInterval(interval) };
  };

  const generateWiFiData = () => {
    return [
      { ssid: 'Building_WiFi_1', rssi: -45 + Math.random() * 10, mac: '00:11:22:33:44:55' },
      { ssid: 'Building_WiFi_2', rssi: -55 + Math.random() * 10, mac: '00:11:22:33:44:56' },
      { ssid: 'Building_WiFi_3', rssi: -65 + Math.random() * 10, mac: '00:11:22:33:44:57' }
    ];
  };

  const generateBluetoothData = () => {
    return [
      { name: 'Beacon_A1', rssi: -50 + Math.random() * 5, mac: 'AA:BB:CC:DD:EE:01' },
      { name: 'Beacon_B2', rssi: -60 + Math.random() * 5, mac: 'AA:BB:CC:DD:EE:02' },
      { name: 'Beacon_C3', rssi: -70 + Math.random() * 5, mac: 'AA:BB:CC:DD:EE:03' }
    ];
  };

  const loadBuildingData = async () => {
    // Simulate loading building floor plans and landmarks
    const mockBuildingData = {
      floors: [
        { id: 1, name: 'Ground Floor', map: '/maps/floor1.svg' },
        { id: 2, name: 'First Floor', map: '/maps/floor2.svg' },
        { id: 3, name: 'Second Floor', map: '/maps/floor3.svg' }
      ],
      landmarks: [
        { id: 1, name: 'Main Entrance', x: 50, y: 100, floor: 1, type: 'entrance' },
        { id: 2, name: 'Elevator A', x: 150, y: 200, floor: 1, type: 'elevator' },
        { id: 3, name: 'Restroom', x: 250, y: 150, floor: 1, type: 'restroom' },
        { id: 4, name: 'Conference Room A', x: 300, y: 250, floor: 2, type: 'room' },
        { id: 5, name: 'Emergency Exit', x: 400, y: 50, floor: 1, type: 'emergency' }
      ]
    };
    
    setBuildingData(mockBuildingData);
    setLandmarks(mockBuildingData.landmarks);
  };

  const setupRealTimeUpdates = () => {
    // Simulate real-time location updates
    const interval = setInterval(() => {
      if (navigationActive && currentPath.length > 0) {
        // Simulate movement along the path
        setUserLocation(prevLocation => {
          const nextPoint = currentPath[0];
          if (nextPoint) {
            const dx = nextPoint.x - prevLocation.x;
            const dy = nextPoint.y - prevLocation.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 5) {
              // Reached waypoint, move to next
              setCurrentPath(prev => prev.slice(1));
              return nextPoint;
            } else {
              // Move towards waypoint
              const speed = 2; // pixels per update
              return {
                ...prevLocation,
                x: prevLocation.x + (dx / distance) * speed,
                y: prevLocation.y + (dy / distance) * speed
              };
            }
          }
          return prevLocation;
        });
      }
    }, 100);

    return () => clearInterval(interval);
  };

  const startNavigation = async (dest) => {
    setDestination(dest);
    setNavigationActive(true);
    setAiAgentStatus('planning');
    
    // AI Agent calculates optimal path
    const path = await calculatePath(userLocation, dest);
    setCurrentPath(path);
    
    setAiAgentStatus('navigating');
    
    // Add to navigation history
    setNavigationHistory(prev => [...prev, {
      id: Date.now(),
      from: userLocation,
      to: dest,
      timestamp: new Date(),
      status: 'active'
    }]);
  };

  const calculatePath = async (start, end) => {
    // Simulate AI-powered path calculation
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Simple path calculation (in real app, this would use A* or similar algorithm)
    const path = [];
    const steps = 20;
    
    for (let i = 1; i <= steps; i++) {
      const progress = i / steps;
      path.push({
        x: start.x + (end.x - start.x) * progress,
        y: start.y + (end.y - start.y) * progress,
        floor: start.floor // Simplified - real app would handle floor changes
      });
    }
    
    return path;
  };

  const stopNavigation = () => {
    setNavigationActive(false);
    setCurrentPath([]);
    setDestination(null);
    setAiAgentStatus('ready');
    
    // Update navigation history
    setNavigationHistory(prev => 
      prev.map(nav => 
        nav.status === 'active' ? { ...nav, status: 'completed' } : nav
      )
    );
  };

  const toggleEmergencyMode = () => {
    setEmergencyMode(!emergencyMode);
    if (!emergencyMode) {
      // Find nearest emergency exit
      const emergencyExits = landmarks.filter(l => l.type === 'emergency');
      if (emergencyExits.length > 0) {
        const nearest = emergencyExits.reduce((closest, exit) => {
          const distToCurrent = Math.sqrt(
            Math.pow(exit.x - userLocation.x, 2) + 
            Math.pow(exit.y - userLocation.y, 2)
          );
          const distToClosest = closest ? Math.sqrt(
            Math.pow(closest.x - userLocation.x, 2) + 
            Math.pow(closest.y - userLocation.y, 2)
          ) : Infinity;
          
          return distToCurrent < distToClosest ? exit : closest;
        }, null);
        
        if (nearest) {
          startNavigation(nearest);
        }
      }
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🧭 NaviAgent</h1>
          <p>AI-Powered Indoor Navigation</p>
          <div className="status-indicators">
            <div className={`status-indicator ${aiAgentStatus}`}>
              <span className="status-dot"></span>
              AI Agent: {aiAgentStatus}
            </div>
            <div className={`status-indicator ${navigationActive ? 'active' : 'idle'}`}>
              <span className="status-dot"></span>
              Navigation: {navigationActive ? 'Active' : 'Idle'}
            </div>
          </div>
        </div>
        
        <div className="emergency-controls">
          <button 
            className={`emergency-btn ${emergencyMode ? 'active' : ''}`}
            onClick={toggleEmergencyMode}
          >
            🚨 Emergency
          </button>
        </div>
      </header>

      <main className="app-main">
        <div className="navigation-container">
          <div className="map-section">
            <NavigationMap 
              userLocation={userLocation}
              destination={destination}
              currentPath={currentPath}
              landmarks={landmarks}
              buildingData={buildingData}
              onDestinationSelect={startNavigation}
              emergencyMode={emergencyMode}
            />
          </div>
          
          <div className="controls-section">
            <div className="control-panel">
              <AIAgent 
                ref={aiAgentRef}
                status={aiAgentStatus}
                userLocation={userLocation}
                sensorData={sensorData}
                onNavigationRequest={startNavigation}
                emergencyMode={emergencyMode}
              />
              
              <VoiceGuidance 
                enabled={voiceEnabled}
                onToggle={setVoiceEnabled}
                currentPath={currentPath}
                userLocation={userLocation}
                destination={destination}
                navigationActive={navigationActive}
              />
              
              <div className="navigation-controls">
                <button 
                  className="nav-btn primary"
                  onClick={() => navigationActive ? stopNavigation() : null}
                  disabled={!navigationActive}
                >
                  {navigationActive ? '⏹️ Stop Navigation' : '▶️ Start Navigation'}
                </button>
                
                <button 
                  className="nav-btn secondary"
                  onClick={() => setUserLocation({ x: 50, y: 100, floor: 1 })}
                >
                  📍 Reset Location
                </button>
              </div>
            </div>
          </div>
        </div>
        
        <div className="info-panels">
          <div className="panel-row">
            <SensorPanel 
              sensorData={sensorData}
              userLocation={userLocation}
            />
            
            <PathPlanner 
              currentPath={currentPath}
              userLocation={userLocation}
              destination={destination}
              onPathUpdate={setCurrentPath}
            />
          </div>
          
          {emergencyMode && (
            <EmergencyNavigation 
              userLocation={userLocation}
              landmarks={landmarks}
              onEmergencyRoute={startNavigation}
              active={emergencyMode}
            />
          )}
        </div>
      </main>
      
      <footer className="app-footer">
        <div className="footer-info">
          <p>NaviAgent v1.0.0 | AI-Powered Indoor Navigation | Real-time GPS-free positioning</p>
          <div className="tech-stack">
            <span>🤖 TensorFlow.js</span>
            <span>⚛️ React</span>
            <span>🗺️ Computer Vision</span>
            <span>📡 Sensor Fusion</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;