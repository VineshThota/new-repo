import React, { useState, useEffect } from 'react';

const SensorPanel = ({ sensorData, userLocation }) => {
  const [sensorHistory, setSensorHistory] = useState({
    accelerometer: [],
    gyroscope: [],
    wifi: [],
    bluetooth: []
  });
  const [positioningAccuracy, setPositioningAccuracy] = useState(0);
  const [movementState, setMovementState] = useState('stationary');

  useEffect(() => {
    if (sensorData) {
      updateSensorHistory();
      calculatePositioningAccuracy();
      detectMovementState();
    }
  }, [sensorData]);

  const updateSensorHistory = () => {
    const timestamp = Date.now();
    
    setSensorHistory(prev => ({
      accelerometer: [...prev.accelerometer.slice(-50), {
        timestamp,
        x: sensorData.accelerometer.x,
        y: sensorData.accelerometer.y,
        z: sensorData.accelerometer.z,
        magnitude: Math.sqrt(
          sensorData.accelerometer.x ** 2 + 
          sensorData.accelerometer.y ** 2 + 
          sensorData.accelerometer.z ** 2
        )
      }],
      gyroscope: [...prev.gyroscope.slice(-50), {
        timestamp,
        x: sensorData.gyroscope.x,
        y: sensorData.gyroscope.y,
        z: sensorData.gyroscope.z
      }],
      wifi: [...prev.wifi.slice(-20), {
        timestamp,
        signals: sensorData.wifi.length,
        avgRssi: sensorData.wifi.reduce((sum, ap) => sum + ap.rssi, 0) / sensorData.wifi.length || 0
      }],
      bluetooth: [...prev.bluetooth.slice(-20), {
        timestamp,
        beacons: sensorData.bluetooth.length,
        avgRssi: sensorData.bluetooth.reduce((sum, beacon) => sum + beacon.rssi, 0) / sensorData.bluetooth.length || 0
      }]
    }));
  };

  const calculatePositioningAccuracy = () => {
    const wifiSignals = sensorData.wifi.length;
    const bluetoothBeacons = sensorData.bluetooth.length;
    const avgWifiRssi = sensorData.wifi.reduce((sum, ap) => sum + Math.abs(ap.rssi), 0) / wifiSignals || 100;
    const avgBluetoothRssi = sensorData.bluetooth.reduce((sum, beacon) => sum + Math.abs(beacon.rssi), 0) / bluetoothBeacons || 100;
    
    // Calculate accuracy based on signal strength and quantity
    let accuracy = 0;
    
    // WiFi contribution (0-40%)
    if (wifiSignals > 0) {
      const wifiScore = Math.min(wifiSignals / 5, 1) * 0.4;
      const signalQuality = Math.max(0, (100 - avgWifiRssi) / 100) * 0.4;
      accuracy += wifiScore + signalQuality;
    }
    
    // Bluetooth contribution (0-30%)
    if (bluetoothBeacons > 0) {
      const bluetoothScore = Math.min(bluetoothBeacons / 3, 1) * 0.3;
      const beaconQuality = Math.max(0, (100 - avgBluetoothRssi) / 100) * 0.3;
      accuracy += bluetoothScore + beaconQuality;
    }
    
    // Movement stability contribution (0-30%)
    const recentAccel = sensorHistory.accelerometer.slice(-10);
    if (recentAccel.length > 5) {
      const stability = calculateStability(recentAccel.map(a => a.magnitude));
      accuracy += stability * 0.3;
    }
    
    setPositioningAccuracy(Math.min(accuracy, 1));
  };

  const calculateStability = (values) => {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    // Lower standard deviation = higher stability
    return Math.max(0, 1 - (stdDev / 5));
  };

  const detectMovementState = () => {
    const recentAccel = sensorHistory.accelerometer.slice(-10);
    if (recentAccel.length < 5) return;
    
    const avgMagnitude = recentAccel.reduce((sum, a) => sum + a.magnitude, 0) / recentAccel.length;
    const variance = recentAccel.reduce((sum, a) => sum + Math.pow(a.magnitude - avgMagnitude, 2), 0) / recentAccel.length;
    
    if (variance < 0.1) {
      setMovementState('stationary');
    } else if (variance < 0.5) {
      setMovementState('walking');
    } else {
      setMovementState('running');
    }
  };

  const renderMiniChart = (data, color, label) => {
    if (!data || data.length < 2) return null;
    
    const maxValue = Math.max(...data.map(d => d.magnitude || d.avgRssi || d.signals || d.beacons || 0));
    const minValue = Math.min(...data.map(d => d.magnitude || d.avgRssi || d.signals || d.beacons || 0));
    const range = maxValue - minValue || 1;
    
    const points = data.slice(-20).map((point, index) => {
      const x = (index / 19) * 100;
      const value = point.magnitude || point.avgRssi || point.signals || point.beacons || 0;
      const y = 100 - ((value - minValue) / range) * 100;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <div className="mini-chart">
        <div className="chart-label">{label}</div>
        <svg width="100" height="40" viewBox="0 0 100 100">
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
          />
        </svg>
      </div>
    );
  };

  const getMovementIcon = (state) => {
    switch (state) {
      case 'stationary': return '🧘';
      case 'walking': return '🚶';
      case 'running': return '🏃';
      default: return '📍';
    }
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy > 0.8) return '#4CAF50';
    if (accuracy > 0.6) return '#FF9800';
    if (accuracy > 0.4) return '#FF5722';
    return '#f44336';
  };

  const getSignalStrengthBars = (rssi) => {
    const strength = Math.max(0, Math.min(4, Math.floor((100 + rssi) / 25)));
    return Array.from({ length: 4 }, (_, i) => (
      <div 
        key={i} 
        className={`signal-bar ${i < strength ? 'active' : ''}`}
        style={{ height: `${(i + 1) * 25}%` }}
      />
    ));
  };

  return (
    <div className="sensor-panel">
      <div className="sensor-header">
        <h3>📡 Sensor Data</h3>
        <div className="positioning-accuracy">
          <span>Accuracy: </span>
          <div className="accuracy-meter">
            <div 
              className="accuracy-fill"
              style={{ 
                width: `${positioningAccuracy * 100}%`,
                backgroundColor: getAccuracyColor(positioningAccuracy)
              }}
            />
          </div>
          <span>{(positioningAccuracy * 100).toFixed(1)}%</span>
        </div>
      </div>
      
      <div className="sensor-content">
        <div className="sensor-grid">
          <div className="sensor-section">
            <h4>📱 Motion Sensors</h4>
            
            <div className="motion-status">
              <div className="movement-indicator">
                <span className="movement-icon">{getMovementIcon(movementState)}</span>
                <span className="movement-text">{movementState}</span>
              </div>
            </div>
            
            <div className="sensor-readings">
              <div className="reading-item">
                <label>Accelerometer (m/s²):</label>
                <div className="reading-values">
                  <span>X: {sensorData.accelerometer.x.toFixed(2)}</span>
                  <span>Y: {sensorData.accelerometer.y.toFixed(2)}</span>
                  <span>Z: {sensorData.accelerometer.z.toFixed(2)}</span>
                </div>
              </div>
              
              <div className="reading-item">
                <label>Gyroscope (rad/s):</label>
                <div className="reading-values">
                  <span>X: {sensorData.gyroscope.x.toFixed(3)}</span>
                  <span>Y: {sensorData.gyroscope.y.toFixed(3)}</span>
                  <span>Z: {sensorData.gyroscope.z.toFixed(3)}</span>
                </div>
              </div>
            </div>
            
            {renderMiniChart(sensorHistory.accelerometer, '#2196F3', 'Acceleration')}
          </div>
          
          <div className="sensor-section">
            <h4>📶 WiFi Signals</h4>
            <div className="wifi-summary">
              <span>Detected: {sensorData.wifi.length} networks</span>
              <span>Avg RSSI: {(sensorData.wifi.reduce((sum, ap) => sum + ap.rssi, 0) / sensorData.wifi.length || 0).toFixed(1)} dBm</span>
            </div>
            
            <div className="wifi-list">
              {sensorData.wifi.slice(0, 3).map((ap, index) => (
                <div key={index} className="wifi-item">
                  <span className="wifi-ssid">{ap.ssid}</span>
                  <div className="signal-strength">
                    {getSignalStrengthBars(ap.rssi)}
                  </div>
                  <span className="wifi-rssi">{ap.rssi} dBm</span>
                </div>
              ))}
            </div>
            
            {renderMiniChart(sensorHistory.wifi, '#4CAF50', 'WiFi Signals')}
          </div>
          
          <div className="sensor-section">
            <h4>🔵 Bluetooth Beacons</h4>
            <div className="bluetooth-summary">
              <span>Detected: {sensorData.bluetooth.length} beacons</span>
              <span>Avg RSSI: {(sensorData.bluetooth.reduce((sum, beacon) => sum + beacon.rssi, 0) / sensorData.bluetooth.length || 0).toFixed(1)} dBm</span>
            </div>
            
            <div className="bluetooth-list">
              {sensorData.bluetooth.slice(0, 3).map((beacon, index) => (
                <div key={index} className="bluetooth-item">
                  <span className="beacon-name">{beacon.name}</span>
                  <div className="signal-strength">
                    {getSignalStrengthBars(beacon.rssi)}
                  </div>
                  <span className="beacon-rssi">{beacon.rssi} dBm</span>
                </div>
              ))}
            </div>
            
            {renderMiniChart(sensorHistory.bluetooth, '#9C27B0', 'Bluetooth')}
          </div>
          
          <div className="sensor-section">
            <h4>📍 Position Info</h4>
            <div className="position-info">
              <div className="position-item">
                <label>Current Location:</label>
                <span>({userLocation.x.toFixed(1)}, {userLocation.y.toFixed(1)})</span>
              </div>
              <div className="position-item">
                <label>Floor:</label>
                <span>{userLocation.floor}</span>
              </div>
              <div className="position-item">
                <label>Positioning Method:</label>
                <span>
                  {sensorData.wifi.length > 0 && sensorData.bluetooth.length > 0 ? 'WiFi + Bluetooth' :
                   sensorData.wifi.length > 0 ? 'WiFi Only' :
                   sensorData.bluetooth.length > 0 ? 'Bluetooth Only' : 'Dead Reckoning'}
                </span>
              </div>
            </div>
            
            <div className="sensor-fusion-status">
              <h5>Sensor Fusion Status</h5>
              <div className="fusion-indicators">
                <div className={`fusion-indicator ${sensorData.wifi.length > 2 ? 'active' : ''}`}>
                  📶 WiFi Triangulation
                </div>
                <div className={`fusion-indicator ${sensorData.bluetooth.length > 2 ? 'active' : ''}`}>
                  🔵 Beacon Positioning
                </div>
                <div className={`fusion-indicator ${movementState !== 'stationary' ? 'active' : ''}`}>
                  🧭 Inertial Navigation
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SensorPanel;