import React, { useState, useEffect, useRef } from 'react';
import './NaviBotDashboard.css';

const NaviBotDashboard = () => {
  const [robotStatus, setRobotStatus] = useState({
    state: 'idle',
    position: { x: 0, y: 0, z: 0, theta: 0 },
    goal: null,
    running: false,
    batteryLevel: 85,
    sensorHealth: {
      lidar: 'healthy',
      camera: 'healthy',
      imu: 'healthy'
    },
    obstacles: [],
    mapData: []
  });

  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [logs, setLogs] = useState([]);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        addLog('Connected to NaviBot AI system', 'info');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setRobotStatus(prevStatus => ({ ...prevStatus, ...data }));
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        addLog('Disconnected from NaviBot AI system', 'warning');
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
      
      wsRef.current.onerror = (error) => {
        addLog(`WebSocket error: ${error}`, 'error');
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Canvas drawing for map visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid
    drawGrid(ctx, width, height);
    
    // Draw obstacles
    robotStatus.obstacles.forEach(obstacle => {
      drawObstacle(ctx, obstacle);
    });
    
    // Draw robot position
    drawRobot(ctx, robotStatus.position);
    
    // Draw goal if set
    if (robotStatus.goal) {
      drawGoal(ctx, robotStatus.goal);
    }
    
    // Draw path if navigating
    if (robotStatus.state === 'navigating' && robotStatus.path) {
      drawPath(ctx, robotStatus.path);
    }
  }, [robotStatus]);

  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    
    const gridSize = 20;
    for (let x = 0; x <= width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    for (let y = 0; y <= height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  const drawRobot = (ctx, position) => {
    const scale = 20; // pixels per meter
    const x = position.x * scale + 300; // offset for centering
    const y = 300 - position.y * scale; // flip Y axis
    
    // Draw robot body
    ctx.fillStyle = '#2196F3';
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw orientation arrow
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(
      x + Math.cos(position.theta) * 12,
      y - Math.sin(position.theta) * 12
    );
    ctx.stroke();
  };

  const drawGoal = (ctx, goal) => {
    const scale = 20;
    const x = goal.x * scale + 300;
    const y = 300 - goal.y * scale;
    
    ctx.fillStyle = '#4CAF50';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw target symbol
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 6, y);
    ctx.lineTo(x + 6, y);
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x, y + 6);
    ctx.stroke();
  };

  const drawObstacle = (ctx, obstacle) => {
    const scale = 20;
    const x = obstacle.x * scale + 300;
    const y = 300 - obstacle.y * scale;
    
    ctx.fillStyle = '#F44336';
    ctx.fillRect(x - 5, y - 5, 10, 10);
  };

  const drawPath = (ctx, path) => {
    if (path.length < 2) return;
    
    const scale = 20;
    ctx.strokeStyle = '#FF9800';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    
    ctx.beginPath();
    const startPoint = path[0];
    ctx.moveTo(
      startPoint.x * scale + 300,
      300 - startPoint.y * scale
    );
    
    for (let i = 1; i < path.length; i++) {
      const point = path[i];
      ctx.lineTo(
        point.x * scale + 300,
        300 - point.y * scale
      );
    }
    
    ctx.stroke();
    ctx.setLineDash([]);
  };

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prevLogs => [
      { timestamp, message, type, id: Date.now() },
      ...prevLogs.slice(0, 49) // Keep only last 50 logs
    ]);
  };

  const sendCommand = (command, params = {}) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, ...params }));
      addLog(`Command sent: ${command}`, 'info');
    } else {
      addLog('Cannot send command: Not connected', 'error');
    }
  };

  const handleNavigateTo = () => {
    const x = parseFloat(document.getElementById('goalX').value);
    const y = parseFloat(document.getElementById('goalY').value);
    
    if (!isNaN(x) && !isNaN(y)) {
      sendCommand('navigate_to', { x, y, z: 0 });
    }
  };

  const getStateColor = (state) => {
    const colors = {
      idle: '#9E9E9E',
      navigating: '#2196F3',
      exploring: '#FF9800',
      obstacle_avoidance: '#F44336',
      mapping: '#9C27B0'
    };
    return colors[state] || '#9E9E9E';
  };

  const getSensorStatusIcon = (status) => {
    return status === 'healthy' ? '✅' : '❌';
  };

  return (
    <div className="navibot-dashboard">
      <header className="dashboard-header">
        <h1>NaviBot AI Dashboard</h1>
        <div className="connection-status">
          <span className={`status-indicator ${connectionStatus}`}></span>
          {connectionStatus.toUpperCase()}
        </div>
      </header>

      <div className="dashboard-content">
        <div className="left-panel">
          {/* Robot Status */}
          <div className="status-card">
            <h3>Robot Status</h3>
            <div className="status-grid">
              <div className="status-item">
                <label>State:</label>
                <span 
                  className="state-badge"
                  style={{ backgroundColor: getStateColor(robotStatus.state) }}
                >
                  {robotStatus.state.toUpperCase()}
                </span>
              </div>
              <div className="status-item">
                <label>Running:</label>
                <span>{robotStatus.running ? '✅' : '❌'}</span>
              </div>
              <div className="status-item">
                <label>Battery:</label>
                <span>{robotStatus.batteryLevel}%</span>
              </div>
            </div>
          </div>

          {/* Position */}
          <div className="status-card">
            <h3>Position</h3>
            <div className="position-grid">
              <div>X: {robotStatus.position.x.toFixed(2)}m</div>
              <div>Y: {robotStatus.position.y.toFixed(2)}m</div>
              <div>Z: {robotStatus.position.z.toFixed(2)}m</div>
              <div>θ: {(robotStatus.position.theta * 180 / Math.PI).toFixed(1)}°</div>
            </div>
          </div>

          {/* Sensor Health */}
          <div className="status-card">
            <h3>Sensor Health</h3>
            <div className="sensor-grid">
              <div>LiDAR: {getSensorStatusIcon(robotStatus.sensorHealth.lidar)}</div>
              <div>Camera: {getSensorStatusIcon(robotStatus.sensorHealth.camera)}</div>
              <div>IMU: {getSensorStatusIcon(robotStatus.sensorHealth.imu)}</div>
            </div>
          </div>

          {/* Controls */}
          <div className="control-card">
            <h3>Navigation Controls</h3>
            <div className="control-group">
              <div className="input-group">
                <input 
                  type="number" 
                  id="goalX" 
                  placeholder="X coordinate" 
                  step="0.1"
                />
                <input 
                  type="number" 
                  id="goalY" 
                  placeholder="Y coordinate" 
                  step="0.1"
                />
                <button onClick={handleNavigateTo}>Navigate To</button>
              </div>
              <div className="button-group">
                <button onClick={() => sendCommand('start_exploration')}>Start Exploration</button>
                <button onClick={() => sendCommand('stop')}>Stop</button>
                <button onClick={() => sendCommand('emergency_stop')}>Emergency Stop</button>
              </div>
            </div>
          </div>
        </div>

        <div className="center-panel">
          {/* Map Visualization */}
          <div className="map-card">
            <h3>Real-time Map</h3>
            <canvas 
              ref={canvasRef}
              width={600}
              height={400}
              className="map-canvas"
            />
            <div className="map-legend">
              <div><span className="legend-robot"></span> Robot</div>
              <div><span className="legend-goal"></span> Goal</div>
              <div><span className="legend-obstacle"></span> Obstacle</div>
              <div><span className="legend-path"></span> Path</div>
            </div>
          </div>
        </div>

        <div className="right-panel">
          {/* System Logs */}
          <div className="logs-card">
            <h3>System Logs</h3>
            <div className="logs-container">
              {logs.map(log => (
                <div key={log.id} className={`log-entry ${log.type}`}>
                  <span className="log-timestamp">{log.timestamp}</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NaviBotDashboard;