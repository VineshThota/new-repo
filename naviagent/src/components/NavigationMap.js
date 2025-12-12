import React, { useState, useEffect, useRef } from 'react';

const NavigationMap = ({ 
  userLocation, 
  destination, 
  currentPath, 
  landmarks, 
  buildingData, 
  onDestinationSelect, 
  emergencyMode 
}) => {
  const [currentFloor, setCurrentFloor] = useState(1);
  const [mapScale, setMapScale] = useState(1);
  const [mapOffset, setMapOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const mapRef = useRef(null);
  const [hoveredLandmark, setHoveredLandmark] = useState(null);

  const mapWidth = 800;
  const mapHeight = 600;

  useEffect(() => {
    // Auto-follow user location
    if (userLocation && mapRef.current) {
      const centerX = mapWidth / 2 - userLocation.x * mapScale;
      const centerY = mapHeight / 2 - userLocation.y * mapScale;
      setMapOffset({ x: centerX, y: centerY });
    }
  }, [userLocation, mapScale]);

  const handleMapClick = (event) => {
    if (isDragging) return;
    
    const rect = mapRef.current.getBoundingClientRect();
    const clickX = (event.clientX - rect.left - mapOffset.x) / mapScale;
    const clickY = (event.clientY - rect.top - mapOffset.y) / mapScale;
    
    // Find nearest landmark or create destination
    const nearestLandmark = findNearestLandmark(clickX, clickY);
    
    if (nearestLandmark && getDistance(clickX, clickY, nearestLandmark.x, nearestLandmark.y) < 30) {
      onDestinationSelect(nearestLandmark);
    } else {
      // Create custom destination
      const customDestination = {
        id: 'custom',
        name: 'Custom Destination',
        x: clickX,
        y: clickY,
        floor: currentFloor,
        type: 'custom'
      };
      onDestinationSelect(customDestination);
    }
  };

  const findNearestLandmark = (x, y) => {
    const floorLandmarks = landmarks.filter(l => l.floor === currentFloor);
    return floorLandmarks.reduce((nearest, landmark) => {
      const distance = getDistance(x, y, landmark.x, landmark.y);
      if (!nearest || distance < getDistance(x, y, nearest.x, nearest.y)) {
        return landmark;
      }
      return nearest;
    }, null);
  };

  const getDistance = (x1, y1, x2, y2) => {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  };

  const handleMouseDown = (event) => {
    setIsDragging(true);
    setDragStart({ x: event.clientX - mapOffset.x, y: event.clientY - mapOffset.y });
  };

  const handleMouseMove = (event) => {
    if (!isDragging) return;
    
    const newOffset = {
      x: event.clientX - dragStart.x,
      y: event.clientY - dragStart.y
    };
    setMapOffset(newOffset);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (event) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.5, Math.min(3, mapScale * delta));
    setMapScale(newScale);
  };

  const getLandmarkIcon = (type) => {
    const icons = {
      entrance: '🚪',
      elevator: '🛗',
      restroom: '🚻',
      room: '🏢',
      emergency: '🚨',
      stairs: '🪜',
      parking: '🅿️',
      custom: '📍'
    };
    return icons[type] || '📍';
  };

  const getLandmarkColor = (type, isEmergency = false) => {
    if (isEmergency) return '#ff4444';
    
    const colors = {
      entrance: '#4CAF50',
      elevator: '#2196F3',
      restroom: '#9C27B0',
      room: '#FF9800',
      emergency: '#f44336',
      stairs: '#607D8B',
      parking: '#795548',
      custom: '#E91E63'
    };
    return colors[type] || '#757575';
  };

  const renderFloorPlan = () => {
    // Simplified floor plan rendering
    return (
      <g>
        {/* Building outline */}
        <rect 
          x="0" 
          y="0" 
          width="500" 
          height="400" 
          fill="#f5f5f5" 
          stroke="#333" 
          strokeWidth="2"
        />
        
        {/* Rooms */}
        <rect x="50" y="50" width="100" height="80" fill="#e3f2fd" stroke="#1976d2" strokeWidth="1" />
        <rect x="200" y="50" width="120" height="80" fill="#f3e5f5" stroke="#7b1fa2" strokeWidth="1" />
        <rect x="350" y="50" width="100" height="80" fill="#fff3e0" stroke="#f57c00" strokeWidth="1" />
        
        <rect x="50" y="180" width="100" height="100" fill="#e8f5e8" stroke="#388e3c" strokeWidth="1" />
        <rect x="200" y="180" width="120" height="100" fill="#fce4ec" stroke="#c2185b" strokeWidth="1" />
        <rect x="350" y="180" width="100" height="100" fill="#f1f8e9" stroke="#689f38" strokeWidth="1" />
        
        {/* Corridors */}
        <rect x="0" y="150" width="500" height="30" fill="#ffffff" stroke="#bdbdbd" strokeWidth="1" />
        <rect x="170" y="0" width="30" height="400" fill="#ffffff" stroke="#bdbdbd" strokeWidth="1" />
        
        {/* Emergency exits */}
        <rect x="480" y="40" width="20" height="60" fill="#ffebee" stroke="#d32f2f" strokeWidth="2" />
        <rect x="0" y="340" width="60" height="20" fill="#ffebee" stroke="#d32f2f" strokeWidth="2" />
      </g>
    );
  };

  const renderPath = () => {
    if (!currentPath || currentPath.length < 2) return null;
    
    const pathPoints = currentPath
      .filter(point => point.floor === currentFloor)
      .map(point => `${point.x},${point.y}`)
      .join(' ');
    
    if (!pathPoints) return null;
    
    return (
      <g>
        <polyline
          points={pathPoints}
          fill="none"
          stroke={emergencyMode ? '#ff4444' : '#2196F3'}
          strokeWidth="4"
          strokeDasharray={emergencyMode ? '10,5' : 'none'}
          opacity="0.8"
        />
        
        {/* Path direction arrows */}
        {currentPath.filter(point => point.floor === currentFloor).map((point, index) => {
          if (index === 0 || index === currentPath.length - 1) return null;
          
          const prevPoint = currentPath[index - 1];
          const angle = Math.atan2(point.y - prevPoint.y, point.x - prevPoint.x) * 180 / Math.PI;
          
          return (
            <g key={index} transform={`translate(${point.x}, ${point.y}) rotate(${angle})`}>
              <polygon
                points="-5,-3 5,0 -5,3"
                fill={emergencyMode ? '#ff4444' : '#2196F3'}
                opacity="0.7"
              />
            </g>
          );
        })}
      </g>
    );
  };

  const renderLandmarks = () => {
    const floorLandmarks = landmarks.filter(l => l.floor === currentFloor);
    
    return floorLandmarks.map(landmark => {
      const isDestination = destination && landmark.id === destination.id;
      const isEmergencyHighlight = emergencyMode && landmark.type === 'emergency';
      
      return (
        <g key={landmark.id}>
          <circle
            cx={landmark.x}
            cy={landmark.y}
            r={isDestination ? 15 : isEmergencyHighlight ? 12 : 8}
            fill={getLandmarkColor(landmark.type, isEmergencyHighlight)}
            stroke={isDestination ? '#fff' : 'none'}
            strokeWidth={isDestination ? 3 : 0}
            opacity={isEmergencyHighlight ? 1 : 0.8}
            onMouseEnter={() => setHoveredLandmark(landmark)}
            onMouseLeave={() => setHoveredLandmark(null)}
            style={{ cursor: 'pointer' }}
            className={isEmergencyHighlight ? 'emergency-pulse' : ''}
          />
          
          <text
            x={landmark.x}
            y={landmark.y + 25}
            textAnchor="middle"
            fontSize="12"
            fill="#333"
            fontWeight={isDestination ? 'bold' : 'normal'}
          >
            {getLandmarkIcon(landmark.type)} {landmark.name}
          </text>
          
          {isDestination && (
            <circle
              cx={landmark.x}
              cy={landmark.y}
              r={20}
              fill="none"
              stroke="#2196F3"
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.6"
            >
              <animateTransform
                attributeName="transform"
                type="rotate"
                values="0 {landmark.x} {landmark.y};360 {landmark.x} {landmark.y}"
                dur="3s"
                repeatCount="indefinite"
              />
            </circle>
          )}
        </g>
      );
    });
  };

  const renderUserLocation = () => {
    if (!userLocation || userLocation.floor !== currentFloor) return null;
    
    return (
      <g>
        <circle
          cx={userLocation.x}
          cy={userLocation.y}
          r="10"
          fill="#4CAF50"
          stroke="#fff"
          strokeWidth="3"
        />
        
        <circle
          cx={userLocation.x}
          cy={userLocation.y}
          r="15"
          fill="none"
          stroke="#4CAF50"
          strokeWidth="2"
          opacity="0.5"
        >
          <animate
            attributeName="r"
            values="15;25;15"
            dur="2s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0.5;0;0.5"
            dur="2s"
            repeatCount="indefinite"
          />
        </circle>
        
        <text
          x={userLocation.x}
          y={userLocation.y - 20}
          textAnchor="middle"
          fontSize="12"
          fill="#4CAF50"
          fontWeight="bold"
        >
          📍 You are here
        </text>
      </g>
    );
  };

  return (
    <div className="navigation-map">
      <div className="map-controls">
        <div className="floor-selector">
          <label>Floor:</label>
          {buildingData?.floors?.map(floor => (
            <button
              key={floor.id}
              className={`floor-btn ${currentFloor === floor.id ? 'active' : ''}`}
              onClick={() => setCurrentFloor(floor.id)}
            >
              {floor.name}
            </button>
          ))}
        </div>
        
        <div className="zoom-controls">
          <button onClick={() => setMapScale(prev => Math.min(3, prev * 1.2))}>🔍+</button>
          <span>{Math.round(mapScale * 100)}%</span>
          <button onClick={() => setMapScale(prev => Math.max(0.5, prev / 1.2))}>🔍-</button>
        </div>
        
        <div className="map-info">
          {emergencyMode && (
            <div className="emergency-indicator">
              🚨 Emergency Mode Active
            </div>
          )}
        </div>
      </div>
      
      <div 
        className="map-container"
        ref={mapRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onClick={handleMapClick}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        <svg 
          width={mapWidth} 
          height={mapHeight}
          viewBox={`0 0 ${mapWidth} ${mapHeight}`}
          style={{
            transform: `translate(${mapOffset.x}px, ${mapOffset.y}px) scale(${mapScale})`,
            transformOrigin: '0 0'
          }}
        >
          {renderFloorPlan()}
          {renderPath()}
          {renderLandmarks()}
          {renderUserLocation()}
        </svg>
        
        {hoveredLandmark && (
          <div 
            className="landmark-tooltip"
            style={{
              position: 'absolute',
              left: hoveredLandmark.x * mapScale + mapOffset.x + 10,
              top: hoveredLandmark.y * mapScale + mapOffset.y - 10,
              background: 'rgba(0,0,0,0.8)',
              color: 'white',
              padding: '5px 10px',
              borderRadius: '4px',
              fontSize: '12px',
              pointerEvents: 'none',
              zIndex: 1000
            }}
          >
            <strong>{hoveredLandmark.name}</strong>
            <br />
            Type: {hoveredLandmark.type}
            <br />
            Floor: {hoveredLandmark.floor}
          </div>
        )}
      </div>
      
      <div className="map-legend">
        <h4>Legend</h4>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#4CAF50' }}></span>
            Your Location
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#2196F3' }}></span>
            Navigation Path
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#ff4444' }}></span>
            Emergency Route
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#FF9800' }}></span>
            Landmarks
          </div>
        </div>
      </div>
    </div>
  );
};

export default NavigationMap;