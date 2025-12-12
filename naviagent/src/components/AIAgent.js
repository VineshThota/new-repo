import React, { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import * as tf from '@tensorflow/tfjs';

const AIAgent = forwardRef(({ 
  status, 
  userLocation, 
  sensorData, 
  onNavigationRequest, 
  emergencyMode 
}, ref) => {
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [agentThoughts, setAgentThoughts] = useState([]);
  const [learningData, setLearningData] = useState([]);
  const [contextualAwareness, setContextualAwareness] = useState({
    crowdDensity: 'low',
    noiseLevel: 'quiet',
    lightingCondition: 'good',
    timeOfDay: 'day',
    weatherIndoor: 'normal'
  });
  const [voiceCommands, setVoiceCommands] = useState([]);
  const [decisionHistory, setDecisionHistory] = useState([]);
  const [adaptiveStrategies, setAdaptiveStrategies] = useState({
    pathOptimization: 'shortest',
    avoidanceBehavior: 'moderate',
    learningRate: 0.1,
    confidenceThreshold: 0.8
  });

  // AI Agent's internal state and reasoning
  const [agentState, setAgentState] = useState({
    currentGoal: null,
    subGoals: [],
    obstacles: [],
    confidence: 0.0,
    reasoning: '',
    nextAction: null,
    memoryBank: [],
    experienceLevel: 0
  });

  useImperativeHandle(ref, () => ({
    processVoiceCommand: (command) => processVoiceCommand(command),
    updateContext: (context) => updateContextualAwareness(context),
    getRecommendations: () => generateRecommendations(),
    emergencyOverride: () => activateEmergencyProtocol()
  }));

  useEffect(() => {
    initializeAIModels();
    startContinuousLearning();
  }, []);

  useEffect(() => {
    if (status === 'ready' && sensorData) {
      processSensorData(sensorData);
      updateContextualAwareness();
      makeIntelligentDecisions();
    }
  }, [sensorData, userLocation, status]);

  useEffect(() => {
    if (emergencyMode) {
      activateEmergencyProtocol();
    }
  }, [emergencyMode]);

  const initializeAIModels = async () => {
    try {
      // Initialize TensorFlow.js models for various AI tasks
      
      // Path prediction model
      const pathModel = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [6], units: 64, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dense({ units: 16, activation: 'relu' }),
          tf.layers.dense({ units: 2, activation: 'linear' }) // x, y coordinates
        ]
      });

      pathModel.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['mae']
      });

      setModel(pathModel);
      
      addAgentThought('AI models initialized successfully. Ready for intelligent navigation.');
      
    } catch (error) {
      console.error('AI model initialization failed:', error);
      addAgentThought('Warning: AI model initialization failed. Using fallback algorithms.');
    }
  };

  const processSensorData = (data) => {
    // Process accelerometer data for movement detection
    const acceleration = Math.sqrt(
      data.accelerometer.x ** 2 + 
      data.accelerometer.y ** 2 + 
      data.accelerometer.z ** 2
    );

    // Process WiFi signals for positioning
    const wifiFingerprint = data.wifi.map(ap => ({
      mac: ap.mac,
      rssi: ap.rssi,
      distance: rssiToDistance(ap.rssi)
    }));

    // Process Bluetooth beacons
    const bluetoothBeacons = data.bluetooth.map(beacon => ({
      mac: beacon.mac,
      rssi: beacon.rssi,
      distance: rssiToDistance(beacon.rssi)
    }));

    // Update agent's understanding of environment
    setAgentState(prev => ({
      ...prev,
      memoryBank: [...prev.memoryBank.slice(-100), { // Keep last 100 readings
        timestamp: Date.now(),
        location: userLocation,
        acceleration,
        wifiFingerprint,
        bluetoothBeacons
      }]
    }));

    // Make predictions if model is available
    if (model && wifiFingerprint.length >= 3) {
      makePredictions(wifiFingerprint, bluetoothBeacons);
    }
  };

  const rssiToDistance = (rssi) => {
    // Convert RSSI to approximate distance (simplified model)
    const txPower = -59; // Typical TX power at 1m
    if (rssi === 0) return -1.0;
    
    const ratio = (txPower - rssi) / 20.0;
    return Math.pow(10, ratio);
  };

  const makePredictions = async (wifiData, bluetoothData) => {
    try {
      // Prepare input tensor
      const inputData = [
        userLocation.x / 1000, // Normalize coordinates
        userLocation.y / 1000,
        wifiData[0]?.rssi || -100,
        wifiData[1]?.rssi || -100,
        bluetoothData[0]?.rssi || -100,
        bluetoothData[1]?.rssi || -100
      ];

      const inputTensor = tf.tensor2d([inputData]);
      const prediction = await model.predict(inputTensor).data();
      
      const predictedLocation = {
        x: prediction[0] * 1000, // Denormalize
        y: prediction[1] * 1000
      };

      setPredictions(prev => [...prev.slice(-10), {
        timestamp: Date.now(),
        predicted: predictedLocation,
        actual: userLocation,
        confidence: calculateConfidence(predictedLocation, userLocation)
      }]);

      inputTensor.dispose();
      
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  const calculateConfidence = (predicted, actual) => {
    const distance = Math.sqrt(
      Math.pow(predicted.x - actual.x, 2) + 
      Math.pow(predicted.y - actual.y, 2)
    );
    return Math.max(0, 1 - (distance / 100)); // Confidence decreases with distance
  };

  const updateContextualAwareness = () => {
    const currentHour = new Date().getHours();
    const wifiCount = sensorData.wifi.length;
    const bluetoothCount = sensorData.bluetooth.length;
    
    setContextualAwareness(prev => ({
      ...prev,
      timeOfDay: currentHour < 6 || currentHour > 20 ? 'night' : 'day',
      crowdDensity: wifiCount > 10 ? 'high' : wifiCount > 5 ? 'medium' : 'low',
      noiseLevel: bluetoothCount > 8 ? 'noisy' : 'quiet'
    }));
  };

  const makeIntelligentDecisions = () => {
    const currentTime = Date.now();
    const recentPredictions = predictions.slice(-5);
    const avgConfidence = recentPredictions.reduce((sum, p) => sum + p.confidence, 0) / recentPredictions.length || 0;
    
    let reasoning = '';
    let nextAction = null;
    let confidence = avgConfidence;

    // Decision logic based on context and sensor data
    if (emergencyMode) {
      reasoning = 'Emergency mode activated. Prioritizing fastest route to nearest exit.';
      nextAction = 'emergency_navigation';
      confidence = 1.0;
    } else if (contextualAwareness.crowdDensity === 'high') {
      reasoning = 'High crowd density detected. Suggesting alternative routes to avoid congestion.';
      nextAction = 'avoid_crowds';
    } else if (avgConfidence < 0.5) {
      reasoning = 'Low positioning confidence. Recommending landmark-based navigation.';
      nextAction = 'landmark_navigation';
    } else {
      reasoning = 'Normal navigation conditions. Using optimal path planning.';
      nextAction = 'optimal_navigation';
    }

    setAgentState(prev => ({
      ...prev,
      confidence,
      reasoning,
      nextAction,
      experienceLevel: prev.experienceLevel + 0.001 // Gradual learning
    }));

    // Record decision for learning
    setDecisionHistory(prev => [...prev.slice(-50), {
      timestamp: currentTime,
      context: contextualAwareness,
      decision: nextAction,
      reasoning,
      confidence
    }]);

    addAgentThought(reasoning);
  };

  const processVoiceCommand = (command) => {
    const processedCommand = command.toLowerCase().trim();
    
    setVoiceCommands(prev => [...prev.slice(-10), {
      timestamp: Date.now(),
      command: processedCommand,
      processed: true
    }]);

    // Simple voice command processing
    if (processedCommand.includes('navigate to') || processedCommand.includes('go to')) {
      const destination = extractDestination(processedCommand);
      if (destination) {
        addAgentThought(`Voice command received: Navigate to ${destination}`);
        // This would trigger navigation to the specified destination
        // onNavigationRequest(destination);
      }
    } else if (processedCommand.includes('emergency') || processedCommand.includes('help')) {
      addAgentThought('Emergency voice command detected. Activating emergency protocol.');
      activateEmergencyProtocol();
    } else if (processedCommand.includes('stop')) {
      addAgentThought('Stop command received.');
      // Stop current navigation
    }
  };

  const extractDestination = (command) => {
    // Simple destination extraction (in real app, use NLP)
    const destinations = ['elevator', 'exit', 'restroom', 'lobby', 'stairs', 'parking'];
    return destinations.find(dest => command.includes(dest));
  };

  const activateEmergencyProtocol = () => {
    setAgentState(prev => ({
      ...prev,
      currentGoal: 'emergency_exit',
      confidence: 1.0,
      reasoning: 'Emergency protocol activated. Finding fastest route to safety.',
      nextAction: 'emergency_navigation'
    }));

    addAgentThought('🚨 EMERGENCY PROTOCOL ACTIVATED - Finding fastest route to nearest exit');
  };

  const generateRecommendations = () => {
    const recommendations = [];
    
    if (contextualAwareness.crowdDensity === 'high') {
      recommendations.push({
        type: 'route',
        message: 'Consider taking alternative route to avoid crowds',
        priority: 'medium'
      });
    }
    
    if (agentState.confidence < 0.6) {
      recommendations.push({
        type: 'positioning',
        message: 'Move closer to WiFi access points for better positioning',
        priority: 'high'
      });
    }
    
    if (contextualAwareness.timeOfDay === 'night') {
      recommendations.push({
        type: 'safety',
        message: 'Use well-lit pathways for night navigation',
        priority: 'high'
      });
    }
    
    return recommendations;
  };

  const startContinuousLearning = () => {
    // Simulate continuous learning from user behavior
    const learningInterval = setInterval(() => {
      if (agentState.memoryBank.length > 10) {
        // Analyze patterns in user movement and preferences
        const recentMemory = agentState.memoryBank.slice(-10);
        const patterns = analyzeMovementPatterns(recentMemory);
        
        setLearningData(prev => [...prev.slice(-100), {
          timestamp: Date.now(),
          patterns,
          adaptations: generateAdaptations(patterns)
        }]);
      }
    }, 5000);

    return () => clearInterval(learningInterval);
  };

  const analyzeMovementPatterns = (memory) => {
    // Analyze user movement patterns for learning
    const speeds = memory.map((m, i) => {
      if (i === 0) return 0;
      const prev = memory[i - 1];
      const distance = Math.sqrt(
        Math.pow(m.location.x - prev.location.x, 2) + 
        Math.pow(m.location.y - prev.location.y, 2)
      );
      const time = (m.timestamp - prev.timestamp) / 1000; // seconds
      return distance / time;
    }).filter(speed => speed > 0);

    const avgSpeed = speeds.reduce((sum, speed) => sum + speed, 0) / speeds.length || 0;
    
    return {
      averageSpeed: avgSpeed,
      movementVariability: calculateVariability(speeds),
      preferredPaths: extractPreferredPaths(memory)
    };
  };

  const calculateVariability = (values) => {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  };

  const extractPreferredPaths = (memory) => {
    // Extract commonly used paths
    return memory.map(m => ({ x: Math.round(m.location.x / 10) * 10, y: Math.round(m.location.y / 10) * 10 }));
  };

  const generateAdaptations = (patterns) => {
    const adaptations = {};
    
    if (patterns.averageSpeed > 2) {
      adaptations.pathOptimization = 'fastest';
    } else if (patterns.averageSpeed < 1) {
      adaptations.pathOptimization = 'comfortable';
    }
    
    return adaptations;
  };

  const addAgentThought = (thought) => {
    setAgentThoughts(prev => [...prev.slice(-20), {
      timestamp: Date.now(),
      thought,
      type: 'reasoning'
    }]);
  };

  return (
    <div className="ai-agent-panel">
      <div className="agent-header">
        <h3>🤖 AI Navigation Agent</h3>
        <div className={`agent-status ${status}`}>
          <span className="status-indicator"></span>
          {status}
        </div>
      </div>
      
      <div className="agent-content">
        <div className="agent-state">
          <h4>Current State</h4>
          <div className="state-info">
            <div className="confidence-meter">
              <label>Confidence: {(agentState.confidence * 100).toFixed(1)}%</label>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ width: `${agentState.confidence * 100}%` }}
                ></div>
              </div>
            </div>
            
            <div className="reasoning">
              <strong>Reasoning:</strong>
              <p>{agentState.reasoning || 'Analyzing environment...'}</p>
            </div>
            
            <div className="next-action">
              <strong>Next Action:</strong>
              <span className={`action-badge ${agentState.nextAction}`}>
                {agentState.nextAction || 'standby'}
              </span>
            </div>
          </div>
        </div>
        
        <div className="contextual-awareness">
          <h4>Environmental Context</h4>
          <div className="context-grid">
            <div className="context-item">
              <span className="context-label">Crowd Density:</span>
              <span className={`context-value ${contextualAwareness.crowdDensity}`}>
                {contextualAwareness.crowdDensity}
              </span>
            </div>
            <div className="context-item">
              <span className="context-label">Time:</span>
              <span className={`context-value ${contextualAwareness.timeOfDay}`}>
                {contextualAwareness.timeOfDay}
              </span>
            </div>
            <div className="context-item">
              <span className="context-label">Noise Level:</span>
              <span className={`context-value ${contextualAwareness.noiseLevel}`}>
                {contextualAwareness.noiseLevel}
              </span>
            </div>
          </div>
        </div>
        
        <div className="agent-thoughts">
          <h4>Agent Thoughts</h4>
          <div className="thoughts-list">
            {agentThoughts.slice(-5).map((thought, index) => (
              <div key={index} className="thought-item">
                <span className="thought-time">
                  {new Date(thought.timestamp).toLocaleTimeString()}
                </span>
                <span className="thought-text">{thought.thought}</span>
              </div>
            ))}
          </div>
        </div>
        
        <div className="learning-stats">
          <h4>Learning Progress</h4>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Experience Level:</span>
              <span className="stat-value">{agentState.experienceLevel.toFixed(3)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Decisions Made:</span>
              <span className="stat-value">{decisionHistory.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Memory Bank:</span>
              <span className="stat-value">{agentState.memoryBank.length} entries</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default AIAgent;