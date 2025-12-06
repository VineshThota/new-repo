import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import Dashboard from './components/Dashboard';
import ChatBot from './components/ChatBot';
import TrainingModules from './components/TrainingModules';
import ProgressTracker from './components/ProgressTracker';
import LegacyIntegration from './components/LegacyIntegration';
import Navigation from './components/Navigation';
import './App.css';

const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: 'Inter', sans-serif;
`;

const MainContent = styled(motion.main)`
  flex: 1;
  margin-left: 250px;
  padding: 20px;
  overflow-y: auto;
`;

const WelcomeMessage = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 30px;
  margin-bottom: 30px;
  color: white;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

function App() {
  const [user, setUser] = useState({
    name: 'Employee',
    department: 'General',
    transitionStage: 'beginner',
    completedModules: 0,
    totalModules: 12
  });

  const [notifications, setNotifications] = useState([
    {
      id: 1,
      type: 'info',
      message: 'Welcome to your AI Workflow Transition Journey!',
      timestamp: new Date().toISOString()
    },
    {
      id: 2,
      type: 'success',
      message: 'Your first training module is ready to begin.',
      timestamp: new Date().toISOString()
    }
  ]);

  useEffect(() => {
    // Initialize user data from localStorage or API
    const savedUser = localStorage.getItem('transitionUser');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const updateUserProgress = (moduleId, completed) => {
    setUser(prev => {
      const updated = {
        ...prev,
        completedModules: completed ? prev.completedModules + 1 : prev.completedModules - 1
      };
      localStorage.setItem('transitionUser', JSON.stringify(updated));
      return updated;
    });
  };

  const addNotification = (notification) => {
    setNotifications(prev => [{
      ...notification,
      id: Date.now(),
      timestamp: new Date().toISOString()
    }, ...prev]);
  };

  return (
    <Router>
      <AppContainer>
        <Navigation user={user} notifications={notifications} />
        <MainContent
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <WelcomeMessage
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <h1>AI Workflow Transition Assistant</h1>
            <p>Empowering employees to embrace AI-powered autonomous workflows with confidence</p>
            <p>Progress: {user.completedModules}/{user.totalModules} modules completed</p>
          </WelcomeMessage>

          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route 
              path="/dashboard" 
              element={
                <Dashboard 
                  user={user} 
                  notifications={notifications}
                  onUpdateProgress={updateUserProgress}
                />
              } 
            />
            <Route 
              path="/chatbot" 
              element={
                <ChatBot 
                  user={user}
                  onAddNotification={addNotification}
                />
              } 
            />
            <Route 
              path="/training" 
              element={
                <TrainingModules 
                  user={user}
                  onUpdateProgress={updateUserProgress}
                  onAddNotification={addNotification}
                />
              } 
            />
            <Route 
              path="/progress" 
              element={
                <ProgressTracker 
                  user={user}
                  notifications={notifications}
                />
              } 
            />
            <Route 
              path="/integration" 
              element={
                <LegacyIntegration 
                  user={user}
                  onAddNotification={addNotification}
                />
              } 
            />
          </Routes>
        </MainContent>
      </AppContainer>
    </Router>
  );
}

export default App;