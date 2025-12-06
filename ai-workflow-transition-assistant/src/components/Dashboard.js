import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

const DashboardContainer = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  padding: 20px 0;
`;

const Card = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 25px;
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const CardTitle = styled.h3`
  margin: 0 0 20px 0;
  font-size: 1.2rem;
  font-weight: 600;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  overflow: hidden;
  margin: 10px 0;
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #4CAF50, #8BC34A);
  border-radius: 4px;
`;

const StatGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
  margin-top: 20px;
`;

const StatItem = styled.div`
  text-align: center;
  padding: 15px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
`;

const StatValue = styled.div`
  font-size: 2rem;
  font-weight: bold;
  color: #4CAF50;
`;

const StatLabel = styled.div`
  font-size: 0.9rem;
  opacity: 0.8;
  margin-top: 5px;
`;

const QuickActionButton = styled(motion.button)`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 10px;
  padding: 15px 20px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  margin: 5px;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
  }
`;

const ActivityItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  
  &:last-child {
    border-bottom: none;
  }
`;

const ActivityText = styled.div`
  flex: 1;
`;

const ActivityTime = styled.div`
  font-size: 0.8rem;
  opacity: 0.7;
`;

const Dashboard = ({ user, notifications, onUpdateProgress }) => {
  const [aiIntegrationStatus, setAiIntegrationStatus] = useState({
    connected: 3,
    pending: 2,
    failed: 1
  });

  const [weeklyProgress, setWeeklyProgress] = useState({
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [{
      label: 'Training Hours',
      data: [2, 3, 1, 4, 2, 3, 1],
      backgroundColor: 'rgba(76, 175, 80, 0.6)',
      borderColor: 'rgba(76, 175, 80, 1)',
      borderWidth: 1
    }]
  });

  const progressPercentage = (user.completedModules / user.totalModules) * 100;

  const doughnutData = {
    labels: ['Completed', 'In Progress', 'Not Started'],
    datasets: [{
      data: [user.completedModules, 2, user.totalModules - user.completedModules - 2],
      backgroundColor: ['#4CAF50', '#FF9800', '#f44336'],
      borderWidth: 0
    }]
  };

  const recentActivities = [
    { text: 'Completed "Introduction to AI Workflows"', time: '2 hours ago' },
    { text: 'Started "Legacy System Integration"', time: '1 day ago' },
    { text: 'Asked AI Assistant about job security', time: '2 days ago' },
    { text: 'Joined team collaboration session', time: '3 days ago' }
  ];

  return (
    <DashboardContainer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* Progress Overview */}
      <Card
        whileHover={{ scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <CardTitle>Learning Progress</CardTitle>
        <div>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4CAF50' }}>
            {Math.round(progressPercentage)}%
          </div>
          <div style={{ opacity: 0.8, marginBottom: '15px' }}>Overall Completion</div>
          <ProgressBar>
            <ProgressFill
              initial={{ width: 0 }}
              animate={{ width: `${progressPercentage}%` }}
              transition={{ duration: 1, delay: 0.5 }}
            />
          </ProgressBar>
          <div style={{ fontSize: '0.9rem', opacity: 0.8 }}>
            {user.completedModules} of {user.totalModules} modules completed
          </div>
        </div>
      </Card>

      {/* AI Integration Status */}
      <Card
        whileHover={{ scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <CardTitle>AI Integration Status</CardTitle>
        <StatGrid>
          <StatItem>
            <StatValue>{aiIntegrationStatus.connected}</StatValue>
            <StatLabel>Connected</StatLabel>
          </StatItem>
          <StatItem>
            <StatValue>{aiIntegrationStatus.pending}</StatValue>
            <StatLabel>Pending</StatLabel>
          </StatItem>
        </StatGrid>
        <div style={{ marginTop: '20px' }}>
          <Doughnut 
            data={doughnutData} 
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: 'bottom',
                  labels: {
                    color: 'white'
                  }
                }
              }
            }}
          />
        </div>
      </Card>

      {/* Weekly Activity */}
      <Card
        whileHover={{ scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <CardTitle>Weekly Training Activity</CardTitle>
        <Bar 
          data={weeklyProgress}
          options={{
            responsive: true,
            plugins: {
              legend: {
                display: false
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                ticks: {
                  color: 'white'
                },
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              },
              x: {
                ticks: {
                  color: 'white'
                },
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              }
            }
          }}
        />
      </Card>

      {/* Quick Actions */}
      <Card
        whileHover={{ scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <CardTitle>Quick Actions</CardTitle>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <QuickActionButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/training'}
          >
            Continue Training
          </QuickActionButton>
          <QuickActionButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/chatbot'}
          >
            Ask AI Assistant
          </QuickActionButton>
          <QuickActionButton
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => window.location.href = '/integration'}
          >
            Check Integrations
          </QuickActionButton>
        </div>
      </Card>

      {/* Recent Activities */}
      <Card
        whileHover={{ scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <CardTitle>Recent Activities</CardTitle>
        <div>
          {recentActivities.map((activity, index) => (
            <ActivityItem key={index}>
              <ActivityText>{activity.text}</ActivityText>
              <ActivityTime>{activity.time}</ActivityTime>
            </ActivityItem>
          ))}
        </div>
      </Card>

      {/* Notifications */}
      <Card
        whileHover={{ scale: 1.02 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <CardTitle>Recent Notifications</CardTitle>
        <div>
          {notifications.slice(0, 4).map((notification) => (
            <ActivityItem key={notification.id}>
              <ActivityText>
                <div style={{ 
                  color: notification.type === 'success' ? '#4CAF50' : 
                         notification.type === 'warning' ? '#FF9800' : '#2196F3',
                  fontWeight: '500'
                }}>
                  {notification.message}
                </div>
              </ActivityText>
              <ActivityTime>
                {new Date(notification.timestamp).toLocaleDateString()}
              </ActivityTime>
            </ActivityItem>
          ))}
        </div>
      </Card>
    </DashboardContainer>
  );
};

export default Dashboard;