import React, { useState } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('assessment');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleAssessment = async (formData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/assess-resistance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const handleTrainingPlan = async (formData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/generate-training-plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const handleChangeStrategies = async (formData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/change-strategies', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 AI Workflow Automation Assistant</h1>
        <p>Overcome employee resistance to automation with AI-powered insights</p>
      </header>

      <nav className="tab-navigation">
        <button 
          className={activeTab === 'assessment' ? 'active' : ''}
          onClick={() => setActiveTab('assessment')}
        >
          📊 Resistance Assessment
        </button>
        <button 
          className={activeTab === 'training' ? 'active' : ''}
          onClick={() => setActiveTab('training')}
        >
          🎓 Training Plans
        </button>
        <button 
          className={activeTab === 'strategies' ? 'active' : ''}
          onClick={() => setActiveTab('strategies')}
        >
          🎯 Change Strategies
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'assessment' && (
          <AssessmentForm onSubmit={handleAssessment} loading={loading} />
        )}
        {activeTab === 'training' && (
          <TrainingForm onSubmit={handleTrainingPlan} loading={loading} />
        )}
        {activeTab === 'strategies' && (
          <StrategiesForm onSubmit={handleChangeStrategies} loading={loading} />
        )}

        {results && (
          <div className="results-section">
            <h3>✨ AI-Generated Recommendations</h3>
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  );
}

const AssessmentForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    teamData: { size: '', departments: '', avgAge: '', techSavviness: '' },
    currentProcesses: { manual: '', automated: '', complexity: '' },
    automationGoals: { timeline: '', scope: '', expectedBenefits: '' }
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="assessment-form">
      <h2>Team Resistance Assessment</h2>
      
      <div className="form-section">
        <h3>Team Information</h3>
        <input
          type="text"
          placeholder="Team size"
          value={formData.teamData.size}
          onChange={(e) => setFormData({
            ...formData,
            teamData: { ...formData.teamData, size: e.target.value }
          })}
        />
        <input
          type="text"
          placeholder="Departments involved"
          value={formData.teamData.departments}
          onChange={(e) => setFormData({
            ...formData,
            teamData: { ...formData.teamData, departments: e.target.value }
          })}
        />
      </div>

      <div className="form-section">
        <h3>Current Processes</h3>
        <textarea
          placeholder="Describe current manual processes"
          value={formData.currentProcesses.manual}
          onChange={(e) => setFormData({
            ...formData,
            currentProcesses: { ...formData.currentProcesses, manual: e.target.value }
          })}
        />
      </div>

      <div className="form-section">
        <h3>Automation Goals</h3>
        <input
          type="text"
          placeholder="Implementation timeline"
          value={formData.automationGoals.timeline}
          onChange={(e) => setFormData({
            ...formData,
            automationGoals: { ...formData.automationGoals, timeline: e.target.value }
          })}
        />
      </div>

      <button type="submit" disabled={loading}>
        {loading ? '🔄 Analyzing...' : '📊 Assess Resistance'}
      </button>
    </form>
  );
};

const TrainingForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    employeeProfile: { role: '', experience: '', currentSkills: '' },
    skillGaps: { technical: '', soft: '', automation: '' },
    learningPreferences: { style: '', pace: '', format: '' }
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="training-form">
      <h2>Personalized Training Plan</h2>
      
      <div className="form-section">
        <h3>Employee Profile</h3>
        <input
          type="text"
          placeholder="Job role"
          value={formData.employeeProfile.role}
          onChange={(e) => setFormData({
            ...formData,
            employeeProfile: { ...formData.employeeProfile, role: e.target.value }
          })}
        />
        <input
          type="text"
          placeholder="Years of experience"
          value={formData.employeeProfile.experience}
          onChange={(e) => setFormData({
            ...formData,
            employeeProfile: { ...formData.employeeProfile, experience: e.target.value }
          })}
        />
      </div>

      <div className="form-section">
        <h3>Skill Gaps</h3>
        <textarea
          placeholder="Technical skills needed"
          value={formData.skillGaps.technical}
          onChange={(e) => setFormData({
            ...formData,
            skillGaps: { ...formData.skillGaps, technical: e.target.value }
          })}
        />
      </div>

      <button type="submit" disabled={loading}>
        {loading ? '🔄 Generating...' : '🎓 Generate Training Plan'}
      </button>
    </form>
  );
};

const StrategiesForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    organizationSize: '',
    industry: '',
    changeType: '',
    timeline: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="strategies-form">
      <h2>Change Management Strategies</h2>
      
      <div className="form-section">
        <select
          value={formData.organizationSize}
          onChange={(e) => setFormData({ ...formData, organizationSize: e.target.value })}
        >
          <option value="">Select organization size</option>
          <option value="startup">Startup (1-50 employees)</option>
          <option value="small">Small (51-200 employees)</option>
          <option value="medium">Medium (201-1000 employees)</option>
          <option value="large">Large (1000+ employees)</option>
        </select>
        
        <input
          type="text"
          placeholder="Industry"
          value={formData.industry}
          onChange={(e) => setFormData({ ...formData, industry: e.target.value })}
        />
        
        <input
          type="text"
          placeholder="Type of change (e.g., process automation)"
          value={formData.changeType}
          onChange={(e) => setFormData({ ...formData, changeType: e.target.value })}
        />
        
        <input
          type="text"
          placeholder="Implementation timeline"
          value={formData.timeline}
          onChange={(e) => setFormData({ ...formData, timeline: e.target.value })}
        />
      </div>

      <button type="submit" disabled={loading}>
        {loading ? '🔄 Strategizing...' : '🎯 Get Change Strategies'}
      </button>
    </form>
  );
};

export default App;