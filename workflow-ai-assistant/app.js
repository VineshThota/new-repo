const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { OpenAI } = require('openai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Routes

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Workflow Analysis Endpoints
app.post('/api/analyze-workflow', async (req, res) => {
  try {
    const { workflowDescription, currentTools, painPoints } = req.body;
    
    const prompt = `
      Analyze this business workflow and provide automation recommendations:
      
      Workflow Description: ${workflowDescription}
      Current Tools: ${currentTools.join(', ')}
      Pain Points: ${painPoints.join(', ')}
      
      Please provide:
      1. Automation opportunities
      2. Tool integration suggestions
      3. Expected efficiency gains
      4. Implementation complexity (1-10)
      5. ROI estimation
      
      Format as JSON with clear sections.
    `;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });
    
    const analysis = JSON.parse(completion.choices[0].message.content);
    
    res.json({
      success: true,
      analysis,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Workflow analysis error:', error);
    res.status(500).json({ error: 'Failed to analyze workflow' });
  }
});

app.get('/api/automation-suggestions', async (req, res) => {
  try {
    const { industry, teamSize, currentAutomationLevel } = req.query;
    
    const prompt = `
      Based on current LinkedIn trends in AI automation tools, suggest 5 specific automation ideas for:
      
      Industry: ${industry}
      Team Size: ${teamSize}
      Current Automation Level: ${currentAutomationLevel}/10
      
      Focus on trending automation areas like:
      - AI-powered workflow optimization
      - Smart data integration
      - Employee adoption strategies
      - ROI-focused implementations
      
      Provide practical, implementable suggestions with expected benefits.
    `;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.8,
    });
    
    res.json({
      success: true,
      suggestions: completion.choices[0].message.content,
      trendingTopics: ['AI Automation Tools', 'Workflow Optimization', 'Data Integration'],
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Suggestions error:', error);
    res.status(500).json({ error: 'Failed to generate suggestions' });
  }
});

app.post('/api/calculate-roi', async (req, res) => {
  try {
    const { currentCosts, timeSpent, proposedAutomation, implementationCost } = req.body;
    
    // Simple ROI calculation
    const monthlySavings = (timeSpent * 40) - (implementationCost / 12); // Assuming $40/hour
    const annualSavings = monthlySavings * 12;
    const roi = ((annualSavings - implementationCost) / implementationCost) * 100;
    const paybackPeriod = implementationCost / monthlySavings;
    
    res.json({
      success: true,
      roi: {
        monthlySavings: Math.round(monthlySavings),
        annualSavings: Math.round(annualSavings),
        roiPercentage: Math.round(roi),
        paybackMonths: Math.round(paybackPeriod),
        recommendation: roi > 200 ? 'Highly Recommended' : roi > 100 ? 'Recommended' : 'Consider Alternatives'
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('ROI calculation error:', error);
    res.status(500).json({ error: 'Failed to calculate ROI' });
  }
});

// Tool Integration Endpoints
app.get('/api/available-tools', (req, res) => {
  const tools = [
    { id: 'zapier', name: 'Zapier', category: 'Automation', trending: true },
    { id: 'slack', name: 'Slack', category: 'Communication', trending: true },
    { id: 'notion', name: 'Notion', category: 'Documentation', trending: false },
    { id: 'airtable', name: 'Airtable', category: 'Database', trending: true },
    { id: 'hubspot', name: 'HubSpot', category: 'CRM', trending: false },
    { id: 'salesforce', name: 'Salesforce', category: 'CRM', trending: false },
    { id: 'gmail', name: 'Gmail', category: 'Email', trending: true },
    { id: 'sheets', name: 'Google Sheets', category: 'Spreadsheet', trending: true }
  ];
  
  res.json({
    success: true,
    tools,
    trendingCount: tools.filter(t => t.trending).length,
    timestamp: new Date().toISOString()
  });
});

app.post('/api/connect-tool', async (req, res) => {
  try {
    const { toolId, apiKey, configuration } = req.body;
    
    // Simulate tool connection
    const connectionStatus = {
      toolId,
      status: 'connected',
      connectedAt: new Date().toISOString(),
      features: ['Data Sync', 'Webhook Support', 'Real-time Updates']
    };
    
    res.json({
      success: true,
      connection: connectionStatus,
      message: `Successfully connected to ${toolId}`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Tool connection error:', error);
    res.status(500).json({ error: 'Failed to connect tool' });
  }
});

app.get('/api/integration-status', (req, res) => {
  const integrations = [
    { tool: 'Zapier', status: 'healthy', lastSync: '2 minutes ago', dataFlow: 'active' },
    { tool: 'Slack', status: 'healthy', lastSync: '5 minutes ago', dataFlow: 'active' },
    { tool: 'Gmail', status: 'warning', lastSync: '1 hour ago', dataFlow: 'limited' }
  ];
  
  res.json({
    success: true,
    integrations,
    overallHealth: 'good',
    timestamp: new Date().toISOString()
  });
});

// Employee Adoption Endpoints
app.get('/api/training-plan', async (req, res) => {
  try {
    const { role, experience, automationGoals } = req.query;
    
    const prompt = `
      Create a personalized training plan for employee adoption of AI automation tools:
      
      Role: ${role}
      Experience Level: ${experience}
      Automation Goals: ${automationGoals}
      
      Based on LinkedIn trends showing resistance to AI automation, provide:
      1. Step-by-step training modules
      2. Hands-on exercises
      3. Success milestones
      4. Change management tips
      5. Expected timeline
      
      Focus on building confidence and demonstrating value.
    `;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.6,
    });
    
    res.json({
      success: true,
      trainingPlan: completion.choices[0].message.content,
      estimatedDuration: '2-4 weeks',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Training plan error:', error);
    res.status(500).json({ error: 'Failed to generate training plan' });
  }
});

app.post('/api/track-progress', (req, res) => {
  try {
    const { userId, moduleCompleted, timeSpent, satisfaction } = req.body;
    
    // Simulate progress tracking
    const progress = {
      userId,
      moduleCompleted,
      timeSpent,
      satisfaction,
      overallProgress: '65%',
      nextModule: 'Advanced Workflow Design',
      achievements: ['First Automation Created', 'Tool Integration Master']
    };
    
    res.json({
      success: true,
      progress,
      message: 'Progress tracked successfully',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Progress tracking error:', error);
    res.status(500).json({ error: 'Failed to track progress' });
  }
});

app.get('/api/success-metrics', (req, res) => {
  const metrics = {
    adoptionRate: '78%',
    timeToValue: '12 days',
    userSatisfaction: '4.2/5',
    automationsCreated: 156,
    timeSaved: '240 hours/month',
    costSavings: '$18,000/month',
    trendingMetrics: {
      'AI Tool Adoption': '+45%',
      'Workflow Efficiency': '+67%',
      'Employee Confidence': '+52%'
    }
  };
  
  res.json({
    success: true,
    metrics,
    lastUpdated: new Date().toISOString(),
    timestamp: new Date().toISOString()
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Start server
app.listen(PORT, () => {
  console.log(`WorkflowAI Assistant server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});

module.exports = app;