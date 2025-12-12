const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// OpenAI configuration
const { OpenAI } = require('openai');
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'AI Workflow Automation Assistant API is running' });
});

// Resistance Assessment Endpoint
app.post('/api/assess-resistance', async (req, res) => {
  try {
    const { teamData, currentProcesses, automationGoals } = req.body;
    
    const prompt = `
      As an AI change management expert, analyze the following team data and provide a resistance assessment:
      
      Team Data: ${JSON.stringify(teamData)}
      Current Processes: ${JSON.stringify(currentProcesses)}
      Automation Goals: ${JSON.stringify(automationGoals)}
      
      Please provide:
      1. Resistance Risk Level (Low/Medium/High)
      2. Key Risk Factors
      3. Recommended Mitigation Strategies
      4. Timeline for Implementation
      
      Format the response as JSON with these fields: riskLevel, riskFactors, strategies, timeline
    `;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });
    
    const assessment = JSON.parse(completion.choices[0].message.content);
    res.json({ success: true, assessment });
    
  } catch (error) {
    console.error('Error in resistance assessment:', error);
    res.status(500).json({ success: false, error: 'Failed to assess resistance' });
  }
});

// Training Plan Generation Endpoint
app.post('/api/generate-training-plan', async (req, res) => {
  try {
    const { employeeProfile, skillGaps, learningPreferences } = req.body;
    
    const prompt = `
      Create a personalized training plan for an employee with the following profile:
      
      Employee Profile: ${JSON.stringify(employeeProfile)}
      Skill Gaps: ${JSON.stringify(skillGaps)}
      Learning Preferences: ${JSON.stringify(learningPreferences)}
      
      Please provide:
      1. Learning Modules (with descriptions and estimated time)
      2. Recommended Learning Path
      3. Milestones and Checkpoints
      4. Resources and Tools
      
      Format the response as JSON with these fields: modules, learningPath, milestones, resources
    `;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });
    
    const trainingPlan = JSON.parse(completion.choices[0].message.content);
    res.json({ success: true, trainingPlan });
    
  } catch (error) {
    console.error('Error generating training plan:', error);
    res.status(500).json({ success: false, error: 'Failed to generate training plan' });
  }
});

// Change Management Strategies Endpoint
app.post('/api/change-strategies', async (req, res) => {
  try {
    const { organizationSize, industry, changeType, timeline } = req.body;
    
    const prompt = `
      Provide change management strategies for:
      
      Organization Size: ${organizationSize}
      Industry: ${industry}
      Change Type: ${changeType}
      Timeline: ${timeline}
      
      Please provide:
      1. Communication Strategy
      2. Stakeholder Engagement Plan
      3. Training and Support Approach
      4. Success Metrics
      
      Format the response as JSON with these fields: communication, stakeholderPlan, trainingApproach, metrics
    `;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });
    
    const strategies = JSON.parse(completion.choices[0].message.content);
    res.json({ success: true, strategies });
    
  } catch (error) {
    console.error('Error generating change strategies:', error);
    res.status(500).json({ success: false, error: 'Failed to generate change strategies' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});