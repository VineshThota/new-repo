const express = require('express');
const cors = require('cors');
const { OpenAI } = require('openai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize OpenAI (for production, use environment variables)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'your-api-key-here'
});

// Mock knowledge base for RAG implementation
const workflowKnowledgeBase = [
  {
    id: 1,
    category: 'customer_onboarding',
    bestPractices: [
      'Implement automated document collection using digital forms',
      'Set up intelligent routing based on customer type and complexity',
      'Create real-time progress tracking dashboards',
      'Integrate with CRM systems for seamless data flow',
      'Use AI-powered document verification'
    ],
    metrics: {
      averageTimeReduction: 40,
      costSavingsPerMonth: 2400,
      efficiencyImprovement: 35
    }
  },
  {
    id: 2,
    category: 'order_processing',
    bestPractices: [
      'Automate inventory checks and allocation',
      'Implement smart pricing algorithms',
      'Set up automated payment processing',
      'Create intelligent shipping optimization',
      'Use predictive analytics for demand forecasting'
    ],
    metrics: {
      averageTimeReduction: 50,
      costSavingsPerMonth: 3200,
      efficiencyImprovement: 45
    }
  },
  {
    id: 3,
    category: 'employee_training',
    bestPractices: [
      'Implement adaptive learning paths',
      'Use AI-powered content recommendations',
      'Set up automated progress tracking',
      'Create personalized training schedules',
      'Integrate with performance management systems'
    ],
    metrics: {
      averageTimeReduction: 30,
      costSavingsPerMonth: 1800,
      efficiencyImprovement: 25
    }
  },
  {
    id: 4,
    category: 'invoice_management',
    bestPractices: [
      'Automate invoice data extraction using OCR',
      'Implement smart approval workflows',
      'Set up automated payment scheduling',
      'Create exception handling for discrepancies',
      'Use AI for fraud detection'
    ],
    metrics: {
      averageTimeReduction: 60,
      costSavingsPerMonth: 4000,
      efficiencyImprovement: 55
    }
  }
];

// RAG function to retrieve relevant knowledge
function retrieveRelevantKnowledge(processDescription) {
  const description = processDescription.toLowerCase();
  
  // Simple keyword matching for demo (in production, use vector similarity)
  let relevantKnowledge = [];
  
  if (description.includes('customer') || description.includes('onboard')) {
    relevantKnowledge.push(workflowKnowledgeBase[0]);
  }
  if (description.includes('order') || description.includes('purchase') || description.includes('sales')) {
    relevantKnowledge.push(workflowKnowledgeBase[1]);
  }
  if (description.includes('training') || description.includes('employee') || description.includes('learning')) {
    relevantKnowledge.push(workflowKnowledgeBase[2]);
  }
  if (description.includes('invoice') || description.includes('payment') || description.includes('billing')) {
    relevantKnowledge.push(workflowKnowledgeBase[3]);
  }
  
  // If no specific match, return general best practices
  if (relevantKnowledge.length === 0) {
    relevantKnowledge = [workflowKnowledgeBase[0]]; // Default to customer onboarding
  }
  
  return relevantKnowledge;
}

// Generate AI recommendations using RAG
async function generateRecommendations(processDescription, relevantKnowledge) {
  try {
    const context = relevantKnowledge.map(kb => 
      `Category: ${kb.category}\nBest Practices: ${kb.bestPractices.join(', ')}\nMetrics: ${JSON.stringify(kb.metrics)}`
    ).join('\n\n');
    
    const prompt = `
    You are an AI workflow optimization expert. Based on the following process description and industry best practices, provide specific recommendations for improvement.
    
    Process Description: ${processDescription}
    
    Relevant Industry Knowledge:
    ${context}
    
    Please provide:
    1. 4 specific, actionable recommendations for improving this workflow
    2. An estimated current efficiency percentage (between 40-80%)
    3. Potential time and cost savings
    
    Format your response as JSON with the following structure:
    {
      "currentEfficiency": number,
      "recommendations": ["recommendation1", "recommendation2", "recommendation3", "recommendation4"],
      "potentialSavings": "X% time reduction, $Y/month cost savings"
    }
    `;
    
    // For demo purposes, return mock data (in production, use OpenAI API)
    const mockResponse = {
      currentEfficiency: Math.floor(Math.random() * 40) + 40, // 40-80%
      recommendations: relevantKnowledge[0].bestPractices.slice(0, 4),
      potentialSavings: `${relevantKnowledge[0].metrics.averageTimeReduction}% time reduction, $${relevantKnowledge[0].metrics.costSavingsPerMonth}/month cost savings`
    };
    
    return mockResponse;
    
    // Uncomment below for actual OpenAI integration
    /*
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });
    
    return JSON.parse(completion.choices[0].message.content);
    */
  } catch (error) {
    console.error('Error generating recommendations:', error);
    throw error;
  }
}

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'SmartFlow AI Backend Server',
    version: '1.0.0',
    status: 'running'
  });
});

// Analyze workflow endpoint
app.post('/api/analyze-workflow', async (req, res) => {
  try {
    const { processDescription } = req.body;
    
    if (!processDescription) {
      return res.status(400).json({ error: 'Process description is required' });
    }
    
    // Step 1: Retrieve relevant knowledge using RAG
    const relevantKnowledge = retrieveRelevantKnowledge(processDescription);
    
    // Step 2: Generate AI recommendations
    const recommendations = await generateRecommendations(processDescription, relevantKnowledge);
    
    // Step 3: Return analysis results
    res.json({
      success: true,
      analysis: {
        processName: processDescription,
        currentEfficiency: recommendations.currentEfficiency,
        recommendedActions: recommendations.recommendations,
        potentialSavings: recommendations.potentialSavings,
        knowledgeSourcesUsed: relevantKnowledge.length
      }
    });
    
  } catch (error) {
    console.error('Error analyzing workflow:', error);
    res.status(500).json({ 
      error: 'Failed to analyze workflow',
      message: error.message 
    });
  }
});

// Get knowledge base categories
app.get('/api/knowledge-categories', (req, res) => {
  const categories = workflowKnowledgeBase.map(kb => ({
    id: kb.id,
    category: kb.category,
    practiceCount: kb.bestPractices.length
  }));
  
  res.json({ categories });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 SmartFlow AI Server running on port ${PORT}`);
  console.log(`📊 Knowledge base loaded with ${workflowKnowledgeBase.length} categories`);
  console.log(`🔗 API endpoints available at http://localhost:${PORT}/api/`);
});