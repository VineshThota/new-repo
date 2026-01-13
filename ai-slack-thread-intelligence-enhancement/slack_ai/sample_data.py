"""Sample Data Generator Module

Generates realistic Slack conversation data for demonstration purposes.
Includes various channels, users, message types, and conversation patterns.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

def generate_sample_data(num_messages: int = 150) -> List[Dict[str, Any]]:
    """Generate sample Slack conversation data.
    
    Args:
        num_messages: Number of messages to generate
        
    Returns:
        List of message dictionaries
    """
    
    # Sample users with realistic names and roles
    users = [
        {'name': 'sarah.chen', 'role': 'Product Manager'},
        {'name': 'mike.johnson', 'role': 'Senior Developer'},
        {'name': 'emily.davis', 'role': 'UX Designer'},
        {'name': 'alex.rodriguez', 'role': 'DevOps Engineer'},
        {'name': 'jessica.kim', 'role': 'Marketing Manager'},
        {'name': 'david.wilson', 'role': 'Sales Director'},
        {'name': 'lisa.brown', 'role': 'Customer Success'},
        {'name': 'tom.anderson', 'role': 'QA Engineer'},
        {'name': 'maria.garcia', 'role': 'Data Analyst'},
        {'name': 'james.taylor', 'role': 'CTO'}
    ]
    
    # Sample channels with different purposes
    channels = [
        'general',
        'product-development',
        'engineering',
        'marketing',
        'sales',
        'customer-support',
        'random',
        'urgent-issues',
        'project-alpha',
        'pricing-discussion'
    ]
    
    # Message templates by category
    message_templates = {
        'urgent': [
            "URGENT: Production server is down! Need immediate attention.",
            "Critical bug found in the payment system - customers can't checkout!",
            "Emergency: Database connection failing, site is inaccessible.",
            "ASAP: Client meeting moved to 2 PM today, need the demo ready!",
            "Priority issue: API rate limits exceeded, services are failing.",
            "Immediate action needed: Security breach detected in user accounts."
        ],
        'decisions': [
            "After reviewing all options, we've decided to go with the new pricing model.",
            "Team agreed to use React for the frontend framework.",
            "Final decision: We'll launch the beta version next month.",
            "Concluded that we need to hire 2 more developers for this project.",
            "We've chosen AWS over Google Cloud for our infrastructure.",
            "Approved the new design mockups, moving forward with implementation."
        ],
        'questions': [
            "Can someone help me understand the new authentication flow?",
            "What's the status on the mobile app development?",
            "Who's responsible for the database migration this weekend?",
            "When is the deadline for the Q4 feature release?",
            "How do we handle user data privacy in the new system?",
            "Which payment gateway should we integrate first?"
        ],
        'updates': [
            "Just finished the user research interviews, compiling results now.",
            "Code review completed, ready to merge the feature branch.",
            "Marketing campaign is live, seeing good initial engagement.",
            "Customer support tickets down 15% this week, great improvement!",
            "Database optimization complete, queries are 40% faster now.",
            "New hire onboarding went smoothly, they start Monday."
        ],
        'meetings': [
            "Reminder: All-hands meeting tomorrow at 10 AM in the main conference room.",
            "Can we schedule a quick sync about the API integration?",
            "Sprint planning meeting moved to Thursday 2 PM.",
            "Client demo scheduled for Friday, need everyone to review the presentation.",
            "Weekly standup in 15 minutes, join the Zoom link in the calendar.",
            "Quarterly review meeting next week, please prepare your reports."
        ],
        'casual': [
            "Great job on the presentation yesterday, really well done!",
            "Thanks for helping me debug that issue, saved me hours!",
            "Anyone want to grab lunch? Thinking of trying that new place.",
            "Happy Friday everyone! Any fun weekend plans?",
            "Congrats on the successful product launch! 🎉",
            "Coffee machine is broken again... someone please fix it! ☕"
        ],
        'technical': [
            "The new API endpoint is returning 500 errors intermittently.",
            "Memory usage spiked to 90% on server-03, investigating the cause.",
            "Docker container deployment failed, checking the configuration.",
            "SSL certificate expires next week, need to renew it.",
            "Load balancer showing high latency, might need to scale up.",
            "Git merge conflict in the main branch, can someone help resolve?"
        ],
        'pricing': [
            "Competitor analysis shows we're 20% higher than market average.",
            "Customer feedback suggests our premium tier is too expensive.",
            "Proposal: Introduce a freemium model to attract more users.",
            "Revenue projections look good with the new pricing structure.",
            "Should we offer annual discounts to increase customer retention?",
            "Pricing page conversion rate dropped after the recent changes."
        ]
    }
    
    messages = []
    base_time = datetime.now() - timedelta(days=30)
    
    # Generate thread conversations
    thread_conversations = [
        {
            'channel': 'product-development',
            'starter': 'sarah.chen',
            'topic': 'pricing',
            'messages': [
                "We need to discuss the new pricing strategy for Q1.",
                "I've been analyzing competitor pricing, we might be too high.",
                "What's the customer feedback on our current pricing?",
                "Most complaints are about the premium tier being expensive.",
                "Should we consider a freemium model?",
                "That could work, but we need to be careful about feature limits.",
                "Let's schedule a meeting to discuss this in detail."
            ]
        },
        {
            'channel': 'engineering',
            'starter': 'mike.johnson',
            'topic': 'technical',
            'messages': [
                "Production deployment failed last night, investigating.",
                "Found the issue - database connection timeout.",
                "Is this related to the recent schema changes?",
                "Possibly, the new indexes might be causing locks.",
                "I'll rollback the changes and test on staging first.",
                "Good call, let me know if you need help with testing."
            ]
        },
        {
            'channel': 'urgent-issues',
            'starter': 'alex.rodriguez',
            'topic': 'urgent',
            'messages': [
                "URGENT: Payment processing is down!",
                "How many customers are affected?",
                "About 50 transactions failed in the last hour.",
                "I'm checking the payment gateway logs now.",
                "Found it - API key expired, updating now.",
                "Fixed! Payments are processing normally again.",
                "Great work! Let's add monitoring for API key expiration."
            ]
        }
    ]
    
    message_id = 1
    
    # Generate threaded conversations
    for thread in thread_conversations:
        thread_ts = (base_time + timedelta(minutes=random.randint(0, 43200))).isoformat()
        participants = [thread['starter']]
        
        for i, msg_text in enumerate(thread['messages']):
            if i == 0:
                user = thread['starter']
                ts = thread_ts
            else:
                # Add variety in participants
                if random.random() < 0.7:  # 70% chance of new participant
                    available_users = [u['name'] for u in users if u['name'] not in participants[-2:]]
                    if available_users:
                        user = random.choice(available_users)
                        participants.append(user)
                    else:
                        user = random.choice([u['name'] for u in users])
                else:
                    user = random.choice(participants)
                
                ts = (datetime.fromisoformat(thread_ts) + timedelta(minutes=i*5 + random.randint(1, 10))).isoformat()
            
            message = {
                'ts': ts,
                'thread_ts': thread_ts if i > 0 else ts,
                'user': user,
                'text': msg_text,
                'channel': thread['channel'],
                'timestamp': ts,
                'message_id': message_id
            }
            
            messages.append(message)
            message_id += 1
    
    # Generate individual messages
    remaining_messages = num_messages - len(messages)
    
    for i in range(remaining_messages):
        # Select random category and template
        category = random.choice(list(message_templates.keys()))
        template = random.choice(message_templates[category])
        
        # Add some variation to templates
        if random.random() < 0.3:  # 30% chance to modify template
            variations = {
                'urgent': [' Please help!', ' This is blocking deployment.', ' Customer impact is high.'],
                'questions': [' Thanks in advance!', ' Any ideas?', ' Urgent response needed.'],
                'updates': [' Will keep you posted.', ' Let me know if questions.', ' More details to follow.'],
                'casual': [' 😊', ' 👍', ' Hope everyone is doing well!']
            }
            
            if category in variations:
                template += random.choice(variations[category])
        
        # Select appropriate channel based on message category
        channel_mapping = {
            'urgent': ['urgent-issues', 'engineering', 'general'],
            'technical': ['engineering', 'product-development'],
            'pricing': ['pricing-discussion', 'product-development', 'sales'],
            'meetings': ['general', 'product-development', 'engineering'],
            'casual': ['random', 'general'],
            'decisions': ['product-development', 'engineering', 'general'],
            'questions': ['engineering', 'product-development', 'general'],
            'updates': ['general', 'product-development', 'marketing']
        }
        
        channel = random.choice(channel_mapping.get(category, channels))
        user = random.choice(users)['name']
        
        # Generate timestamp
        timestamp = (base_time + timedelta(minutes=random.randint(0, 43200))).isoformat()
        
        message = {
            'ts': timestamp,
            'user': user,
            'text': template,
            'channel': channel,
            'timestamp': timestamp,
            'message_id': message_id
        }
        
        messages.append(message)
        message_id += 1
    
    # Sort messages by timestamp
    messages.sort(key=lambda x: x['timestamp'])
    
    return messages

def generate_user_profiles() -> List[Dict[str, Any]]:
    """Generate sample user profiles."""
    return [
        {
            'id': 'U001',
            'name': 'sarah.chen',
            'real_name': 'Sarah Chen',
            'title': 'Product Manager',
            'email': 'sarah.chen@company.com',
            'timezone': 'America/Los_Angeles',
            'is_admin': True
        },
        {
            'id': 'U002',
            'name': 'mike.johnson',
            'real_name': 'Mike Johnson',
            'title': 'Senior Developer',
            'email': 'mike.johnson@company.com',
            'timezone': 'America/New_York',
            'is_admin': False
        },
        {
            'id': 'U003',
            'name': 'emily.davis',
            'real_name': 'Emily Davis',
            'title': 'UX Designer',
            'email': 'emily.davis@company.com',
            'timezone': 'America/Chicago',
            'is_admin': False
        },
        # Add more users as needed
    ]

def generate_channel_info() -> List[Dict[str, Any]]:
    """Generate sample channel information."""
    return [
        {
            'id': 'C001',
            'name': 'general',
            'purpose': 'Company-wide announcements and general discussion',
            'topic': 'Welcome to our workspace!',
            'member_count': 45,
            'is_private': False
        },
        {
            'id': 'C002',
            'name': 'product-development',
            'purpose': 'Product strategy and development discussions',
            'topic': 'Building amazing products together',
            'member_count': 12,
            'is_private': False
        },
        {
            'id': 'C003',
            'name': 'urgent-issues',
            'purpose': 'Critical issues requiring immediate attention',
            'topic': 'Emergency response channel',
            'member_count': 8,
            'is_private': False
        },
        # Add more channels as needed
    ]

def generate_sample_search_queries() -> List[str]:
    """Generate sample search queries for testing."""
    return [
        "What did we decide about pricing?",
        "Any production issues this week?",
        "When is the next team meeting?",
        "Who is working on the API integration?",
        "Status update on the mobile app",
        "Customer feedback about the new feature",
        "Database performance problems",
        "Marketing campaign results",
        "Budget approval for Q1",
        "Security vulnerability reports"
    ]