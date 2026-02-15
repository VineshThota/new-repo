import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import random

def load_sample_data() -> Dict[str, Any]:
    """Load sample data for demo mode."""
    
    # Generate realistic usage history
    usage_history = []
    base_date = datetime.now() - timedelta(days=180)
    base_usage = 1450
    
    for i in range(6):
        date = base_date + timedelta(days=30 * i)
        # Add some realistic variation
        variation = random.randint(-150, 200)
        usage = max(base_usage + variation + (i * 20), 800)  # Trending upward
        usage_history.append({
            'date': date.strftime('%Y-%m-%d'),
            'usage': usage
        })
    
    # Current month usage
    current_usage = usage_history[-1]['usage'] + random.randint(50, 150)
    
    # Top consuming rules with realistic data
    top_rules = [
        {
            'name': 'Auto-assign issues to team leads',
            'usage': 450,
            'efficiency_score': 65,
            'success_rate': 95
        },
        {
            'name': 'Label issues by component',
            'usage': 380,
            'efficiency_score': 78,
            'success_rate': 97
        },
        {
            'name': 'Update issue status on PR merge',
            'usage': 280,
            'efficiency_score': 82,
            'success_rate': 98
        },
        {
            'name': 'Notify stakeholders on priority change',
            'usage': 220,
            'efficiency_score': 58,
            'success_rate': 92
        },
        {
            'name': 'Auto-close resolved issues',
            'usage': 150,
            'efficiency_score': 88,
            'success_rate': 99
        }
    ]
    
    return {
        'current_usage': current_usage,
        'usage_limit': 1700,
        'usage_change': random.randint(-80, 120),
        'optimization_potential': 42,
        'potential_savings': 615,
        'total_rules': 25,
        'inefficient_rules': 8,
        'usage_history': usage_history,
        'rule_efficiency': {
            'Efficient (70%+)': 12,
            'Moderate (50-70%)': 5,
            'Inefficient (<50%)': 8
        },
        'top_rules': top_rules,
        'rule_categories': {
            'Assignment': 6,
            'Notifications': 8,
            'Status Updates': 5,
            'Labeling': 4,
            'Cleanup': 2
        },
        'trigger_types': {
            'Issue Created': 10,
            'Field Changed': 8,
            'Scheduled': 4,
            'Webhook': 3
        }
    }

def format_usage_data(usage_data: List[Dict]) -> pd.DataFrame:
    """Format usage data for visualization."""
    df = pd.DataFrame(usage_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def calculate_savings_projection(current_usage: int, optimizations: List[Dict]) -> Dict:
    """Calculate projected savings from optimizations."""
    total_savings = sum(opt.get('savings', 0) for opt in optimizations)
    projected_usage = max(current_usage - total_savings, 0)
    
    return {
        'current_usage': current_usage,
        'projected_usage': projected_usage,
        'total_savings': total_savings,
        'percentage_reduction': (total_savings / current_usage * 100) if current_usage > 0 else 0,
        'months_under_limit': calculate_months_under_limit(projected_usage)
    }

def calculate_months_under_limit(usage: int, limit: int = 1700) -> int:
    """Calculate how many months the usage will stay under limit."""
    if usage >= limit:
        return 0
    
    # Assume 5% monthly growth
    monthly_growth = 0.05
    months = 0
    current = usage
    
    while current < limit and months < 24:  # Cap at 24 months
        months += 1
        current *= (1 + monthly_growth)
    
    return months

def generate_rule_recommendations(rules_data: List[Dict]) -> List[Dict]:
    """Generate optimization recommendations based on rule analysis."""
    recommendations = []
    
    for rule in rules_data:
        usage = rule.get('usage', 0)
        efficiency = rule.get('efficiency_score', 50)
        
        if usage > 300 and efficiency < 70:
            # High usage, low efficiency
            recommendations.append({
                'rule_name': rule.get('name', 'Unknown Rule'),
                'current_usage': usage,
                'action': 'Convert to scheduled trigger',
                'savings': int(usage * 0.6),
                'confidence': 90,
                'risk': 'Low',
                'priority': 'High'
            })
        elif usage > 200 and efficiency < 60:
            # Medium usage, low efficiency
            recommendations.append({
                'rule_name': rule.get('name', 'Unknown Rule'),
                'current_usage': usage,
                'action': 'Add scoping conditions',
                'savings': int(usage * 0.4),
                'confidence': 85,
                'risk': 'Low',
                'priority': 'Medium'
            })
        elif efficiency < 50:
            # Low efficiency regardless of usage
            recommendations.append({
                'rule_name': rule.get('name', 'Unknown Rule'),
                'current_usage': usage,
                'action': 'Review and optimize logic',
                'savings': int(usage * 0.3),
                'confidence': 75,
                'risk': 'Medium',
                'priority': 'Medium'
            })
    
    return sorted(recommendations, key=lambda x: x['savings'], reverse=True)

def export_rules_data(rules: List[Dict], format_type: str = 'json') -> str:
    """Export rules data in specified format."""
    if format_type.lower() == 'json':
        return json.dumps(rules, indent=2)
    elif format_type.lower() == 'csv':
        df = pd.DataFrame(rules)
        return df.to_csv(index=False)
    elif format_type.lower() == 'yaml':
        try:
            import yaml
            return yaml.dump(rules, default_flow_style=False)
        except ImportError:
            return "YAML export requires PyYAML: pip install PyYAML"
    else:
        return json.dumps(rules, indent=2)

def validate_jira_credentials(url: str, email: str, token: str) -> Dict[str, Any]:
    """Validate Jira connection credentials."""
    errors = []
    
    if not url:
        errors.append("Jira URL is required")
    elif not url.startswith(('http://', 'https://')):
        errors.append("Jira URL must start with http:// or https://")
    elif '.atlassian.net' not in url:
        errors.append("URL should be an Atlassian domain (*.atlassian.net)")
    
    if not email:
        errors.append("Email is required")
    elif '@' not in email:
        errors.append("Invalid email format")
    
    if not token:
        errors.append("API token is required")
    elif len(token) < 10:
        errors.append("API token seems too short")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def generate_optimization_report(data: Dict, recommendations: List[Dict]) -> str:
    """Generate a comprehensive optimization report."""
    report = f"""
# Jira Automation Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Status
- Monthly Usage: {data['current_usage']:,} executions
- Usage Limit: {data['usage_limit']:,} executions
- Utilization: {(data['current_usage']/data['usage_limit']*100):.1f}%
- Active Rules: {data['total_rules']}
- Inefficient Rules: {data['inefficient_rules']}

## Optimization Potential
- Potential Savings: {data['potential_savings']} executions/month
- Optimization Potential: {data['optimization_potential']}%

## Top Recommendations
"""
    
    for i, rec in enumerate(recommendations[:5], 1):
        report += f"""
### {i}. {rec['rule_name']}
- Current Usage: {rec['current_usage']} executions/month
- Recommended Action: {rec['action']}
- Potential Savings: {rec['savings']} executions/month
- Confidence: {rec['confidence']}%
- Risk Level: {rec['risk']}

"""
    
    total_savings = sum(rec['savings'] for rec in recommendations)
    new_usage = data['current_usage'] - total_savings
    
    report += f"""
## Implementation Impact
- Total Potential Savings: {total_savings:,} executions/month
- Projected Usage After Optimization: {new_usage:,} executions/month
- New Utilization Rate: {(new_usage/data['usage_limit']*100):.1f}%
- Months Until Limit (with 5% growth): {calculate_months_under_limit(new_usage)}

## Next Steps
1. Review and approve high-priority recommendations
2. Implement optimizations in test environment
3. Monitor rule performance for 1-2 weeks
4. Apply optimizations to production
5. Set up ongoing monitoring and alerts
"""
    
    return report

def create_backup_data(rules: List[Dict]) -> Dict:
    """Create backup data structure for rules."""
    return {
        'backup_date': datetime.now().isoformat(),
        'total_rules': len(rules),
        'rules': rules,
        'metadata': {
            'version': '1.0',
            'source': 'jira-automation-optimizer',
            'backup_type': 'full'
        }
    }

def parse_rule_conditions(conditions: List[Dict]) -> str:
    """Parse rule conditions into human-readable format."""
    if not conditions:
        return "No conditions"
    
    condition_strings = []
    for condition in conditions:
        condition_type = condition.get('type', 'unknown')
        value = condition.get('value', '')
        
        if condition_type == 'issue-type':
            condition_strings.append(f"Issue type is {value}")
        elif condition_type == 'priority':
            condition_strings.append(f"Priority is {value}")
        elif condition_type == 'status':
            condition_strings.append(f"Status is {value}")
        elif condition_type == 'project':
            condition_strings.append(f"Project is {value}")
        else:
            condition_strings.append(f"{condition_type}: {value}")
    
    return " AND ".join(condition_strings)

def calculate_rule_complexity_score(rule: Dict) -> int:
    """Calculate complexity score for a rule (1-10)."""
    score = 1
    
    # Add points for conditions
    conditions = rule.get('conditions', [])
    score += min(len(conditions), 3)  # Max 3 points for conditions
    
    # Add points for actions
    actions = rule.get('actions', [])
    score += min(len(actions), 3)  # Max 3 points for actions
    
    # Add points for trigger complexity
    trigger = rule.get('trigger', {})
    trigger_type = trigger.get('type', '')
    if trigger_type in ['scheduled', 'webhook']:
        score += 2
    elif trigger_type in ['field-changed']:
        score += 1
    
    return min(score, 10)

def get_efficiency_color(score: float) -> str:
    """Get color code for efficiency score visualization."""
    if score >= 80:
        return "#2E8B57"  # Green
    elif score >= 60:
        return "#FFD700"  # Gold
    elif score >= 40:
        return "#FF8C00"  # Orange
    else:
        return "#DC143C"  # Red

def format_number(num: int) -> str:
    """Format number with appropriate suffix (K, M)."""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)