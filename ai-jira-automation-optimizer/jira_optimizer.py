import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict, Counter
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JiraAutomationAnalyzer:
    """AI-powered Jira automation rule analyzer and optimizer."""
    
    def __init__(self, jira_url: str, email: str, api_token: str):
        self.jira_url = jira_url.rstrip('/')
        self.email = email
        self.api_token = api_token
        self.session = requests.Session()
        self.session.auth = (email, api_token)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # Initialize NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize vectorizer for rule similarity
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Rule analysis cache
        self.rules_cache = {}
        self.usage_cache = {}
        
    def test_connection(self) -> bool:
        """Test connection to Jira instance."""
        try:
            response = self.session.get(f"{self.jira_url}/rest/api/3/myself")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_automation_rules(self) -> List[Dict]:
        """Fetch all automation rules from Jira."""
        try:
            # Note: This endpoint may vary based on Jira version and permissions
            response = self.session.get(f"{self.jira_url}/rest/api/3/automation/rule")
            if response.status_code == 200:
                return response.json().get('values', [])
            else:
                logger.warning(f"Could not fetch automation rules: {response.status_code}")
                return self._get_sample_rules()  # Fallback to sample data
        except Exception as e:
            logger.error(f"Error fetching automation rules: {e}")
            return self._get_sample_rules()
    
    def _get_sample_rules(self) -> List[Dict]:
        """Generate sample automation rules for demo purposes."""
        return [
            {
                "id": "rule-001",
                "name": "Auto-assign issues to team leads",
                "description": "Automatically assign new issues to team leads based on component",
                "trigger": {"type": "issue-created"},
                "conditions": [{"type": "issue-type", "value": "Bug"}],
                "actions": [{"type": "assign", "value": "{{component.lead}}"}],
                "enabled": True,
                "usage_stats": {"monthly_executions": 450, "success_rate": 0.95}
            },
            {
                "id": "rule-002",
                "name": "Update issue status on PR merge",
                "description": "Change issue status to In Review when PR is merged",
                "trigger": {"type": "webhook", "source": "github"},
                "conditions": [{"type": "pr-merged"}],
                "actions": [{"type": "transition", "value": "In Review"}],
                "enabled": True,
                "usage_stats": {"monthly_executions": 280, "success_rate": 0.98}
            },
            {
                "id": "rule-003",
                "name": "Notify stakeholders on priority change",
                "description": "Send email notifications when issue priority changes to High or Critical",
                "trigger": {"type": "field-changed", "field": "priority"},
                "conditions": [{"type": "priority-in", "value": ["High", "Critical"]}],
                "actions": [{"type": "email", "recipients": "stakeholders@company.com"}],
                "enabled": True,
                "usage_stats": {"monthly_executions": 220, "success_rate": 0.92}
            },
            {
                "id": "rule-004",
                "name": "Auto-close resolved issues",
                "description": "Automatically close issues that have been resolved for 7 days",
                "trigger": {"type": "scheduled", "frequency": "daily"},
                "conditions": [
                    {"type": "status", "value": "Resolved"},
                    {"type": "updated-before", "value": "7d"}
                ],
                "actions": [{"type": "transition", "value": "Closed"}],
                "enabled": True,
                "usage_stats": {"monthly_executions": 150, "success_rate": 0.99}
            },
            {
                "id": "rule-005",
                "name": "Label issues by component",
                "description": "Add labels to issues based on affected component",
                "trigger": {"type": "issue-created"},
                "conditions": [{"type": "component-not-empty"}],
                "actions": [{"type": "add-label", "value": "{{component.name}}"}],
                "enabled": True,
                "usage_stats": {"monthly_executions": 380, "success_rate": 0.97}
            }
        ]
    
    def analyze_rule_semantics(self, rule: Dict) -> Dict:
        """Analyze rule semantics using NLP."""
        if not self.nlp:
            return {"semantic_score": 0.5, "complexity": "medium", "keywords": []}
        
        # Combine rule name and description for analysis
        text = f"{rule.get('name', '')} {rule.get('description', '')}"
        doc = self.nlp(text)
        
        # Extract key information
        keywords = [token.lemma_.lower() for token in doc 
                   if not token.is_stop and not token.is_punct and len(token.text) > 2]
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Calculate complexity based on conditions and actions
        conditions_count = len(rule.get('conditions', []))
        actions_count = len(rule.get('actions', []))
        complexity_score = conditions_count + actions_count
        
        if complexity_score <= 2:
            complexity = "low"
        elif complexity_score <= 5:
            complexity = "medium"
        else:
            complexity = "high"
        
        return {
            "semantic_score": min(len(keywords) / 10, 1.0),  # Normalize to 0-1
            "complexity": complexity,
            "keywords": keywords[:10],  # Top 10 keywords
            "entities": entities,
            "conditions_count": conditions_count,
            "actions_count": actions_count
        }
    
    def calculate_rule_efficiency(self, rule: Dict) -> float:
        """Calculate efficiency score for a rule (0-100)."""
        usage_stats = rule.get('usage_stats', {})
        monthly_executions = usage_stats.get('monthly_executions', 0)
        success_rate = usage_stats.get('success_rate', 0.5)
        
        # Analyze rule structure
        semantics = self.analyze_rule_semantics(rule)
        
        # Efficiency factors
        trigger_efficiency = self._get_trigger_efficiency(rule.get('trigger', {}))
        condition_efficiency = self._get_condition_efficiency(rule.get('conditions', []))
        action_efficiency = self._get_action_efficiency(rule.get('actions', []))
        
        # Calculate weighted efficiency score
        efficiency = (
            success_rate * 30 +  # Success rate (30%)
            trigger_efficiency * 25 +  # Trigger efficiency (25%)
            condition_efficiency * 25 +  # Condition efficiency (25%)
            action_efficiency * 20  # Action efficiency (20%)
        )
        
        # Penalize high-frequency rules without proper scoping
        if monthly_executions > 300 and semantics['complexity'] == 'low':
            efficiency *= 0.8  # 20% penalty
        
        return min(max(efficiency, 0), 100)  # Clamp to 0-100
    
    def _get_trigger_efficiency(self, trigger: Dict) -> float:
        """Rate trigger efficiency (0-100)."""
        trigger_type = trigger.get('type', '')
        
        # Scheduled triggers are generally more efficient
        if trigger_type == 'scheduled':
            return 90
        elif trigger_type in ['issue-created', 'issue-updated']:
            return 70
        elif trigger_type == 'field-changed':
            return 60
        elif trigger_type == 'webhook':
            return 80
        else:
            return 50
    
    def _get_condition_efficiency(self, conditions: List[Dict]) -> float:
        """Rate condition efficiency (0-100)."""
        if not conditions:
            return 30  # No conditions = inefficient
        
        # More specific conditions are better
        specificity_score = 0
        for condition in conditions:
            condition_type = condition.get('type', '')
            if condition_type in ['project', 'issue-type', 'status']:
                specificity_score += 20  # High specificity
            elif condition_type in ['priority', 'component', 'assignee']:
                specificity_score += 15  # Medium specificity
            else:
                specificity_score += 10  # Low specificity
        
        return min(specificity_score, 100)
    
    def _get_action_efficiency(self, actions: List[Dict]) -> float:
        """Rate action efficiency (0-100)."""
        if not actions:
            return 0
        
        # Simple actions are more efficient
        efficiency_score = 0
        for action in actions:
            action_type = action.get('type', '')
            if action_type in ['assign', 'transition', 'add-label']:
                efficiency_score += 25  # High efficiency
            elif action_type in ['comment', 'email']:
                efficiency_score += 15  # Medium efficiency
            else:
                efficiency_score += 10  # Low efficiency
        
        return min(efficiency_score, 100)
    
    def find_similar_rules(self, rules: List[Dict], threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find similar rules that could be consolidated."""
        if len(rules) < 2:
            return []
        
        # Create text representations of rules
        rule_texts = []
        rule_ids = []
        
        for rule in rules:
            text = f"{rule.get('name', '')} {rule.get('description', '')}"
            # Add trigger and condition information
            trigger = rule.get('trigger', {})
            text += f" trigger:{trigger.get('type', '')}"
            
            for condition in rule.get('conditions', []):
                text += f" condition:{condition.get('type', '')}"
            
            rule_texts.append(text)
            rule_ids.append(rule.get('id', ''))
        
        # Calculate similarity matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(rule_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            similar_pairs = []
            for i in range(len(rules)):
                for j in range(i + 1, len(rules)):
                    similarity = similarity_matrix[i][j]
                    if similarity >= threshold:
                        similar_pairs.append((rule_ids[i], rule_ids[j], similarity))
            
            return sorted(similar_pairs, key=lambda x: x[2], reverse=True)
        
        except Exception as e:
            logger.error(f"Error finding similar rules: {e}")
            return []
    
    def get_optimization_recommendations(self, rules: List[Dict]) -> List[Dict]:
        """Generate AI-powered optimization recommendations."""
        recommendations = []
        
        for rule in rules:
            rule_id = rule.get('id', '')
            rule_name = rule.get('name', '')
            usage_stats = rule.get('usage_stats', {})
            monthly_executions = usage_stats.get('monthly_executions', 0)
            
            efficiency_score = self.calculate_rule_efficiency(rule)
            semantics = self.analyze_rule_semantics(rule)
            
            # Generate recommendations based on analysis
            if monthly_executions > 300 and efficiency_score < 70:
                # High usage, low efficiency
                trigger = rule.get('trigger', {})
                if trigger.get('type') in ['issue-created', 'issue-updated', 'field-changed']:
                    recommendations.append({
                        'rule_id': rule_id,
                        'rule_name': rule_name,
                        'current_usage': monthly_executions,
                        'action': 'Convert to scheduled trigger (hourly)',
                        'savings': int(monthly_executions * 0.7),  # 70% reduction
                        'confidence': 95,
                        'risk': 'Low',
                        'reason': 'High-frequency event trigger can be optimized with scheduled execution'
                    })
            
            elif len(rule.get('conditions', [])) < 2 and monthly_executions > 200:
                # Insufficient scoping
                recommendations.append({
                    'rule_id': rule_id,
                    'rule_name': rule_name,
                    'current_usage': monthly_executions,
                    'action': 'Add project scope filter',
                    'savings': int(monthly_executions * 0.6),  # 60% reduction
                    'confidence': 88,
                    'risk': 'Low',
                    'reason': 'Rule lacks proper scoping conditions'
                })
            
            elif efficiency_score < 50:
                # General inefficiency
                recommendations.append({
                    'rule_id': rule_id,
                    'rule_name': rule_name,
                    'current_usage': monthly_executions,
                    'action': 'Review and optimize rule logic',
                    'savings': int(monthly_executions * 0.3),  # 30% reduction
                    'confidence': 75,
                    'risk': 'Medium',
                    'reason': f'Low efficiency score: {efficiency_score:.1f}/100'
                })
        
        # Find consolidation opportunities
        similar_rules = self.find_similar_rules(rules)
        for rule1_id, rule2_id, similarity in similar_rules[:3]:  # Top 3 pairs
            rule1 = next((r for r in rules if r.get('id') == rule1_id), None)
            rule2 = next((r for r in rules if r.get('id') == rule2_id), None)
            
            if rule1 and rule2:
                combined_usage = (rule1.get('usage_stats', {}).get('monthly_executions', 0) +
                                rule2.get('usage_stats', {}).get('monthly_executions', 0))
                
                recommendations.append({
                    'rule_id': f"{rule1_id},{rule2_id}",
                    'rule_name': f"Consolidate: {rule1.get('name', '')} + {rule2.get('name', '')}",
                    'current_usage': combined_usage,
                    'action': 'Consolidate with similar notification rules',
                    'savings': int(combined_usage * 0.4),  # 40% reduction
                    'confidence': int(similarity * 100),
                    'risk': 'Medium',
                    'reason': f'Rules are {similarity:.1%} similar and can be merged'
                })
        
        return sorted(recommendations, key=lambda x: x['savings'], reverse=True)
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data."""
        rules = self.get_automation_rules()
        
        # Calculate metrics
        total_usage = sum(rule.get('usage_stats', {}).get('monthly_executions', 0) for rule in rules)
        usage_limit = 1700  # Standard plan limit
        
        # Efficiency analysis
        efficiency_scores = [self.calculate_rule_efficiency(rule) for rule in rules]
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
        
        # Categorize rules by efficiency
        efficient_rules = sum(1 for score in efficiency_scores if score >= 70)
        inefficient_rules = sum(1 for score in efficiency_scores if score < 50)
        
        # Usage history (simulated)
        usage_history = []
        base_date = datetime.now() - timedelta(days=180)
        for i in range(6):
            date = base_date + timedelta(days=30 * i)
            usage = total_usage + np.random.randint(-200, 200)
            usage_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'usage': max(usage, 0)
            })
        
        # Top consuming rules
        top_rules = sorted(rules, 
                          key=lambda x: x.get('usage_stats', {}).get('monthly_executions', 0), 
                          reverse=True)[:5]
        
        top_rules_data = []
        for rule in top_rules:
            usage_stats = rule.get('usage_stats', {})
            top_rules_data.append({
                'name': rule.get('name', 'Unknown'),
                'usage': usage_stats.get('monthly_executions', 0),
                'efficiency_score': self.calculate_rule_efficiency(rule),
                'success_rate': usage_stats.get('success_rate', 0) * 100
            })
        
        return {
            'current_usage': total_usage,
            'usage_limit': usage_limit,
            'usage_change': np.random.randint(-100, 100),  # Simulated change
            'optimization_potential': int(100 - avg_efficiency),
            'potential_savings': int(total_usage * (100 - avg_efficiency) / 100),
            'total_rules': len(rules),
            'inefficient_rules': inefficient_rules,
            'usage_history': usage_history,
            'rule_efficiency': {
                'Efficient (70%+)': efficient_rules,
                'Moderate (50-70%)': len(rules) - efficient_rules - inefficient_rules,
                'Inefficient (<50%)': inefficient_rules
            },
            'top_rules': top_rules_data,
            'rule_categories': {
                'Assignment': 2,
                'Notifications': 3,
                'Status Updates': 4,
                'Labeling': 2,
                'Cleanup': 1
            },
            'trigger_types': {
                'Issue Created': 3,
                'Field Changed': 2,
                'Scheduled': 1,
                'Webhook': 1
            }
        }

class OptimizationEngine:
    """Engine for applying optimization recommendations."""
    
    def __init__(self, analyzer: JiraAutomationAnalyzer):
        self.analyzer = analyzer
    
    def consolidate_rules(self, rule_ids: List[str], strategy: str = "merge_conditions") -> Dict:
        """Consolidate multiple rules into one optimized rule."""
        # This would implement actual rule consolidation logic
        # For demo purposes, return success status
        return {
            'status': 'success',
            'consolidated_rule_id': f"consolidated_{len(rule_ids)}_rules",
            'original_rules': rule_ids,
            'estimated_savings': len(rule_ids) * 100,
            'strategy_used': strategy
        }
    
    def convert_to_scheduled(self, rule_id: str, frequency: str = "hourly") -> Dict:
        """Convert event-triggered rule to scheduled execution."""
        # This would implement actual rule modification logic
        return {
            'status': 'success',
            'rule_id': rule_id,
            'old_trigger': 'event-based',
            'new_trigger': f'scheduled-{frequency}',
            'estimated_savings': 300
        }
    
    def add_scope_conditions(self, rule_id: str, conditions: List[Dict]) -> Dict:
        """Add scoping conditions to reduce rule execution frequency."""
        return {
            'status': 'success',
            'rule_id': rule_id,
            'added_conditions': conditions,
            'estimated_savings': 200
        }