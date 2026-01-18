import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import List, Dict, Tuple
import re

class SalesforceNavigationAssistant:
    def __init__(self):
        self.common_tasks = {
            "Lead Management": {
                "Create New Lead": ["Navigate to Leads tab", "Click 'New' button", "Fill required fields: Company, Last Name, Status", "Save record"],
                "Convert Lead": ["Open Lead record", "Click 'Convert' button", "Select Account/Contact options", "Create Opportunity if needed", "Click 'Convert'"],
                "Import Leads": ["Go to Setup", "Search 'Data Import Wizard'", "Select 'Leads'", "Upload CSV file", "Map fields", "Start import"]
            },
            "Opportunity Management": {
                "Create Opportunity": ["Go to Opportunities tab", "Click 'New'", "Fill: Opportunity Name, Account, Close Date, Stage", "Save"],
                "Update Stage": ["Open Opportunity", "Change Stage field", "Update Probability if needed", "Add notes in Chatter", "Save"],
                "Create Quote": ["Open Opportunity", "Go to Quotes related list", "Click 'New Quote'", "Add products", "Save and send"]
            },
            "Account Management": {
                "Create Account": ["Navigate to Accounts", "Click 'New'", "Enter Account Name and Type", "Add billing address", "Save"],
                "Merge Duplicate Accounts": ["Go to Accounts", "Search for duplicates", "Select accounts to merge", "Choose master record", "Merge"],
                "View Account Hierarchy": ["Open Account record", "Click 'View Hierarchy' button", "Explore parent/child relationships"]
            },
            "Contact Management": {
                "Add Contact": ["Go to Contacts tab", "Click 'New'", "Link to Account", "Fill contact details", "Save"],
                "Mass Update Contacts": ["Create Contact list view", "Select contacts", "Choose 'Mass Update'", "Select fields to update", "Apply changes"]
            },
            "Reports & Dashboards": {
                "Create Report": ["Go to Reports tab", "Click 'New Report'", "Choose report type", "Add fields and filters", "Run and save"],
                "Build Dashboard": ["Go to Dashboards", "Click 'New Dashboard'", "Add components", "Configure data sources", "Save and share"],
                "Schedule Report": ["Open report", "Click 'Subscribe'", "Set frequency and recipients", "Save subscription"]
            },
            "Data Management": {
                "Export Data": ["Go to Setup", "Search 'Data Export'", "Select objects", "Choose export format", "Request export"],
                "Clean Duplicate Data": ["Go to Setup", "Search 'Duplicate Management'", "Create matching rules", "Run duplicate jobs", "Review and merge"]
            }
        }
        
        self.navigation_shortcuts = {
            "Quick Access": {
                "Global Search": "Click search bar (top) or press Ctrl+K",
                "App Launcher": "Click 9-dot grid icon (top-left)",
                "Setup": "Click gear icon → Setup",
                "Recent Items": "Click dropdown arrow next to object tabs"
            },
            "Keyboard Shortcuts": {
                "New Record": "Alt + N",
                "Save": "Ctrl + S",
                "Global Search": "Ctrl + K",
                "Help": "F1 or Ctrl + ?"
            }
        }
        
        self.common_pain_points = {
            "Complex Navigation": {
                "issue": "Users get lost in Salesforce's multi-level navigation",
                "solution": "Use breadcrumbs, bookmark frequently used pages, customize navigation bar",
                "ai_tip": "Create a personal navigation map of your most-used features"
            },
            "Too Many Clicks": {
                "issue": "Simple tasks require multiple clicks and page loads",
                "solution": "Use Quick Actions, Lightning Components, and keyboard shortcuts",
                "ai_tip": "Set up custom Quick Actions for your most common workflows"
            },
            "Information Overload": {
                "issue": "Pages show too much information, causing confusion",
                "solution": "Customize page layouts, use record types, create focused list views",
                "ai_tip": "Hide irrelevant fields and sections to reduce cognitive load"
            },
            "Poor Search Experience": {
                "issue": "Difficulty finding records and information quickly",
                "solution": "Use Global Search, create custom search layouts, use filters",
                "ai_tip": "Learn search operators like 'Account:' or 'Opportunity:' for better results"
            },
            "Inconsistent UI": {
                "issue": "Different parts of Salesforce look and behave differently",
                "solution": "Migrate to Lightning Experience, standardize customizations",
                "ai_tip": "Focus on Lightning Experience for consistent modern interface"
            }
        }
    
    def get_task_guidance(self, category: str, task: str) -> List[str]:
        """Get step-by-step guidance for a specific task"""
        if category in self.common_tasks and task in self.common_tasks[category]:
            return self.common_tasks[category][task]
        return ["Task not found. Please select from available options."]
    
    def search_tasks(self, query: str) -> Dict[str, List[str]]:
        """Search for tasks based on user query"""
        results = {}
        query_lower = query.lower()
        
        for category, tasks in self.common_tasks.items():
            matching_tasks = []
            for task_name, steps in tasks.items():
                if (query_lower in task_name.lower() or 
                    query_lower in category.lower() or
                    any(query_lower in step.lower() for step in steps)):
                    matching_tasks.append(task_name)
            
            if matching_tasks:
                results[category] = matching_tasks
        
        return results
    
    def get_ai_recommendations(self, user_role: str, experience_level: str) -> List[str]:
        """Generate AI-powered recommendations based on user profile"""
        recommendations = []
        
        if experience_level == "Beginner":
            recommendations.extend([
                "🎯 Start with the Trailhead learning platform for hands-on tutorials",
                "📚 Focus on mastering one object (Leads/Accounts) before moving to others",
                "🔍 Use Global Search (Ctrl+K) instead of navigating through tabs",
                "📌 Bookmark your most-used pages for quick access"
            ])
        
        if user_role == "Sales Rep":
            recommendations.extend([
                "⚡ Set up Quick Actions for common tasks like logging calls",
                "📊 Create personal dashboard with your key metrics",
                "🔔 Configure opportunity alerts for important deals",
                "📱 Use Salesforce Mobile app for on-the-go updates"
            ])
        elif user_role == "Sales Manager":
            recommendations.extend([
                "📈 Build team performance dashboards and reports",
                "🎯 Set up territory management for your team",
                "📋 Create approval processes for discounts and deals",
                "👥 Use Chatter for team collaboration and updates"
            ])
        elif user_role == "Marketing":
            recommendations.extend([
                "🎯 Master Campaign management and ROI tracking",
                "📊 Set up lead scoring and qualification rules",
                "📧 Integrate with marketing automation tools",
                "📈 Create lead source and conversion reports"
            ])
        
        if experience_level == "Advanced":
            recommendations.extend([
                "⚙️ Explore Process Builder for workflow automation",
                "🔧 Learn basic Apex and Lightning Component development",
                "🔗 Set up advanced integrations with third-party tools",
                "📊 Create custom objects and fields for unique business needs"
            ])
        
        return recommendations
    
    def analyze_user_efficiency(self, tasks_completed: int, time_spent: int, errors_made: int) -> Dict:
        """Analyze user efficiency and provide improvement suggestions"""
        if time_spent == 0:
            return {"efficiency_score": 0, "suggestions": ["Please provide valid time data"]}
        
        tasks_per_hour = (tasks_completed / time_spent) * 60
        error_rate = (errors_made / max(tasks_completed, 1)) * 100
        
        # Calculate efficiency score (0-100)
        efficiency_score = min(100, max(0, (tasks_per_hour * 10) - (error_rate * 2)))
        
        suggestions = []
        
        if efficiency_score < 30:
            suggestions.extend([
                "🚨 Consider additional Salesforce training",
                "📚 Focus on learning keyboard shortcuts",
                "🎯 Simplify your page layouts to reduce complexity"
            ])
        elif efficiency_score < 60:
            suggestions.extend([
                "⚡ Set up Quick Actions for repetitive tasks",
                "🔍 Learn advanced search techniques",
                "📊 Use list views to organize your data better"
            ])
        else:
            suggestions.extend([
                "🌟 Great job! Consider mentoring other users",
                "🔧 Explore automation tools like Process Builder",
                "📈 Focus on advanced reporting and analytics"
            ])
        
        if error_rate > 20:
            suggestions.append("⚠️ High error rate detected - consider slowing down and double-checking entries")
        
        return {
            "efficiency_score": round(efficiency_score, 1),
            "tasks_per_hour": round(tasks_per_hour, 1),
            "error_rate": round(error_rate, 1),
            "suggestions": suggestions
        }

def main():
    st.set_page_config(
        page_title="AI Salesforce Navigation Assistant",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧭 AI-Powered Salesforce Navigation Assistant")
    st.markdown("""
    **Simplify Salesforce complexity with intelligent navigation guidance and workflow optimization.**
    
    This tool addresses the common complaint that Salesforce is "too complex" and "difficult to navigate" 
    by providing AI-powered assistance for common tasks and workflows.
    """)
    
    assistant = SalesforceNavigationAssistant()
    
    # Sidebar for user profile
    st.sidebar.header("👤 User Profile")
    user_role = st.sidebar.selectbox(
        "Your Role",
        ["Sales Rep", "Sales Manager", "Marketing", "Admin", "Support", "Other"]
    )
    
    experience_level = st.sidebar.selectbox(
        "Experience Level",
        ["Beginner", "Intermediate", "Advanced"]
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Task Guidance", 
        "🔍 Smart Search", 
        "💡 AI Recommendations", 
        "📊 Efficiency Analysis",
        "🆘 Pain Point Solutions"
    ])
    
    with tab1:
        st.header("🎯 Step-by-Step Task Guidance")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Task Category")
            selected_category = st.selectbox(
                "Category",
                list(assistant.common_tasks.keys())
            )
            
            if selected_category:
                st.subheader("Select Specific Task")
                selected_task = st.selectbox(
                    "Task",
                    list(assistant.common_tasks[selected_category].keys())
                )
        
        with col2:
            if selected_category and selected_task:
                st.subheader(f"📋 How to: {selected_task}")
                steps = assistant.get_task_guidance(selected_category, selected_task)
                
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i}:** {step}")
                
                # Add helpful tips
                st.info("💡 **Pro Tip:** Use keyboard shortcuts and bookmarks to speed up these processes!")
        
        # Quick navigation shortcuts
        st.subheader("⚡ Quick Navigation Shortcuts")
        
        shortcut_col1, shortcut_col2 = st.columns(2)
        
        with shortcut_col1:
            st.markdown("**🚀 Quick Access**")
            for action, shortcut in assistant.navigation_shortcuts["Quick Access"].items():
                st.markdown(f"• **{action}:** {shortcut}")
        
        with shortcut_col2:
            st.markdown("**⌨️ Keyboard Shortcuts**")
            for action, shortcut in assistant.navigation_shortcuts["Keyboard Shortcuts"].items():
                st.markdown(f"• **{action}:** `{shortcut}`")
    
    with tab2:
        st.header("🔍 Smart Task Search")
        
        search_query = st.text_input(
            "Search for tasks or workflows",
            placeholder="e.g., 'create lead', 'opportunity', 'report'"
        )
        
        if search_query:
            results = assistant.search_tasks(search_query)
            
            if results:
                st.success(f"Found {sum(len(tasks) for tasks in results.values())} matching tasks:")
                
                for category, tasks in results.items():
                    st.subheader(f"📂 {category}")
                    for task in tasks:
                        with st.expander(f"📋 {task}"):
                            steps = assistant.get_task_guidance(category, task)
                            for i, step in enumerate(steps, 1):
                                st.markdown(f"**{i}.** {step}")
            else:
                st.warning("No matching tasks found. Try different keywords or browse the Task Guidance tab.")
    
    with tab3:
        st.header("💡 AI-Powered Recommendations")
        
        recommendations = assistant.get_ai_recommendations(user_role, experience_level)
        
        st.subheader(f"🎯 Personalized Tips for {experience_level} {user_role}")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {recommendation}")
        
        # Learning path suggestions
        st.subheader("📚 Suggested Learning Path")
        
        if experience_level == "Beginner":
            learning_path = [
                "Complete Salesforce Basics Trailhead module",
                "Master navigation and basic record management",
                "Learn to create and manage leads/contacts",
                "Understand reports and dashboards basics"
            ]
        elif experience_level == "Intermediate":
            learning_path = [
                "Advanced reporting and analytics",
                "Workflow automation with Process Builder",
                "Custom fields and page layouts",
                "Integration basics"
            ]
        else:
            learning_path = [
                "Apex programming fundamentals",
                "Lightning Component development",
                "Advanced integration patterns",
                "Salesforce architecture and best practices"
            ]
        
        for i, step in enumerate(learning_path, 1):
            st.markdown(f"**Phase {i}:** {step}")
    
    with tab4:
        st.header("📊 Efficiency Analysis")
        
        st.subheader("📈 Track Your Salesforce Productivity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tasks_completed = st.number_input(
                "Tasks Completed (last hour)",
                min_value=0,
                max_value=100,
                value=5
            )
        
        with col2:
            time_spent = st.number_input(
                "Time Spent (minutes)",
                min_value=1,
                max_value=120,
                value=60
            )
        
        with col3:
            errors_made = st.number_input(
                "Errors/Corrections Made",
                min_value=0,
                max_value=50,
                value=1
            )
        
        if st.button("🔍 Analyze Efficiency"):
            analysis = assistant.analyze_user_efficiency(tasks_completed, time_spent, errors_made)
            
            # Display efficiency score with gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = analysis["efficiency_score"],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Efficiency Score"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Tasks/Hour",
                    f"{analysis['tasks_per_hour']:.1f}",
                    help="Number of tasks completed per hour"
                )
            
            with metric_col2:
                st.metric(
                    "Error Rate",
                    f"{analysis['error_rate']:.1f}%",
                    help="Percentage of tasks that required corrections"
                )
            
            with metric_col3:
                st.metric(
                    "Efficiency Score",
                    f"{analysis['efficiency_score']:.1f}/100",
                    help="Overall efficiency rating"
                )
            
            # Display suggestions
            st.subheader("🎯 Improvement Suggestions")
            for suggestion in analysis["suggestions"]:
                st.markdown(f"• {suggestion}")
    
    with tab5:
        st.header("🆘 Common Pain Point Solutions")
        
        st.markdown("""
        Based on user feedback and research, here are solutions to the most common Salesforce frustrations:
        """)
        
        for pain_point, details in assistant.common_pain_points.items():
            with st.expander(f"❗ {pain_point}"):
                st.markdown(f"**Problem:** {details['issue']}")
                st.markdown(f"**Solution:** {details['solution']}")
                st.info(f"🤖 **AI Tip:** {details['ai_tip']}")
        
        # Additional resources
        st.subheader("📚 Additional Resources")
        
        resources = {
            "🎓 Trailhead": "https://trailhead.salesforce.com - Free learning platform",
            "📖 Help Documentation": "https://help.salesforce.com - Official documentation",
            "👥 Community": "https://trailblazercommunity.salesforce.com - User community",
            "🎥 YouTube Channel": "Salesforce official channel for video tutorials",
            "📱 Mobile App": "Salesforce mobile app for on-the-go access"
        }
        
        for resource, description in resources.items():
            st.markdown(f"• **{resource}:** {description}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🎯 About This Tool:**
    This AI-powered assistant addresses Salesforce's complexity by providing intelligent navigation guidance, 
    workflow optimization, and personalized recommendations. Based on extensive user research showing that 
    52% of sales professionals report their CRM costs them opportunities due to complexity.
    
    **📊 Data Sources:** User experience research, Salesforce best practices, and AI-driven workflow analysis.
    """)

if __name__ == "__main__":
    main()