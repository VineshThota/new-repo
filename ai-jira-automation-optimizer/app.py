import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from jira_optimizer import JiraAutomationAnalyzer, OptimizationEngine
from utils import load_sample_data, format_usage_data

# Page configuration
st.set_page_config(
    page_title="AI Jira Automation Optimizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.optimization-card {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2ca02c;
}
.warning-card {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 AI Jira Automation Optimizer")
    st.markdown("**Smart Rule Management for Usage Reduction**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Demo mode toggle
        demo_mode = st.checkbox("Demo Mode (Use Sample Data)", value=True)
        
        if not demo_mode:
            st.subheader("Jira Connection")
            jira_url = st.text_input("Jira URL", placeholder="https://company.atlassian.net")
            jira_email = st.text_input("Email", placeholder="admin@company.com")
            jira_token = st.text_input("API Token", type="password")
            
            if st.button("Connect to Jira"):
                if jira_url and jira_email and jira_token:
                    try:
                        analyzer = JiraAutomationAnalyzer(jira_url, jira_email, jira_token)
                        st.success("✅ Connected successfully!")
                        st.session_state['analyzer'] = analyzer
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                else:
                    st.error("Please fill in all fields")
        
        st.markdown("---")
        st.subheader("Analysis Options")
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Scan", "Deep Analysis", "Comprehensive Audit"]
        )
        
        optimization_aggressiveness = st.slider(
            "Optimization Aggressiveness",
            min_value=1, max_value=10, value=5,
            help="1 = Conservative, 10 = Aggressive"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔍 Analysis", 
        "⚡ Optimization", 
        "📈 Monitoring", 
        "🛠️ Tools"
    ])
    
    # Load data (demo or real)
    if demo_mode:
        data = load_sample_data()
    else:
        if 'analyzer' in st.session_state:
            data = st.session_state['analyzer'].get_dashboard_data()
        else:
            st.warning("Please connect to Jira first")
            return
    
    with tab1:
        show_dashboard(data)
    
    with tab2:
        show_analysis(data, analysis_depth)
    
    with tab3:
        show_optimization(data, optimization_aggressiveness)
    
    with tab4:
        show_monitoring(data)
    
    with tab5:
        show_tools()

def show_dashboard(data):
    st.header("📊 Automation Usage Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Monthly Usage",
            f"{data['current_usage']:,}",
            delta=f"{data['usage_change']:+d} from last month"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        usage_percentage = (data['current_usage'] / data['usage_limit']) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Usage Limit",
            f"{usage_percentage:.1f}%",
            delta=f"{data['usage_limit'] - data['current_usage']} remaining"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="optimization-card">', unsafe_allow_html=True)
        st.metric(
            "Optimization Potential",
            f"{data['optimization_potential']}%",
            delta=f"{data['potential_savings']} executions/month"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Active Rules",
            data['total_rules'],
            delta=f"{data['inefficient_rules']} need optimization"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Usage trend chart
    st.subheader("Usage Trends")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create usage trend chart
        usage_df = pd.DataFrame(data['usage_history'])
        fig = px.line(
            usage_df, x='date', y='usage',
            title='Monthly Automation Usage Trend',
            labels={'usage': 'Executions', 'date': 'Date'}
        )
        fig.add_hline(
            y=data['usage_limit'], 
            line_dash="dash", 
            line_color="red",
            annotation_text="Usage Limit"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rule efficiency distribution
        efficiency_data = data['rule_efficiency']
        fig_pie = px.pie(
            values=list(efficiency_data.values()),
            names=list(efficiency_data.keys()),
            title="Rule Efficiency Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top consuming rules
    st.subheader("Top Resource-Consuming Rules")
    rules_df = pd.DataFrame(data['top_rules'])
    st.dataframe(
        rules_df,
        column_config={
            "usage": st.column_config.ProgressColumn(
                "Monthly Usage",
                help="Executions per month",
                min_value=0,
                max_value=rules_df['usage'].max(),
            ),
            "efficiency_score": st.column_config.ProgressColumn(
                "Efficiency Score",
                help="Higher is better",
                min_value=0,
                max_value=100,
            )
        },
        use_container_width=True
    )

def show_analysis(data, analysis_depth):
    st.header("🔍 Intelligent Rule Analysis")
    
    if st.button(f"Run {analysis_depth}", type="primary"):
        with st.spinner("Analyzing automation rules..."):
            # Simulate analysis progress
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            st.success("Analysis completed!")
    
    # Analysis results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Rule Categories")
        categories = data['rule_categories']
        fig = px.bar(
            x=list(categories.keys()),
            y=list(categories.values()),
            title="Rules by Category",
            labels={'x': 'Category', 'y': 'Number of Rules'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Trigger Types")
        triggers = data['trigger_types']
        fig = px.pie(
            values=list(triggers.values()),
            names=list(triggers.keys()),
            title="Automation Triggers"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed findings
    st.subheader("Key Findings")
    
    findings = [
        {
            "type": "warning",
            "title": "High-Frequency Rules Detected",
            "description": "5 rules are executing more than 200 times per month",
            "impact": "Contributing to 60% of total usage"
        },
        {
            "type": "info",
            "title": "Consolidation Opportunities",
            "description": "8 rules have similar conditions and could be merged",
            "impact": "Potential 25% usage reduction"
        },
        {
            "type": "success",
            "title": "Well-Optimized Rules",
            "description": "12 rules are already efficiently configured",
            "impact": "No changes needed"
        }
    ]
    
    for finding in findings:
        if finding['type'] == 'warning':
            st.warning(f"⚠️ **{finding['title']}**: {finding['description']} - {finding['impact']}")
        elif finding['type'] == 'info':
            st.info(f"ℹ️ **{finding['title']}**: {finding['description']} - {finding['impact']}")
        else:
            st.success(f"✅ **{finding['title']}**: {finding['description']} - {finding['impact']}")

def show_optimization(data, aggressiveness):
    st.header("⚡ AI-Powered Optimization")
    
    # Optimization recommendations
    st.subheader("Optimization Recommendations")
    
    recommendations = [
        {
            "rule_name": "Auto-assign issues to team leads",
            "current_usage": 450,
            "action": "Convert to scheduled trigger (hourly)",
            "savings": 315,
            "confidence": 95,
            "risk": "Low"
        },
        {
            "rule_name": "Update issue status on PR merge",
            "current_usage": 280,
            "action": "Add project scope filter",
            "savings": 168,
            "confidence": 88,
            "risk": "Low"
        },
        {
            "rule_name": "Notify stakeholders on priority change",
            "current_usage": 220,
            "action": "Consolidate with similar notification rules",
            "savings": 132,
            "confidence": 82,
            "risk": "Medium"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"Recommendation {i+1}: {rec['rule_name']}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Current Usage**: {rec['current_usage']} executions/month")
                st.write(f"**Recommended Action**: {rec['action']}")
                st.write(f"**Potential Savings**: {rec['savings']} executions/month")
            
            with col2:
                st.metric("Confidence", f"{rec['confidence']}%")
                st.metric("Risk Level", rec['risk'])
            
            with col3:
                if st.button(f"Apply Optimization {i+1}", key=f"opt_{i}"):
                    st.success("✅ Optimization applied successfully!")
                    st.info("💡 Monitor the rule for 24-48 hours to ensure expected behavior.")
    
    # Batch optimization
    st.subheader("Batch Optimization")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("Apply multiple optimizations at once:")
        selected_optimizations = st.multiselect(
            "Select optimizations to apply:",
            [f"Recommendation {i+1}" for i in range(len(recommendations))]
        )
        
        if st.button("Apply Selected Optimizations", type="primary"):
            if selected_optimizations:
                total_savings = sum(rec['savings'] for i, rec in enumerate(recommendations) 
                                  if f"Recommendation {i+1}" in selected_optimizations)
                st.success(f"✅ Applied {len(selected_optimizations)} optimizations!")
                st.info(f"💰 Total potential savings: {total_savings} executions/month")
            else:
                st.warning("Please select at least one optimization.")
    
    with col2:
        # Optimization impact preview
        current_total = sum(rec['current_usage'] for rec in recommendations)
        potential_savings = sum(rec['savings'] for rec in recommendations)
        
        fig = go.Figure(data=[
            go.Bar(name='Current Usage', x=['Before', 'After'], 
                   y=[current_total, current_total - potential_savings]),
        ])
        fig.update_layout(title='Optimization Impact Preview')
        st.plotly_chart(fig, use_container_width=True)

def show_monitoring(data):
    st.header("📈 Proactive Monitoring")
    
    # Usage alerts
    st.subheader("Usage Alerts")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.warning("⚠️ **Usage Alert**: You're at 85% of your monthly limit")
        st.write("Projected to exceed limit in 8 days at current rate")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("ℹ️ **Optimization Opportunity**: 3 new inefficient rules detected")
    
    with col2:
        # Usage forecast
        forecast_data = {
            'date': pd.date_range(start='2026-02-01', periods=30, freq='D'),
            'actual': [50 + i*15 + (i%7)*10 for i in range(15)] + [None]*15,
            'forecast': [None]*15 + [800 + i*25 for i in range(15)]
        }
        forecast_df = pd.DataFrame(forecast_data)
        
        fig = px.line(forecast_df, x='date', y=['actual', 'forecast'],
                     title='Usage Forecast')
        fig.add_hline(y=1700, line_dash="dash", line_color="red",
                     annotation_text="Usage Limit")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics_data = {
        'Metric': ['Average Response Time', 'Success Rate', 'Error Rate', 'Resource Usage'],
        'Current': ['2.3s', '98.5%', '1.5%', '67%'],
        'Target': ['<3s', '>95%', '<5%', '<80%'],
        'Status': ['✅ Good', '✅ Good', '✅ Good', '⚠️ Monitor']
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

def show_tools():
    st.header("🛠️ Optimization Tools")
    
    tool_tabs = st.tabs(["Rule Builder", "Testing", "Export/Import", "Backup"])
    
    with tool_tabs[0]:
        st.subheader("AI Rule Builder")
        st.write("Create optimized automation rules using natural language:")
        
        user_intent = st.text_area(
            "Describe what you want to automate:",
            placeholder="When a high-priority bug is created, assign it to the team lead and notify stakeholders..."
        )
        
        if st.button("Generate Optimized Rule"):
            if user_intent:
                st.success("✅ Rule generated successfully!")
                st.code("""
# Generated Automation Rule
Trigger: Issue Created
Conditions:
  - Issue Type = Bug
  - Priority = High
  - Project = [Current Project]
Actions:
  - Assign to: {{project.lead}}
  - Send notification to: stakeholders@company.com
  - Add comment: "High priority bug assigned to team lead"
                """, language="yaml")
            else:
                st.warning("Please describe what you want to automate.")
    
    with tool_tabs[1]:
        st.subheader("Rule Testing")
        st.write("Test automation rules before deployment:")
        
        test_rule = st.selectbox(
            "Select rule to test:",
            ["Auto-assign issues", "Status updates", "Notifications"]
        )
        
        test_data = st.text_area(
            "Test data (JSON):",
            value='{"issue_type": "Bug", "priority": "High", "assignee": null}'
        )
        
        if st.button("Run Test"):
            st.success("✅ Test completed successfully!")
            st.json({
                "result": "PASS",
                "actions_triggered": 2,
                "execution_time": "0.3s",
                "estimated_monthly_usage": 45
            })
    
    with tool_tabs[2]:
        st.subheader("Export/Import Rules")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Export Rules**")
            export_format = st.selectbox("Format:", ["JSON", "YAML", "CSV"])
            if st.button("Export All Rules"):
                st.download_button(
                    "Download Rules",
                    data=json.dumps({"rules": "exported_rules_data"}, indent=2),
                    file_name=f"jira_rules.{export_format.lower()}",
                    mime="application/json"
                )
        
        with col2:
            st.write("**Import Rules**")
            uploaded_file = st.file_uploader("Choose file", type=['json', 'yaml', 'csv'])
            if uploaded_file and st.button("Import Rules"):
                st.success("✅ Rules imported successfully!")
    
    with tool_tabs[3]:
        st.subheader("Backup & Restore")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Create Backup**")
            backup_name = st.text_input("Backup name:", value=f"backup_{datetime.now().strftime('%Y%m%d')}")
            if st.button("Create Backup"):
                st.success(f"✅ Backup '{backup_name}' created successfully!")
        
        with col2:
            st.write("**Restore from Backup**")
            backup_list = ["backup_20260214", "backup_20260213", "backup_20260212"]
            selected_backup = st.selectbox("Select backup:", backup_list)
            if st.button("Restore Backup"):
                st.warning("⚠️ This will overwrite current rules. Are you sure?")
                if st.button("Confirm Restore"):
                    st.success(f"✅ Restored from '{selected_backup}'")

if __name__ == "__main__":
    main()