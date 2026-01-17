#!/usr/bin/env python3
"""
Streamlit Dashboard for Salesforce AI Data Quality Enhancement

Interactive web interface for managing Salesforce data quality,
including duplicate detection, data cleansing, and quality reporting.

Author: AI Product Enhancement System
Date: 2026-01-17
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time

from salesforce_ai_cleaner import SalesforceDataCleaner, DataQualityReport

# Page configuration
st.set_page_config(
    page_title="Salesforce AI Data Quality Enhancement",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = SalesforceDataCleaner()
    st.session_state.connected = False
    st.session_state.duplicates = []
    st.session_state.quality_reports = {}

def connect_to_salesforce():
    """Connect to Salesforce and update session state."""
    with st.spinner("Connecting to Salesforce..."):
        success = st.session_state.cleaner.connect()
        st.session_state.connected = success
        return success

def display_connection_status():
    """Display Salesforce connection status."""
    if st.session_state.connected:
        st.success("✅ Connected to Salesforce")
    else:
        st.error("❌ Not connected to Salesforce")
        if st.button("Connect to Salesforce"):
            if connect_to_salesforce():
                st.rerun()

def create_quality_score_gauge(score, title):
    """Create a gauge chart for quality scores."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def display_dashboard():
    """Display the main dashboard."""
    st.markdown('<div class="main-header">🔧 Salesforce AI Data Quality Enhancement</div>', 
                unsafe_allow_html=True)
    
    # Connection status
    display_connection_status()
    
    if not st.session_state.connected:
        st.info("Please connect to Salesforce to access data quality features.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard Overview", "Duplicate Detection", "Data Quality Assessment", 
         "Batch Processing", "Settings"]
    )
    
    if page == "Dashboard Overview":
        display_overview()
    elif page == "Duplicate Detection":
        display_duplicate_detection()
    elif page == "Data Quality Assessment":
        display_quality_assessment()
    elif page == "Batch Processing":
        display_batch_processing()
    elif page == "Settings":
        display_settings()

def display_overview():
    """Display dashboard overview with key metrics."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Records Processed Today",
            value="2,847",
            delta="+12%"
        )
    
    with col2:
        st.metric(
            label="Duplicates Found",
            value="156",
            delta="-8%"
        )
    
    with col3:
        st.metric(
            label="Data Quality Score",
            value="8.4/10",
            delta="+0.3"
        )
    
    with col4:
        st.metric(
            label="Auto-Merged Records",
            value="89",
            delta="+15%"
        )
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Quality Trends")
        
        # Sample data for demonstration
        dates = pd.date_range(start='2026-01-01', end='2026-01-17', freq='D')
        quality_scores = [7.2, 7.4, 7.6, 7.8, 8.0, 8.1, 8.3, 8.2, 8.4, 8.6, 
                         8.5, 8.7, 8.8, 8.6, 8.9, 8.7, 8.4]
        
        df_trends = pd.DataFrame({
            'Date': dates,
            'Quality Score': quality_scores
        })
        
        fig = px.line(df_trends, x='Date', y='Quality Score', 
                     title='Data Quality Score Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Object Type Distribution")
        
        # Sample data
        object_data = {
            'Object Type': ['Contact', 'Account', 'Lead', 'Opportunity', 'Case'],
            'Record Count': [15420, 8930, 12340, 5670, 3210],
            'Quality Score': [8.2, 8.7, 7.9, 8.5, 8.1]
        }
        
        df_objects = pd.DataFrame(object_data)
        
        fig = px.bar(df_objects, x='Object Type', y='Record Count',
                    color='Quality Score', color_continuous_scale='RdYlGn',
                    title='Records by Object Type')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    activity_data = {
        'Time': ['2 minutes ago', '15 minutes ago', '1 hour ago', '3 hours ago', '5 hours ago'],
        'Action': ['Merged 3 duplicate Contacts', 'Cleaned 45 Account records', 
                  'Found 12 potential Lead duplicates', 'Updated Contact matching rules',
                  'Completed quality assessment for Opportunities'],
        'Status': ['✅ Success', '✅ Success', '⚠️ Review Required', '✅ Success', '✅ Success']
    }
    
    df_activity = pd.DataFrame(activity_data)
    st.dataframe(df_activity, use_container_width=True, hide_index=True)

def display_duplicate_detection():
    """Display duplicate detection interface."""
    st.header("🔍 Duplicate Detection")
    
    # Object selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        object_type = st.selectbox(
            "Select Object Type",
            ["Contact", "Account", "Lead", "Opportunity"]
        )
    
    with col2:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05
        )
    
    with col3:
        if st.button("Find Duplicates", type="primary"):
            with st.spinner(f"Searching for {object_type} duplicates..."):
                duplicates = st.session_state.cleaner.find_duplicates(object_type, threshold)
                st.session_state.duplicates = duplicates
    
    # Display results
    if st.session_state.duplicates:
        st.success(f"Found {len(st.session_state.duplicates)} potential duplicate groups")
        
        for i, group in enumerate(st.session_state.duplicates):
            with st.expander(f"Duplicate Group {i+1} (Confidence: {group.confidence:.1%})"):
                
                # Display matching fields
                st.write(f"**Matching Fields:** {', '.join(group.matching_fields)}")
                
                # Create comparison table
                records_data = []
                for record in group.records:
                    record_info = {
                        'ID': record.get('Id', 'N/A'),
                        'Name': record.get('Name') or f"{record.get('FirstName', '')} {record.get('LastName', '')}".strip(),
                        'Email': record.get('Email', 'N/A'),
                        'Phone': record.get('Phone', 'N/A'),
                        'Company': record.get('Company') or record.get('Account', {}).get('Name', 'N/A')
                    }
                    records_data.append(record_info)
                
                df_records = pd.DataFrame(records_data)
                st.dataframe(df_records, use_container_width=True, hide_index=True)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"Preview Merge {i+1}", key=f"preview_{i}"):
                        preview = st.session_state.cleaner.preview_merge(group.records)
                        st.json(preview)
                
                with col2:
                    if st.button(f"Auto Merge {i+1}", key=f"merge_{i}", type="primary"):
                        if st.session_state.cleaner.merge_records(group.records):
                            st.success("Records merged successfully!")
                        else:
                            st.error("Failed to merge records")
                
                with col3:
                    if st.button(f"Skip Group {i+1}", key=f"skip_{i}"):
                        st.info("Group skipped")
    
    elif hasattr(st.session_state, 'duplicates') and len(st.session_state.duplicates) == 0:
        st.info("No duplicates found with the current threshold.")

def display_quality_assessment():
    """Display data quality assessment interface."""
    st.header("📈 Data Quality Assessment")
    
    # Object selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        object_type = st.selectbox(
            "Select Object Type for Assessment",
            ["Contact", "Account", "Lead", "Opportunity", "Case"]
        )
    
    with col2:
        if st.button("Run Assessment", type="primary"):
            with st.spinner(f"Assessing {object_type} data quality..."):
                report = st.session_state.cleaner.assess_data_quality(object_type)
                st.session_state.quality_reports[object_type] = report
    
    # Display results
    if object_type in st.session_state.quality_reports:
        report = st.session_state.quality_reports[object_type]
        
        # Overall score
        st.subheader("Overall Quality Score")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig = create_quality_score_gauge(report.overall_score * 10, "Overall Score")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_quality_score_gauge(report.completeness, "Completeness")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = create_quality_score_gauge(report.accuracy, "Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            fig = create_quality_score_gauge(report.consistency, "Consistency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Field-level scores
        st.subheader("Field-Level Quality Scores")
        
        if report.field_scores:
            df_fields = pd.DataFrame([
                {'Field': field, 'Completeness %': score}
                for field, score in report.field_scores.items()
            ])
            
            fig = px.bar(df_fields, x='Field', y='Completeness %',
                        title='Field Completeness Scores',
                        color='Completeness %',
                        color_continuous_scale='RdYlGn')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        
        for i, recommendation in enumerate(report.recommendations, 1):
            st.write(f"{i}. {recommendation}")
        
        # Duplicate rate
        if report.duplicate_rate > 0:
            st.warning(f"⚠️ Duplicate Rate: {report.duplicate_rate:.1f}% - Consider running duplicate detection")

def display_batch_processing():
    """Display batch processing interface."""
    st.header("⚙️ Batch Processing")
    
    st.info("Run automated data cleansing across multiple object types.")
    
    # Object selection
    objects = st.multiselect(
        "Select Objects to Process",
        ["Contact", "Account", "Lead", "Opportunity", "Case"],
        default=["Contact", "Account"]
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        auto_merge_threshold = st.slider(
            "Auto-merge Threshold",
            min_value=0.8,
            max_value=1.0,
            value=0.95,
            step=0.01
        )
    
    with col2:
        standardize_formats = st.checkbox("Standardize Data Formats", value=True)
        validate_emails = st.checkbox("Validate Email Addresses", value=True)
        enrich_missing_data = st.checkbox("Enrich Missing Data", value=False)
    
    # Run batch processing
    if st.button("Start Batch Processing", type="primary"):
        if not objects:
            st.error("Please select at least one object type.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing data..."):
                results = st.session_state.cleaner.clean_data(
                    objects,
                    auto_merge_threshold=auto_merge_threshold,
                    standardize_formats=standardize_formats,
                    validate_emails=validate_emails,
                    enrich_missing_data=enrich_missing_data
                )
                
                # Simulate progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f'Processing... {i+1}%')
                    time.sleep(0.01)
            
            st.success("Batch processing completed!")
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", results['total_records'])
            
            with col2:
                st.metric("Merged Duplicates", results['merged_duplicates'])
            
            with col3:
                st.metric("Standardized Fields", results['standardized_fields'])
            
            with col4:
                st.metric("Fixed Errors", results['data_errors'])

def display_settings():
    """Display settings and configuration."""
    st.header("⚙️ Settings")
    
    # Matching rules configuration
    st.subheader("Matching Rules Configuration")
    
    object_type = st.selectbox(
        "Configure Rules for Object Type",
        ["Contact", "Account", "Lead"]
    )
    
    if object_type in st.session_state.cleaner.matching_rules:
        rules = st.session_state.cleaner.matching_rules[object_type]
        
        st.write(f"**Current rules for {object_type}:**")
        
        # Display current matching fields
        for field, config in rules['matching_fields'].items():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**{field}**")
            
            with col2:
                st.write(f"Weight: {config['weight']}")
            
            with col3:
                if config.get('exact_match'):
                    st.write("Exact Match")
                else:
                    st.write(f"Fuzzy: {config.get('fuzzy_threshold', 0.8)}")
        
        st.write(f"**Minimum Score:** {rules['minimum_score']}")
    
    # Connection settings
    st.subheader("Connection Settings")
    
    st.info("Salesforce credentials are loaded from environment variables (.env file)")
    
    # Test connection
    if st.button("Test Connection"):
        if connect_to_salesforce():
            st.success("✅ Connection successful!")
        else:
            st.error("❌ Connection failed. Please check your credentials.")
    
    # Export/Import settings
    st.subheader("Export/Import Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Configuration"):
            config = {
                'matching_rules': st.session_state.cleaner.matching_rules,
                'export_date': datetime.now().isoformat()
            }
            st.download_button(
                label="Download Config JSON",
                data=json.dumps(config, indent=2),
                file_name=f"salesforce_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_file = st.file_uploader("Import Configuration", type="json")
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                if 'matching_rules' in config:
                    st.session_state.cleaner.set_matching_rules(config['matching_rules'])
                    st.success("Configuration imported successfully!")
                else:
                    st.error("Invalid configuration file format.")
            except Exception as e:
                st.error(f"Error importing configuration: {str(e)}")

def main():
    """Main application entry point."""
    display_dashboard()

if __name__ == "__main__":
    main()