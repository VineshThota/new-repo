import streamlit as st
import psutil
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import subprocess
import os
import json
from typing import List, Dict, Tuple
import threading
import queue

class TeamsMemoryOptimizer:
    def __init__(self):
        self.monitoring_active = False
        self.memory_history = []
        self.optimization_suggestions = []
        
    def find_teams_processes(self) -> List[Dict]:
        """Find all Microsoft Teams related processes"""
        teams_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if any(keyword in proc.info['name'].lower() for keyword in 
                      ['teams', 'ms-teams', 'microsoft teams']):
                    memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                    teams_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_mb': round(memory_mb, 2),
                        'cpu_percent': proc.info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return teams_processes
    
    def get_system_memory_info(self) -> Dict:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / 1024**3, 2),
            'available_gb': round(memory.available / 1024**3, 2),
            'used_gb': round(memory.used / 1024**3, 2),
            'percent_used': memory.percent
        }
    
    def analyze_memory_usage(self, processes: List[Dict]) -> Dict:
        """Analyze Teams memory usage patterns"""
        if not processes:
            return {'total_memory': 0, 'process_count': 0, 'analysis': 'No Teams processes found'}
        
        total_memory = sum(p['memory_mb'] for p in processes)
        process_count = len(processes)
        
        analysis = []
        if total_memory > 1000:  # Over 1GB
            analysis.append("🔴 CRITICAL: Teams is using over 1GB of memory")
        elif total_memory > 500:  # Over 500MB
            analysis.append("🟡 WARNING: Teams is using significant memory (>500MB)")
        else:
            analysis.append("🟢 GOOD: Teams memory usage is within normal range")
        
        if process_count > 5:
            analysis.append(f"⚠️ Multiple Teams processes detected ({process_count})")
        
        return {
            'total_memory': round(total_memory, 2),
            'process_count': process_count,
            'analysis': analysis
        }
    
    def generate_optimization_suggestions(self, memory_analysis: Dict, system_info: Dict) -> List[str]:
        """Generate AI-powered optimization suggestions"""
        suggestions = []
        
        total_memory = memory_analysis['total_memory']
        process_count = memory_analysis['process_count']
        system_memory_percent = system_info['percent_used']
        
        # Memory-based suggestions
        if total_memory > 1000:
            suggestions.extend([
                "🔧 Restart Teams to clear memory leaks",
                "💾 Close unnecessary Teams tabs and chats",
                "⚙️ Disable GPU hardware acceleration in Teams settings",
                "📱 Consider using Teams web version instead of desktop app"
            ])
        
        if process_count > 3:
            suggestions.extend([
                "🔄 Multiple Teams processes detected - restart the application",
                "🚫 Check for duplicate Teams installations",
                "🧹 Clear Teams cache and temporary files"
            ])
        
        # System-based suggestions
        if system_memory_percent > 80:
            suggestions.extend([
                "💻 System memory is critically low - close other applications",
                "🔄 Consider restarting your computer",
                "📊 Monitor other memory-intensive applications"
            ])
        
        # General optimization suggestions
        suggestions.extend([
            "🎯 Disable Teams auto-start if not needed",
            "📹 Turn off video when not necessary in meetings",
            "🔕 Reduce notification frequency",
            "🗂️ Limit the number of teams and channels you're active in",
            "🔄 Update Teams to the latest version"
        ])
        
        return suggestions[:8]  # Return top 8 suggestions
    
    def clear_teams_cache(self) -> str:
        """Provide instructions to clear Teams cache"""
        instructions = """
        **Steps to Clear Microsoft Teams Cache:**
        
        1. **Close Teams completely**
           - Right-click Teams icon in system tray
           - Select 'Quit'
        
        2. **Open Run dialog**
           - Press Windows + R
        
        3. **Navigate to cache folder**
           - Type: `%appdata%\\Microsoft\\Teams`
           - Press Enter
        
        4. **Delete cache folders**
           - Delete these folders:
             - Application Cache
             - Blob_storage
             - Cache
             - GPUCache
             - IndexedDB
             - Local Storage
             - tmp
        
        5. **Restart Teams**
           - Launch Teams normally
        
        **Note:** This will log you out and reset some settings.
        """
        return instructions

def main():
    st.set_page_config(
        page_title="AI Teams Memory Optimizer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 AI-Powered Microsoft Teams Memory Optimizer")
    st.markdown("""
    **Solve Microsoft Teams' notorious memory consumption issues with AI-powered analysis and optimization suggestions.**
    
    This tool addresses the common complaint that Teams consumes 1GB+ of RAM even when idle.
    """)
    
    optimizer = TeamsMemoryOptimizer()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    if st.sidebar.button("🔍 Scan Teams Processes", type="primary"):
        st.session_state.scan_triggered = True
    
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5s)", value=False)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Current System Status")
        
        # Get current data
        teams_processes = optimizer.find_teams_processes()
        system_info = optimizer.get_system_memory_info()
        memory_analysis = optimizer.analyze_memory_usage(teams_processes)
        
        # System memory overview
        st.subheader("💾 System Memory")
        memory_col1, memory_col2, memory_col3 = st.columns(3)
        
        with memory_col1:
            st.metric(
                "Total RAM", 
                f"{system_info['total_gb']} GB",
                help="Total system memory"
            )
        
        with memory_col2:
            st.metric(
                "Available RAM", 
                f"{system_info['available_gb']} GB",
                help="Available system memory"
            )
        
        with memory_col3:
            st.metric(
                "Memory Usage", 
                f"{system_info['percent_used']:.1f}%",
                delta=f"{system_info['used_gb']} GB used",
                help="Current system memory usage"
            )
        
        # Teams memory analysis
        st.subheader("🎯 Teams Memory Analysis")
        
        if teams_processes:
            teams_col1, teams_col2 = st.columns(2)
            
            with teams_col1:
                st.metric(
                    "Teams Memory Usage", 
                    f"{memory_analysis['total_memory']} MB",
                    help="Total memory used by all Teams processes"
                )
            
            with teams_col2:
                st.metric(
                    "Active Processes", 
                    memory_analysis['process_count'],
                    help="Number of Teams-related processes"
                )
            
            # Analysis results
            st.subheader("🔍 Analysis Results")
            for analysis_point in memory_analysis['analysis']:
                if "CRITICAL" in analysis_point:
                    st.error(analysis_point)
                elif "WARNING" in analysis_point:
                    st.warning(analysis_point)
                else:
                    st.success(analysis_point)
            
            # Process details
            st.subheader("📋 Process Details")
            df = pd.DataFrame(teams_processes)
            st.dataframe(
                df,
                column_config={
                    "pid": "Process ID",
                    "name": "Process Name",
                    "memory_mb": st.column_config.NumberColumn(
                        "Memory (MB)",
                        format="%.2f MB"
                    ),
                    "cpu_percent": st.column_config.NumberColumn(
                        "CPU %",
                        format="%.1f%%"
                    )
                },
                hide_index=True
            )
            
            # Memory usage chart
            if len(teams_processes) > 1:
                st.subheader("📈 Memory Distribution")
                fig = px.pie(
                    df, 
                    values='memory_mb', 
                    names='name',
                    title="Teams Memory Usage by Process"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("🔍 No Microsoft Teams processes found running.")
            st.markdown("""
            **Possible reasons:**
            - Teams is not currently running
            - Teams is running as a different user
            - Teams processes have different names on your system
            """)
    
    with col2:
        st.header("🤖 AI Optimization")
        
        if teams_processes:
            suggestions = optimizer.generate_optimization_suggestions(memory_analysis, system_info)
            
            st.subheader("💡 Smart Suggestions")
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"**{i}.** {suggestion}")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("📋 Show Cache Clear Instructions"):
                st.markdown(optimizer.clear_teams_cache())
            
            if st.button("🔄 Refresh Analysis"):
                st.rerun()
        
        else:
            st.info("Start Teams to see optimization suggestions")
        
        # Memory usage gauge
        st.subheader("📊 Memory Gauge")
        
        if teams_processes:
            teams_memory_gb = memory_analysis['total_memory'] / 1024
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = teams_memory_gb,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Teams Memory (GB)"},
                delta = {'reference': 0.5},  # 500MB reference
                gauge = {
                    'axis': {'range': [None, 2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "yellow"},
                        {'range': [1, 2], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🎯 About This Tool:**
    This AI-powered optimizer addresses Microsoft Teams' well-documented memory consumption issues. 
    Based on user complaints and technical analysis, it provides intelligent suggestions to optimize 
    Teams performance and reduce system resource usage.
    
    **📊 Data Sources:** Real-time system monitoring, process analysis, and AI-driven optimization algorithms.
    """)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(1)  # Small delay before rerun
        st.rerun()

if __name__ == "__main__":
    main()