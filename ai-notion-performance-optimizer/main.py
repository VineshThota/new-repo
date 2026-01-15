#!/usr/bin/env python3
"""
AI-Powered Notion Database Performance Optimizer

This application analyzes Notion database configurations and provides
intelligent recommendations to improve loading times and performance.

Author: AI Agent Product Enhancement System
Date: 2026-01-15
"""

import json
import time
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class PropertyType(Enum):
    TITLE = "title"
    TEXT = "text"
    NUMBER = "number"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    PERSON = "person"
    FILES = "files"
    CHECKBOX = "checkbox"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    FORMULA = "formula"
    ROLLUP = "rollup"
    RELATION = "relation"
    CREATED_TIME = "created_time"
    CREATED_BY = "created_by"
    LAST_EDITED_TIME = "last_edited_time"
    LAST_EDITED_BY = "last_edited_by"

class PerformanceImpact(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class DatabaseProperty:
    name: str
    type: PropertyType
    is_visible: bool = True
    has_filter: bool = False
    has_sort: bool = False
    options_count: int = 0  # For select/multi-select
    formula_complexity: int = 0  # 0-5 scale
    rollup_depth: int = 0  # Number of relation hops

@dataclass
class DatabaseConfig:
    name: str
    page_count: int
    properties: List[DatabaseProperty]
    has_inline_databases: int = 0
    linked_databases_count: int = 0
    average_page_size_mb: float = 0.1
    has_complex_views: bool = False
    relation_chains_depth: int = 0

@dataclass
class OptimizationRecommendation:
    title: str
    description: str
    impact: PerformanceImpact
    effort: str  # "Low", "Medium", "High"
    expected_improvement: str
    implementation_steps: List[str]
    priority_score: float

class NotionPerformanceAnalyzer:
    """AI-powered analyzer for Notion database performance optimization."""
    
    def __init__(self):
        self.performance_weights = {
            'page_count': 0.25,
            'visible_properties': 0.20,
            'complex_filters': 0.15,
            'formula_complexity': 0.15,
            'rollup_complexity': 0.10,
            'inline_databases': 0.10,
            'page_size': 0.05
        }
    
    def calculate_performance_score(self, config: DatabaseConfig) -> float:
        """Calculate overall performance score (0-100, higher is better)."""
        score = 100.0
        
        # Page count impact
        if config.page_count > 10000:
            score -= 30
        elif config.page_count > 5000:
            score -= 20
        elif config.page_count > 1000:
            score -= 10
        
        # Visible properties impact
        visible_props = sum(1 for prop in config.properties if prop.is_visible)
        if visible_props > 20:
            score -= 25
        elif visible_props > 15:
            score -= 15
        elif visible_props > 10:
            score -= 8
        
        # Complex filters and sorts
        complex_filters = sum(1 for prop in config.properties 
                            if prop.has_filter and prop.type in [PropertyType.FORMULA, PropertyType.ROLLUP, PropertyType.TEXT])
        score -= complex_filters * 5
        
        # Formula complexity
        total_formula_complexity = sum(prop.formula_complexity for prop in config.properties if prop.type == PropertyType.FORMULA)
        score -= total_formula_complexity * 3
        
        # Rollup depth
        max_rollup_depth = max((prop.rollup_depth for prop in config.properties if prop.type == PropertyType.ROLLUP), default=0)
        score -= max_rollup_depth * 8
        
        # Inline databases
        score -= config.has_inline_databases * 10
        
        # Page size
        if config.average_page_size_mb > 2.0:
            score -= 20
        elif config.average_page_size_mb > 1.0:
            score -= 10
        
        return max(0, min(100, score))
    
    def generate_recommendations(self, config: DatabaseConfig) -> List[OptimizationRecommendation]:
        """Generate AI-powered optimization recommendations."""
        recommendations = []
        
        # Page count optimization
        if config.page_count > 5000:
            recommendations.append(OptimizationRecommendation(
                title="Implement Database Archiving Strategy",
                description=f"Your database has {config.page_count:,} pages, which significantly impacts performance. Consider archiving old or unused pages.",
                impact=PerformanceImpact.HIGH,
                effort="Medium",
                expected_improvement="30-50% faster loading",
                implementation_steps=[
                    "Add a 'Created time' filter to identify old pages",
                    "Create an 'Archive' status property",
                    "Move pages older than 1 year to archived status",
                    "Create separate views for active vs archived data"
                ],
                priority_score=9.0
            ))
        
        # Visible properties optimization
        visible_props = [prop for prop in config.properties if prop.is_visible]
        if len(visible_props) > 15:
            recommendations.append(OptimizationRecommendation(
                title="Hide Unnecessary Properties",
                description=f"You have {len(visible_props)} visible properties. Hiding less important ones can improve responsiveness.",
                impact=PerformanceImpact.MEDIUM,
                effort="Low",
                expected_improvement="15-25% faster loading",
                implementation_steps=[
                    "Identify properties used less than 20% of the time",
                    "Hide properties that are only needed for specific workflows",
                    "Create custom views with only essential properties",
                    "Use property groups to organize remaining visible properties"
                ],
                priority_score=7.5
            ))
        
        # Complex filter optimization
        complex_filters = [prop for prop in config.properties 
                         if prop.has_filter and prop.type in [PropertyType.FORMULA, PropertyType.ROLLUP, PropertyType.TEXT]]
        if complex_filters:
            recommendations.append(OptimizationRecommendation(
                title="Optimize Complex Filters",
                description=f"Found {len(complex_filters)} complex filters on formula/rollup/text properties that slow down database loading.",
                impact=PerformanceImpact.HIGH,
                effort="Medium",
                expected_improvement="25-40% faster filtering",
                implementation_steps=[
                    "Replace text filters with select/multi-select properties where possible",
                    "Add simple property filters before complex ones",
                    "Consider pre-calculating formula results as select options",
                    "Use status properties instead of formula-based filters"
                ],
                priority_score=8.5
            ))
        
        # Formula complexity optimization
        complex_formulas = [prop for prop in config.properties 
                          if prop.type == PropertyType.FORMULA and prop.formula_complexity > 3]
        if complex_formulas:
            recommendations.append(OptimizationRecommendation(
                title="Simplify Complex Formulas",
                description=f"Found {len(complex_formulas)} complex formulas that may be causing performance issues.",
                impact=PerformanceImpact.MEDIUM,
                effort="High",
                expected_improvement="20-35% faster calculation",
                implementation_steps=[
                    "Break down complex formulas into simpler components",
                    "Use intermediate properties for multi-step calculations",
                    "Consider replacing formulas with manual updates for static data",
                    "Cache frequently used formula results"
                ],
                priority_score=6.5
            ))
        
        # Rollup optimization
        deep_rollups = [prop for prop in config.properties 
                       if prop.type == PropertyType.ROLLUP and prop.rollup_depth > 2]
        if deep_rollups:
            recommendations.append(OptimizationRecommendation(
                title="Reduce Rollup Chain Depth",
                description=f"Found {len(deep_rollups)} rollups with deep reference chains that impact performance.",
                impact=PerformanceImpact.HIGH,
                effort="High",
                expected_improvement="30-50% faster rollup calculation",
                implementation_steps=[
                    "Flatten deep rollup chains by creating intermediate databases",
                    "Use direct relations instead of rollup chains where possible",
                    "Consider denormalizing frequently accessed rollup data",
                    "Implement periodic rollup result caching"
                ],
                priority_score=8.0
            ))
        
        # Inline database optimization
        if config.has_inline_databases > 3:
            recommendations.append(OptimizationRecommendation(
                title="Convert Inline Databases to Linked Views",
                description=f"You have {config.has_inline_databases} inline databases on high-traffic pages, causing performance issues.",
                impact=PerformanceImpact.HIGH,
                effort="Medium",
                expected_improvement="40-60% faster page loading",
                implementation_steps=[
                    "Move each inline database to its own dedicated page",
                    "Create linked database views pointing to the original databases",
                    "Configure views to show only necessary data",
                    "Use database templates for consistent structure"
                ],
                priority_score=9.5
            ))
        
        # Page size optimization
        if config.average_page_size_mb > 1.5:
            recommendations.append(OptimizationRecommendation(
                title="Optimize Page Content Size",
                description=f"Average page size of {config.average_page_size_mb:.1f}MB is causing slow loading times.",
                impact=PerformanceImpact.MEDIUM,
                effort="Medium",
                expected_improvement="25-40% faster page loading",
                implementation_steps=[
                    "Compress large images before uploading",
                    "Move large content blocks to sub-pages",
                    "Reduce the number of embedded widgets per page",
                    "Use toggles to hide non-essential content"
                ],
                priority_score=7.0
            ))
        
        # Sort recommendations by priority score
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        return recommendations
    
    def estimate_performance_improvement(self, config: DatabaseConfig, applied_recommendations: List[str]) -> Dict[str, float]:
        """Estimate performance improvement after applying recommendations."""
        current_score = self.calculate_performance_score(config)
        
        # Simulate improvements based on applied recommendations
        improvement_factors = {
            "Implement Database Archiving Strategy": 0.35,
            "Hide Unnecessary Properties": 0.20,
            "Optimize Complex Filters": 0.30,
            "Simplify Complex Formulas": 0.25,
            "Reduce Rollup Chain Depth": 0.40,
            "Convert Inline Databases to Linked Views": 0.50,
            "Optimize Page Content Size": 0.30
        }
        
        total_improvement = sum(improvement_factors.get(rec, 0) for rec in applied_recommendations)
        improved_score = min(100, current_score + (total_improvement * 20))
        
        return {
            "current_score": current_score,
            "improved_score": improved_score,
            "improvement_percentage": ((improved_score - current_score) / current_score) * 100 if current_score > 0 else 0,
            "estimated_load_time_reduction": min(80, total_improvement * 60)  # Max 80% reduction
        }

def create_sample_database() -> DatabaseConfig:
    """Create a sample database configuration for demonstration."""
    properties = [
        DatabaseProperty("Title", PropertyType.TITLE, True, True, True),
        DatabaseProperty("Status", PropertyType.SELECT, True, True, True, options_count=5),
        DatabaseProperty("Priority", PropertyType.SELECT, True, True, False, options_count=3),
        DatabaseProperty("Assignee", PropertyType.PERSON, True, True, False),
        DatabaseProperty("Due Date", PropertyType.DATE, True, True, True),
        DatabaseProperty("Tags", PropertyType.MULTI_SELECT, True, True, False, options_count=15),
        DatabaseProperty("Description", PropertyType.TEXT, True, True, False),
        DatabaseProperty("Progress", PropertyType.FORMULA, True, False, False, formula_complexity=4),
        DatabaseProperty("Related Tasks", PropertyType.ROLLUP, True, True, False, rollup_depth=3),
        DatabaseProperty("Time Spent", PropertyType.NUMBER, True, False, True),
        DatabaseProperty("Budget", PropertyType.NUMBER, False, False, False),
        DatabaseProperty("Notes", PropertyType.TEXT, False, False, False),
        DatabaseProperty("Files", PropertyType.FILES, True, False, False),
        DatabaseProperty("URL", PropertyType.URL, False, False, False),
        DatabaseProperty("Email", PropertyType.EMAIL, False, False, False),
        DatabaseProperty("Created", PropertyType.CREATED_TIME, True, True, True),
        DatabaseProperty("Modified", PropertyType.LAST_EDITED_TIME, False, False, False),
    ]
    
    return DatabaseConfig(
        name="Project Management Database",
        page_count=8500,
        properties=properties,
        has_inline_databases=5,
        linked_databases_count=2,
        average_page_size_mb=1.8,
        has_complex_views=True,
        relation_chains_depth=3
    )

def main():
    st.set_page_config(
        page_title="AI Notion Performance Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ AI-Powered Notion Database Performance Optimizer")
    st.markdown("""
    This tool analyzes your Notion database configuration and provides intelligent recommendations 
    to improve loading times and overall performance based on Notion's best practices and AI analysis.
    """)
    
    # Initialize analyzer
    analyzer = NotionPerformanceAnalyzer()
    
    # Sidebar for configuration
    st.sidebar.header("Database Configuration")
    
    # Option to use sample data or input custom data
    use_sample = st.sidebar.checkbox("Use Sample Database", value=True)
    
    if use_sample:
        config = create_sample_database()
        st.sidebar.success("Using sample project management database")
    else:
        # Custom database configuration
        st.sidebar.subheader("Basic Information")
        db_name = st.sidebar.text_input("Database Name", "My Database")
        page_count = st.sidebar.number_input("Number of Pages", min_value=1, max_value=100000, value=1000)
        inline_dbs = st.sidebar.number_input("Inline Databases Count", min_value=0, max_value=20, value=0)
        avg_page_size = st.sidebar.slider("Average Page Size (MB)", 0.1, 5.0, 0.5, 0.1)
        
        # For simplicity, create a basic config
        config = DatabaseConfig(
            name=db_name,
            page_count=page_count,
            properties=[],  # Would need more complex UI for full property configuration
            has_inline_databases=inline_dbs,
            average_page_size_mb=avg_page_size
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📊 Analysis for: {config.name}")
        
        # Performance score
        performance_score = analyzer.calculate_performance_score(config)
        
        # Create performance gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=performance_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Performance Score"},
            delta={'reference': 80},
            gauge={
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
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Performance interpretation
        if performance_score >= 80:
            st.success(f"🎉 Excellent performance! Your database is well-optimized.")
        elif performance_score >= 60:
            st.warning(f"⚠️ Good performance with room for improvement.")
        elif performance_score >= 40:
            st.warning(f"🔧 Performance issues detected. Optimization recommended.")
        else:
            st.error(f"🚨 Critical performance issues. Immediate optimization required.")
    
    with col2:
        st.header("📈 Database Stats")
        
        stats_data = {
            "Metric": ["Total Pages", "Visible Properties", "Inline Databases", "Avg Page Size", "Performance Score"],
            "Value": [
                f"{config.page_count:,}",
                f"{sum(1 for prop in config.properties if prop.is_visible)}",
                f"{config.has_inline_databases}",
                f"{config.average_page_size_mb:.1f} MB",
                f"{performance_score:.1f}/100"
            ]
        }
        
        st.table(pd.DataFrame(stats_data))
    
    # Recommendations section
    st.header("🎯 AI-Powered Optimization Recommendations")
    
    recommendations = analyzer.generate_recommendations(config)
    
    if not recommendations:
        st.success("🎉 Your database is already well-optimized! No major recommendations at this time.")
    else:
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec.title} (Priority: {rec.priority_score:.1f}/10)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    impact_color = {
                        PerformanceImpact.LOW: "🟢",
                        PerformanceImpact.MEDIUM: "🟡",
                        PerformanceImpact.HIGH: "🟠",
                        PerformanceImpact.CRITICAL: "🔴"
                    }
                    st.write(f"**Impact:** {impact_color[rec.impact]} {rec.impact.name}")
                
                with col2:
                    st.write(f"**Effort:** {rec.effort}")
                
                with col3:
                    st.write(f"**Expected Improvement:** {rec.expected_improvement}")
                
                st.write(f"**Description:** {rec.description}")
                
                st.write("**Implementation Steps:**")
                for step in rec.implementation_steps:
                    st.write(f"• {step}")
    
    # Performance improvement simulator
    st.header("🔮 Performance Improvement Simulator")
    
    if recommendations:
        st.write("Select which recommendations you plan to implement:")
        
        selected_recommendations = []
        for rec in recommendations:
            if st.checkbox(f"{rec.title}", key=f"sim_{rec.title}"):
                selected_recommendations.append(rec.title)
        
        if selected_recommendations:
            improvements = analyzer.estimate_performance_improvement(config, selected_recommendations)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Before/After comparison
                comparison_data = {
                    "Metric": ["Performance Score", "Estimated Load Time Reduction"],
                    "Before": [f"{improvements['current_score']:.1f}/100", "0%"],
                    "After": [f"{improvements['improved_score']:.1f}/100", f"{improvements['estimated_load_time_reduction']:.1f}%"]
                }
                st.table(pd.DataFrame(comparison_data))
            
            with col2:
                # Improvement chart
                fig_improvement = go.Figure(data=[
                    go.Bar(name='Current', x=['Performance Score'], y=[improvements['current_score']], marker_color='lightcoral'),
                    go.Bar(name='After Optimization', x=['Performance Score'], y=[improvements['improved_score']], marker_color='lightgreen')
                ])
                fig_improvement.update_layout(
                    title='Performance Score Improvement',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 100]),
                    height=300
                )
                st.plotly_chart(fig_improvement, use_container_width=True)
            
            improvement_pct = improvements['improvement_percentage']
            if improvement_pct > 0:
                st.success(f"🚀 Implementing these recommendations could improve your database performance by {improvement_pct:.1f}%!")
    
    # Additional resources
    st.header("📚 Additional Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Notion Official Guides:**
        - [Database Optimization](https://notion.com/help/optimize-database-load-times-and-performance)
        - [Database Best Practices](https://notion.com/help/guides/database-best-practices)
        """)
    
    with col2:
        st.markdown("""
        **Performance Tips:**
        - Use linked databases instead of inline
        - Hide unnecessary properties
        - Limit complex formulas and rollups
        - Archive old data regularly
        """)
    
    with col3:
        st.markdown("""
        **Quick Wins:**
        - Set database load limits (10-25 pages)
        - Use simple property filters first
        - Compress images before uploading
        - Organize with toggles and sub-pages
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    🤖 AI-Powered Notion Performance Optimizer | Built with Streamlit & Python<br>
    Helping teams optimize their Notion databases for better performance
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()