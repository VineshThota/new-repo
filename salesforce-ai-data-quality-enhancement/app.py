import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple

class SalesforceDataQualityAI:
    """AI-powered Salesforce data quality enhancement system"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.company_standardizations = {
            'inc': 'Inc.',
            'incorporated': 'Inc.',
            'corp': 'Corp.',
            'corporation': 'Corp.',
            'llc': 'LLC',
            'ltd': 'Ltd.',
            'limited': 'Ltd.',
            'co': 'Co.',
            'company': 'Company'
        }
        
    def standardize_company_name(self, company_name: str) -> str:
        """Standardize company names using NLP techniques"""
        if not company_name or pd.isna(company_name):
            return ""
            
        # Clean and normalize
        name = str(company_name).strip()
        name = re.sub(r'[^\w\s&.-]', '', name)  # Remove special chars except &, ., -
        name = ' '.join(name.split())  # Normalize whitespace
        
        # Apply standardizations
        words = name.lower().split()
        standardized_words = []
        
        for word in words:
            if word in self.company_standardizations:
                standardized_words.append(self.company_standardizations[word])
            else:
                standardized_words.append(word.title())
                
        return ' '.join(standardized_words)
    
    def standardize_phone_number(self, phone: str) -> str:
        """Standardize phone numbers to consistent format"""
        if not phone or pd.isna(phone):
            return ""
            
        # Extract digits only
        digits = re.sub(r'\D', '', str(phone))
        
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            return phone  # Return original if can't standardize
    
    def standardize_email(self, email: str) -> str:
        """Standardize email addresses"""
        if not email or pd.isna(email):
            return ""
            
        email = str(email).strip().lower()
        # Basic email validation
        if '@' in email and '.' in email.split('@')[1]:
            return email
        return ""
    
    def calculate_data_quality_score(self, record: Dict) -> Dict:
        """Calculate comprehensive data quality score for a record"""
        score = 0
        max_score = 0
        issues = []
        
        # Required fields check
        required_fields = ['first_name', 'last_name', 'email', 'company']
        for field in required_fields:
            max_score += 20
            if field in record and record[field] and not pd.isna(record[field]):
                score += 20
            else:
                issues.append(f"Missing {field.replace('_', ' ').title()}")
        
        # Email validation
        max_score += 10
        if 'email' in record and record['email']:
            email = str(record['email'])
            if '@' in email and '.' in email.split('@')[1]:
                score += 10
            else:
                issues.append("Invalid email format")
        
        # Phone validation
        max_score += 10
        if 'phone' in record and record['phone']:
            phone_digits = re.sub(r'\D', '', str(record['phone']))
            if len(phone_digits) >= 10:
                score += 10
            else:
                issues.append("Invalid phone number")
        
        quality_percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        return {
            'score': quality_percentage,
            'issues': issues,
            'grade': self._get_quality_grade(quality_percentage)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def find_duplicates(self, df: pd.DataFrame, threshold: int = 85) -> List[Dict]:
        """Find duplicate records using fuzzy matching and ML techniques"""
        duplicates = []
        processed_indices = set()
        
        for i, record1 in df.iterrows():
            if i in processed_indices:
                continue
                
            matches = []
            
            for j, record2 in df.iterrows():
                if i >= j or j in processed_indices:
                    continue
                    
                similarity_score = self._calculate_similarity(record1, record2)
                
                if similarity_score >= threshold:
                    matches.append({
                        'index': j,
                        'record': record2.to_dict(),
                        'similarity': similarity_score
                    })
                    processed_indices.add(j)
            
            if matches:
                duplicates.append({
                    'primary_index': i,
                    'primary_record': record1.to_dict(),
                    'matches': matches
                })
                processed_indices.add(i)
        
        return duplicates
    
    def _calculate_similarity(self, record1: pd.Series, record2: pd.Series) -> float:
        """Calculate similarity between two records using multiple algorithms"""
        scores = []
        
        # Name similarity
        name1 = f"{record1.get('first_name', '')} {record1.get('last_name', '')}".strip()
        name2 = f"{record2.get('first_name', '')} {record2.get('last_name', '')}".strip()
        if name1 and name2:
            scores.append(fuzz.ratio(name1.lower(), name2.lower()))
        
        # Email similarity
        email1 = str(record1.get('email', '')).lower()
        email2 = str(record2.get('email', '')).lower()
        if email1 and email2:
            scores.append(fuzz.ratio(email1, email2))
        
        # Company similarity
        company1 = str(record1.get('company', '')).lower()
        company2 = str(record2.get('company', '')).lower()
        if company1 and company2:
            scores.append(fuzz.ratio(company1, company2))
        
        # Phone similarity
        phone1 = re.sub(r'\D', '', str(record1.get('phone', '')))
        phone2 = re.sub(r'\D', '', str(record2.get('phone', '')))
        if phone1 and phone2 and len(phone1) >= 10 and len(phone2) >= 10:
            scores.append(fuzz.ratio(phone1[-10:], phone2[-10:]))
        
        return np.mean(scores) if scores else 0
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive data cleaning to the dataset"""
        cleaned_df = df.copy()
        
        # Standardize company names
        if 'company' in cleaned_df.columns:
            cleaned_df['company'] = cleaned_df['company'].apply(self.standardize_company_name)
        
        # Standardize phone numbers
        if 'phone' in cleaned_df.columns:
            cleaned_df['phone'] = cleaned_df['phone'].apply(self.standardize_phone_number)
        
        # Standardize emails
        if 'email' in cleaned_df.columns:
            cleaned_df['email'] = cleaned_df['email'].apply(self.standardize_email)
        
        # Standardize names
        for col in ['first_name', 'last_name']:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: str(x).strip().title() if x and not pd.isna(x) else ""
                )
        
        return cleaned_df

def create_sample_data() -> pd.DataFrame:
    """Create sample Salesforce-like data with quality issues"""
    data = {
        'first_name': ['John', 'john', 'Jane', 'jane', 'Bob', 'Robert', 'Alice', '', 'Mike'],
        'last_name': ['Smith', 'smith', 'Doe', 'doe', 'Johnson', 'Johnson', 'Williams', 'Brown', 'Davis'],
        'email': ['john.smith@acme.com', 'j.smith@acme.com', 'jane.doe@techcorp.com', 
                 'jane.doe@tech-corp.com', 'bob@company.com', 'robert.johnson@company.com',
                 'alice@startup.io', 'invalid-email', 'mike.davis@bigcorp.com'],
        'phone': ['555-123-4567', '(555) 123-4567', '555.987.6543', '5559876543',
                 '555-555-5555', '555-555-5555', '123-456-7890', '', '555-999-8888'],
        'company': ['Acme Inc', 'ACME INCORPORATED', 'TechCorp LLC', 'Tech Corp, LLC',
                   'Big Company', 'Big Company Inc.', 'Startup Co', 'Unknown Corp', 'BigCorp Ltd']
    }
    return pd.DataFrame(data)

def main():
    st.set_page_config(
        page_title="Salesforce AI Data Quality Enhancement",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Salesforce AI Data Quality Enhancement")
    st.markdown("""
    **AI-powered solution to fix Salesforce data quality issues**
    
    This tool addresses the critical pain point of messy CRM data that breaks automation and AI functionality.
    Upload your Salesforce data or use our sample dataset to see the AI in action.
    """)
    
    # Initialize the AI system
    ai_system = SalesforceDataQualityAI()
    
    # Sidebar for options
    st.sidebar.header("Options")
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    duplicate_threshold = st.sidebar.slider("Duplicate Detection Threshold", 70, 95, 85)
    
    # Data input
    if use_sample_data:
        df = create_sample_data()
        st.info("Using sample Salesforce data with common quality issues")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file or use sample data")
            return
    
    # Display original data
    st.header("📊 Original Data")
    st.dataframe(df, use_container_width=True)
    
    # Data quality analysis
    st.header("🔍 Data Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quality Scores")
        quality_scores = []
        for _, record in df.iterrows():
            quality_info = ai_system.calculate_data_quality_score(record.to_dict())
            quality_scores.append(quality_info)
        
        # Create quality distribution chart
        scores = [q['score'] for q in quality_scores]
        grades = [q['grade'] for q in quality_scores]
        
        fig = px.histogram(x=grades, title="Data Quality Grade Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        avg_score = np.mean(scores)
        st.metric("Average Quality Score", f"{avg_score:.1f}%")
    
    with col2:
        st.subheader("Common Issues")
        all_issues = []
        for q in quality_scores:
            all_issues.extend(q['issues'])
        
        if all_issues:
            issue_counts = pd.Series(all_issues).value_counts()
            fig = px.bar(x=issue_counts.values, y=issue_counts.index, 
                        orientation='h', title="Most Common Data Issues")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No data quality issues found!")
    
    # Duplicate detection
    st.header("🔍 Duplicate Detection")
    
    with st.spinner("Analyzing duplicates using AI algorithms..."):
        duplicates = ai_system.find_duplicates(df, threshold=duplicate_threshold)
    
    if duplicates:
        st.warning(f"Found {len(duplicates)} potential duplicate groups")
        
        for i, dup_group in enumerate(duplicates):
            with st.expander(f"Duplicate Group {i+1} (Similarity: {dup_group['matches'][0]['similarity']:.1f}%)"):
                st.write("**Primary Record:**")
                st.json(dup_group['primary_record'])
                
                st.write("**Matching Records:**")
                for match in dup_group['matches']:
                    st.write(f"Similarity: {match['similarity']:.1f}%")
                    st.json(match['record'])
    else:
        st.success("No duplicates found!")
    
    # Data cleaning
    st.header("🧹 AI-Powered Data Cleaning")
    
    if st.button("Clean Data with AI", type="primary"):
        with st.spinner("Applying AI-powered data standardization..."):
            cleaned_df = ai_system.clean_dataset(df)
        
        st.success("Data cleaning completed!")
        
        # Show before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before Cleaning")
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("After AI Cleaning")
            st.dataframe(cleaned_df, use_container_width=True)
        
        # Calculate improvement metrics
        original_scores = [ai_system.calculate_data_quality_score(record.to_dict())['score'] 
                          for _, record in df.iterrows()]
        cleaned_scores = [ai_system.calculate_data_quality_score(record.to_dict())['score'] 
                         for _, record in cleaned_df.iterrows()]
        
        improvement = np.mean(cleaned_scores) - np.mean(original_scores)
        
        st.metric(
            "Quality Improvement", 
            f"+{improvement:.1f}%",
            delta=f"{improvement:.1f}%"
        )
        
        # Download cleaned data
        csv = cleaned_df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data",
            data=csv,
            file_name=f"cleaned_salesforce_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Implementation guide
    st.header("🚀 Implementation Guide")
    
    with st.expander("How to integrate with Salesforce"):
        st.markdown("""
        ### Integration Steps:
        
        1. **API Setup**: Configure Salesforce REST API credentials
        2. **Data Export**: Export your Salesforce data using SOQL queries
        3. **AI Processing**: Run this tool on your exported data
        4. **Data Import**: Import cleaned data back to Salesforce
        5. **Automation**: Set up scheduled data quality checks
        
        ### Required Salesforce Permissions:
        - API Enabled
        - Modify All Data (for updates)
        - View All Data (for analysis)
        
        ### Recommended Frequency:
        - Daily: For high-volume orgs
        - Weekly: For medium-volume orgs
        - Monthly: For low-volume orgs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Built with**: Python, Streamlit, scikit-learn, fuzzywuzzy, pandas
    
    **Addresses Salesforce Pain Points**: Data quality issues, duplicate records, 
    inconsistent data entry, automation failures, manual cleanup time
    """)

if __name__ == "__main__":
    main()