#!/usr/bin/env python3
"""
Salesforce AI Data Quality Enhancement Tool

This module provides AI-powered data quality improvements for Salesforce CRM,
including duplicate detection, data standardization, and automated cleansing.

Author: AI Product Enhancement System
Date: 2026-01-17
"""

import os
import re
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import phonetics
import requests
from simple_salesforce import Salesforce
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('salesforce_cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DuplicateGroup:
    """Represents a group of potential duplicate records."""
    records: List[Dict[str, Any]]
    confidence: float
    matching_fields: List[str]
    suggested_master: Dict[str, Any]

@dataclass
class DataQualityReport:
    """Represents data quality assessment results."""
    overall_score: float
    completeness: float
    accuracy: float
    consistency: float
    duplicate_rate: float
    field_scores: Dict[str, float]
    recommendations: List[str]

class SalesforceDataCleaner:
    """AI-powered Salesforce data quality enhancement tool."""
    
    def __init__(self, config_path: str = '.env'):
        """Initialize the Salesforce Data Cleaner.
        
        Args:
            config_path: Path to configuration file with Salesforce credentials
        """
        self.sf = None
        self.db_path = 'data_quality.db'
        self.ml_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Default matching rules
        self.matching_rules = {
            'Contact': {
                'required_fields': ['Email'],
                'matching_fields': {
                    'Email': {'weight': 0.4, 'exact_match': True},
                    'FirstName': {'weight': 0.2, 'fuzzy_threshold': 0.8},
                    'LastName': {'weight': 0.3, 'fuzzy_threshold': 0.9},
                    'Phone': {'weight': 0.1, 'normalize': True}
                },
                'minimum_score': 0.8
            },
            'Account': {
                'required_fields': ['Name'],
                'matching_fields': {
                    'Name': {'weight': 0.5, 'fuzzy_threshold': 0.85},
                    'Website': {'weight': 0.3, 'exact_match': True},
                    'Phone': {'weight': 0.2, 'normalize': True}
                },
                'minimum_score': 0.75
            },
            'Lead': {
                'required_fields': ['Email'],
                'matching_fields': {
                    'Email': {'weight': 0.4, 'exact_match': True},
                    'FirstName': {'weight': 0.2, 'fuzzy_threshold': 0.8},
                    'LastName': {'weight': 0.3, 'fuzzy_threshold': 0.9},
                    'Company': {'weight': 0.1, 'fuzzy_threshold': 0.8}
                },
                'minimum_score': 0.8
            }
        }
        
        self._init_database()
        self._train_ml_model()
    
    def connect(self) -> bool:
        """Connect to Salesforce using credentials from environment variables.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            username = os.getenv('SF_USERNAME')
            password = os.getenv('SF_PASSWORD')
            security_token = os.getenv('SF_SECURITY_TOKEN')
            domain = os.getenv('SF_DOMAIN', 'login')
            
            if not all([username, password, security_token]):
                logger.error("Missing Salesforce credentials in environment variables")
                return False
            
            self.sf = Salesforce(
                username=username,
                password=password,
                security_token=security_token,
                domain=domain
            )
            
            logger.info("Successfully connected to Salesforce")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Salesforce: {str(e)}")
            return False
    
    def _init_database(self):
        """Initialize SQLite database for caching and audit trail."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                records TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS merge_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                master_id TEXT NOT NULL,
                merged_ids TEXT NOT NULL,
                object_type TEXT NOT NULL,
                merged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_type TEXT NOT NULL,
                overall_score REAL NOT NULL,
                completeness REAL NOT NULL,
                accuracy REAL NOT NULL,
                consistency REAL NOT NULL,
                report_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _train_ml_model(self):
        """Train machine learning model for duplicate detection."""
        # In a real implementation, this would load training data
        # For demo purposes, we'll create a simple model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Generate synthetic training data for demonstration
        X_train = np.random.rand(1000, 10)  # 10 features
        y_train = np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 30% duplicates
        
        self.ml_model.fit(X_train, y_train)
        logger.info("ML model trained successfully")
    
    def normalize_phone(self, phone: str) -> str:
        """Normalize phone number to standard format.
        
        Args:
            phone: Raw phone number string
            
        Returns:
            str: Normalized phone number
        """
        if not phone:
            return ''
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Handle different formats
        if len(digits) == 10:
            return f"+1{digits}"  # US format
        elif len(digits) == 11 and digits.startswith('1'):
            return f"+{digits}"
        elif len(digits) > 11:
            return f"+{digits}"  # International format
        
        return phone  # Return original if can't normalize
    
    def normalize_email(self, email: str) -> str:
        """Normalize email address.
        
        Args:
            email: Raw email string
            
        Returns:
            str: Normalized email
        """
        if not email:
            return ''
        
        return email.lower().strip()
    
    def normalize_name(self, name: str) -> str:
        """Normalize person/company name.
        
        Args:
            name: Raw name string
            
        Returns:
            str: Normalized name
        """
        if not name:
            return ''
        
        # Remove extra whitespace and standardize capitalization
        normalized = ' '.join(name.strip().split())
        
        # Handle common abbreviations
        replacements = {
            ' Inc.': ' Inc',
            ' LLC.': ' LLC',
            ' Corp.': ' Corp',
            ' Ltd.': ' Ltd'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized.title()
    
    def calculate_similarity(self, record1: Dict, record2: Dict, object_type: str) -> float:
        """Calculate similarity score between two records.
        
        Args:
            record1: First record dictionary
            record2: Second record dictionary
            object_type: Salesforce object type (Contact, Account, Lead)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if object_type not in self.matching_rules:
            return 0.0
        
        rules = self.matching_rules[object_type]
        total_score = 0.0
        total_weight = 0.0
        
        for field, config in rules['matching_fields'].items():
            if field not in record1 or field not in record2:
                continue
            
            val1 = str(record1[field] or '')
            val2 = str(record2[field] or '')
            
            if not val1 or not val2:
                continue
            
            # Normalize values based on field type
            if field in ['Phone']:
                val1 = self.normalize_phone(val1)
                val2 = self.normalize_phone(val2)
            elif field in ['Email']:
                val1 = self.normalize_email(val1)
                val2 = self.normalize_email(val2)
            elif field in ['FirstName', 'LastName', 'Name', 'Company']:
                val1 = self.normalize_name(val1)
                val2 = self.normalize_name(val2)
            
            # Calculate field similarity
            if config.get('exact_match', False):
                field_score = 1.0 if val1 == val2 else 0.0
            else:
                # Use fuzzy matching
                fuzzy_score = fuzz.ratio(val1, val2) / 100.0
                threshold = config.get('fuzzy_threshold', 0.8)
                field_score = fuzzy_score if fuzzy_score >= threshold else 0.0
                
                # Add phonetic matching for names
                if field in ['FirstName', 'LastName']:
                    soundex1 = phonetics.soundex(val1)
                    soundex2 = phonetics.soundex(val2)
                    if soundex1 == soundex2:
                        field_score = max(field_score, 0.8)
            
            weight = config['weight']
            total_score += field_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def find_duplicates(self, object_type: str, threshold: float = None) -> List[DuplicateGroup]:
        """Find potential duplicate records using AI algorithms.
        
        Args:
            object_type: Salesforce object type (Contact, Account, Lead)
            threshold: Minimum similarity threshold (uses default if None)
            
        Returns:
            List[DuplicateGroup]: List of potential duplicate groups
        """
        if not self.sf:
            logger.error("Not connected to Salesforce")
            return []
        
        if object_type not in self.matching_rules:
            logger.error(f"No matching rules defined for {object_type}")
            return []
        
        threshold = threshold or self.matching_rules[object_type]['minimum_score']
        
        try:
            # Query records from Salesforce
            fields = list(self.matching_rules[object_type]['matching_fields'].keys())
            fields.append('Id')
            
            query = f"SELECT {', '.join(fields)} FROM {object_type} LIMIT 1000"
            result = self.sf.query(query)
            records = result['records']
            
            logger.info(f"Retrieved {len(records)} {object_type} records")
            
            # Find duplicates using pairwise comparison
            duplicate_groups = []
            processed_ids = set()
            
            for i, record1 in enumerate(records):
                if record1['Id'] in processed_ids:
                    continue
                
                group_records = [record1]
                matching_fields = []
                
                for j, record2 in enumerate(records[i+1:], i+1):
                    if record2['Id'] in processed_ids:
                        continue
                    
                    similarity = self.calculate_similarity(record1, record2, object_type)
                    
                    if similarity >= threshold:
                        group_records.append(record2)
                        processed_ids.add(record2['Id'])
                        
                        # Track which fields matched
                        for field in self.matching_rules[object_type]['matching_fields']:
                            if field in record1 and field in record2:
                                val1 = str(record1[field] or '')
                                val2 = str(record2[field] or '')
                                if val1 and val2 and fuzz.ratio(val1, val2) > 80:
                                    if field not in matching_fields:
                                        matching_fields.append(field)
                
                if len(group_records) > 1:
                    # Determine master record (most complete)
                    master_record = max(group_records, 
                                      key=lambda r: sum(1 for v in r.values() if v))
                    
                    duplicate_group = DuplicateGroup(
                        records=group_records,
                        confidence=min(0.99, threshold + 0.1),  # Adjust confidence
                        matching_fields=matching_fields,
                        suggested_master=master_record
                    )
                    
                    duplicate_groups.append(duplicate_group)
                    processed_ids.add(record1['Id'])
            
            logger.info(f"Found {len(duplicate_groups)} potential duplicate groups")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []
    
    def assess_data_quality(self, object_type: str) -> DataQualityReport:
        """Assess data quality for a Salesforce object.
        
        Args:
            object_type: Salesforce object type to assess
            
        Returns:
            DataQualityReport: Comprehensive data quality assessment
        """
        if not self.sf:
            logger.error("Not connected to Salesforce")
            return DataQualityReport(0, 0, 0, 0, 0, {}, [])
        
        try:
            # Get object metadata
            obj_desc = getattr(self.sf, object_type).describe()
            fields = [f['name'] for f in obj_desc['fields'] if f['type'] in 
                     ['string', 'email', 'phone', 'textarea']]
            
            # Query sample records
            query = f"SELECT {', '.join(fields[:20])} FROM {object_type} LIMIT 1000"
            result = self.sf.query(query)
            records = result['records']
            
            if not records:
                return DataQualityReport(0, 0, 0, 0, 0, {}, ["No records found"])
            
            # Calculate completeness
            field_completeness = {}
            total_completeness = 0
            
            for field in fields[:20]:  # Limit to first 20 fields
                non_empty = sum(1 for r in records if r.get(field))
                completeness = (non_empty / len(records)) * 100
                field_completeness[field] = completeness
                total_completeness += completeness
            
            avg_completeness = total_completeness / len(field_completeness)
            
            # Estimate accuracy (basic validation)
            accuracy_score = 85.0  # Placeholder - would implement real validation
            
            # Check for email format validity
            email_fields = [f for f in fields if 'email' in f.lower()]
            if email_fields:
                email_field = email_fields[0]
                valid_emails = sum(1 for r in records 
                                 if r.get(email_field) and 
                                 re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(r[email_field])))
                email_accuracy = (valid_emails / len(records)) * 100
                accuracy_score = (accuracy_score + email_accuracy) / 2
            
            # Estimate consistency
            consistency_score = 80.0  # Placeholder
            
            # Calculate duplicate rate
            duplicates = self.find_duplicates(object_type)
            duplicate_records = sum(len(group.records) for group in duplicates)
            duplicate_rate = (duplicate_records / len(records)) * 100
            
            # Overall score
            overall_score = (avg_completeness * 0.4 + 
                           accuracy_score * 0.3 + 
                           consistency_score * 0.2 + 
                           (100 - duplicate_rate) * 0.1) / 10
            
            # Generate recommendations
            recommendations = []
            if avg_completeness < 70:
                recommendations.append("Improve data completeness by making key fields required")
            if duplicate_rate > 10:
                recommendations.append("Implement duplicate prevention rules")
            if accuracy_score < 80:
                recommendations.append("Add data validation rules for better accuracy")
            
            return DataQualityReport(
                overall_score=overall_score,
                completeness=avg_completeness,
                accuracy=accuracy_score,
                consistency=consistency_score,
                duplicate_rate=duplicate_rate,
                field_scores=field_completeness,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return DataQualityReport(0, 0, 0, 0, 0, {}, [f"Error: {str(e)}"])
    
    def preview_merge(self, records: List[Dict]) -> Dict[str, Any]:
        """Preview what a merge operation would look like.
        
        Args:
            records: List of records to merge
            
        Returns:
            Dict: Preview of merged record
        """
        if not records:
            return {}
        
        # Start with the most complete record as base
        master = max(records, key=lambda r: sum(1 for v in r.values() if v))
        merged = master.copy()
        
        # Fill in missing fields from other records
        for record in records:
            for field, value in record.items():
                if not merged.get(field) and value:
                    merged[field] = value
        
        return {
            'master_id': master['Id'],
            'merged_record': merged,
            'source_records': [r['Id'] for r in records],
            'changes_made': [f for f in merged if merged[f] != master.get(f)]
        }
    
    def merge_records(self, records: List[Dict]) -> bool:
        """Merge duplicate records in Salesforce.
        
        Args:
            records: List of records to merge
            
        Returns:
            bool: True if merge successful
        """
        if not self.sf or len(records) < 2:
            return False
        
        try:
            # Get merge preview
            preview = self.preview_merge(records)
            master_id = preview['master_id']
            
            # Update master record with merged data
            update_data = {k: v for k, v in preview['merged_record'].items() 
                          if k != 'Id' and k in preview['changes_made']}
            
            if update_data:
                getattr(self.sf, records[0]['attributes']['type']).update(master_id, update_data)
            
            # Delete duplicate records
            for record in records:
                if record['Id'] != master_id:
                    getattr(self.sf, record['attributes']['type']).delete(record['Id'])
            
            # Log merge in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO merge_history (master_id, merged_ids, object_type) VALUES (?, ?, ?)",
                (master_id, ','.join(r['Id'] for r in records if r['Id'] != master_id),
                 records[0]['attributes']['type'])
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully merged {len(records)} records into {master_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging records: {str(e)}")
            return False
    
    def clean_data(self, objects: List[str], **kwargs) -> Dict[str, int]:
        """Run comprehensive data cleansing process.
        
        Args:
            objects: List of Salesforce objects to clean
            **kwargs: Additional options (auto_merge_threshold, etc.)
            
        Returns:
            Dict: Summary of cleansing results
        """
        results = {
            'total_records': 0,
            'merged_duplicates': 0,
            'standardized_fields': 0,
            'data_errors': 0
        }
        
        auto_merge_threshold = kwargs.get('auto_merge_threshold', 0.95)
        
        for obj_type in objects:
            logger.info(f"Cleaning {obj_type} data...")
            
            # Find and merge duplicates
            duplicates = self.find_duplicates(obj_type)
            
            for group in duplicates:
                results['total_records'] += len(group.records)
                
                if group.confidence >= auto_merge_threshold:
                    if self.merge_records(group.records):
                        results['merged_duplicates'] += len(group.records) - 1
        
        logger.info(f"Data cleaning completed: {results}")
        return results
    
    def set_matching_rules(self, rules: Dict[str, Dict]):
        """Set custom matching rules for duplicate detection.
        
        Args:
            rules: Dictionary of matching rules by object type
        """
        self.matching_rules.update(rules)
        logger.info("Updated matching rules")


if __name__ == "__main__":
    # Example usage
    cleaner = SalesforceDataCleaner()
    
    if cleaner.connect():
        # Find duplicates
        duplicates = cleaner.find_duplicates('Contact')
        print(f"Found {len(duplicates)} potential duplicate groups")
        
        # Assess data quality
        quality_report = cleaner.assess_data_quality('Contact')
        print(f"Data Quality Score: {quality_report.overall_score:.1f}/10")
        
        # Run automated cleaning
        results = cleaner.clean_data(['Contact'], auto_merge_threshold=0.95)
        print(f"Cleaning results: {results}")
    else:
        print("Failed to connect to Salesforce")