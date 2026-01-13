"""AI Slack Thread Intelligence Enhancement Package

This package provides AI-powered solutions for Slack's information overload problems:
- Thread summarization
- Priority classification
- Context search
- Sample data generation
"""

from .thread_summarizer import ThreadSummarizer
from .priority_classifier import PriorityClassifier
from .context_search import ContextSearch
from .sample_data import generate_sample_data

__version__ = "1.0.0"
__author__ = "AI Enhancement Team"

__all__ = [
    "ThreadSummarizer",
    "PriorityClassifier", 
    "ContextSearch",
    "generate_sample_data"
]