# Setup and Usage Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VineshThota/new-repo.git
cd new-repo/salesforce-ai-data-quality-enhancement
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 📊 Using the Application

### Sample Data Mode
1. Keep "Use Sample Data" checked in the sidebar
2. The app loads realistic Salesforce data with common quality issues
3. Explore the different sections to see AI analysis in action

### Upload Your Own Data
1. Uncheck "Use Sample Data" in the sidebar
2. Click "Browse files" and upload a CSV file
3. Ensure your CSV has columns like: `first_name`, `last_name`, `email`, `phone`, `company`

### Key Features

#### 🔍 Data Quality Analysis
- **Quality Scores**: See letter grades (A-F) for each record
- **Common Issues**: Visual breakdown of data problems
- **Average Score**: Overall data quality metric

#### 🔍 Duplicate Detection
- Adjust "Duplicate Detection Threshold" (70-95%)
- AI algorithms find similar records using:
  - Fuzzy string matching
  - Levenshtein distance
  - Multi-field comparison
- View detailed similarity scores and matching records

#### 🧹 AI-Powered Data Cleaning
- Click "Clean Data with AI" to apply standardization
- See before/after comparison
- Download cleaned data as CSV
- Track quality improvement metrics

## 🔧 Advanced Configuration

### Customizing Duplicate Detection

Edit the `_calculate_similarity` method in `app.py` to adjust:
- Field weights (name vs email vs phone importance)
- Matching algorithms (fuzzy ratio, token sort, etc.)
- Threshold sensitivity

```python
# Example: Increase email weight in similarity calculation
if email1 and email2:
    email_score = fuzz.ratio(email1, email2)
    scores.append(email_score * 1.5)  # 1.5x weight for email
```

### Adding Custom Standardizations

Modify the `company_standardizations` dictionary:

```python
self.company_standardizations = {
    'inc': 'Inc.',
    'incorporated': 'Inc.',
    'corp': 'Corp.',
    # Add your custom mappings
    'tech': 'Technology',
    'intl': 'International'
}
```

### Custom Data Quality Rules

Add new validation rules in `calculate_data_quality_score`:

```python
# Example: Industry field validation
if 'industry' in record and record['industry']:
    valid_industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing']
    if record['industry'] in valid_industries:
        score += 5
    else:
        issues.append("Invalid industry category")
```

## 🔗 Salesforce Integration

### API Setup

1. **Create Connected App in Salesforce**
   - Setup → App Manager → New Connected App
   - Enable OAuth settings
   - Add required scopes: `api`, `refresh_token`

2. **Get API Credentials**
   - Consumer Key
   - Consumer Secret
   - Username
   - Password + Security Token

3. **Install Salesforce Python Library**
```bash
pip install simple-salesforce
```

### Data Export from Salesforce

```python
from simple_salesforce import Salesforce

# Connect to Salesforce
sf = Salesforce(
    username='your_username@company.com',
    password='your_password',
    security_token='your_security_token'
)

# Export contacts
query = "SELECT Id, FirstName, LastName, Email, Phone, Account.Name FROM Contact LIMIT 1000"
results = sf.query(query)

# Convert to DataFrame
import pandas as pd
contacts = pd.DataFrame(results['records'])
```

### Data Import Back to Salesforce

```python
# Update records after cleaning
for index, record in cleaned_df.iterrows():
    if record['Id']:  # If record has Salesforce ID
        sf.Contact.update(record['Id'], {
            'FirstName': record['first_name'],
            'LastName': record['last_name'],
            'Email': record['email'],
            'Phone': record['phone']
        })
```

## 🐳 Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t salesforce-data-quality .

# Run container
docker run -p 8501:8501 salesforce-data-quality
```

## ☁️ Cloud Deployment

### Streamlit Cloud
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy the app directly from your repository

### Heroku Deployment

1. **Create Procfile**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Deploy**
```bash
heroku create your-app-name
git push heroku main
```

### AWS/GCP/Azure
- Use container services (ECS, Cloud Run, Container Instances)
- Deploy the Docker image created above
- Configure load balancing and auto-scaling as needed

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'fuzzywuzzy'"**
```bash
pip install fuzzywuzzy python-Levenshtein
```

**"Streamlit app not loading"**
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

**"Memory error with large datasets"**
- Process data in chunks
- Increase system memory
- Use sampling for initial analysis

**"Slow duplicate detection"**
- Reduce dataset size for testing
- Increase duplicate threshold
- Consider parallel processing for production

### Performance Optimization

1. **For Large Datasets (10K+ records)**
   - Implement batch processing
   - Use multiprocessing for duplicate detection
   - Add progress bars for long operations

2. **Memory Management**
   - Process data in chunks
   - Use generators instead of loading all data
   - Clear intermediate variables

3. **Speed Improvements**
   - Cache expensive operations with `@st.cache_data`
   - Use vectorized pandas operations
   - Consider using faster similarity libraries

## 📈 Monitoring and Analytics

### Track Data Quality Metrics

```python
# Add to your workflow
metrics = {
    'timestamp': datetime.now(),
    'total_records': len(df),
    'avg_quality_score': np.mean(quality_scores),
    'duplicates_found': len(duplicates),
    'records_cleaned': len(cleaned_df)
}

# Log to file or database
with open('quality_metrics.json', 'a') as f:
    json.dump(metrics, f, default=str)
    f.write('\n')
```

### Automated Reporting

```python
# Generate weekly quality reports
def generate_quality_report(df):
    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_records': len(df),
        'quality_distribution': df['quality_grade'].value_counts().to_dict(),
        'top_issues': df['issues'].value_counts().head(5).to_dict()
    }
    return report
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Email**: Contact the maintainer for enterprise support

---

**Built with ❤️ to solve real Salesforce data quality problems**