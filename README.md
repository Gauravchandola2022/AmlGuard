# AML 360 - Anti-Money Laundering Transaction Monitoring System

AML 360 is a comprehensive, production-grade Anti-Money Laundering (AML) transaction monitoring system that implements real-world compliance requirements with advanced machine learning capabilities, explainable AI, and RAG-powered investigation tools.

## 🚀 Features

### Core Capabilities
- **Real-time Transaction Scoring**: 5 deterministic rules covering high-risk countries, suspicious keywords, large amounts, structuring patterns, and rounded amounts
- **Machine Learning Integration**: RandomForest/XGBoost models with SHAP explainability for reducing false positives
- **Vector Database & RAG**: Chroma-powered semantic search with intelligent chatbot for transaction explanations
- **Comprehensive Dashboard**: Multi-page Streamlit interface with KPIs, analytics, and investigation tools
- **Batch Processing**: Handle 100k+ transactions with efficient pandas-based processing
- **Audit Trail**: Complete logging and compliance tracking

### Rule Engine
1. **Beneficiary High-Risk Country** - Flags transactions to sanctioned/high-risk jurisdictions
2. **Suspicious Keywords** - Detects money laundering indicators in payment instructions
3. **Large Amount Detection** - Identifies transactions over $1M USD equivalent
4. **Structuring Detection** - 3-day rolling window analysis for deposit splitting patterns
5. **Rounded Amount Detection** - Flags unnaturally rounded transaction amounts

### Technical Stack
- **Backend**: FastAPI, Python, SQLite/PostgreSQL
- **Frontend**: Streamlit with multi-page architecture
- **ML/AI**: scikit-learn, XGBoost, SHAP, sentence-transformers
- **Vector DB**: ChromaDB for semantic search
- **Processing**: Pandas, NumPy for high-performance data processing

## 📋 Prerequisites

- Python 3.8+
- 8GB RAM minimum (16GB recommended for large datasets)
- 2GB free disk space

## 🔧 Installation & Setup

### 1. Environment Setup
```bash
# Clone repository (or extract files)
cd aml360

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit fastapi uvicorn pandas numpy scikit-learn xgboost
pip install shap sentence-transformers chromadb sqlite3 plotly
pip install python-multipart pytest requests pydantic
