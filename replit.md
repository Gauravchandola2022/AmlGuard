# AML 360 - Anti-Money Laundering Transaction Monitoring System

## Overview

AML 360 is a production-grade Anti-Money Laundering (AML) transaction monitoring system that combines deterministic rule-based scoring with machine learning capabilities. The system processes financial transactions in real-time or batch mode, identifies suspicious activities using 5 core compliance rules, and provides explainable ML predictions to reduce false positives. It features a comprehensive Streamlit dashboard for investigation, a vector database for semantic search, and a RAG-powered chatbot for transaction explanations.

**Core Purpose**: Detect and flag potentially suspicious financial transactions for AML compliance teams through automated scoring, ML enhancement, and investigative tools.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application
- **Design Pattern**: Dashboard-based UI with separate pages for different workflows
- **Key Pages**: 
  - Transaction upload and batch processing
  - KPI dashboard with analytics
  - Investigation tools with drill-down capabilities
  - RAG chatbot for transaction explanations
- **Visualization**: Plotly for interactive charts and graphs
- **State Management**: Streamlit session state for user interactions

### Backend Architecture

#### Rule Engine (Deterministic Scoring)
- **Pattern**: Pipeline processor with 5 independent rule evaluators
- **Rules Implemented**:
  1. High-risk country detection (3-tier risk scoring: Level 1/2/3)
  2. Suspicious keyword matching in payment instructions
  3. Large amount detection (>$1M USD equivalent)
  4. Structuring pattern detection (3-day rolling window)
  5. Rounded amount detection (configurable trailing zero threshold)
- **Score Aggregation**: Additive scoring model with configurable thresholds
- **Currency Handling**: Real-time conversion to USD using exchange rate API

#### Machine Learning Layer
- **Algorithm**: RandomForest classifier for binary classification (suspicious/not suspicious)
- **Feature Engineering**: Extracts rule-based features, transaction metadata, and temporal patterns
- **Explainability**: SHAP (SHapley Additive exPlanations) for model interpretability
- **Training Strategy**: Time-series split for temporal validation
- **Model Persistence**: Joblib serialization with versioning
- **Class Imbalance**: Handles via class weights and synthetic data generation

#### Referential Service
- **Pattern**: Microservice architecture (FastAPI)
- **Endpoints**: 
  - `/api/exchange-rates` - Real-time currency conversion rates
  - `/api/high-risk-countries` - 3-tier country risk classification
  - `/api/suspicious-keywords` - AML keyword dictionary
- **Caching**: In-memory cache with TTL (1 hour default)
- **Versioning**: API version prefix (`/v1/`) for backwards compatibility

#### Data Processing Pipeline
- **Batch Processing**: Pandas-based vectorized operations for 100k+ transactions
- **Validation**: Comprehensive error handling with fallback defaults
- **Edge Cases**: Handles missing currencies, invalid amounts, unknown countries
- **Audit Trail**: Complete logging with PII masking at all stages

### Data Storage Solutions

#### Primary Database (SQLite)
- **Schema**: `flagged_transactions` table with transaction metadata, scores, and explanations
- **Fields**: transaction_id (unique), scores, SHAP summaries, RAG retrievals, timestamps
- **Access Pattern**: Context manager for connection pooling
- **Rationale**: SQLite chosen for simplicity, portability, and sufficient performance for moderate transaction volumes

#### Vector Database (ChromaDB)
- **Purpose**: Semantic search over flagged transactions
- **Embeddings**: Sentence transformers (all-MiniLM-L6-v2 model)
- **Collection**: Persistent storage for transaction embeddings and metadata
- **Query Pattern**: Similarity search for RAG retrieval
- **Rationale**: Enables natural language querying of historical suspicious transactions

### Security & Privacy Architecture

#### PII Masking
- **Pattern**: Multi-layer masking strategy
- **Techniques**:
  - Name masking (show first/last chars only)
  - Account number masking (last 4 digits only)
  - Email obfuscation
  - Hashed identifiers for audit trails (SHA-256)
- **Application**: Applied to logs, displays, and audit records

#### Audit Logging
- **Pattern**: Separate audit trail from application logs
- **Storage**: Date-partitioned log files with rotation
- **Content**: Transaction scoring events, model predictions, user actions
- **Format**: Structured JSON with hashed PII for compliance tracking

### Design Patterns & Rationale

#### Microservice for Referentials
- **Problem**: Centralized reference data management with caching
- **Solution**: Separate FastAPI service with versioned endpoints
- **Pros**: Independent scaling, cache isolation, easy updates
- **Cons**: Additional service to manage, network latency

#### RAG for Explainability
- **Problem**: Provide compliance officers with context for flagged transactions
- **Solution**: Vector database + semantic search + chatbot interface
- **Pros**: Natural language queries, historical context, pattern discovery
- **Cons**: Requires embedding model, storage overhead

#### Hybrid Scoring (Rules + ML)
- **Problem**: Balance precision (low false positives) with recall (catch suspicious activity)
- **Solution**: Deterministic rules provide baseline, ML reduces false positives
- **Pros**: Explainable baseline, ML enhancement, configurable thresholds
- **Cons**: Two models to maintain, feature drift management

## External Dependencies

### Third-Party Services
- **Referential Data API**: Internal FastAPI service for exchange rates, country risk levels, and keyword dictionaries
  - Endpoints: `/api/exchange-rates`, `/api/high-risk-countries`, `/api/suspicious-keywords`
  - Default fallback data available if service unavailable

### APIs & Integrations
- **Currency Exchange**: Uses referential service with static rates (could integrate live forex API)
- **Country Risk Data**: 3-tier classification system (Level 1/2/3 risk countries)
- **No external ML APIs**: All ML inference happens locally

### Databases
- **SQLite**: Primary relational database for flagged transactions and metadata
- **ChromaDB**: Vector database for embeddings and semantic search
- **Note**: System designed to work with SQLite but can be adapted to PostgreSQL for production scale

### Key Python Packages
- **Web Framework**: FastAPI (backend service), Streamlit (frontend)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn, XGBoost (optional), SHAP (explainability)
- **Vector Store**: ChromaDB, sentence-transformers (embeddings)
- **Visualization**: Plotly
- **Utilities**: joblib (model persistence), requests (API calls)

### ML Models & Data
- **Embedding Model**: all-MiniLM-L6-v2 (sentence transformers) for semantic search
- **Classification Model**: RandomForest (scikit-learn) trained on transaction features
- **Training Data**: Supports synthetic data generation for initial model training
- **Model Storage**: Joblib-serialized models with versioning in `models/` directory