# AML 360 Demo Script - 3-Minute Walkthrough

This demo script provides a step-by-step walkthrough of the AML 360 system's key features for presentations and evaluations.

## 🎯 Demo Objectives
- Demonstrate real-time transaction scoring and rule engine
- Show ML model integration with explainability (SHAP)
- Highlight RAG-powered chatbot for investigations
- Showcase comprehensive dashboard and analytics
- Prove system can handle 100k+ transactions

## ⏱️ Quick Setup (30 seconds)

### Terminal Commands
```bash
# Terminal 1: Start referential service
uvicorn backend.referentials_service:app --reload --port 8001

# Terminal 2: Start main application  
streamlit run ui/app_streamlit.py --server.port 5000
