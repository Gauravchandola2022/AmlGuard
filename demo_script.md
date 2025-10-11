# AML 360 Demo Script - 3-Minute Walkthrough

This demo script provides a step-by-step walkthrough of the AML 360 system's key features for presentations and evaluations.

## 🎯 Demo Objectives
- Demonstrate real-time transaction scoring with 5 deterministic rules
- Show ML model integration with explainability
- Highlight RAG-powered chatbot for transaction explanations
- Showcase comprehensive dashboard and analytics
- Prove batch processing capability for 100k+ transactions

## ⏱️ Quick Setup (30 seconds)

**Note**: On Replit, services start automatically. Otherwise:

### Terminal Commands
```bash
# Terminal 1: Start referential service
uv run uvicorn backend.referentials_service:app --host 0.0.0.0 --port 8001

# Terminal 2: Start main application  
uv run streamlit run ui/app_streamlit.py --server.port 5000
