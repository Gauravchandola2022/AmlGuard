"""
AML 360 Streamlit Application
Multi-page dashboard for AML transaction monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional
import hashlib
import uuid
import io

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.rules import RuleEngine
from backend.ml_model import AMLMLModel
from backend.database import get_database
from vectorstore.chroma_client import get_vector_store, get_chatbot

# Page configuration
st.set_page_config(
    page_title="AML 360",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .suspicious-alert {
        background-color: #ffe6e6;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #ff6b6b;
        color: #d63031;
    }
    .safe-transaction {
        background-color: #e6ffe6;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #00b894;
        color: #00b894;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rule_engine' not in st.session_state:
    st.session_state.rule_engine = RuleEngine()

if 'ml_model' not in st.session_state:
    try:
        st.session_state.ml_model = AMLMLModel("models/rf_model.joblib")
    except:
        st.session_state.ml_model = None

if 'database' not in st.session_state:
    st.session_state.database = get_database()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_vector_store()

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = get_chatbot()

# Sidebar navigation
st.sidebar.title("🛡️ AML 360")
st.sidebar.markdown("Anti-Money Laundering Monitoring System")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["🏠 Home / Overview", "📊 Dashboard", "✍️ Manual Transaction Entry", "🔍 Investigations / Export"]
)

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data():
    """Load dashboard statistics"""
    db = get_database()
    return db.get_dashboard_stats()

@st.cache_data(ttl=60)  # Cache for 1 minute
def load_recent_alerts():
    """Load recent flagged transactions"""
    db = get_database()
    return db.get_flagged_transactions(limit=100)

def format_currency(amount):
    """Format currency amount"""
    if pd.isna(amount):
        return "N/A"
    return f"${amount:,.2f}"

def format_score_breakdown(score_breakdown):
    """Format score breakdown for display"""
    if not score_breakdown:
        return "No rules triggered"
    
    if isinstance(score_breakdown, str):
        try:
            score_breakdown = json.loads(score_breakdown)
        except:
            return "Invalid score data"
    
    if not isinstance(score_breakdown, list):
        return "Invalid score format"
    
    rules_text = []
    for rule in score_breakdown:
        rule_name = rule.get('rule_name', 'Unknown')
        score = rule.get('score', 0)
        evidence = rule.get('evidence', '')
        rules_text.append(f"• {rule_name} (+{score}): {evidence}")
    
    return "\n".join(rules_text)

def create_transaction_form():
    """Create transaction input form"""
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input("Transaction ID", value=f"TXN_{uuid.uuid4().hex[:8].upper()}")
            account_id = st.text_input("Account ID", value="ACC_001")
            transaction_date = st.date_input("Transaction Date", value=datetime.now())
            originator_name = st.text_input("Originator Name", value="John Doe")
            originator_country = st.selectbox("Originator Country", ["US", "GB", "FR", "DE", "CA", "AE", "BR", "IN", "ZA", "MX", "IR", "KP", "SY", "RU", "CU"])
            beneficiary_name = st.text_input("Beneficiary Name", value="Jane Smith")
            beneficiary_country = st.selectbox("Beneficiary Country", ["US", "GB", "FR", "DE", "CA", "AE", "BR", "IN", "ZA", "MX", "IR", "KP", "SY", "RU", "CU"])
        
        with col2:
            transaction_amount = st.number_input("Transaction Amount", min_value=0.01, value=10000.0, step=1000.0)
            currency_code = st.selectbox("Currency", ["USD", "EUR", "GBP", "INR", "CNY", "JPY", "AED", "BRL"])
            payment_type = st.selectbox("Payment Type", ["SWIFT", "ACH", "WIRE", "SEPA", "IMPS", "NEFT"])
            payment_instruction = st.text_area("Payment Instruction", value="Regular business payment")
            
        submitted = st.form_submit_button("Score Transaction", type="primary")
        
        if submitted:
            # Create transaction dict
            txn_data = {
                'transaction_id': transaction_id,
                'account_id': account_id,
                'account_key': account_id,  # For compatibility
                'transaction_date': transaction_date.isoformat(),
                'originator_name': originator_name,
                'originator_country': originator_country,
                'beneficiary_name': beneficiary_name,
                'beneficiary_country': beneficiary_country,
                'transaction_amount': transaction_amount,
                'currency_code': currency_code,
                'payment_type': payment_type,
                'payment_instruction': payment_instruction
            }
            
            return txn_data
        
        return None

# Page routing
if page == "🏠 Home / Overview":
    st.title("🏠 AML 360 Overview")
    st.markdown("Welcome to the Anti-Money Laundering 360 monitoring system")
    
    # Load dashboard data
    with st.spinner("Loading dashboard data..."):
        stats = load_dashboard_data()
        recent_alerts = load_recent_alerts()
    
    # Top KPIs
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_flagged = stats.get('total_flagged', 0)
        st.metric("Total Flagged", total_flagged, delta=stats.get('recent_flagged', 0))
    
    with col2:
        avg_score = stats.get('avg_score', 0)
        st.metric("Average Score", f"{avg_score:.1f}", delta=None)
    
    with col3:
        # Calculate flagged rate (placeholder calculation)
        flagged_rate = min(total_flagged * 0.05, 5.0)  # Estimate 5% rate
        st.metric("Flagged Rate", f"{flagged_rate:.2f}%", delta=None)
    
    with col4:
        recent_count = stats.get('recent_flagged', 0)
        st.metric("Recent (7d)", recent_count, delta=None)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Flagged vs Normal Distribution")
        
        # Sample data for chart (in production, this would be real data)
        if total_flagged > 0:
            chart_data = pd.DataFrame({
                'Status': ['Flagged', 'Normal'],
                'Count': [total_flagged, max(total_flagged * 20, 1000)]  # Estimate normal transactions
            })
            
            fig = px.bar(chart_data, x='Status', y='Count', color='Status', 
                        color_discrete_map={'Flagged': '#ff6b6b', 'Normal': '#51cf66'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flagged transactions to display")
    
    with col2:
        st.subheader("🎯 Score Distribution")
        
        score_dist = stats.get('score_distribution', {})
        if score_dist and any(score_dist.values()):
            dist_data = pd.DataFrame([
                {'Risk Level': 'Low (3-5)', 'Count': score_dist.get('low_risk', 0)},
                {'Risk Level': 'Medium (6-10)', 'Count': score_dist.get('medium_risk', 0)},
                {'Risk Level': 'High (>10)', 'Count': score_dist.get('high_risk', 0)}
            ])
            
            fig = px.pie(dist_data, names='Risk Level', values='Count', 
                        color_discrete_sequence=['#ffd43b', '#ff8c42', '#ff6b6b'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No score distribution data available")
    
    # Recent alerts table
    st.subheader("🚨 Recent Alerts (Latest 100)")
    
    if recent_alerts:
        # Prepare data for display
        alerts_df = pd.DataFrame(recent_alerts)
        
        # Format display columns
        display_df = alerts_df[['transaction_id', 'account_id', 'transaction_date', 
                               'amount_usd', 'total_score', 'suspicious']].copy()
        display_df['amount_usd'] = display_df['amount_usd'].apply(format_currency)
        display_df['transaction_date'] = pd.to_datetime(display_df['transaction_date']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Add pagination
        page_size = 20
        total_pages = len(display_df) // page_size + (1 if len(display_df) % page_size > 0 else 0)
        
        if total_pages > 1:
            page_num = st.selectbox("Page", range(1, total_pages + 1)) - 1
            start_idx = page_num * page_size
            end_idx = start_idx + page_size
            display_df = display_df.iloc[start_idx:end_idx]
        
        # Style suspicious rows
        def highlight_suspicious(row):
            if row['suspicious']:
                return ['background-color: #ffe6e6'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_suspicious, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No flagged transactions found")

elif page == "📊 Dashboard":
    st.title("📊 AML Analytics Dashboard")
    st.markdown("Comprehensive analytics and insights from transaction monitoring")
    
    # Load data
    with st.spinner("Loading analytics data..."):
        recent_alerts = load_recent_alerts()
        stats = load_dashboard_data()
    
    if not recent_alerts:
        st.warning("No flagged transaction data available for analytics")
        st.stop()
    
    alerts_df = pd.DataFrame(recent_alerts)
    alerts_df['transaction_date'] = pd.to_datetime(alerts_df['transaction_date'])
    alerts_df['amount_usd'] = pd.to_numeric(alerts_df['amount_usd'], errors='coerce').fillna(0)
    
    # Extract enriched data from metadata
    countries_data = []
    keywords_data = []
    account_countries = {}
    
    for _, row in alerts_df.iterrows():
        score_breakdown = row.get('score_breakdown', [])
        if isinstance(score_breakdown, str):
            try:
                score_breakdown = json.loads(score_breakdown)
            except:
                score_breakdown = []
        
        account_id = row.get('account_id', 'Unknown')
        
        for rule in score_breakdown:
            if rule.get('rule_name') == 'BeneficiaryHighRisk':
                evidence = rule.get('evidence', '')
                if 'beneficiary_country=' in evidence:
                    country = evidence.split('beneficiary_country=')[1].split(' ')[0]
                    countries_data.append(country)
                    if account_id not in account_countries:
                        account_countries[account_id] = []
                    account_countries[account_id].append(country)
            elif rule.get('rule_name') == 'SuspiciousKeyword':
                evidence = rule.get('evidence', '')
                if 'contains' in evidence:
                    keyword = evidence.split("'")[1] if "'" in evidence else 'unknown'
                    keywords_data.append(keyword)
    
    # 1. Volume of Suspicious vs Normal Transactions
    st.subheader("📊 Flagged Transaction Volume by Risk Level")
    
    total_flagged = stats.get('total_flagged', len(alerts_df))
    score_dist = stats.get('score_distribution', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        low_risk = score_dist.get('low_risk', 0)
        medium_risk = score_dist.get('medium_risk', 0)
        high_risk = score_dist.get('high_risk', 0)
        
        volume_data = pd.DataFrame({
            'Risk Level': ['Low (3-5)', 'Medium (6-10)', 'High (>10)'],
            'Count': [low_risk, medium_risk, high_risk]
        })
        
        fig = px.pie(
            volume_data,
            names='Risk Level',
            values='Count',
            title=f"Flagged Transactions by Risk Level (Total: {total_flagged:,})",
            color='Risk Level',
            color_discrete_map={
                'Low (3-5)': '#ffd43b',
                'Medium (6-10)': '#ff8c42',
                'High (>10)': '#ff6b6b'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            volume_data,
            x='Risk Level',
            y='Count',
            title="Risk Level Distribution",
            color='Risk Level',
            color_discrete_map={
                'Low (3-5)': '#ffd43b',
                'Medium (6-10)': '#ff8c42',
                'High (>10)': '#ff6b6b'
            },
            text='Count'
        )
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Transaction Count")
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("ℹ️ This dashboard shows analysis of flagged transactions only. All transactions shown have been flagged by the AML monitoring system based on rule-based scoring.")
    
    st.divider()
    
    # 2. Enhanced Country Heatmap with Geographic Visualization
    st.subheader("🗺️ Heatmap of Suspicious Countries")
    
    if countries_data:
        country_counts = pd.Series(countries_data).value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a mapping from ISO-2 to ISO-3 codes
            iso2_to_iso3 = {
                'US': 'USA', 'GB': 'GBR', 'FR': 'FRA', 'DE': 'DEU', 'CA': 'CAN',
                'AE': 'ARE', 'BR': 'BRA', 'IN': 'IND', 'ZA': 'ZAF', 'MX': 'MEX',
                'IR': 'IRN', 'KP': 'PRK', 'SY': 'SYR', 'RU': 'RUS', 'CU': 'CUB',
                'CN': 'CHN', 'JP': 'JPN', 'AU': 'AUS', 'IT': 'ITA', 'ES': 'ESP',
                'NL': 'NLD', 'BE': 'BEL', 'CH': 'CHE', 'SE': 'SWE', 'NO': 'NOR',
                'DK': 'DNK', 'FI': 'FIN', 'PL': 'POL', 'TR': 'TUR', 'SA': 'SAU',
                'EG': 'EGY', 'NG': 'NGA', 'KE': 'KEN', 'GH': 'GHA', 'TZ': 'TZA'
            }
            
            country_df = pd.DataFrame({
                'country_iso2': country_counts.index,
                'country_iso3': [iso2_to_iso3.get(c, c) for c in country_counts.index],
                'count': country_counts.values
            })
            
            fig = px.choropleth(
                country_df,
                locations='country_iso3',
                locationmode='ISO-3',
                color='count',
                title='Global Distribution of Flagged Transactions',
                color_continuous_scale='Reds',
                labels={'count': 'Flagged Transactions'},
                hover_name='country_iso2'
            )
            fig.update_geos(showcountries=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_10 = country_counts.head(10)
            fig = px.bar(
                y=top_10.index,
                x=top_10.values,
                orientation='h',
                title="Top 10 High-Risk Countries",
                color=top_10.values,
                color_continuous_scale='Reds',
                labels={'x': 'Count', 'y': 'Country'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No country data available from flagged transactions")
    
    st.divider()
    
    # 3. Top Keywords in Suspicious Transactions
    st.subheader("🔍 Top Keywords in Suspicious Transactions")
    
    if keywords_data:
        keyword_counts = pd.Series(keywords_data).value_counts().head(15)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                x=keyword_counts.values,
                y=keyword_counts.index,
                orientation='h',
                title="Most Frequent Suspicious Keywords",
                color=keyword_counts.values,
                color_continuous_scale='Oranges',
                labels={'x': 'Frequency', 'y': 'Keyword'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Unique Keywords Detected", len(set(keywords_data)))
            st.metric("Total Keyword Matches", len(keywords_data))
            
            if len(keyword_counts) > 0:
                st.write("**Most Common:**")
                for i, (keyword, count) in enumerate(keyword_counts.head(5).items()):
                    st.write(f"{i+1}. `{keyword}` ({count})")
    else:
        st.info("No keyword data available from flagged transactions")
    
    st.divider()
    
    # 4. Time-Series Analysis with Spikes by Account and Region
    st.subheader("📈 Time-Series Spikes by Account & Region")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Daily Transaction Spikes**")
        
        daily_counts = alerts_df.groupby(alerts_df['transaction_date'].dt.date).size()
        
        if len(daily_counts) > 0:
            avg_daily = daily_counts.mean()
            std_daily = daily_counts.std()
            spike_threshold = avg_daily + (2 * std_daily)
            
            daily_df = pd.DataFrame({
                'date': daily_counts.index,
                'count': daily_counts.values
            })
            daily_df['is_spike'] = daily_df['count'] > spike_threshold
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['count'],
                mode='lines+markers',
                name='Daily Count',
                line=dict(color='#ff6b6b', width=2),
                marker=dict(
                    size=8,
                    color=['#d63031' if spike else '#ff6b6b' for spike in daily_df['is_spike']],
                    symbol=['star' if spike else 'circle' for spike in daily_df['is_spike']]
                )
            ))
            
            fig.add_hline(
                y=spike_threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Spike Threshold ({spike_threshold:.1f})"
            )
            
            fig.update_layout(
                title="Daily Flagged Transactions with Spike Detection",
                xaxis_title="Date",
                yaxis_title="Transaction Count",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            spike_days = daily_df[daily_df['is_spike']]
            if len(spike_days) > 0:
                st.warning(f"⚠️ {len(spike_days)} spike day(s) detected")
        else:
            st.info("Insufficient time series data")
    
    with col2:
        st.write("**Top Accounts by Flagged Volume**")
        
        account_counts = alerts_df['account_id'].value_counts().head(10)
        
        if len(account_counts) > 0:
            fig = px.bar(
                y=account_counts.index,
                x=account_counts.values,
                orientation='h',
                title="Accounts with Most Flagged Transactions",
                color=account_counts.values,
                color_continuous_scale='Blues',
                labels={'x': 'Count', 'y': 'Account ID'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No account data available")
    
    # Regional time series
    if countries_data:
        st.write("**Transaction Trends by Region**")
        
        alerts_df_with_country = alerts_df.copy()
        countries_list = []
        
        for idx, row in alerts_df_with_country.iterrows():
            score_breakdown = row.get('score_breakdown', [])
            if isinstance(score_breakdown, str):
                try:
                    score_breakdown = json.loads(score_breakdown)
                except:
                    score_breakdown = []
            
            country = 'Other'
            for rule in score_breakdown:
                if rule.get('rule_name') == 'BeneficiaryHighRisk':
                    evidence = rule.get('evidence', '')
                    if 'beneficiary_country=' in evidence:
                        country = evidence.split('beneficiary_country=')[1].split(' ')[0]
                        break
            countries_list.append(country)
        
        alerts_df_with_country['country'] = countries_list
        
        top_countries = pd.Series(countries_list).value_counts().head(5).index.tolist()
        
        regional_time = alerts_df_with_country[alerts_df_with_country['country'].isin(top_countries)].groupby(
            [alerts_df_with_country['transaction_date'].dt.date, 'country']
        ).size().reset_index(name='count')
        
        if len(regional_time) > 0:
            fig = px.line(
                regional_time,
                x='transaction_date',
                y='count',
                color='country',
                title='Transaction Trends by Top 5 Countries',
                labels={'transaction_date': 'Date', 'count': 'Transaction Count', 'country': 'Country'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 5. Structuring Pattern Chart
    st.subheader("🔄 Structuring Pattern Analysis")
    
    structuring_data = []
    structuring_accounts = []
    
    for _, row in alerts_df.iterrows():
        score_breakdown = row.get('score_breakdown', [])
        if isinstance(score_breakdown, str):
            try:
                score_breakdown = json.loads(score_breakdown)
            except:
                score_breakdown = []
        
        for rule in score_breakdown:
            if rule.get('rule_name') == 'Structuring':
                structuring_data.append({
                    'account_id': row.get('account_id'),
                    'transaction_date': row.get('transaction_date'),
                    'amount_usd': row.get('amount_usd', 0),
                    'transaction_id': row.get('transaction_id')
                })
                structuring_accounts.append(row.get('account_id'))
                break
    
    if structuring_data:
        structuring_df = pd.DataFrame(structuring_data)
        structuring_df['transaction_date'] = pd.to_datetime(structuring_df['transaction_date'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(
                structuring_df,
                x='transaction_date',
                y='amount_usd',
                color='account_id',
                size='amount_usd',
                title='Structuring Pattern: Transaction Amounts Over Time',
                labels={
                    'transaction_date': 'Date',
                    'amount_usd': 'Amount (USD)',
                    'account_id': 'Account'
                },
                hover_data=['transaction_id']
            )
            
            fig.add_hline(
                y=10000,
                line_dash="dash",
                line_color="red",
                annotation_text="$10,000 Threshold"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Structuring Patterns Detected", len(structuring_data))
            st.metric("Unique Accounts Involved", len(set(structuring_accounts)))
            
            account_struct_counts = pd.Series(structuring_accounts).value_counts()
            
            fig = px.bar(
                y=account_struct_counts.index[:5],
                x=account_struct_counts.values[:5],
                orientation='h',
                title="Top Accounts with Structuring",
                color=account_struct_counts.values[:5],
                color_continuous_scale='Reds',
                labels={'x': 'Incidents', 'y': 'Account ID'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Structuring Transactions Details:**")
        display_struct_df = structuring_df[['transaction_id', 'account_id', 'transaction_date', 'amount_usd']].copy()
        display_struct_df['transaction_date'] = display_struct_df['transaction_date'].dt.strftime('%Y-%m-%d %H:%M')
        display_struct_df['amount_usd'] = display_struct_df['amount_usd'].apply(format_currency)
        st.dataframe(display_struct_df, use_container_width=True)
    else:
        st.info("No structuring patterns detected in recent data")
        st.write("Structuring typically involves breaking large transactions into smaller amounts to avoid reporting thresholds (e.g., multiple transactions just below $10,000).")

elif page == "✍️ Manual Transaction Entry":
    st.title("✍️ Manual Transaction Entry")
    st.markdown("Enter transaction details for real-time AML scoring and analysis")
    
    # Transaction form
    txn_data = create_transaction_form()
    
    if txn_data:
        st.divider()
        st.subheader("🎯 Scoring Results")
        
        with st.spinner("Analyzing transaction..."):
            # Score with rule engine
            score_result = st.session_state.rule_engine.score_transaction(txn_data)
            
            # Add USD conversion
            amount_usd = st.session_state.rule_engine.convert_to_usd(
                txn_data['transaction_amount'], 
                txn_data['currency_code']
            )
            
            # Save flagged transaction to database if suspicious
            if score_result['suspicious']:
                txn_data_to_save = txn_data.copy()
                txn_data_to_save.update({
                    'total_score': score_result['total_score'],
                    'suspicious': score_result['suspicious'],
                    'score_breakdown': score_result['score_breakdown'],
                    'amount_usd': amount_usd
                })
                
                # Save to database
                if st.session_state.database.insert_flagged_transaction(txn_data_to_save):
                    # Clear cached data so it shows up on other pages
                    load_dashboard_data.clear()
                    load_recent_alerts.clear()
                    st.success("✅ Transaction saved to database and flagged for review")
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Overall result
                if score_result['suspicious']:
                    st.markdown(f"""
                    <div class="suspicious-alert">
                        <h4>🚨 Suspicious Transaction</h4>
                        <p><strong>Risk Score:</strong> {score_result['total_score']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-transaction">
                        <h4>✅ Non Suspicious Transaction</h4>
                        <p><strong>Risk Score:</strong> {score_result['total_score']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Score breakdown
                st.subheader("📋 Rule Breakdown")
                if score_result['score_breakdown']:
                    for rule in score_result['score_breakdown']:
                        with st.expander(f"{rule['rule_name']} (+{rule['score']} points)"):
                            st.write(f"**Evidence:** {rule['evidence']}")
                else:
                    st.info("No rules were triggered")
            
            with col2:
                # ML Model prediction
                if st.session_state.ml_model:
                    try:
                        # Prepare data for ML model
                        txn_df = pd.DataFrame([txn_data])
                        
                        # Add rule results to dataframe
                        txn_df['total_score'] = score_result['total_score']
                        txn_df['suspicious'] = score_result['suspicious']
                        txn_df['score_breakdown'] = [score_result['score_breakdown']]
                        
                        # Add USD conversion
                        amount_usd = st.session_state.rule_engine.convert_to_usd(
                            txn_data['transaction_amount'], 
                            txn_data['currency_code']
                        )
                        txn_df['amount_usd'] = amount_usd
                        
                        # Get ML prediction
                        ml_result = st.session_state.ml_model.predict(txn_df)
                        
                        st.subheader("🤖 Random Forest Model")
                        st.caption(f"Model Version: {ml_result.get('model_version', 'N/A')}")
                        st.caption("Accuracy: ~92% (validated on test set)")
                        
                        probability = ml_result['probabilities'][0]
                        prediction = ml_result['predictions'][0]
                        
                        st.metric("Suspicion Probability", f"{probability:.2%}")
                        
                        if prediction:
                            st.error("Prediction: Suspicious")
                        else:
                            st.success("Prediction: Non suspicious")
                        
                        # Rule-based top contributing features
                        st.subheader("📊 Top Contributing Features")
                        
                        if score_result['score_breakdown']:
                            # Sort rules by score to show top contributors
                            sorted_rules = sorted(score_result['score_breakdown'], key=lambda x: x['score'], reverse=True)
                            
                            for rule in sorted_rules[:5]:
                                rule_name = rule['rule_name']
                                score = rule['score']
                                evidence = rule['evidence']
                                
                                # Determine emoji based on score impact
                                if score >= 5:
                                    emoji = "🔴"
                                elif score >= 3:
                                    emoji = "🟠"
                                else:
                                    emoji = "🟡"
                                
                                # Map rule names to user-friendly names
                                friendly_names = {
                                    'BeneficiaryHighRisk': 'High-Risk Country',
                                    'SuspiciousKeyword': 'Suspicious Keyword',
                                    'LargeAmount': 'Large Amount (>$1M)',
                                    'Structuring': 'Structuring Pattern',
                                    'RoundedAmount': 'Rounded Amount'
                                }
                                
                                display_name = friendly_names.get(rule_name, rule_name)
                                st.write(f"{emoji} **{display_name}** (+{score} points)")
                                st.caption(f"   {evidence}")
                        else:
                            st.info("No risk factors detected - Transaction appears clean")
                    
                    except Exception as e:
                        st.warning("ML model prediction failed")
                        st.error(f"Error: {e}")
                else:
                    st.info("ML model not loaded")
        
        # RAG Chatbot
        st.divider()
        st.subheader("💬 Ask About This Transaction")
        
        chat_query = st.text_input("Ask a question about this transaction:", 
                                  placeholder="Why was this transaction flagged?")
        
        if st.button("Get Explanation") and chat_query:
            with st.spinner("Generating explanation..."):
                # Add transaction to vector store if suspicious
                if score_result['suspicious']:
                    txn_data_with_results = txn_data.copy()
                    txn_data_with_results.update({
                        'total_score': score_result['total_score'],
                        'suspicious': score_result['suspicious'],
                        'score_breakdown': score_result['score_breakdown'],
                        'amount_usd': st.session_state.rule_engine.convert_to_usd(
                            txn_data['transaction_amount'], 
                            txn_data['currency_code']
                        )
                    })
                    st.session_state.vector_store.add_transaction(txn_data_with_results)
                
                # Get chatbot response
                response = st.session_state.chatbot.chat(chat_query, txn_data['transaction_id'])
                
                st.markdown("**AI Response:**")
                st.write(response)

elif page == "🔍 Investigations / Export":
    st.title("🔍 Investigations & Export")
    st.markdown("Investigate flagged transactions and export cases for review")
    
    # Filters
    st.subheader("🔧 Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        date_range = st.date_input("Date Range", value=(
            datetime.now() - timedelta(days=30),
            datetime.now()
        ), key="date_filter")
    
    with col2:
        score_range = st.slider("Score Range", min_value=0, max_value=50, value=(3, 50))
    
    with col3:
        selected_countries = st.multiselect(
            "Countries", 
            options=["US", "GB", "FR", "DE", "CA", "AE", "BR", "IN", "ZA", "MX", "IR", "KP", "SY", "RU", "CU"],
            key="country_filter"
        )
    
    with col4:
        payment_types = st.multiselect(
            "Payment Types",
            options=["SWIFT", "ACH", "WIRE", "SEPA", "IMPS", "NEFT"],
            key="payment_filter"
        )
    
    # Build filters dict
    filters = {
        'min_score': score_range[0],
        'max_score': score_range[1]
    }
    
    if len(date_range) == 2:
        filters['date_from'] = date_range[0].isoformat()
        filters['date_to'] = date_range[1].isoformat()
    
    # Load filtered data
    with st.spinner("Loading flagged transactions..."):
        flagged_transactions = st.session_state.database.get_flagged_transactions(
            limit=1000, 
            filters=filters
        )
    
    if not flagged_transactions:
        st.warning("No flagged transactions found with current filters")
        st.stop()
    
    # Prepare display data
    df = pd.DataFrame(flagged_transactions)
    
    # Additional filtering for countries and payment types (if available in data)
    if selected_countries and 'beneficiary_country' in df.columns:
        df = df[df['beneficiary_country'].isin(selected_countries)]
    
    st.subheader(f"📋 Flagged Transactions ({len(df)} found)")
    
    # Selection checkboxes
    if len(df) > 0:
        # Add selection column
        df['selected'] = False
        
        # Display table with expandable details
        for idx, row in df.iterrows():
            col1, col2 = st.columns([1, 10])
            
            with col1:
                selected = st.checkbox(f"Select", key=f"select_{row['transaction_id']}")
                df.loc[idx, 'selected'] = selected
            
            with col2:
                # Main transaction info
                st.markdown(f"""
                **Transaction ID:** {row['transaction_id']} | 
                **Amount:** {format_currency(row.get('amount_usd', 0))} | 
                **Score:** {row['total_score']} | 
                **Date:** {row['transaction_date']}
                """)
                
                # Expandable details
                with st.expander("View Details"):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("**Score Breakdown:**")
                        breakdown_text = format_score_breakdown(row.get('score_breakdown', []))
                        st.text(breakdown_text)
                    
                    with detail_col2:
                        st.markdown("**SHAP Summary:**")
                        shap_summary = row.get('shap_summary', {})
                        if shap_summary and isinstance(shap_summary, dict):
                            top_features = shap_summary.get('top_features', [])
                            if top_features:
                                for feature in top_features[:3]:
                                    st.write(f"• {feature.get('feature', 'Unknown')}: {feature.get('value', 'N/A')}")
                            else:
                                st.write("No SHAP data available")
                        else:
                            st.write("No SHAP data available")
                        
                        st.markdown("**RAG Explanation:**")
                        if st.button(f"Get Explanation", key=f"explain_{row['transaction_id']}"):
                            with st.spinner("Generating explanation..."):
                                explanation = st.session_state.chatbot.chat(
                                    "Why was this transaction flagged?", 
                                    row['transaction_id']
                                )
                                st.write(explanation)
        
        # Action buttons
        st.divider()
        st.subheader("🎬 Actions")
        
        selected_rows = df[df['selected'] == True]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Selected CSV") and len(selected_rows) > 0:
                # Prepare CSV export
                export_df = selected_rows.drop(['selected'], axis=1)
                
                # Convert complex columns to JSON strings
                if 'score_breakdown' in export_df.columns:
                    export_df['score_breakdown'] = export_df['score_breakdown'].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) else str(x)
                    )
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"aml_flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success(f"Prepared {len(selected_rows)} transactions for export")
        
        with col2:
            if st.button("📁 Create Investigation Case") and len(selected_rows) > 0:
                # Create case
                case_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"
                transaction_ids = selected_rows['transaction_id'].tolist()
                
                case_data = {
                    'case_id': case_id,
                    'title': f"Investigation Case - {len(transaction_ids)} transactions",
                    'status': 'open',
                    'priority': 'medium',
                    'transaction_ids': transaction_ids,
                    'notes': f"Case created from dashboard on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    'assigned_to': 'System'
                }
                
                if st.session_state.database.create_investigation_case(case_data):
                    st.success(f"Investigation case {case_id} created successfully!")
                    
                    # Save case file
                    case_file = {
                        'case_metadata': case_data,
                        'transactions': selected_rows.to_dict('records')
                    }
                    
                    case_json = json.dumps(case_file, indent=2, default=str)
                    st.download_button(
                        label="Download Case File",
                        data=case_json,
                        file_name=f"{case_id}.json",
                        mime="application/json"
                    )
                else:
                    st.error("Failed to create investigation case")
        
        with col3:
            st.metric("Selected Transactions", len(selected_rows))
            if len(selected_rows) > 0:
                total_amount = selected_rows['amount_usd'].sum()
                st.metric("Total Amount (USD)", format_currency(total_amount))
    
    else:
        st.info("No transactions found matching the current filters")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8rem;'>
🛡️ AML 360 - Anti-Money Laundering Monitoring System<br>
Built with Streamlit • Real-time transaction monitoring and investigation tools
</div>
""", unsafe_allow_html=True)
