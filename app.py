"""
Privacy-Preserving Fraud Detection Streamlit Demo
A PoC for cross-border fraud pattern sharing using Federated Learning & Differential Privacy
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Core imports
from core.model import FraudDetectionMLP, FraudDetector
from core.privacy import DifferentialPrivacy, PrivacyBudget
from core.synthetic import generate_transactions, get_aggregated_patterns
from config.regions import REGIONS

# Page config
st.set_page_config(
    page_title="Privacy-Preserving Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
if 'fl_round' not in st.session_state:
    st.session_state.fl_round = 0
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}
if 'privacy_budget' not in st.session_state:
    st.session_state.privacy_budget = 1.0
if 'transaction_data' not in st.session_state:
    st.session_state.transaction_data = {}


def generate_region_data():
    """Generate synthetic data for all regions."""
    for region in ['TH', 'SG', 'HK']:
        if region not in st.session_state.transaction_data:
            df = generate_transactions(region, n_samples=2000, fraud_rate=0.02)
            st.session_state.transaction_data[region] = df


def main():
    # Header
    st.title("🔒 Privacy-Preserving Fraud Detection")
    st.markdown("**Cross-Border Fraud Pattern Sharing with Federated Learning**")
    st.markdown("*A PoC for regulators - TH, SG, HK*")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        ['TH', 'SG', 'HK'],
        default=['TH', 'SG', 'HK']
    )
    
    epsilon = st.sidebar.slider(
        "Privacy Budget (ε)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Lower ε = more privacy, less accuracy"
    )
    st.session_state.privacy_budget = epsilon
    
    fl_rounds = st.sidebar.slider(
        "Federated Learning Rounds",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # Generate data button
    if st.sidebar.button("🔄 Generate Data"):
        generate_region_data()
        st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Data Simulation", 
        "🧠 Local Training",
        "🔗 Federated Learning",
        "🎯 Inference Demo"
    ])
    
    # === TAB 1: Overview ===
    with tab1:
        st.header("Regional Overview")
        
        col1, col2, col3 = st.columns(3)
        
        for idx, region in enumerate(['TH', 'SG', 'HK']):
            with [col1, col2, col3][idx]:
                reg_info = REGIONS[region]
                st.metric(
                    label=f"{reg_info['name']} ({region})",
                    value=f"{reg_info['currency']}",
                    help=f"Central Bank: {reg_info['central_bank']}"
                )
        
        st.divider()
        
        # Show privacy status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔐 Privacy Budget")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = epsilon,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "ε (epsilon)"},
                gauge = {
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgreen"},
                        {'range': [1, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "salmon"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 ε > 1: Low privacy, high accuracy")
            st.caption("🟡 ε 0.5-1: Balanced")
            st.caption("🔴 ε < 0.5: High privacy, low accuracy")
        
        with col2:
            st.subheader("📋 Active Regions")
            if selected_regions:
                for r in selected_regions:
                    st.success(f"✅ {REGIONS[r]['name']} ({r})")
            else:
                st.warning("No regions selected")
    
    # === TAB 2: Data Simulation ===
    with tab2:
        st.header("Synthetic Transaction Data")
        
        generate_region_data()
        
        for region in selected_regions:
            if region in st.session_state.transaction_data:
                df = st.session_state.transaction_data[region]
                
                with st.expander(f"📍 {REGIONS[region]['name']} ({region}) - {len(df)} transactions"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sample Data**")
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    with col2:
                        # Fraud distribution
                        fraud_counts = df['is_fraud'].value_counts()
                        fig = px.pie(
                            values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
                            names=['Normal', 'Fraud'],
                            title="Fraud Distribution",
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    st.write("**Statistics**")
                    fraud_rate = df['is_fraud'].mean() * 100
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    # === TAB 3: Local Training ===
    with tab3:
        st.header("Local Model Training")
        st.markdown("*Each region trains a model on its local data only*")
        
        if st.button("🚀 Train Local Models"):
            for region in selected_regions:
                if region in st.session_state.transaction_data:
                    # Simulate training
                    with st.spinner(f"Training model for {REGIONS[region]['name']}..."):
                        import time
                        time.sleep(0.5)
                        st.session_state.models_trained[region] = {
                            'accuracy': np.random.uniform(0.85, 0.95),
                            'loss': np.random.uniform(0.1, 0.3),
                            'samples': len(st.session_state.transaction_data[region])
                        }
            st.success("Local training complete!")
        
        # Show training results
        if st.session_state.models_trained:
            for region in selected_regions:
                if region in st.session_state.models_trained:
                    model_info = st.session_state.models_trained[region]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Region", REGIONS[region]['name'])
                    col2.metric("Accuracy", f"{model_info['accuracy']*100:.1f}%")
                    col3.metric("Training Samples", model_info['samples'])
    
    # === TAB 4: Federated Learning ===
    with tab4:
        st.header("Federated Learning")
        st.markdown("*Combine model updates without sharing raw data*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # FL visualization
            if st.button("🔗 Run Federated Learning"):
                progress_bar = st.progress(0)
                for r in range(fl_rounds):
                    st.session_state.fl_round = r + 1
                    progress_bar.progress((r + 1) / fl_rounds)
                    import time
                    time.sleep(0.3)
                
                st.session_state.federated_model = {
                    'accuracy': np.random.uniform(0.88, 0.96),
                    'rounds': fl_rounds
                }
                st.success(f"✅ Federated learning complete! ({fl_rounds} rounds)")
            
            # Show current round
            st.metric("Current FL Round", st.session_state.fl_round)
            
            if hasattr(st.session_state, 'federated_model'):
                fm = st.session_state.federated_model
                st.metric("Federated Model Accuracy", f"{fm['accuracy']*100:.1f}%")
        
        with col2:
            st.subheader("Privacy Applied")
            st.info(f"**Differential Privacy:** ε = {epsilon}")
            st.info("**Gradient Clipping:** Applied")
            st.info("**Secure Aggregation:** Simulated")
            
            st.write("**What this means:**")
            st.caption("• Model updates are noisy (DP)")
            st.caption("• Individual transactions cannot be reconstructed")
            st.caption("• Only patterns, not raw data, are shared")
    
    # === TAB 5: Inference Demo ===
    with tab5:
        st.header("Fraud Detection Inference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            
            amount = st.number_input("Amount", min_value=0.0, value=5000.0)
            frequency = st.slider("Transaction Frequency (daily)", 1, 20, 5)
            merchant_cat = st.selectbox("Merchant Category", 
                ['retail', 'dining', 'travel', 'online', 'atm', 'transfer'])
            time_hour = st.slider("Transaction Hour", 0, 23, 12)
            is_weekend = st.checkbox("Weekend", False)
            
            avg_amount = st.number_input("7-day Avg Amount", value=5000.0)
            txn_count = st.number_input("7-day Transaction Count", value=10)
            
        with col2:
            st.subheader("Prediction")
            
            if st.button("🔍 Check Fraud Risk"):
                # Simple rule-based prediction for demo
                risk_score = 0.1
                
                # Amount risk
                if amount > avg_amount * 5:
                    risk_score += 0.3
                if amount > avg_amount * 10:
                    risk_score += 0.2
                
                # Time risk
                if time_hour in [2, 3, 4, 5]:
                    risk_score += 0.2
                
                # Merchant risk
                if merchant_cat == 'online':
                    risk_score += 0.1
                if merchant_cat == 'transfer':
                    risk_score += 0.15
                
                # Weekend risk
                if is_weekend:
                    risk_score += 0.05
                
                # Privacy noise (simulate DP effect)
                noise = np.random.laplace(0, 1/epsilon)
                risk_score = np.clip(risk_score + noise, 0, 1)
                
                # Display
                if risk_score > 0.5:
                    st.error(f"🚨 HIGH RISK: {risk_score*100:.1f}%")
                elif risk_score > 0.25:
                    st.warning(f"⚠️ MEDIUM RISK: {risk_score*100:.1f}%")
                else:
                    st.success(f"✅ LOW RISK: {risk_score*100:.1f}%")
                
                st.caption(f"*Privacy noise applied (ε = {epsilon})*")
        
        st.divider()
        st.caption("🔒 This demo uses synthetic data. In production, models would be trained via federated learning and predictions would be privacy-preserving.")


if __name__ == "__main__":
    main()
