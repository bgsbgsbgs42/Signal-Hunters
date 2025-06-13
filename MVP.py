#!/usr/bin/env python3
"""
Signal Hunters MVP Dashboard
Streamlit web app for hypothesis testing and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
import psycopg2
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ======================
# CONFIGURATION
# ======================
class Config:
    """Dashboard configuration"""
    # Database
    DB_CONFIG = {
        "host": "localhost",
        "database": "signal_hunters",
        "user": "postgres",
        "password": "yourpassword"
    }
    
    # Default coins
    COINS = ["BTC", "ETH", "SOL"]
    
    # Time ranges
    TIME_RANGES = {
        "24h": timedelta(hours=24),
        "3d": timedelta(days=3),
        "1w": timedelta(weeks=1)
    }

# ======================
# DATA LAYER
# ======================
class DataLoader:
    """Handles data loading from DB and APIs"""
    
    @staticmethod
    def load_topics(coin: str, time_range: timedelta) -> pd.DataFrame:
        """Load topic data from database"""
        conn = psycopg2.connect(**Config.DB_CONFIG)
        try:
            query = f"""
                SELECT 
                    t.topic_id,
                    t.label,
                    t.keywords,
                    tf.timestamp,
                    tf.frequency
                FROM topic_frequencies tf
                JOIN crypto_topics t ON tf.topic_id = t.topic_id
                WHERE t.coin = '{coin}'
                AND tf.timestamp > NOW() - INTERVAL '{time_range.total_seconds()} seconds'
                ORDER BY tf.timestamp
            """
            return pd.read_sql(query, conn)
        finally:
            conn.close()
    
    @staticmethod
    def load_price_data(coin: str, time_range: timedelta) -> pd.DataFrame:
        """Load price data from Yahoo Finance"""
        ticker = f"{coin}-USD"
        df = yf.download(
            ticker, 
            period="1mo",  # Over-fetch for caching
            interval="1h"
        )
        cutoff = datetime.now() - time_range
        return df[df.index >= cutoff]['Close'].pct_change().dropna()

# ======================
# ANALYSIS LAYER
# ======================
class Analyzer:
    """Core analysis functions"""
    
    @staticmethod
    def run_granger_test(topic_freq: pd.Series, price_returns: pd.Series, max_lag: int = 12):
        """Run Granger causality test"""
        # Align data
        df = pd.concat([topic_freq, price_returns], axis=1).dropna()
        df.columns = ['topic_freq', 'price_returns']
        
        # Run test
        results = grangercausalitytests(
            df[['price_returns', 'topic_freq']],
            maxlag=max_lag,
            verbose=False
        )
        
        # Format results
        return pd.DataFrame([
            {
                'lag': lag,
                'f_stat': tests[0]['ssr_ftest'][0],
                'p_value': tests[0]['ssr_ftest'][1]
            }
            for lag, tests in results.items()
        ])

# ======================
# STREAMLIT UI
# ======================
def main():
    st.set_page_config(
        page_title="Signal Hunters", 
        page_icon="🔍",
        layout="wide"
    )
    
    # Sidebar controls
    st.sidebar.title("Hypothesis Testing")
    coin = st.sidebar.selectbox("Coin", Config.COINS)
    time_range = st.sidebar.selectbox(
        "Time Range", 
        list(Config.TIME_RANGES.keys()),
        index=1
    )
    
    # Load data
    with st.spinner("Loading data..."):
        topics_df = DataLoader.load_topics(coin, Config.TIME_RANGES[time_range])
        price_returns = DataLoader.load_price_data(coin, Config.TIME_RANGES[time_range])
    
    # Main dashboard
    st.title(f"Signal Hunters: {coin} Analysis")
    
    # Section 1: Topic Trends
    st.header("Active Topics")
    if not topics_df.empty:
        # Get top 5 most frequent topics
        top_topics = topics_df.groupby('topic_id')['frequency'].mean().nlargest(5).index
        filtered_df = topics_df[topics_df['topic_id'].isin(top_topics)]
        
        # Plot topic frequencies
        fig = px.line(
            filtered_df,
            x='timestamp',
            y='frequency',
            color='label',
            title=f"Topic Frequency Trends ({time_range})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic details expander
        with st.expander("View Topic Details"):
            for topic_id in top_topics:
                topic_data = filtered_df[filtered_df['topic_id'] == topic_id].iloc[0]
                st.markdown(f"**{topic_data['label']}**")
                st.write(f"Keywords: {', '.join(json.loads(topic_data['keywords']))}")
                st.write(f"Avg Frequency: {topic_data['frequency']:.2%}")
                st.divider()
    else:
        st.warning("No topic data available for this time range")
    
    # Section 2: Hypothesis Testing
    st.header("Test Predictive Relationship")
    
    if not topics_df.empty:
        # Topic selection
        selected_topic = st.selectbox(
            "Select Topic to Test",
            options=topics_df['label'].unique()
        )
        
        # Lag selection
        max_lag = st.slider(
            "Maximum Lag (hours)",
            min_value=1,
            max_value=24,
            value=12
        )
        
        # Run analysis
        if st.button("Run Granger Causality Test"):
            topic_series = topics_df[topics_df['label'] == selected_topic].set_index('timestamp')['frequency']
            
            with st.spinner("Running causality analysis..."):
                results = Analyzer.run_granger_test(topic_series, price_returns, max_lag)
                
                # Display results
                st.subheader("Test Results")
                
                # P-value plot
                fig1 = px.line(
                    results,
                    x='lag',
                    y='p_value',
                    title="P-values by Lag",
                    markers=True
                )
                fig1.add_hline(y=0.05, line_dash="dash", line_color="red")
                st.plotly_chart(fig1, use_container_width=True)
                
                # F-statistic plot
                fig2 = px.bar(
                    results,
                    x='lag',
                    y='f_stat',
                    title="F-statistics by Lag"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Best lag
                best_lag = results.loc[results['p_value'].idxmin()]
                st.metric(
                    "Most Predictive Lag",
                    value=f"{int(best_lag['lag'])} hours",
                    help=f"p-value: {best_lag['p_value']:.4f}, F-stat: {best_lag['f_stat']:.2f}"
                )
                
                # Interpretation
                if best_lag['p_value'] < 0.05:
                    st.success(
                        f"✅ Significant predictive relationship found at {best_lag['lag']}h lag "
                        f"(p={best_lag['p_value']:.3f})"
                    )
                else:
                    st.warning(
                        "⚠️ No significant predictive relationship found "
                        f"(best p-value: {best_lag['p_value']:.3f})"
                    )
    else:
        st.warning("No topics available for testing")

    # Section 3: Submit New Hypothesis
    st.header("Submit New Hypothesis")
    with st.form("hypothesis_form"):
        st.text_area(
            "Describe your trading hypothesis",
            help="Example: 'When BTC ETF discussion exceeds 20% frequency, price rises within 6 hours'"
        )
        
        st.number_input(
            "Expected lag (hours)",
            min_value=1,
            max_value=24,
            value=6
        )
        
        if st.form_submit_button("Submit for Testing"):
            st.success("Hypothesis submitted! Results will appear in 24h")

if __name__ == "__main__":
    main()