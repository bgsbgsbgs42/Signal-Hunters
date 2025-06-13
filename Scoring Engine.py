#!/usr/bin/env python3
"""
Signal Hunters Scoring Engine
Quantifies signal quality based on lag, correlation, and novelty metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import pearsonr, zscore
from typing import Dict, Tuple
import psycopg2
from psycopg2.extras import execute_values

class SignalScorer:
    """Quantitative scoring of trading signals"""
    
    def __init__(self):
        # Scoring weights (sum to 1.0)
        self.weights = {
            'lag_score': 0.4,     # Shorter lags preferred
            'corr_score': 0.3,    # Higher correlation better
            'novelty_score': 0.3  # Newer signals rewarded
        }
        
        # Database connection
        self.conn = psycopg2.connect(
            host="localhost",
            database="signal_hunters",
            user="postgres",
            password="yourpassword"
        )
    
    def calculate_composite_score(self, 
                                lag: int, 
                                correlation: float, 
                                first_observed: datetime) -> float:
        """
        Computes overall signal score (0-100 scale)
        
        Args:
            lag: Optimal predictive lag in hours
            correlation: Pearson r with price
            first_observed: When signal was first detected
        """
        # Component scores (0-1 scale)
        lag_s = self._calculate_lag_score(lag)
        corr_s = self._calculate_correlation_score(correlation)
        novelty_s = self._calculate_novelty_score(first_observed)
        
        # Weighted composite
        composite = (
            self.weights['lag_score'] * lag_s +
            self.weights['corr_score'] * corr_s +
            self.weights['novelty_score'] * novelty_s
        )
        
        return round(composite * 100, 2)
    
    def _calculate_lag_score(self, lag: int) -> float:
        """Convert lag hours to score (shorter = better)"""
        return max(0, 1 - (lag / 24))  # Linear decay over 24h
    
    def _calculate_correlation_score(self, corr: float) -> float:
        """Convert correlation to score (absolute value)"""
        return min(1.0, abs(corr) * 2)  # 0.5 correlation → 1.0 score
    
    def _calculate_novelty_score(self, first_seen: datetime) -> float:
        """Reward newer signals with exponential decay"""
        days_old = (datetime.now() - first_seen).total_seconds() / 86400
        return np.exp(-days_old / 7)  # Half-life of 7 days
    
    def score_existing_signals(self) -> pd.DataFrame:
        """Score all signals in database"""
        with self.conn.cursor() as cur:
            # Get all test results
            cur.execute("""
                SELECT 
                    s.signal_id,
                    s.hypothesis,
                    s.first_observed,
                    r.optimal_lag,
                    r.p_value,
                    r.correlation
                FROM user_signals s
                JOIN signal_results r ON s.signal_id = r.signal_id
                WHERE r.p_value < 0.05
            """)
            signals = cur.fetchall()
        
        # Calculate scores
        scored_signals = []
        for sig in signals:
            score = self.calculate_composite_score(
                lag=sig[3],
                correlation=sig[5],
                first_observed=sig[2]
            )
            scored_signals.append({
                'signal_id': sig[0],
                'hypothesis': sig[1],
                'score': score,
                'lag': sig[3],
                'correlation': sig[5],
                'days_old': (datetime.now() - sig[2]).days
            })
        
        return pd.DataFrame(scored_signals)
    
    def update_leaderboard(self):
        """Refresh scores in database"""
        scored = self.score_existing_signals()
        
        with self.conn.cursor() as cur:
            # Clear old scores
            cur.execute("TRUNCATE TABLE signal_leaderboard")
            
            # Insert new scores
            execute_values(
                cur,
                """INSERT INTO signal_leaderboard 
                (signal_id, score, last_updated) VALUES %s""",
                [(
                    row['signal_id'],
                    row['score'],
                    datetime.now()
                ) for _, row in scored.iterrows()]
            )
            self.conn.commit()
        
        return scored.sort_values('score', ascending=False)

# ======================
# DATABASE SCHEMA UPGRADE
# ======================
def setup_database():
    """Ensure required tables exist"""
    conn = psycopg2.connect(
        host="localhost",
        database="signal_hunters",
        user="postgres",
        password="yourpassword"
    )
    
    try:
        with conn.cursor() as cur:
            # Results table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_results (
                    result_id SERIAL PRIMARY KEY,
                    signal_id INTEGER REFERENCES user_signals(signal_id),
                    optimal_lag INTEGER NOT NULL,
                    p_value FLOAT NOT NULL,
                    correlation FLOAT NOT NULL,
                    test_timestamp TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Leaderboard
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signal_leaderboard (
                    rank_id SERIAL PRIMARY KEY,
                    signal_id INTEGER REFERENCES user_signals(signal_id),
                    score FLOAT NOT NULL,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS leaderboard_score_idx 
                ON signal_leaderboard (score DESC);
            """)
            
            conn.commit()
    finally:
        conn.close()

# ======================
# STREAMLIT INTEGRATION
# ======================
def show_leaderboard():
    """Display in dashboard"""
    import streamlit as st
    
    scorer = SignalScorer()
    leaderboard = scorer.update_leaderboard()
    
    st.title("Signal Leaderboard")
    st.dataframe(
        leaderboard[['hypothesis', 'score', 'lag', 'correlation', 'days_old']],
        column_config={
            "score": st.column_config.ProgressColumn(
                "Score",
                help="Composite signal quality (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "lag": "Optimal Lag (h)",
            "correlation": "Correlation",
            "days_old": "Days Active"
        },
        hide_index=True,
        use_container_width=True
    )

# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    # Initialize database
    setup_database()
    
    # Create scorer instance
    scorer = SignalScorer()
    
    # Example signal evaluation
    example_score = scorer.calculate_composite_score(
        lag=6,
        correlation=0.45,
        first_observed=datetime.now() - timedelta(days=3)
    )
    print(f"Example signal score: {example_score}")
    
    # Update production leaderboard
    leaderboard = scorer.update_leaderboard()
    print("\nTop 5 Signals:")
    print(leaderboard.head(5))