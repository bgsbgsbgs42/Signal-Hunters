# Signal Hunters - Implementation Guide 🚀

## Phase 1: Quick Prototype Setup (30 minutes)

Let's get a working prototype running with minimal dependencies first.

### Step 1: Create Project Structure

```bash
mkdir signal-hunters
cd signal-hunters

# Create basic structure
mkdir -p {src/{data,ml,analysis,dashboard},config,tests,logs}
touch {requirements.txt,.env.example,setup_database.py}
```

### Step 2: Create Requirements File

Create `requirements.txt`:

```txt
# Core dependencies
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0

# Database
psycopg2-binary>=2.9.0
sqlalchemy>=1.4.0

# Data sources
yfinance>=0.2.0
requests>=2.31.0

# ML/Statistics
scikit-learn>=1.3.0
statsmodels>=0.14.0
scipy>=1.11.0

# NLP (optional for MVP)
sentence-transformers>=2.2.0
bertopic>=0.15.0

# Optional production dependencies
# kafka-python>=2.0.0
# redis>=4.6.0
# tensorflow>=2.13.0
```

### Step 3: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Create Minimal Database Setup

Create `setup_database.py`:

```python
#!/usr/bin/env python3
"""
Minimal database setup for Signal Hunters MVP
"""
import psycopg2
import os
from datetime import datetime

def create_database():
    """Create database and basic tables"""
    
    # Database connection (modify as needed)
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",  # Connect to default DB first
        user="postgres",
        password=os.getenv("DB_PASSWORD", "password")
    )
    conn.autocommit = True
    
    with conn.cursor() as cur:
        # Create database if it doesn't exist
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'signal_hunters'")
        if not cur.fetchone():
            cur.execute("CREATE DATABASE signal_hunters")
            print("✓ Created signal_hunters database")
    
    conn.close()
    
    # Connect to our new database
    conn = psycopg2.connect(
        host="localhost",
        database="signal_hunters",
        user="postgres",
        password=os.getenv("DB_PASSWORD", "password")
    )
    
    with conn.cursor() as cur:
        # Create basic tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_signals (
                signal_id SERIAL PRIMARY KEY,
                hypothesis TEXT NOT NULL,
                coin VARCHAR(10) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE TABLE IF NOT EXISTS signal_results (
                result_id SERIAL PRIMARY KEY,
                signal_id INTEGER REFERENCES user_signals(signal_id),
                optimal_lag INTEGER,
                p_value FLOAT,
                correlation FLOAT,
                test_timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE TABLE IF NOT EXISTS signal_leaderboard (
                rank_id SERIAL PRIMARY KEY,
                signal_id INTEGER REFERENCES user_signals(signal_id),
                score FLOAT NOT NULL,
                last_updated TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Insert sample data
        cur.execute("""
            INSERT INTO user_signals (hypothesis, coin) VALUES
            ('High sentiment leads to price increase', 'BTC'),
            ('Volume spikes predict volatility', 'ETH'),
            ('Weekend discussions affect Monday prices', 'SOL')
            ON CONFLICT DO NOTHING;
        """)
        
        conn.commit()
        print("✓ Created database tables and sample data")
    
    conn.close()

if __name__ == "__main__":
    create_database()
```

### Step 5: Create MVP Dashboard

Create `mvp_dashboard.py`:

```python
#!/usr/bin/env python3
"""
Signal Hunters MVP Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="Signal Hunters MVP",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_sample_data(coin="BTC", days=30):
    """Load sample market data"""
    ticker = f"{coin}-USD"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get price data
    df = yf.download(ticker, start=start_date, end=end_date, interval="1h")
    df = df.reset_index()
    
    # Generate synthetic social data for demo
    np.random.seed(42)
    n_points = len(df)
    
    df['sentiment'] = np.random.uniform(-0.5, 0.5, n_points) + 0.1 * np.sin(np.arange(n_points) * 0.1)
    df['social_volume'] = np.random.poisson(50, n_points) + 20 * np.abs(df['sentiment'])
    df['price_returns'] = df['Close'].pct_change()
    
    return df

def run_simple_correlation_test(sentiment_data, price_returns, max_lag=12):
    """Simple correlation analysis with different lags"""
    results = []
    
    for lag in range(1, max_lag + 1):
        # Shift sentiment by lag hours
        lagged_sentiment = sentiment_data.shift(lag)
        
        # Calculate correlation with future returns
        valid_data = pd.concat([lagged_sentiment, price_returns], axis=1).dropna()
        
        if len(valid_data) > 10:
            corr, p_value = pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
            results.append({
                'lag': lag,
                'correlation': corr,
                'p_value': p_value
            })
    
    return pd.DataFrame(results)

def main():
    st.title("🔍 Signal Hunters MVP")
    st.markdown("**Discover predictive relationships between social sentiment and crypto prices**")
    
    # Sidebar
    st.sidebar.header("Analysis Settings")
    coin = st.sidebar.selectbox("Select Coin", ["BTC", "ETH", "SOL"])
    days_back = st.sidebar.slider("Days of History", 7, 60, 30)
    max_lag = st.sidebar.slider("Maximum Lag (hours)", 1, 24, 12)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_sample_data(coin, days_back)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{coin} Price Chart")
        price_fig = px.line(df, x='Datetime', y='Close', title=f"{coin} Price")
        st.plotly_chart(price_fig, use_container_width=True)
        
        st.subheader("Social Sentiment")
        sentiment_fig = px.line(df, x='Datetime', y='sentiment', title="Social Sentiment")
        sentiment_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    with col2:
        st.subheader("Social Volume")
        volume_fig = px.bar(df, x='Datetime', y='social_volume', title="Social Media Posts")
        st.plotly_chart(volume_fig, use_container_width=True)
        
        st.subheader("Price Returns")
        returns_fig = px.line(df, x='Datetime', y='price_returns', title="Hourly Returns")
        returns_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(returns_fig, use_container_width=True)
    
    # Analysis section
    st.header("Predictive Analysis")
    
    if st.button("Run Correlation Analysis", type="primary"):
        with st.spinner("Analyzing predictive relationships..."):
            results = run_simple_correlation_test(
                df['sentiment'], 
                df['price_returns'], 
                max_lag
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation by lag
                corr_fig = px.line(
                    results, 
                    x='lag', 
                    y='correlation',
                    title="Correlation by Lag",
                    markers=True
                )
                corr_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(corr_fig, use_container_width=True)
            
            with col2:
                # P-values
                p_fig = px.bar(
                    results, 
                    x='lag', 
                    y='p_value',
                    title="Statistical Significance (p-values)"
                )
                p_fig.add_hline(y=0.05, line_dash="dash", line_color="red")
                st.plotly_chart(p_fig, use_container_width=True)
            
            # Best result
            best_lag = results.loc[results['correlation'].abs().idxmax()]
            
            st.subheader("Best Predictive Relationship")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Optimal Lag", 
                    f"{int(best_lag['lag'])} hours",
                    help="Time delay for best prediction"
                )
            
            with col2:
                st.metric(
                    "Correlation", 
                    f"{best_lag['correlation']:.3f}",
                    help="Strength of relationship (-1 to 1)"
                )
            
            with col3:
                st.metric(
                    "P-value", 
                    f"{best_lag['p_value']:.3f}",
                    help="Statistical significance (< 0.05 is good)"
                )
            
            # Interpretation
            if best_lag['p_value'] < 0.05:
                if best_lag['correlation'] > 0:
                    st.success(f"✅ **Strong Signal Found!** Positive sentiment predicts price increases {best_lag['lag']} hours later")
                else:
                    st.success(f"✅ **Strong Signal Found!** Negative sentiment predicts price decreases {best_lag['lag']} hours later")
            else:
                st.warning("⚠️ No statistically significant relationship found. Try different parameters or more data.")
    
    # Instructions
    st.header("How It Works")
    st.markdown("""
    1. **Data Collection**: We analyze historical price data and simulated social sentiment
    2. **Lag Analysis**: Test if sentiment at time T predicts price moves at T+1, T+2, etc.
    3. **Statistical Testing**: Use correlation analysis to measure relationship strength
    4. **Signal Validation**: P-values < 0.05 indicate statistically significant relationships
    
    **Next Steps**: 
    - Add real social media data (Reddit, Twitter, Discord)
    - Implement advanced NLP for topic modeling
    - Add Granger causality testing
    - Build backtesting engine
    """)

if __name__ == "__main__":
    main()
```

### Step 6: Create Environment File

Create `.env.example`:

```env
# Database
DB_HOST=localhost
DB_NAME=signal_hunters
DB_USER=postgres
DB_PASSWORD=your_password

# APIs (for future phases)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
DISCORD_TOKEN=your_discord_token
TELEGRAM_API_ID=your_telegram_api_id
TELEGRAM_API_HASH=your_telegram_api_hash
```

## Running the Prototype

### Step 1: Install PostgreSQL (Optional for MVP)

The MVP can run without a database initially. If you want the full experience:

**On macOS:**
```bash
brew install postgresql
brew services start postgresql
```

**On Ubuntu:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**On Windows:**
Download from [PostgreSQL website](https://www.postgresql.org/download/windows/)

### Step 2: Setup Database (Optional)

```bash
# Set your database password
export DB_PASSWORD=your_password

# Run setup
python setup_database.py
```

### Step 3: Launch the Dashboard

```bash
streamlit run mvp_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## What You'll See

1. **Price Charts**: Real crypto price data from Yahoo Finance
2. **Sentiment Analysis**: Simulated social sentiment data
3. **Correlation Testing**: Statistical analysis of predictive relationships
4. **Interactive Controls**: Adjust time periods and analysis parameters

## Next Steps

Once your MVP is running:

1. **Add Real Data**: Implement Reddit/Twitter APIs
2. **Advanced NLP**: Add BERTopic for topic modeling
3. **Granger Testing**: Implement statistical causality tests
4. **Backtesting**: Add trading strategy validation
5. **Production**: Deploy with Docker/Kubernetes

## Troubleshooting

**Common Issues:**

1. **Package Installation Errors:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **Database Connection:**
   - Ensure PostgreSQL is running
   - Check username/password in `.env`
   - Try connecting manually: `psql -U postgres`

3. **Port Conflicts:**
   ```bash
   streamlit run mvp_dashboard.py --server.port 8502
   ```

4. **Memory Issues:**
   - Reduce data time periods
   - Close other applications
   - Consider cloud deployment

Ready to discover your first alpha signal? 🚀