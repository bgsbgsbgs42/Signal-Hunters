# Signal Hunters 🔍

A real-time trading signal discovery platform that uses advanced NLP and machine learning to identify predictive relationships between social media sentiment and cryptocurrency price movements.

## 🌟 Features

- **Real-time Data Ingestion**: Streams from Reddit, Discord, Telegram, and market APIs
- **Advanced Topic Modeling**: BERTopic-powered semantic analysis of social discussions
- **Granger Causality Testing**: Statistical validation of predictive relationships
- **Anomaly Detection**: Multi-model approach using Isolation Forests and LSTM autoencoders
- **Walk-Forward Backtesting**: Rigorous hypothesis validation with performance metrics
- **Signal Scoring**: Quantitative ranking based on lag, correlation, and novelty
- **Interactive Dashboard**: Streamlit-based web interface for hypothesis testing

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Kafka Streams   │───▶│   PostgreSQL    │
│                 │    │                  │    │   TimescaleDB   │
│ • Reddit API    │    │ • Message Queue  │    │                 │
│ • Discord Bot   │    │ • Data Pipeline  │    │ • Social Posts  │
│ • Telegram API  │    │ • NLP Processing │    │ • Market Data   │
│ • Market APIs   │    │                  │    │ • Topic Models  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ML Pipeline    │    │   Signal Engine  │    │   Dashboard     │
│                 │    │                  │    │                 │
│ • Topic Models  │◀───│ • Granger Tests  │◀───│ • Streamlit UI  │
│ • Sentiment     │    │ • Backtesting    │    │ • Visualization │
│ • Anomaly Det.  │    │ • Scoring        │    │ • Hypothesis    │
│ • Forecasting   │    │ • Leaderboard    │    │   Testing       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- PostgreSQL 12+ with TimescaleDB extension
- Apache Kafka (optional for production)
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster ML training)

### API Keys Required
- Reddit API credentials
- Discord bot token
- Telegram API credentials
- Exchange API keys (Binance recommended)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/signal-hunters.git
cd signal-hunters

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Install PostgreSQL and TimescaleDB
# Ubuntu/Debian:
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install timescaledb-postgresql-12

# macOS:
brew install postgresql timescaledb

# Start PostgreSQL
sudo systemctl start postgresql

# Create database
sudo -u postgres createdb signal_hunters
sudo -u postgres psql -d signal_hunters -c "CREATE EXTENSION timescaledb;"
```

### 3. Configuration

Create a `.env` file in the project root:

```env
# Database
DB_HOST=localhost
DB_NAME=signal_hunters
DB_USER=postgres
DB_PASSWORD=yourpassword

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=SignalHunters/1.0

# Discord
DISCORD_TOKEN=your_discord_token

# Telegram
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash

# Exchange APIs
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret

# Kafka (optional)
KAFKA_BROKERS=localhost:9092
```

### 4. Initialize Database Schema

```bash
python setup_database.py
```

### 5. Start the MVP Dashboard

```bash
streamlit run MVP.py
```

The dashboard will be available at `http://localhost:8501`

## 📁 Project Structure

```
signal-hunters/
├── README.md
├── requirements.txt
├── .env.example
├── setup_database.py
├── MVP.py                          # Streamlit dashboard
├── Data Ingestion.py              # Real-time data collectors
├── Full Implementation.py         # Complete pipeline
├── Granger Causality.py          # Statistical testing
├── Scoring Engine.py             # Signal evaluation
├── Unified Real Time NLP pipeline.py  # Topic modeling
├── Anomaly Detection.py          # ML-based detection
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── processors.py
│   │   └── connectors.py
│   ├── ml/
│   │   ├── topics.py
│   │   ├── anomalies.py
│   │   └── forecasting.py
│   ├── analysis/
│   │   ├── granger.py
│   │   ├── backtesting.py
│   │   └── scoring.py
│   └── dashboard/
│       ├── app.py
│       ├── components.py
│       └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_ml.py
│   └── test_analysis.py
└── docs/
    ├── installation.md
    ├── api_reference.md
    └── user_guide.md
```

## 🔧 Implementation Phases

### Phase 1: MVP Dashboard (Start Here)
- [ ] Run `MVP.py` for basic hypothesis testing
- [ ] Test with sample data
- [ ] Verify database connectivity

### Phase 2: Data Pipeline
- [ ] Configure API credentials
- [ ] Start data ingestion (`Data Ingestion.py`)
- [ ] Monitor data flow in dashboard

### Phase 3: ML Pipeline
- [ ] Train topic models on historical data
- [ ] Implement real-time NLP processing
- [ ] Add anomaly detection

### Phase 4: Signal Engine
- [ ] Deploy Granger causality testing
- [ ] Implement backtesting engine
- [ ] Create signal scoring system

### Phase 5: Production
- [ ] Deploy with Docker
- [ ] Set up monitoring and alerts
- [ ] Scale with Kubernetes

## 📊 Usage Examples

### Testing a Hypothesis

```python
from src.analysis.granger import GrangerCausalityAnalyzer

# Example: Does Bitcoin ETF discussion predict price movements?
analyzer = GrangerCausalityAnalyzer()
result = await analyzer.analyze_topic(
    coin="BTC",
    topic_freq=etf_discussion_frequency
)

print(f"Optimal lag: {result['optimal_lag']} hours")
print(f"P-value: {result['p_value']:.4f}")
```

### Running Backtest

```python
from src.analysis.backtesting import BacktestingEngine

hypothesis = {
    'condition': 'sentiment > 0.8 AND volume > 1.5x_avg',
    'action': 'long',
    'hold_period': 6  # hours
}

engine = BacktestingEngine()
results = engine.test_hypothesis(hypothesis, historical_data)
print(f"Sharpe ratio: {results['sharpe']:.2f}")
```

## 🔍 Monitoring

The dashboard provides real-time monitoring of:

- **Data Flow**: Message rates from each source
- **Topic Trends**: Emerging discussion themes
- **Signal Performance**: Live P&L tracking
- **Anomaly Alerts**: Unusual market/social patterns

## ⚠️ Important Notes

### Data Sources
- Ensure compliance with API terms of service
- Implement rate limiting to avoid API bans
- Consider data privacy and user consent

### Trading Risks
- This is for research and educational purposes
- Past performance doesn't guarantee future results
- Always validate signals before live trading
- Use proper risk management

### Performance
- Topic models require significant memory (4GB+ recommended)
- GPU acceleration improves training speed
- Consider distributed processing for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BERTopic for semantic topic modeling
- Statsmodels for Granger causality testing
- Streamlit for rapid dashboard development
- TimescaleDB for time-series optimization


---

**⚡ Ready to discover the next alpha signal? Start with Phase 1 and let's build something amazing!**
