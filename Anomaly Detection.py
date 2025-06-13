#!/usr/bin/env python3
"""
Signal Hunters Anomaly & Backtesting Module
Combines isolation forests, LSTM autoencoders, and walk-forward validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from prophet import Prophet
from neuralprophet import NeuralProphet
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

class AnomalyDetector:
    """Detects unusual activity in social+market data"""
    
    def __init__(self):
        # Models
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.01,  # 1% anomaly rate
            random_state=42
        )
        
        self.lstm_ae = self._build_lstm_autoencoder()
        
        # Thresholds
        self.volume_z_threshold = 3.0
        self.sentiment_spike_threshold = 0.5
    
    def _build_lstm_autoencoder(self, time_steps=24, n_features=5):
        """LSTM-based anomaly detector"""
        model = Sequential([
            LSTM(64, input_shape=(time_steps, n_features), return_sequences=True),
            LSTM(32, return_sequences=False),
            RepeatVector(time_steps),
            LSTM(32, return_sequences=True),
            LSTM(64, return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        model.compile(optimizer='adam', loss='mae')
        return model
    
    def detect_iso_anomalies(self, features: pd.DataFrame):
        """Isolation Forest detection"""
        return self.iso_forest.fit_predict(features)
    
    def detect_lstm_anomalies(self, time_series: np.ndarray):
        """LSTM reconstruction error detection"""
        # Normalize
        mean, std = time_series.mean(axis=0), time_series.std(axis=0)
        normalized = (time_series - mean) / std
        
        # Train (in production, pre-train on historical data)
        self.lstm_ae.fit(
            normalized, normalized,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Predict and calculate errors
        pred = self.lstm_ae.predict(normalized)
        errors = np.mean(np.abs(pred - normalized), axis=1)
        return errors > np.quantile(errors, 0.99)  # Top 1% as anomalies
    
    def detect_composite_anomalies(self, social_data: dict, market_data: dict):
        """Combined anomaly scoring"""
        # Feature engineering
        features = pd.DataFrame({
            'social_volume': social_data['post_count_24h'],
            'sentiment_change': social_data['sentiment_pct_change'],
            'liquidity': market_data['bid_ask_spread'],
            'volume_ratio': market_data['volume'] / market_data['avg_volume_30d']
        })
        
        # Z-score based detection
        z_scores = features.apply(lambda x: (x - x.mean()) / x.std())
        volume_anomaly = z_scores['social_volume'].abs() > self.volume_z_threshold
        sentiment_anomaly = features['sentiment_change'].abs() > self.sentiment_spike_threshold
        
        # Model-based detection
        iso_anomalies = self.detect_iso_anomalies(features)
        lstm_anomalies = self.detect_lstm_anomalies(features.values.reshape(1, -1, features.shape[1]))
        
        # Composite score
        return {
            'timestamp': datetime.now(),
            'is_anomaly': any([volume_anomaly, sentiment_anomaly, iso_anomalies, lstm_anomalies]),
            'scores': {
                'volume_z': float(z_scores['social_volume']),
                'sentiment_change': float(features['sentiment_change']),
                'iso_score': float(iso_anomalies.mean()),
                'lstm_error': float(lstm_anomalies.mean())
            }
        }

class ForecastEvaluator:
    """Benchmark forecasting models"""
    
    def __init__(self):
        self.prophet = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        self.neuralprophet = NeuralProphet(
            n_lags=24,
            n_forecasts=12,
            num_hidden_layers=2
        )
    
    def run_prophet_forecast(self, history: pd.DataFrame):
        """Prophet baseline forecast"""
        df = history.reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
        self.prophet.fit(df)
        future = self.prophet.make_future_dataframe(periods=12, freq='H')
        return self.prophet.predict(future)
    
    def run_neuralprophet_forecast(self, history: pd.DataFrame):
        """NeuralProphet forecast"""
        df = history.reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
        self.neuralprophet.fit(df, freq='H')
        return self.neuralprophet.make_future_dataframe(df, periods=12)

class BacktestingEngine:
    """Walk-forward hypothesis testing"""
    
    def __init__(self):
        self.metrics = {
            'sharpe': None,
            'max_dd': None,
            'win_rate': None,
            'pearson': None,
            'spearman': None,
            'granger_p': None
        }
    
    class SignalStrategy(Strategy):
        """Generic strategy from user signals"""
        def init(self):
            self.signal = self.I(self.data.signal)
        
        def next(self):
            if crossover(self.signal, 0.7):  # Example threshold
                self.buy()
            elif crossover(0.3, self.signal):
                self.sell()
    
    def run_backtest(self, hypothesis: dict, data: pd.DataFrame):
        """Execute walk-forward backtest"""
        # Prepare data
        data['signal'] = self._generate_signal_column(hypothesis, data)
        
        # Walk-forward parameters
        train_size = int(len(data) * 0.7)
        test_size = len(data) - train_size
        n_windows = 5
        
        results = []
        for i in range(n_windows):
            # Split data
            train = data.iloc[i*test_size : train_size+i*test_size]
            test = data.iloc[train_size+i*test_size : train_size+(i+1)*test_size]
            
            # Run backtest
            bt = Backtest(
                test,
                self.SignalStrategy,
                commission=0.001,
                exclusive_orders=True
            )
            stats = bt.run()
            
            # Calculate metrics
            results.append({
                'window': i,
                'sharpe': stats['Sharpe Ratio'],
                'max_dd': stats['Max. Drawdown [%]'],
                'win_rate': stats['Win Rate [%]'],
                **self._calculate_correlations(train, test)
            })
        
        return pd.DataFrame(results)
    
    def _generate_signal_column(self, hypothesis: dict, data: pd.DataFrame):
        """Convert hypothesis into trading signals"""
        # Example: "If sentiment > 0.8 and volume > 1.5x avg → 1 else 0"
        condition = (
            (data['sentiment'] > hypothesis['sentiment_thresh']) & 
            (data['volume'] > hypothesis['volume_multiplier'] * data['avg_volume'])
        )
        return np.where(condition, 1, 0)
    
    def _calculate_correlations(self, train: pd.DataFrame, test: pd.DataFrame):
        """Statistical relationship metrics"""
        # Pearson/Spearman
        pearson = pearsonr(train['signal'], train['close'].pct_change().shift(-1).dropna())
        spearman = spearmanr(train['signal'], train['close'].pct_change().shift(-1).dropna())
        
        # Granger causality
        try:
            gc_result = grangercausalitytests(
                pd.DataFrame({
                    'price': test['close'].pct_change().dropna(),
                    'signal': test['signal'].shift(1).dropna()
                }).dropna(),
                maxlag=12,
                verbose=False
            )
            p_values = [v[0]['ssr_ftest'][1] for v in gc_result.values()]
            granger_p = min(p_values)
        except:
            granger_p = 1.0
        
        return {
            'pearson': pearson[0],
            'spearman': spearman[0],
            'granger_p': granger_p
        }

# ======================
# INTEGRATION LAYER
# ======================
class SignalAnalysisPipeline:
    """End-to-end signal evaluation"""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.forecaster = ForecastEvaluator()
        self.backtester = BacktestingEngine()
    
    def evaluate_hypothesis(self, hypothesis: dict, data: pd.DataFrame):
        """Full evaluation workflow"""
        # Step 1: Anomaly detection
        anomaly_result = self.anomaly_detector.detect_composite_anomalies(
            social_data={
                'post_count_24h': data['post_count'].iloc[-24:].sum(),
                'sentiment_pct_change': data['sentiment'].pct_change().iloc[-1]
            },
            market_data={
                'bid_ask_spread': data['spread'].mean(),
                'volume': data['volume'].iloc[-1],
                'avg_volume_30d': data['volume'].rolling(30).mean().iloc[-1]
            }
        )
        
        # Step 2: Forecasting benchmarks
        prophet_fcst = self.forecaster.run_prophet_forecast(data[['timestamp', 'close']])
        neural_fcst = self.forecaster.run_neuralprophet_forecast(data[['timestamp', 'close']])
        
        # Step 3: Backtesting
        backtest_results = self.backtester.run_backtest(hypothesis, data)
        
        return {
            'anomaly': anomaly_result,
            'forecasts': {
                'prophet': prophet_fcst[['ds', 'yhat']].tail(12).to_dict('records'),
                'neuralprophet': neural_fcst.tail(12).to_dict('records')
            },
            'backtest': backtest_results.mean().to_dict(),
            'metrics': {
                'avg_sharpe': backtest_results['sharpe'].mean(),
                'max_drawdown': backtest_results['max_dd'].max(),
                'pearson_mean': backtest_results['pearson'].mean(),
                'granger_min_p': backtest_results['granger_p'].min()
            }
        }

# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    # Mock data - replace with real data pipeline
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=1000, freq='H'),
        'close': np.cumsum(np.random.randn(1000)) + 100,
        'volume': np.abs(np.random.randn(1000) * 1000) + 5000,
        'sentiment': np.random.uniform(-1, 1, 1000),
        'post_count': np.random.poisson(50, 1000),
        'spread': np.random.uniform(0.01, 0.05, 1000)
    })
    mock_data['avg_volume'] = mock_data['volume'].rolling(30).mean()
    
    # Example hypothesis
    example_hypothesis = {
        'sentiment_thresh': 0.8,
        'volume_multiplier': 1.5,
        'hold_period': 6
    }
    
    # Run analysis
    pipeline = SignalAnalysisPipeline()
    results = pipeline.evaluate_hypothesis(example_hypothesis, mock_data)
    
    print("\n=== Evaluation Results ===")
    print(f"Anomaly Detected: {results['anomaly']['is_anomaly']}")
    print(f"Avg Sharpe Ratio: {results['metrics']['avg_sharpe']:.2f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
    print(f"Pearson Correlation: {results['metrics']['pearson_mean']:.2f}")
    print(f"Granger P-value: {results['metrics']['granger_min_p']:.4f}")