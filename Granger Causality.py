import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stationary import adfuller
from typing import Dict, Tuple
import yfinance as yf  # For price data
from datetime import datetime, timedelta

class GrangerCausalityAnalyzer:
    """Real-time Granger causality testing between topics and prices"""
    
    def __init__(self):
        self.max_lags = 24  # Test up to 24 hours back
        self.significance_level = 0.05
        self.coin_pairs = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'SOL': 'SOL-USD'
        }
        
    async def fetch_price_data(self, coin: str, hours: int = 48) -> pd.DataFrame:
        """Get recent hourly price data from Yahoo Finance"""
        ticker = self.coin_pairs[coin]
        end = datetime.now()
        start = end - timedelta(hours=hours)
        df = yf.download(ticker, start=start, end=end, interval='60m')
        return df['Close'].pct_change().dropna()  # Returns

    def _check_stationarity(self, series: pd.Series) -> bool:
        """Augmented Dickey-Fuller test for stationarity"""
        result = adfuller(series)
        return result[1] < 0.05  # p-value < significance level

    def prepare_causality_data(self, 
                             topic_freq: pd.Series, 
                             price_returns: pd.Series) -> pd.DataFrame:
        """
        Aligns topic frequency and price data
        Returns DataFrame with both series at hourly intervals
        """
        # Resample to hourly
        df = pd.concat([
            topic_freq.resample('H').mean().ffill(),
            price_returns.resample('H').last()
        ], axis=1)
        df.columns = ['topic_freq', 'price_returns']
        return df.dropna()

    def run_granger_test(self, 
                        topic_freq: pd.Series, 
                        price_returns: pd.Series) -> Dict[int, Dict]:
        """
        Full Granger causality testing with multiple lags
        Returns dict of test results for each lag
        """
        # Ensure stationarity
        if not self._check_stationarity(topic_freq):
            topic_freq = topic_freq.diff().dropna()
        if not self._check_stationarity(price_returns):
            price_returns = price_returns.diff().dropna()
        
        # Prepare dataset
        data = self.prepare_causality_data(topic_freq, price_returns)
        
        # Run Granger test
        return grangercausalitytests(
            data[['price_returns', 'topic_freq']], 
            maxlag=self.max_lags, 
            verbose=False
        )

    def find_optimal_lag(self, test_results: Dict) -> Tuple[int, float]:
        """Identifies the lag with strongest causal relationship"""
        significant_lags = []
        for lag, tests in test_results.items():
            p_value = tests[0]['ssr_ftest'][1]
            if p_value < self.significance_level:
                significant_lags.append((lag, p_value))
        
        if not significant_lags:
            return None, None
        
        # Return lag with lowest p-value
        return min(significant_lags, key=lambda x: x[1])

    async def analyze_topic(self, 
                          coin: str, 
                          topic_freq: pd.Series) -> Dict:
        """Full analysis pipeline for one topic"""
        # Get price data
        price_returns = await self.fetch_price_data(coin)
        
        # Run Granger test
        test_results = self.run_granger_test(topic_freq, price_returns)
        
        # Find best lag
        best_lag, best_p = self.find_optimal_lag(test_results)
        
        return {
            'coin': coin,
            'optimal_lag': best_lag,
            'p_value': best_p,
            'test_results': {
                lag: {
                    'f_stat': tests[0]['ssr_ftest'][0],
                    'p_value': tests[0]['ssr_ftest'][1]
                } 
                for lag, tests in test_results.items()
            }
        }

# ======================
# INTEGRATION WITH TOPIC PIPELINE
# ======================
class EnhancedTopicPipeline(TopicPipeline):
    """Adds causality testing to the real-time pipeline"""
    
    def __init__(self):
        super().__init__()
        self.granger = GrangerCausalityAnalyzer()
        self.causality_results = {}
        
    async def process_stream(self, data_handler):
        """Extended processing with causality checks"""
        while True:
            # Get new messages
            async with data_handler.lock:
                batch = data_handler.buffer
                data_handler.buffer = []
            
            if batch:
                df = pd.DataFrame(batch)
                df = self.clean_data(df)
                self.update_models(df)
                await self.run_causality_checks()
                self.save_to_db(df)
            
            await asyncio.sleep(3600)  # Run hourly for causality checks
            
    async def run_causality_checks(self):
        """Check all active topics for predictive power"""
        for coin, model in self.models.items():
            # Get topic frequencies over time
            topic_freq = self._get_topic_frequencies(coin, model)
            
            # Test each significant topic
            for topic_id in topic_freq.columns:
                result = await self.granger.analyze_topic(
                    coin, 
                    topic_freq[topic_id]
                )
                self.causality_results[f"{coin}_{topic_id}"] = result
                self._evaluate_causality_result(result)
    
    def _get_topic_frequencies(self, coin: str, model) -> pd.DataFrame:
        """Extracts hourly topic frequencies from model"""
        # This would come from your topic frequency tracking
        # Mock implementation:
        freq = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=48, freq='H'),
            'topic_23': np.random.rand(48) * 0.2,
            'topic_45': np.random.rand(48) * 0.1
        }).set_index('timestamp')
        return freq
    
    def _evaluate_causality_result(self, result: Dict):
        """Decide if a relationship is tradable"""
        if result['p_value'] < 0.01 and result['optimal_lag'] <= 6:
            print(f"STRONG SIGNAL: {result['coin']} topic predicts price {result['optimal_lag']}h ahead")
            self._trigger_trading_signal(result)
    
    def _trigger_trading_signal(self, result: Dict):
        """Example trading signal handler"""
        # Implement your trading logic here
        pass

# ======================
# DATABASE INTEGRATION
# ======================
class CausalityDB(DatabaseManager):
    """Extends database with causality results storage"""
    
    def __init__(self):
        super().__init__()
        self._create_causality_tables()
    
    def _create_causality_tables(self):
        """Add tables for causality results"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS granger_results (
                    id SERIAL PRIMARY KEY,
                    coin VARCHAR(10) NOT NULL,
                    topic_id INTEGER NOT NULL,
                    optimal_lag INTEGER,
                    p_value FLOAT,
                    test_results JSONB,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    FOREIGN KEY (topic_id) REFERENCES crypto_topics(topic_id)
                );
                
                CREATE INDEX IF NOT EXISTS granger_coin_time_idx 
                ON granger_results (coin, timestamp);
            """)
            self.conn.commit()
    
    def save_causality_result(self, result: Dict):
        """Store Granger test results"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO granger_results
                (coin, topic_id, optimal_lag, p_value, test_results)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                result['coin'],
                int(result['topic_id'].split('_')[1]),
                result['optimal_lag'],
                result['p_value'],
                json.dumps(result['test_results'])
            ))
            self.conn.commit()

# ======================
# MAIN EXECUTION
# ======================
async def main():
    print("=== Starting Enhanced Signal Hunters Pipeline ===")
    
    # Initialize components
    data_handler = RealTimeData()
    pipeline = EnhancedTopicPipeline()
    db = CausalityDB()
    
    # Create tasks
    tasks = [
        asyncio.create_task(data_handler.reddit_stream()),
        asyncio.create_task(data_handler.discord_stream()),
        asyncio.create_task(data_handler.telegram_stream()),
        asyncio.create_task(pipeline.process_stream(data_handler))
    ]
    
    # Run indefinitely
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())