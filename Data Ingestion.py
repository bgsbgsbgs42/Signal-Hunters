#!/usr/bin/env python3
"""
Signal Hunters Data Ingestion Pipeline
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional

# Data connectors
from praw import Reddit
from discord.ext import commands
from telethon import TelegramClient
from ccxt import binance, kraken
import websockets

# Processing
from kafka import KafkaProducer
from sentence_transformers import SentenceTransformer
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestionFramework:
    """Unified data collection and processing pipeline"""
    
    def __init__(self):
        # Initialize connectors
        self.reddit = Reddit(
            client_id='YOUR_CLIENT_ID',
            client_secret='YOUR_CLIENT_SECRET',
            user_agent='SignalHunters/1.0'
        )
        
        self.discord_bot = commands.Bot(command_prefix='!')
        self.telegram = TelegramClient('session', 'API_ID', 'API_HASH')
        
        # Market data clients
        self.binance = binance({'enableRateLimit': True})
        self.kraken = kraken()
        
        # Message queue
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # NLP models
        self.sentiment_model = SentenceTransformer('finbert-tone')
        self.topic_model = None  # Load after training
        
    async def start_reddit_stream(self):
        """Continuous Reddit post collection"""
        subreddit = self.reddit.subreddit('CryptoCurrency+Bitcoin+ethereum')
        for post in subreddit.stream.submissions():
            try:
                message = self._process_reddit_post(post)
                self._send_to_kafka('reddit_posts', message)
            except Exception as e:
                logger.error(f"Reddit error: {str(e)}")
    
    async def start_discord_listener(self):
        """Discord message processor"""
        @self.discord_bot.event
        async def on_message(message):
            if message.author.bot:
                return
            
            try:
                processed = self._process_discord_message(message)
                self._send_to_kafka('discord_messages', processed)
            except Exception as e:
                logger.error(f"Discord error: {str(e)}")
        
        await self.discord_bot.start('YOUR_DISCORD_TOKEN')
    
    async def start_telegram_monitor(self):
        """Telegram channel watcher"""
        @self.telegram.on(events.NewMessage(chats=['Cryptocom', 'WhaleAlert']))
        async def handler(event):
            try:
                message = self._process_telegram_message(event.message)
                self._send_to_kafka('telegram_messages', message)
            except Exception as e:
                logger.error(f"Telegram error: {str(e)}")
        
        await self.telegram.start()
        await self.telegram.run_until_disconnected()
    
    async def fetch_market_data(self):
        """OHLCV data collector"""
        while True:
            try:
                for symbol in ['BTC/USDT', 'ETH/USDT']:
                    data = self.binance.fetch_ohlcv(symbol, '1h', limit=1000)
                    self._send_to_kafka('market_data', {
                        'symbol': symbol,
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                await asyncio.sleep(3600)  # Hourly updates
            except Exception as e:
                logger.error(f"Market data error: {str(e)}")
                await asyncio.sleep(60)
    
    def _send_to_kafka(self, topic: str, message: Dict):
        """Reliable message queuing"""
        try:
            future = self.producer.send(topic, value=message)
            future.add_callback(
                lambda _: logger.debug(f"Sent to {topic}"),
                lambda e: logger.error(f"Failed to send to {topic}: {str(e)}")
            )
            self.producer.flush()
        except Exception as e:
            logger.critical(f"Kafka producer error: {str(e)}")
            self._fallback_storage(message)
    
    def _fallback_storage(self, message: Dict):
        """Local storage when queue fails"""
        with open('fallback.ndjson', 'a') as f:
            f.write(json.dumps(message) + '\n')
        logger.info("Used fallback storage")

class DataProcessor:
    """Real-time data transformation"""
    
    def __init__(self):
        # Initialize NLP models
        self.topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=25
        )
        
        # Initialize DB connection
        self.db = TimescaleDB()
    
    async def consume_messages(self):
        """Process messages from Kafka"""
        consumer = KafkaConsumer(
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        consumer.subscribe(['reddit_posts', 'discord_messages', 'telegram_messages'])
        
        for message in consumer:
            try:
                await self._process_message(message.value)
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
    
    async def _process_message(self, message: Dict):
        """Apply NLP and store results"""
        # Clean text
        text = self._clean_text(message['content'])
        
        # Sentiment analysis
        embedding = self.sentiment_model.encode(text)
        sentiment = self._classify_sentiment(embedding)
        
        # Topic modeling
        topics, _ = self.topic_model.transform([text])
        
        # Store in database
        await self.db.insert_social_post(
            source=message['source'],
            text=text,
            sentiment=sentiment,
            topic_id=topics[0],
            timestamp=message['timestamp']
        )

class TimescaleDB:
    """Time-series optimized storage"""
    
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="signal_hunters",
            user="postgres",
            password="yourpassword",
            host="localhost"
        )
        self._create_tables()
    
    def _create_tables(self):
        """Initialize database schema"""
        with self.conn.cursor() as cur:
            # Social media data
            cur.execute("""
                CREATE TABLE IF NOT EXISTS social_posts (
                    id BIGSERIAL PRIMARY KEY,
                    source VARCHAR(16) NOT NULL,
                    text TEXT NOT NULL,
                    sentiment FLOAT,
                    topic_id INTEGER,
                    timestamp TIMESTAMPTZ NOT NULL,
                    coin VARCHAR(8)
                );
                SELECT create_hypertable('social_posts', 'timestamp');
            """)
            
            # Market data
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol VARCHAR(16) NOT NULL,
                    open FLOAT NOT NULL,
                    high FLOAT NOT NULL,
                    low FLOAT NOT NULL,
                    close FLOAT NOT NULL,
                    volume FLOAT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL
                );
                SELECT create_hypertable('market_data', 'timestamp');
            """)
            self.conn.commit()

async def main():
    """Start all pipeline components"""
    framework = DataIngestionFramework()
    processor = DataProcessor()
    
    tasks = [
        asyncio.create_task(framework.start_reddit_stream()),
        asyncio.create_task(framework.start_discord_listener()),
        asyncio.create_task(framework.start_telegram_monitor()),
        asyncio.create_task(framework.fetch_market_data()),
        asyncio.create_task(processor.consume_messages())
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())