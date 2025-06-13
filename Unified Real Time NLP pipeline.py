#!/usr/bin/env python3
"""
Signal Hunters - Real-Time Topic Modeling Pipeline
Now with WebSocket streams for live social media data
"""

import asyncio
import websockets
import json
import re
import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
import aiohttp
import discord
from telethon import TelegramClient
import nest_asyncio
nest_asyncio.apply()  # For Jupyter compatibility

# ======================
# CONFIGURATION
# ======================
class Config:
    """Real-time pipeline configuration"""
    # Model Parameters
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MIN_TOPIC_SIZE = 15
    TOP_N_WORDS = 8
    
    # Real-Time Sources
    REDDIT_WS_URL = "wss://reddit-stream.xyz/socket"
    DISCORD_TOKEN = "your_discord_token"
    TELEGRAM_API = {
        "api_id": 12345,
        "api_hash": "your_api_hash"
    }
    
    # Database
    DB_CONFIG = {
        "host": "localhost",
        "database": "signal_hunters",
        "user": "postgres",
        "password": "yourpassword"
    }

# ======================
# REAL-TIME DATA INGESTION
# ======================
class RealTimeData:
    """Handles streaming data from multiple sources"""
    
    def __init__(self):
        self.buffer = []
        self.lock = asyncio.Lock()
        
    async def reddit_stream(self):
        """Connect to Reddit WebSocket stream"""
        async with websockets.connect(Config.REDDIT_WS_URL) as ws:
            await ws.send(json.dumps({
                "type": "subscribe",
                "subreddits": ["CryptoCurrency", "Bitcoin", "ethereum"]
            }))
            
            async for message in ws:
                data = json.loads(message)
                async with self.lock:
                    self.buffer.append({
                        "text": data["title"] + " " + data["body"],
                        "coin": self._detect_coin(data["body"]),
                        "source": "reddit",
                        "timestamp": datetime.utcnow()
                    })

    async def discord_stream(self):
        """Connect to Discord gateway"""
        client = discord.Client()
        
        @client.event
        async def on_message(message):
            if message.channel.name in ["crypto-talk", "trading-signals"]:
                async with self.lock:
                    self.buffer.append({
                        "text": message.content,
                        "coin": self._detect_coin(message.content),
                        "source": "discord",
                        "timestamp": datetime.utcnow()
                    })
        
        await client.start(Config.DISCORD_TOKEN)

    async def telegram_stream(self):
        """Monitor Telegram channels"""
        client = TelegramClient(
            'signal_hunters', 
            Config.TELEGRAM_API["api_id"], 
            Config.TELEGRAM_API["api_hash"]
        )
        
        @client.on(events.NewMessage(chats=["Cryptocom", "WhaleAlert"]))
        async def handler(event):
            async with self.lock:
                self.buffer.append({
                    "text": event.message.text,
                    "coin": self._detect_coin(event.message.text),
                    "source": "telegram",
                    "timestamp": datetime.utcnow()
                })
        
        await client.start()
        await client.run_until_disconnected()

    @staticmethod
    def _detect_coin(text):
        """Simple coin mention detector"""
        text = text.lower()
        if "bitcoin" in text or "btc" in text:
            return "BTC"
        elif "ethereum" in text or "eth" in text:
            return "ETH"
        elif "solana" in text or "sol" in text:
            return "SOL"
        return "OTHER"

# ======================
# PROCESSING PIPELINE
# ======================
class TopicPipeline:
    """Real-time topic modeling processor"""
    
    def __init__(self):
        self.models = {}
        self.embeddings = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Create coin-specific BERTopic instances"""
        for coin in ["BTC", "ETH", "SOL"]:
            self.models[coin] = BERTopic(
                embedding_model=Config.EMBEDDING_MODEL,
                umap_model=UMAP(n_components=5, random_state=42),
                hdbscan_model=HDBSCAN(min_cluster_size=Config.MIN_TOPIC_SIZE),
                top_n_words=Config.TOP_N_WORDS,
                verbose=True
            )

    async def process_stream(self, data_handler):
        """Continuous processing loop"""
        while True:
            # Get new messages
            async with data_handler.lock:
                batch = data_handler.buffer
                data_handler.buffer = []
            
            if batch:
                df = pd.DataFrame(batch)
                df = self.clean_data(df)
                self.update_models(df)
                self.save_to_db(df)
            
            await asyncio.sleep(60)  # Process every minute

    def clean_data(self, df):
        """Clean incoming real-time data"""
        tqdm.pandas(desc="Cleaning text")
        df['text'] = df['text'].progress_apply(self._clean_text)
        return df[df['text'].str.len() > 20]  # Filter short messages

    @staticmethod
    def _clean_text(text):
        """Text normalization"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\s$]', '', text)
        return text.strip()

    def update_models(self, df):
        """Incremental model updates"""
        for coin, model in self.models.items():
            coin_data = df[df['coin'] == coin]
            if len(coin_data) > 0:
                model.partial_fit(coin_data['text'].tolist())
                print(f"Updated {coin} model with {len(coin_data)} new docs")

    def save_to_db(self, df):
        """Store results in PostgreSQL"""
        conn = psycopg2.connect(**Config.DB_CONFIG)
        try:
            with conn.cursor() as cur:
                # Insert raw messages
                execute_values(
                    cur,
                    """INSERT INTO social_messages 
                    (text, coin, source, timestamp) VALUES %s""",
                    [(
                        row['text'],
                        row['coin'],
                        row['source'],
                        row['timestamp']
                    ) for _, row in df.iterrows()]
                )
                
                # Insert topic updates
                for coin, model in self.models.items():
                    topics = model.get_topic_info()
                    for _, row in topics.iterrows():
                        cur.execute("""
                            INSERT INTO topic_updates 
                            (coin, topic_id, keywords, timestamp)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT (coin, topic_id) 
                            DO UPDATE SET keywords = EXCLUDED.keywords
                        """, (
                            coin,
                            row['Topic'],
                            json.dumps(row['Name'].split(', '))
                        ))
                
                conn.commit()
        finally:
            conn.close()

# ======================
# MAIN EXECUTION
# ======================
async def main():
    print("=== Starting Real-Time Signal Hunters Pipeline ===")
    
    # Initialize components
    data_handler = RealTimeData()
    pipeline = TopicPipeline()
    
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