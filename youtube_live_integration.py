"""
youtube_live_integration.py
--------------------------
Live YouTube data integration for TikTok learning system.
Captures live streams, comments, and new content from your YouTube channel.

Features:
- Live stream chat monitoring
- New video detection
- Comment stream processing
- Automatic transcription and embedding
- Real-time vector database updates
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Database and embeddings
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer

# Transcription
import whisper
import yt_dlp

# Rate limiting
import aiohttp
from asyncio import Semaphore

logger = logging.getLogger(__name__)

class YouTubeLiveIntegration:
    """Live YouTube data integration for TikTok learning system"""
    
    def __init__(self, api_key: str, channel_id: str, db_config: Dict[str, str]):
        self.api_key = api_key
        self.channel_id = channel_id
        self.db_config = db_config
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.whisper_model = whisper.load_model("base")
        self.semaphore = Semaphore(5)  # Rate limiting
        
        # Track processed content to avoid duplicates
        self.processed_videos = set()
        self.processed_comments = set()
        self.last_check = datetime.now()
    
    async def monitor_live_streams(self) -> List[Dict[str, Any]]:
        """Monitor live streams from your channel"""
        try:
            # Get live broadcasts
            request = self.youtube.search().list(
                part="snippet",
                channelId=self.channel_id,
                eventType="live",
                type="video",
                maxResults=10
            )
            response = request.execute()
            
            live_streams = []
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                
                # Get live chat ID
                video_response = self.youtube.videos().list(
                    part="liveStreamingDetails",
                    id=video_id
                ).execute()
                
                if video_response['items']:
                    live_details = video_response['items'][0].get('liveStreamingDetails', {})
                    live_chat_id = live_details.get('activeLiveChatId')
                    
                    if live_chat_id:
                        live_streams.append({
                            'video_id': video_id,
                            'title': item['snippet']['title'],
                            'description': item['snippet']['description'],
                            'live_chat_id': live_chat_id,
                            'started_at': live_details.get('actualStartTime'),
                            'thumbnail': item['snippet']['thumbnails']['default']['url']
                        })
            
            return live_streams
            
        except HttpError as e:
            logger.error(f"YouTube API error in monitor_live_streams: {e}")
            return []
    
    async def capture_live_chat(self, live_chat_id: str, video_id: str) -> List[Dict[str, Any]]:
        """Capture live chat messages"""
        try:
            chat_messages = []
            next_page_token = None
            
            while True:
                request = self.youtube.liveChatMessages().list(
                    liveChatId=live_chat_id,
                    part="snippet,authorDetails",
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    message_id = item['id']
                    if message_id not in self.processed_comments:
                        snippet = item['snippet']
                        author = item['authorDetails']
                        
                        chat_messages.append({
                            'message_id': message_id,
                            'video_id': video_id,
                            'author_name': author['displayName'],
                            'author_channel_id': author.get('channelId', ''),
                            'message': snippet['displayMessage'],
                            'timestamp': snippet['publishedAt'],
                            'type': snippet['type'],
                            'is_moderator': author.get('isChatModerator', False),
                            'is_owner': author.get('isChatOwner', False)
                        })
                        
                        self.processed_comments.add(message_id)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                
                # Rate limiting
                await asyncio.sleep(1)
            
            return chat_messages
            
        except HttpError as e:
            logger.error(f"YouTube API error in capture_live_chat: {e}")
            return []
    
    async def get_new_videos(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get new videos from your channel"""
        try:
            # Calculate time threshold
            published_after = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
            
            request = self.youtube.search().list(
                part="snippet",
                channelId=self.channel_id,
                publishedAfter=published_after,
                type="video",
                order="date",
                maxResults=50
            )
            response = request.execute()
            
            new_videos = []
            for item in response.get('items', []):
                video_id = item['id']['videoId']
                
                if video_id not in self.processed_videos:
                    new_videos.append({
                        'video_id': video_id,
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'published_at': item['snippet']['publishedAt'],
                        'thumbnail': item['snippet']['thumbnails']['default']['url'],
                        'channel_title': item['snippet']['channelTitle']
                    })
                    
                    self.processed_videos.add(video_id)
            
            return new_videos
            
        except HttpError as e:
            logger.error(f"YouTube API error in get_new_videos: {e}")
            return []
    
    async def get_video_comments(self, video_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get comments from a specific video"""
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_results:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    order="time",
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    comment_id = item['id']
                    if comment_id not in self.processed_comments:
                        snippet = item['snippet']['topLevelComment']['snippet']
                        
                        comments.append({
                            'comment_id': comment_id,
                            'video_id': video_id,
                            'author_name': snippet['authorDisplayName'],
                            'author_channel_id': snippet.get('authorChannelId', {}).get('value', ''),
                            'text': snippet['textDisplay'],
                            'like_count': snippet['likeCount'],
                            'published_at': snippet['publishedAt'],
                            'updated_at': snippet['updatedAt']
                        })
                        
                        self.processed_comments.add(comment_id)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return comments
            
        except HttpError as e:
            logger.error(f"YouTube API error in get_video_comments: {e}")
            return []
    
    async def transcribe_video(self, video_id: str) -> Optional[str]:
        """Transcribe video audio using Whisper"""
        try:
            # Download audio using yt-dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'temp_audio_{video_id}.%(ext)s',
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=True)
                audio_file = ydl.prepare_filename(info)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_file)
            transcript = result['text']
            
            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing video {video_id}: {e}")
            return None
    
    async def process_and_store_content(self, content_type: str, content_data: Dict[str, Any]):
        """Process content and store in vector database"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            
            # Create table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS youtube_live_content (
                    id SERIAL PRIMARY KEY,
                    content_type VARCHAR(50) NOT NULL,
                    content_id VARCHAR(100) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    video_id VARCHAR(50),
                    author_name VARCHAR(255),
                    timestamp TIMESTAMP,
                    processed_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(content_type, content_id)
                )
            """)
            
            # Create index for vector similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS youtube_live_content_embedding_idx 
                ON youtube_live_content USING ivfflat (embedding vector_cosine_ops)
            """)
            
            # Prepare content for embedding
            if content_type == "live_chat":
                text_content = content_data['message']
                metadata = {
                    'author': content_data['author_name'],
                    'is_moderator': content_data.get('is_moderator', False),
                    'is_owner': content_data.get('is_owner', False),
                    'message_type': content_data.get('type', 'text')
                }
            elif content_type == "comment":
                text_content = content_data['text']
                metadata = {
                    'author': content_data['author_name'],
                    'like_count': content_data['like_count']
                }
            elif content_type == "video_transcript":
                text_content = content_data['transcript']
                metadata = {
                    'title': content_data['title'],
                    'description': content_data['description']
                }
            else:
                return
            
            # Generate embedding
            embedding = self.embedding_model.encode([text_content])[0].tolist()
            
            # Store in database
            await conn.execute("""
                INSERT INTO youtube_live_content 
                (content_type, content_id, content, embedding, metadata, video_id, author_name, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (content_type, content_id) DO NOTHING
            """, 
                content_type,
                content_data.get('message_id') or content_data.get('comment_id') or content_data.get('video_id'),
                text_content,
                embedding,
                json.dumps(metadata),
                content_data.get('video_id'),
                content_data.get('author_name', ''),
                datetime.fromisoformat(content_data.get('timestamp', datetime.now().isoformat()).replace('Z', '+00:00'))
            )
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Error storing content: {e}")
    
    async def run_live_monitoring(self, check_interval: int = 60):
        """Main loop for live monitoring"""
        logger.info("Starting YouTube live monitoring...")
        
        while True:
            try:
                # Monitor live streams
                live_streams = await self.monitor_live_streams()
                
                for stream in live_streams:
                    logger.info(f"Processing live stream: {stream['title']}")
                    
                    # Capture live chat
                    chat_messages = await self.capture_live_chat(
                        stream['live_chat_id'], 
                        stream['video_id']
                    )
                    
                    # Process chat messages
                    for message in chat_messages:
                        await self.process_and_store_content("live_chat", message)
                
                # Check for new videos
                new_videos = await self.get_new_videos(hours_back=1)
                
                for video in new_videos:
                    logger.info(f"Processing new video: {video['title']}")
                    
                    # Get video comments
                    comments = await self.get_video_comments(video['video_id'])
                    
                    # Process comments
                    for comment in comments:
                        await self.process_and_store_content("comment", comment)
                    
                    # Transcribe video (optional, resource intensive)
                    # transcript = await self.transcribe_video(video['video_id'])
                    # if transcript:
                    #     video['transcript'] = transcript
                    #     await self.process_and_store_content("video_transcript", video)
                
                logger.info(f"Live monitoring cycle completed. Sleeping for {check_interval} seconds...")
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in live monitoring: {e}")
                await asyncio.sleep(check_interval)

async def main():
    """Example usage"""
    # Configuration
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")
    
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "tiktok_learning"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "")
    }
    
    if not YOUTUBE_API_KEY or not CHANNEL_ID:
        print("Please set YOUTUBE_API_KEY and YOUTUBE_CHANNEL_ID environment variables")
        return
    
    # Initialize and run
    youtube_integration = YouTubeLiveIntegration(YOUTUBE_API_KEY, CHANNEL_ID, db_config)
    await youtube_integration.run_live_monitoring(check_interval=30)

if __name__ == "__main__":
    asyncio.run(main())
