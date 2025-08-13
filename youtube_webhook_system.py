"""
youtube_webhook_system.py
-------------------------
Real-time YouTube webhook system using PubSubHubbub for instant notifications.
More efficient than polling - gets notified immediately when content is published.

Features:
- Real-time notifications for new videos
- Webhook verification and security
- Automatic subscription management
- Integration with existing TikTok learning system
"""

import asyncio
import hashlib
import hmac
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import httpx
import asyncpg

from youtube_live_integration import YouTubeLiveIntegration

logger = logging.getLogger(__name__)

class YouTubeWebhookSystem:
    """Real-time YouTube webhook system using PubSubHubbub"""
    
    def __init__(self, 
                 channel_id: str, 
                 webhook_url: str, 
                 secret: str,
                 youtube_integration: YouTubeLiveIntegration):
        self.channel_id = channel_id
        self.webhook_url = webhook_url
        self.secret = secret.encode('utf-8')
        self.youtube_integration = youtube_integration
        
        # PubSubHubbub hub URL for YouTube
        self.hub_url = "https://pubsubhubbub.appspot.com/subscribe"
        self.topic_url = f"https://www.youtube.com/xml/feeds/videos.xml?channel_id={channel_id}"
        
        # Subscription management
        self.subscription_active = False
        self.lease_seconds = 86400  # 24 hours
        
    async def subscribe_to_channel(self) -> bool:
        """Subscribe to YouTube channel notifications"""
        try:
            subscription_data = {
                'hub.callback': self.webhook_url,
                'hub.topic': self.topic_url,
                'hub.verify': 'async',
                'hub.mode': 'subscribe',
                'hub.lease_seconds': str(self.lease_seconds),
                'hub.secret': self.secret.decode('utf-8')
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.hub_url,
                    data=subscription_data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                
                if response.status_code == 202:
                    logger.info(f"Subscription request sent for channel {self.channel_id}")
                    return True
                else:
                    logger.error(f"Subscription failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error subscribing to channel: {e}")
            return False
    
    async def unsubscribe_from_channel(self) -> bool:
        """Unsubscribe from YouTube channel notifications"""
        try:
            subscription_data = {
                'hub.callback': self.webhook_url,
                'hub.topic': self.topic_url,
                'hub.verify': 'async',
                'hub.mode': 'unsubscribe'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.hub_url,
                    data=subscription_data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                
                if response.status_code == 202:
                    logger.info(f"Unsubscription request sent for channel {self.channel_id}")
                    self.subscription_active = False
                    return True
                else:
                    logger.error(f"Unsubscription failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error unsubscribing from channel: {e}")
            return False
    
    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """Verify webhook signature for security"""
        if not signature.startswith('sha1='):
            return False
        
        expected_signature = hmac.new(
            self.secret,
            body,
            hashlib.sha1
        ).hexdigest()
        
        received_signature = signature[5:]  # Remove 'sha1=' prefix
        
        return hmac.compare_digest(expected_signature, received_signature)
    
    def parse_notification(self, xml_content: str) -> Optional[Dict[str, Any]]:
        """Parse YouTube notification XML"""
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'yt': 'http://www.youtube.com/xml/schemas/2015'
            }
            
            # Extract video information
            entry = root.find('atom:entry', namespaces)
            if entry is None:
                return None
            
            video_id = entry.find('yt:videoId', namespaces)
            title = entry.find('atom:title', namespaces)
            published = entry.find('atom:published', namespaces)
            updated = entry.find('atom:updated', namespaces)
            author = entry.find('atom:author/atom:name', namespaces)
            
            if video_id is not None:
                return {
                    'video_id': video_id.text,
                    'title': title.text if title is not None else '',
                    'published_at': published.text if published is not None else '',
                    'updated_at': updated.text if updated is not None else '',
                    'author': author.text if author is not None else '',
                    'channel_id': self.channel_id
                }
            
            return None
            
        except ET.ParseError as e:
            logger.error(f"Error parsing XML notification: {e}")
            return None
    
    async def process_video_notification(self, video_data: Dict[str, Any]):
        """Process new video notification"""
        try:
            logger.info(f"Processing new video: {video_data['title']}")
            
            # Get detailed video information
            video_details = {
                'video_id': video_data['video_id'],
                'title': video_data['title'],
                'description': '',  # Will be fetched separately
                'published_at': video_data['published_at'],
                'thumbnail': f"https://img.youtube.com/vi/{video_data['video_id']}/default.jpg",
                'channel_title': video_data['author']
            }
            
            # Get video comments
            comments = await self.youtube_integration.get_video_comments(
                video_data['video_id'], 
                max_results=50
            )
            
            # Process comments
            for comment in comments:
                await self.youtube_integration.process_and_store_content("comment", comment)
            
            # Optional: Transcribe video (resource intensive)
            # transcript = await self.youtube_integration.transcribe_video(video_data['video_id'])
            # if transcript:
            #     video_details['transcript'] = transcript
            #     await self.youtube_integration.process_and_store_content("video_transcript", video_details)
            
            logger.info(f"Processed video {video_data['video_id']} with {len(comments)} comments")
            
        except Exception as e:
            logger.error(f"Error processing video notification: {e}")

# FastAPI webhook endpoint
app = FastAPI(title="YouTube Webhook System")

# Global webhook system instance
webhook_system: Optional[YouTubeWebhookSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize webhook system on startup"""
    global webhook_system
    
    import os
    
    # Configuration
    CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")
    WEBHOOK_URL = os.getenv("YOUTUBE_WEBHOOK_URL")  # Your public webhook URL
    WEBHOOK_SECRET = os.getenv("YOUTUBE_WEBHOOK_SECRET", "your-secret-key")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    
    if not all([CHANNEL_ID, WEBHOOK_URL, YOUTUBE_API_KEY]):
        logger.error("Missing required environment variables")
        return
    
    # Database configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DB", "tiktok_learning"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "")
    }
    
    # Initialize YouTube integration
    youtube_integration = YouTubeLiveIntegration(YOUTUBE_API_KEY, CHANNEL_ID, db_config)
    
    # Initialize webhook system
    webhook_system = YouTubeWebhookSystem(
        CHANNEL_ID, 
        WEBHOOK_URL, 
        WEBHOOK_SECRET,
        youtube_integration
    )
    
    # Subscribe to channel notifications
    success = await webhook_system.subscribe_to_channel()
    if success:
        logger.info("Successfully subscribed to YouTube notifications")
    else:
        logger.error("Failed to subscribe to YouTube notifications")

@app.get("/youtube/webhook")
async def verify_webhook(request: Request):
    """Handle webhook verification from YouTube"""
    global webhook_system
    
    if not webhook_system:
        raise HTTPException(status_code=500, detail="Webhook system not initialized")
    
    # Get verification parameters
    hub_mode = request.query_params.get('hub.mode')
    hub_challenge = request.query_params.get('hub.challenge')
    hub_topic = request.query_params.get('hub.topic')
    hub_lease_seconds = request.query_params.get('hub.lease_seconds')
    
    # Verify the subscription
    if (hub_mode == 'subscribe' and 
        hub_topic == webhook_system.topic_url and 
        hub_challenge):
        
        webhook_system.subscription_active = True
        logger.info(f"Webhook verified for channel {webhook_system.channel_id}")
        logger.info(f"Lease duration: {hub_lease_seconds} seconds")
        
        return PlainTextResponse(hub_challenge)
    
    elif hub_mode == 'unsubscribe' and hub_challenge:
        webhook_system.subscription_active = False
        logger.info(f"Webhook unsubscribed for channel {webhook_system.channel_id}")
        return PlainTextResponse(hub_challenge)
    
    else:
        raise HTTPException(status_code=400, detail="Invalid verification request")

@app.post("/youtube/webhook")
async def handle_notification(request: Request, background_tasks: BackgroundTasks):
    """Handle YouTube notification"""
    global webhook_system
    
    if not webhook_system:
        raise HTTPException(status_code=500, detail="Webhook system not initialized")
    
    # Get request body and signature
    body = await request.body()
    signature = request.headers.get('X-Hub-Signature', '')
    
    # Verify signature for security
    if not webhook_system.verify_webhook_signature(body, signature):
        logger.warning("Invalid webhook signature")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Parse notification
    xml_content = body.decode('utf-8')
    video_data = webhook_system.parse_notification(xml_content)
    
    if video_data:
        # Process notification in background
        background_tasks.add_task(
            webhook_system.process_video_notification, 
            video_data
        )
        logger.info(f"Queued processing for video: {video_data['title']}")
    
    return {"status": "ok"}

@app.get("/youtube/status")
async def get_webhook_status():
    """Get webhook system status"""
    global webhook_system
    
    if not webhook_system:
        return {"status": "not_initialized"}
    
    return {
        "status": "active" if webhook_system.subscription_active else "inactive",
        "channel_id": webhook_system.channel_id,
        "topic_url": webhook_system.topic_url,
        "webhook_url": webhook_system.webhook_url
    }

@app.post("/youtube/resubscribe")
async def resubscribe_webhook():
    """Manually resubscribe to YouTube notifications"""
    global webhook_system
    
    if not webhook_system:
        raise HTTPException(status_code=500, detail="Webhook system not initialized")
    
    success = await webhook_system.subscribe_to_channel()
    
    return {
        "success": success,
        "message": "Resubscription successful" if success else "Resubscription failed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
