"""
youtube_setup_guide.py
----------------------
Setup and configuration guide for YouTube Live Integration.
Includes API setup, webhook configuration, and testing utilities.
"""

import os
import json
from typing import Dict, List, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class YouTubeSetupGuide:
    """Guide for setting up YouTube API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test YouTube API connection"""
        try:
            # Test with a simple request
            request = self.youtube.channels().list(
                part="snippet,statistics",
                mine=True
            )
            response = request.execute()
            
            if response.get('items'):
                channel = response['items'][0]
                return {
                    "success": True,
                    "channel_id": channel['id'],
                    "channel_title": channel['snippet']['title'],
                    "subscriber_count": channel['statistics'].get('subscriberCount', 'Hidden'),
                    "video_count": channel['statistics']['videoCount'],
                    "view_count": channel['statistics']['viewCount']
                }
            else:
                return {
                    "success": False,
                    "error": "No channel found. Make sure API key has proper permissions."
                }
                
        except HttpError as e:
            return {
                "success": False,
                "error": f"YouTube API Error: {e}"
            }
    
    def get_channel_info(self, channel_id: str = None) -> Dict[str, Any]:
        """Get detailed channel information"""
        try:
            if channel_id:
                request = self.youtube.channels().list(
                    part="snippet,statistics,contentDetails",
                    id=channel_id
                )
            else:
                request = self.youtube.channels().list(
                    part="snippet,statistics,contentDetails",
                    mine=True
                )
            
            response = request.execute()
            
            if response.get('items'):
                channel = response['items'][0]
                return {
                    "success": True,
                    "channel_id": channel['id'],
                    "title": channel['snippet']['title'],
                    "description": channel['snippet']['description'],
                    "published_at": channel['snippet']['publishedAt'],
                    "thumbnail": channel['snippet']['thumbnails']['default']['url'],
                    "statistics": channel['statistics'],
                    "uploads_playlist": channel['contentDetails']['relatedPlaylists']['uploads']
                }
            else:
                return {
                    "success": False,
                    "error": "Channel not found"
                }
                
        except HttpError as e:
            return {
                "success": False,
                "error": f"YouTube API Error: {e}"
            }
    
    def check_live_streaming_enabled(self, channel_id: str = None) -> Dict[str, Any]:
        """Check if live streaming is enabled for the channel"""
        try:
            if channel_id:
                request = self.youtube.channels().list(
                    part="status",
                    id=channel_id
                )
            else:
                request = self.youtube.channels().list(
                    part="status",
                    mine=True
                )
            
            response = request.execute()
            
            if response.get('items'):
                status = response['items'][0]['status']
                return {
                    "success": True,
                    "live_streaming_enabled": status.get('isLinked', False),
                    "privacy_status": status.get('privacyStatus', 'unknown'),
                    "made_for_kids": status.get('madeForKids', False)
                }
            else:
                return {
                    "success": False,
                    "error": "Channel status not found"
                }
                
        except HttpError as e:
            return {
                "success": False,
                "error": f"YouTube API Error: {e}"
            }
    
    def get_recent_videos(self, channel_id: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get recent videos from channel"""
        try:
            if channel_id:
                request = self.youtube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    type="video",
                    order="date",
                    maxResults=max_results
                )
            else:
                # Get uploads playlist first
                channel_info = self.get_channel_info()
                if not channel_info["success"]:
                    return []
                
                uploads_playlist = channel_info["uploads_playlist"]
                request = self.youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist,
                    maxResults=max_results
                )
            
            response = request.execute()
            
            videos = []
            for item in response.get('items', []):
                if 'videoId' in item['snippet']['resourceId']:
                    video_id = item['snippet']['resourceId']['videoId']
                else:
                    video_id = item['id']['videoId']
                
                videos.append({
                    "video_id": video_id,
                    "title": item['snippet']['title'],
                    "description": item['snippet']['description'],
                    "published_at": item['snippet']['publishedAt'],
                    "thumbnail": item['snippet']['thumbnails']['default']['url']
                })
            
            return videos
            
        except HttpError as e:
            print(f"YouTube API Error: {e}")
            return []

def setup_youtube_integration():
    """Interactive setup for YouTube integration"""
    print("🎥 YouTube Live Integration Setup")
    print("=" * 50)
    
    # Step 1: API Key
    print("\n1. YouTube API Key Setup")
    print("-" * 30)
    print("To get your YouTube API key:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Create a new project or select existing one")
    print("3. Enable YouTube Data API v3")
    print("4. Create credentials (API Key)")
    print("5. Restrict the key to YouTube Data API v3")
    
    api_key = input("\nEnter your YouTube API Key: ").strip()
    
    if not api_key:
        print("❌ API Key is required!")
        return
    
    # Test API connection
    print("\n2. Testing API Connection...")
    print("-" * 30)
    
    setup_guide = YouTubeSetupGuide(api_key)
    connection_test = setup_guide.test_api_connection()
    
    if connection_test["success"]:
        print("✅ API Connection successful!")
        print(f"Channel: {connection_test['channel_title']}")
        print(f"Channel ID: {connection_test['channel_id']}")
        print(f"Subscribers: {connection_test['subscriber_count']}")
        print(f"Videos: {connection_test['video_count']}")
        
        channel_id = connection_test['channel_id']
    else:
        print(f"❌ API Connection failed: {connection_test['error']}")
        return
    
    # Check live streaming
    print("\n3. Checking Live Streaming Status...")
    print("-" * 30)
    
    live_status = setup_guide.check_live_streaming_enabled()
    if live_status["success"]:
        if live_status["live_streaming_enabled"]:
            print("✅ Live streaming is enabled!")
        else:
            print("⚠️  Live streaming is not enabled.")
            print("To enable live streaming:")
            print("1. Go to YouTube Studio")
            print("2. Go to Settings > Channel > Features")
            print("3. Enable live streaming")
    else:
        print(f"❌ Could not check live streaming status: {live_status['error']}")
    
    # Get recent videos
    print("\n4. Recent Videos Preview...")
    print("-" * 30)
    
    recent_videos = setup_guide.get_recent_videos(channel_id, max_results=5)
    if recent_videos:
        print("Recent videos found:")
        for i, video in enumerate(recent_videos, 1):
            print(f"{i}. {video['title'][:50]}...")
    else:
        print("No recent videos found.")
    
    # Generate configuration
    print("\n5. Configuration Generated")
    print("-" * 30)
    
    config = {
        "YOUTUBE_API_KEY": api_key,
        "YOUTUBE_CHANNEL_ID": channel_id,
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "tiktok_learning",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": ""
    }
    
    print("Add these to your .env file:")
    print()
    for key, value in config.items():
        print(f"{key}={value}")
    
    # Save configuration
    with open("youtube_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to youtube_config.json")
    
    # Usage instructions
    print("\n6. Usage Instructions")
    print("-" * 30)
    print("To start live monitoring:")
    print("1. Set environment variables from above")
    print("2. Run: python youtube_live_integration.py")
    print("3. The system will monitor:")
    print("   - Live stream chats")
    print("   - New video uploads")
    print("   - New comments")
    print("   - Automatic transcription (optional)")
    
    print("\n🎉 Setup complete! Your YouTube integration is ready.")

def test_live_monitoring():
    """Test the live monitoring functionality"""
    from youtube_live_integration import YouTubeLiveIntegration
    import asyncio
    
    # Load configuration
    try:
        with open("youtube_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Configuration file not found. Run setup first.")
        return
    
    db_config = {
        "host": config["POSTGRES_HOST"],
        "port": config["POSTGRES_PORT"],
        "database": config["POSTGRES_DB"],
        "user": config["POSTGRES_USER"],
        "password": config["POSTGRES_PASSWORD"]
    }
    
    async def test():
        integration = YouTubeLiveIntegration(
            config["YOUTUBE_API_KEY"],
            config["YOUTUBE_CHANNEL_ID"],
            db_config
        )
        
        print("Testing live stream monitoring...")
        live_streams = await integration.monitor_live_streams()
        print(f"Found {len(live_streams)} live streams")
        
        print("Testing new video detection...")
        new_videos = await integration.get_new_videos(hours_back=24)
        print(f"Found {len(new_videos)} new videos in last 24 hours")
        
        if new_videos:
            video = new_videos[0]
            print(f"Testing comment retrieval for: {video['title']}")
            comments = await integration.get_video_comments(video['video_id'], max_results=5)
            print(f"Found {len(comments)} comments")
    
    asyncio.run(test())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_live_monitoring()
    else:
        setup_youtube_integration()
