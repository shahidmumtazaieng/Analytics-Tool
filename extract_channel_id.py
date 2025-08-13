"""
extract_channel_id.py
---------------------
Script to extract YouTube channel ID for Bilal Sirbuland channel.
This will help configure the system with the correct channel ID.
"""

import os
import requests
from typing import Optional, Dict, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class YouTubeChannelExtractor:
    """Extract YouTube channel information"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def extract_channel_id_from_handle(self, handle: str) -> Optional[Dict[str, Any]]:
        """Extract channel ID from handle like @BilalSirbuland"""
        try:
            # Remove @ if present
            handle = handle.replace('@', '')
            
            # Search for channel by handle
            request = self.youtube.search().list(
                part="snippet",
                q=handle,
                type="channel",
                maxResults=10
            )
            response = request.execute()
            
            # Look for exact match
            for item in response.get('items', []):
                channel_title = item['snippet']['title']
                channel_id = item['snippet']['channelId']
                
                # Check if this matches our channel
                if 'bilal' in channel_title.lower() and 'sirbuland' in channel_title.lower():
                    return self.get_detailed_channel_info(channel_id)
            
            return None
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return None
    
    def extract_channel_id_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract channel ID from URL"""
        try:
            # Extract handle from URL
            if '@' in url:
                handle = url.split('@')[1].split('/')[0]
                return self.extract_channel_id_from_handle(handle)
            
            return None
            
        except Exception as e:
            print(f"Error extracting from URL: {e}")
            return None
    
    def get_detailed_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get detailed channel information"""
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            if response.get('items'):
                channel = response['items'][0]
                snippet = channel['snippet']
                statistics = channel['statistics']
                
                return {
                    "channel_id": channel_id,
                    "title": snippet['title'],
                    "description": snippet['description'],
                    "custom_url": snippet.get('customUrl', ''),
                    "published_at": snippet['publishedAt'],
                    "thumbnail_url": snippet['thumbnails']['default']['url'],
                    "high_res_thumbnail": snippet['thumbnails'].get('high', {}).get('url', ''),
                    "subscriber_count": statistics.get('subscriberCount', 'Hidden'),
                    "video_count": statistics['videoCount'],
                    "view_count": statistics['viewCount'],
                    "uploads_playlist": channel['contentDetails']['relatedPlaylists']['uploads']
                }
            
            return None
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return None
    
    def search_by_keywords(self, keywords: str) -> Optional[Dict[str, Any]]:
        """Search for channel by keywords"""
        try:
            request = self.youtube.search().list(
                part="snippet",
                q=keywords,
                type="channel",
                maxResults=10
            )
            response = request.execute()
            
            print(f"Found {len(response.get('items', []))} channels for '{keywords}':")
            
            for i, item in enumerate(response.get('items', []), 1):
                channel_title = item['snippet']['title']
                channel_id = item['snippet']['channelId']
                description = item['snippet']['description'][:100] + "..." if len(item['snippet']['description']) > 100 else item['snippet']['description']
                
                print(f"{i}. {channel_title}")
                print(f"   ID: {channel_id}")
                print(f"   Description: {description}")
                print()
                
                # Check if this looks like the right channel
                if ('bilal' in channel_title.lower() and 'sirbuland' in channel_title.lower()) or \
                   ('tiktok' in description.lower() and 'shop' in description.lower()):
                    print(f"🎯 This looks like the target channel!")
                    return self.get_detailed_channel_info(channel_id)
            
            return None
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return None

def main():
    """Main function to extract channel information"""
    print("🎥 YouTube Channel ID Extractor for Bilal Sirbuland")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("❌ YOUTUBE_API_KEY environment variable not set!")
        print("\nTo get a YouTube API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable YouTube Data API v3")
        print("4. Create credentials (API Key)")
        print("5. Set YOUTUBE_API_KEY environment variable")
        return
    
    extractor = YouTubeChannelExtractor(api_key)
    
    # Method 1: Try extracting from handle
    print("\n1. Trying to extract from handle @BilalSirbuland...")
    channel_info = extractor.extract_channel_id_from_handle("@BilalSirbuland")
    
    if not channel_info:
        print("❌ Could not find channel by handle")
        
        # Method 2: Try extracting from URL
        print("\n2. Trying to extract from URL...")
        channel_info = extractor.extract_channel_id_from_url("https://www.youtube.com/@BilalSirbuland")
    
    if not channel_info:
        print("❌ Could not find channel by URL")
        
        # Method 3: Search by keywords
        print("\n3. Searching by keywords...")
        channel_info = extractor.search_by_keywords("Bilal Sirbuland TikTok Shop")
    
    if channel_info:
        print("\n✅ Channel Information Found!")
        print("=" * 40)
        print(f"Channel ID: {channel_info['channel_id']}")
        print(f"Title: {channel_info['title']}")
        print(f"Custom URL: {channel_info.get('custom_url', 'Not set')}")
        print(f"Subscribers: {channel_info['subscriber_count']}")
        print(f"Videos: {channel_info['video_count']}")
        print(f"Total Views: {channel_info['view_count']}")
        print(f"Thumbnail: {channel_info['thumbnail_url']}")
        print(f"Description: {channel_info['description'][:200]}...")
        
        # Generate environment variables
        print("\n🔧 Environment Variables to Add:")
        print("=" * 40)
        print(f"YOUTUBE_CHANNEL_ID={channel_info['channel_id']}")
        print(f"YOUTUBE_CHANNEL_NAME={channel_info['title']}")
        print(f"YOUTUBE_CHANNEL_URL=https://www.youtube.com/@BilalSirbuland")
        print(f"YOUTUBE_SUBSCRIBER_COUNT={channel_info['subscriber_count']}")
        print(f"YOUTUBE_VIDEO_COUNT={channel_info['video_count']}")
        print(f"YOUTUBE_CHANNEL_PROFILE_IMAGE={channel_info['high_res_thumbnail']}")
        print(f"YOUTUBE_DESCRIPTION={channel_info['description'][:100]}...")
        
        # Save to file
        with open("youtube_channel_config.txt", "w") as f:
            f.write("# YouTube Channel Configuration for Bilal Sirbuland\n")
            f.write(f"YOUTUBE_CHANNEL_ID={channel_info['channel_id']}\n")
            f.write(f"YOUTUBE_CHANNEL_NAME={channel_info['title']}\n")
            f.write(f"YOUTUBE_CHANNEL_HANDLE=@BilalSirbuland\n")
            f.write(f"YOUTUBE_CHANNEL_URL=https://www.youtube.com/@BilalSirbuland\n")
            f.write(f"YOUTUBE_SUBSCRIBER_COUNT={channel_info['subscriber_count']}\n")
            f.write(f"YOUTUBE_VIDEO_COUNT={channel_info['video_count']}\n")
            f.write(f"YOUTUBE_CHANNEL_PROFILE_IMAGE={channel_info['high_res_thumbnail']}\n")
            f.write(f"YOUTUBE_DESCRIPTION={channel_info['description']}\n")
        
        print(f"\n💾 Configuration saved to youtube_channel_config.txt")
        
    else:
        print("\n❌ Could not find the channel!")
        print("This might be because:")
        print("1. The channel handle has changed")
        print("2. The channel is private or restricted")
        print("3. API quota is exceeded")
        print("4. The search terms don't match exactly")
        
        print("\n🔧 Manual Configuration:")
        print("You can manually set these environment variables:")
        print("YOUTUBE_CHANNEL_ID=UCYourActualChannelId")
        print("YOUTUBE_CHANNEL_NAME=Bilal Sirbuland")
        print("YOUTUBE_CHANNEL_HANDLE=@BilalSirbuland")
        print("YOUTUBE_CHANNEL_URL=https://www.youtube.com/@BilalSirbuland")

if __name__ == "__main__":
    main()
