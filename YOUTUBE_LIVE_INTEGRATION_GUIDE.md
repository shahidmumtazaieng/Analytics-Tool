# 🎥 **YOUTUBE LIVE INTEGRATION GUIDE**

Complete guide for integrating live YouTube data into your TikTok learning system.

## **📋 OVERVIEW**

This integration captures live data from YouTube channels and automatically processes it into your vector database:

### **🎯 What Gets Captured:**
- ✅ **Live Stream Chat** - Real-time comments during live streams
- ✅ **New Video Uploads** - Automatic detection of new content
- ✅ **Video Comments** - New comments on existing videos
- ✅ **Video Transcripts** - Automatic transcription using Whisper
- ✅ **Community Posts** - YouTube community tab updates

### **🔄 Two Integration Methods:**

#### **Method 1: Polling (Simple)**
- Checks for new content every 30-60 seconds
- Uses YouTube Data API v3
- Good for testing and small channels

#### **Method 2: Webhooks (Recommended)**
- Real-time notifications using PubSubHubbub
- Instant processing of new content
- More efficient for active channels

## **🚀 QUICK SETUP**

### **Step 1: Get YouTube API Key**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project or select existing one
3. Enable **YouTube Data API v3**
4. Create credentials (API Key)
5. Restrict key to YouTube Data API v3

### **Step 2: Run Setup Script**

```bash
cd Backend
python youtube_setup_guide.py
```

This will:
- Test your API connection
- Get your channel information
- Check live streaming status
- Generate configuration file

### **Step 3: Configure Environment**

Add to your `.env` file:
```bash
# YouTube Integration
YOUTUBE_API_KEY=your_youtube_api_key_here
YOUTUBE_CHANNEL_ID=your_channel_id_here

# For webhook method (optional)
YOUTUBE_WEBHOOK_URL=https://yourdomain.com/youtube/webhook
YOUTUBE_WEBHOOK_SECRET=your_secret_key_here

# Database (existing)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tiktok_learning
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

### **Step 4: Install Dependencies**

```bash
pip install google-api-python-client
pip install yt-dlp
pip install openai-whisper
pip install fastapi uvicorn  # For webhook method
```

## **📊 METHOD 1: POLLING INTEGRATION**

### **Simple Polling Setup**

```python
from youtube_live_integration import YouTubeLiveIntegration
import asyncio
import os

async def main():
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
    
    # Initialize integration
    youtube_integration = YouTubeLiveIntegration(
        YOUTUBE_API_KEY, 
        CHANNEL_ID, 
        db_config
    )
    
    # Start monitoring (checks every 30 seconds)
    await youtube_integration.run_live_monitoring(check_interval=30)

if __name__ == "__main__":
    asyncio.run(main())
```

### **Run Polling Integration**

```bash
python youtube_live_integration.py
```

## **🔔 METHOD 2: WEBHOOK INTEGRATION (RECOMMENDED)**

### **Webhook Setup Requirements**

1. **Public URL**: Your server must be accessible from the internet
2. **HTTPS**: YouTube requires HTTPS for webhooks
3. **Domain**: Use ngrok for testing or proper domain for production

### **Testing with ngrok**

```bash
# Install ngrok
npm install -g ngrok

# Start your webhook server
python youtube_webhook_system.py

# In another terminal, expose it
ngrok http 8001

# Use the HTTPS URL for YOUTUBE_WEBHOOK_URL
# Example: https://abc123.ngrok.io/youtube/webhook
```

### **Production Webhook Setup**

```bash
# Set your production webhook URL
export YOUTUBE_WEBHOOK_URL=https://yourdomain.com/youtube/webhook

# Start webhook server
python youtube_webhook_system.py
```

### **Webhook Endpoints**

- `GET /youtube/webhook` - Verification endpoint
- `POST /youtube/webhook` - Notification handler
- `GET /youtube/status` - Check webhook status
- `POST /youtube/resubscribe` - Manual resubscription

## **🔧 INTEGRATION WITH TIKTOK LEARNING SYSTEM**

### **Database Schema**

The integration creates this table:

```sql
CREATE TABLE youtube_live_content (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50) NOT NULL,  -- 'live_chat', 'comment', 'video_transcript'
    content_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    video_id VARCHAR(50),
    author_name VARCHAR(255),
    timestamp TIMESTAMP,
    processed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(content_type, content_id)
);
```

### **Vector Search Integration**

```python
# Search live YouTube content
async def search_youtube_content(query: str, limit: int = 5):
    conn = await asyncpg.connect(**db_config)
    
    # Generate query embedding
    embedding = embedding_model.encode([query])[0].tolist()
    
    # Search similar content
    results = await conn.fetch("""
        SELECT content, metadata, video_id, author_name,
               1 - (embedding <=> $1::vector) as similarity
        FROM youtube_live_content
        WHERE content_type = 'comment'
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """, embedding, limit)
    
    await conn.close()
    return results
```

### **Integration with Existing Agents**

```python
# Enhance your TikTok agents with live YouTube data
class EnhancedTikTokAgent:
    async def get_context(self, query: str):
        # Get static knowledge (existing)
        static_context = await self.search_static_knowledge(query)
        
        # Get live YouTube data (new)
        live_context = await self.search_youtube_content(query)
        
        # Combine contexts
        combined_context = static_context + live_context
        
        return combined_context
```

## **⚙️ CONFIGURATION OPTIONS**

### **Content Filtering**

```python
# Filter content by relevance
TIKTOK_KEYWORDS = [
    "tiktok shop", "tiktok seller", "tiktok commerce",
    "social commerce", "live shopping", "creator economy",
    "product hunting", "dropshipping", "tiktok ads"
]

def is_tiktok_relevant(content: str) -> bool:
    content_lower = content.lower()
    relevance_score = sum(1 for keyword in TIKTOK_KEYWORDS if keyword in content_lower)
    return relevance_score >= 2
```

### **Rate Limiting**

```python
# Respect YouTube API quotas
RATE_LIMITS = {
    "comments_per_video": 100,
    "videos_per_check": 10,
    "api_calls_per_day": 10000,
    "check_interval": 30  # seconds
}
```

### **Transcription Settings**

```python
# Whisper model options
WHISPER_MODELS = {
    "tiny": "Fast, less accurate",
    "base": "Balanced (recommended)",
    "small": "Better accuracy",
    "medium": "High accuracy, slower",
    "large": "Best accuracy, very slow"
}
```

## **📊 MONITORING & ANALYTICS**

### **Check Integration Status**

```bash
# Test API connection
python youtube_setup_guide.py test

# Check webhook status
curl https://yourdomain.com/youtube/status

# View recent content
python -c "
import asyncio
from youtube_live_integration import YouTubeLiveIntegration
# ... check recent content
"
```

### **Database Queries**

```sql
-- Check content volume
SELECT content_type, COUNT(*) as count
FROM youtube_live_content
GROUP BY content_type;

-- Recent activity
SELECT content_type, author_name, LEFT(content, 100) as preview
FROM youtube_live_content
WHERE processed_at > NOW() - INTERVAL '1 hour'
ORDER BY processed_at DESC;

-- Top contributors
SELECT author_name, COUNT(*) as contributions
FROM youtube_live_content
WHERE content_type = 'comment'
GROUP BY author_name
ORDER BY contributions DESC
LIMIT 10;
```

## **🔒 SECURITY CONSIDERATIONS**

### **API Key Security**
- ✅ Store API keys in environment variables
- ✅ Restrict API key to YouTube Data API v3 only
- ✅ Monitor API usage in Google Cloud Console

### **Webhook Security**
- ✅ Use HTTPS for webhook URLs
- ✅ Verify webhook signatures
- ✅ Use strong webhook secrets
- ✅ Rate limit webhook endpoints

### **Data Privacy**
- ✅ Only process public comments and content
- ✅ Respect user privacy settings
- ✅ Follow YouTube Terms of Service
- ✅ Implement data retention policies

## **🚨 TROUBLESHOOTING**

### **Common Issues**

#### **API Quota Exceeded**
```
Error: quotaExceeded
Solution: Reduce check frequency or upgrade quota
```

#### **Webhook Not Receiving Notifications**
```
Check:
1. URL is publicly accessible
2. HTTPS is working
3. Signature verification is correct
4. Subscription is active
```

#### **Transcription Errors**
```
Error: Audio download failed
Solution: Check yt-dlp version and video availability
```

### **Debug Commands**

```bash
# Test API connection
python youtube_setup_guide.py

# Test webhook locally
curl -X GET "http://localhost:8001/youtube/status"

# Check database content
psql -d tiktok_learning -c "SELECT COUNT(*) FROM youtube_live_content;"
```

## **📈 PERFORMANCE OPTIMIZATION**

### **Efficient Processing**
- ✅ Use async/await for concurrent processing
- ✅ Batch database operations
- ✅ Cache frequently accessed data
- ✅ Process transcription in background

### **Resource Management**
- ✅ Limit concurrent API calls
- ✅ Clean up temporary audio files
- ✅ Monitor memory usage for Whisper
- ✅ Use connection pooling for database

## **🎯 NEXT STEPS**

1. **Start with polling method** for testing
2. **Set up webhooks** for production
3. **Monitor content quality** and relevance
4. **Integrate with existing agents** gradually
5. **Add custom filtering** for your specific needs

## **💡 ADVANCED FEATURES**

### **Multi-Channel Support**
```python
# Monitor multiple channels
CHANNELS = [
    "your_main_channel_id",
    "partner_channel_id",
    "competitor_channel_id"
]
```

### **Content Classification**
```python
# Classify content by topic
TOPIC_CLASSIFIERS = {
    "product_hunting": ["product", "hunting", "research"],
    "compliance": ["policy", "violation", "banned"],
    "marketing": ["ads", "promotion", "campaign"]
}
```

### **Real-time Alerts**
```python
# Send alerts for important content
async def check_for_alerts(content: str):
    if "urgent" in content.lower() or "breaking" in content.lower():
        await send_notification(content)
```

Your YouTube live integration is now ready to enhance your TikTok learning system with real-time data! 🚀
