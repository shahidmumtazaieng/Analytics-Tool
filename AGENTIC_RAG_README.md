# 🚀 TikTok Shop Agentic RAG System v2.0

## Overview

The **TikTok Shop Agentic RAG System** is an advanced AI-powered learning platform that replaces traditional fine-tuning approaches with dynamic, intelligent agents. This system combines multiple cutting-edge technologies to provide comprehensive TikTok Shop insights and guidance.

## 🌟 Key Features

### 🤖 Agentic Architecture
- **Pydantic AI Agents**: Specialized agents for different TikTok Shop domains
- **Dynamic Tool Selection**: Intelligent routing based on query context
- **Multi-Agent Coordination**: Collaborative problem-solving approach

### 🕸️ Knowledge Graph Integration
- **Neo4j + Graphiti**: Temporal knowledge relationships
- **Entity Mapping**: Connected insights across different concepts
- **Relationship Discovery**: Uncover hidden patterns and connections

### 📊 Multi-Source Data Integration
- **YouTube Transcripts**: Real course content and tutorials
- **Facebook Groups**: Community discussions and experiences
- **Vector Database**: Semantic similarity search with pgvector
- **Hybrid Search**: Combined vector + keyword matching

### 🎯 TikTok Shop Specialization
- **Product Hunting**: Trending products and market analysis
- **Compliance**: Policy guidance and violation prevention
- **Reinstatement**: Account recovery strategies
- **Strategy**: Marketing and growth optimization
- **Trends**: Algorithm changes and viral patterns

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js 14)                   │
│  Enhanced Chat Interface + Tool Transparency + Real-time   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 FastAPI Backend                             │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  Agentic RAG    │   LangGraph     │   Traditional   │   │
│  │    System       │   Multi-Agent   │     Tools       │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Data Layer                                  │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  PostgreSQL     │     Neo4j       │    Firebase     │   │
│  │  + pgvector     │  + Graphiti     │   (Auth/Data)   │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern async Python web framework
- **Pydantic AI**: Agentic framework for intelligent agents
- **LangGraph**: Multi-agent orchestration (existing tools)
- **PostgreSQL + pgvector**: Vector similarity search
- **Neo4j + Graphiti**: Knowledge graph and temporal relationships
- **Gemini API**: Cost-effective AI processing
- **Firebase Admin**: Authentication and user management

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Modern styling
- **Enhanced Chat Interface**: Tool transparency and source attribution

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp Backend/.env.example Backend/.env

# Edit with your API keys
nano Backend/.env
```

Required environment variables:
```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:pass@localhost:5432/tiktok_agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 2. Database Setup

```bash
# Start databases with Docker
cd Backend
docker-compose up -d postgres neo4j

# Run setup script
python setup_agentic_rag.py
```

### 3. Install Dependencies

```bash
# Backend
cd Backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 4. Start Services

```bash
# Backend
cd Backend
uvicorn app:app --reload

# Frontend (new terminal)
cd frontend
npm run dev
```

## 📊 Data Ingestion

### YouTube Transcripts

```bash
curl -X POST "http://localhost:8000/api/tiktok-learning/ingest/youtube" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "transcripts": [
      {
        "video_id": "abc123",
        "title": "TikTok Shop Success Guide",
        "transcript": "Complete transcript content...",
        "channel": "TikTok Experts",
        "duration": 600,
        "views": 50000,
        "tags": ["tiktok shop", "e-commerce"]
      }
    ]
  }'
```

### Facebook Groups Data

```bash
curl -X POST "http://localhost:8000/api/tiktok-learning/ingest/facebook" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "facebook_data": [
      {
        "id": "post_123",
        "type": "post",
        "content": "Just got my account reinstated...",
        "author": "SuccessfulSeller",
        "group_name": "TikTok Shop Masters",
        "likes": 127,
        "engagement_score": 0.85
      }
    ]
  }'
```

## 🎯 Usage Examples

### Basic Chat Request

```javascript
const response = await fetch('/api/chat/tiktok_learning', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: "What are the best product hunting strategies for TikTok Shop?",
    context: {
      rag_enabled: true,
      use_knowledge_graph: true,
      focus_category: "product_hunting"
    }
  })
});
```

### Enhanced Response Format

```json
{
  "response": "Based on successful TikTok Shop sellers...",
  "sources": [
    {
      "type": "youtube_transcript",
      "title": "Product Hunting Masterclass",
      "relevance_score": 0.95,
      "content_preview": "The key to finding winning products..."
    }
  ],
  "tools_used": ["tiktok_vector_search", "tiktok_graph_search"],
  "agent_reasoning": "Used product hunting category search...",
  "knowledge_graph_insights": ["Connected to viral trends data"],
  "confidence": 0.92,
  "enhanced_features": {
    "agentic_rag": true,
    "knowledge_graph": true,
    "multi_source_data": true
  }
}
```

## 🔧 Configuration

### Agent Specialization

The system includes specialized agents for:

1. **Product Hunting Agent**: Trend analysis, profit calculations
2. **Compliance Agent**: Policy guidance, violation prevention
3. **Reinstatement Agent**: Account recovery strategies
4. **Strategy Agent**: Marketing optimization, growth hacking
5. **Trends Agent**: Algorithm changes, viral patterns

### Knowledge Categories

- `product_hunting`: Product research and trend analysis
- `compliance`: TikTok Shop policies and guidelines
- `reinstatement`: Account recovery and appeals
- `strategy`: Marketing and growth strategies
- `trends`: Current trends and algorithm insights
- `general`: General TikTok Shop questions

## 📈 Performance Features

### Intelligent Caching
- Vector search result caching
- Knowledge graph query optimization
- Session-based context retention

### Scalable Architecture
- Async database operations
- Connection pooling
- Horizontal scaling support

### Real-time Insights
- Live knowledge retrieval
- Dynamic agent selection
- Streaming response support

## 🔒 Security

- Firebase Authentication integration
- JWT token validation
- Rate limiting and abuse prevention
- Secure API key management
- User data isolation

## 📊 Monitoring

### System Health
```bash
curl http://localhost:8000/api/tiktok-learning/status
```

### Performance Metrics
- Response times by tool
- Knowledge source utilization
- Agent selection patterns
- User engagement analytics

## 🚀 Deployment

### Docker Deployment

```bash
# Full stack deployment
docker-compose --profile full-stack up -d

# With monitoring
docker-compose --profile monitoring up -d
```

### Production Considerations

1. **Database Optimization**: Proper indexing and connection pooling
2. **Caching Layer**: Redis for improved performance
3. **Load Balancing**: Multiple backend instances
4. **Monitoring**: Grafana + Prometheus setup
5. **Backup Strategy**: Automated database backups

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting guide
2. Review system logs
3. Contact the development team

---

**TikTok Shop Agentic RAG System v2.0** - Empowering sellers with intelligent, multi-source AI insights.
