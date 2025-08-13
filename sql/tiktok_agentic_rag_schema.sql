-- TikTok Shop Agentic RAG Database Schema
-- PostgreSQL with pgvector extension for vector similarity search
-- Enhanced schema for YouTube transcripts and Facebook group data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS ingestion_logs CASCADE;
DROP TABLE IF EXISTS knowledge_sources CASCADE;

-- Documents table for storing source documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL, -- e.g., "youtube:video_id" or "facebook:post_id"
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- TikTok-specific fields
    source_type TEXT CHECK (source_type IN ('youtube_transcript', 'facebook_post', 'facebook_comment', 'general')),
    category TEXT CHECK (category IN ('product_hunting', 'compliance', 'reinstatement', 'strategy', 'trends', 'general')),
    
    -- Search optimization
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED
);

-- Chunks table for storing document chunks with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI/Gemini embedding dimension
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- TikTok-specific fields
    tiktok_keywords TEXT[] DEFAULT '{}',
    keyword_count INTEGER DEFAULT 0,
    tiktok_relevance FLOAT DEFAULT 0.0,
    
    -- Constraints
    CONSTRAINT chunks_chunk_index_check CHECK (chunk_index >= 0),
    CONSTRAINT chunks_token_count_check CHECK (token_count >= 0),
    CONSTRAINT chunks_tiktok_relevance_check CHECK (tiktok_relevance >= 0.0 AND tiktok_relevance <= 1.0)
);

-- Knowledge sources table for tracking ingestion sources
CREATE TABLE knowledge_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type TEXT NOT NULL CHECK (source_type IN ('youtube_transcript', 'facebook_post', 'facebook_comment')),
    source_id TEXT NOT NULL, -- External ID (video_id, post_id, etc.)
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    category TEXT NOT NULL CHECK (category IN ('product_hunting', 'compliance', 'reinstatement', 'strategy', 'trends', 'general')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    
    -- Unique constraint on source
    UNIQUE(source_type, source_id)
);

-- Sessions table for chat sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Session tracking
    message_count INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Messages table for chat history
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Agentic RAG specific fields
    tools_used TEXT[] DEFAULT '{}',
    sources_used JSONB DEFAULT '[]',
    confidence_score FLOAT,
    agent_reasoning TEXT,
    
    -- Performance tracking
    retrieval_time FLOAT DEFAULT 0.0,
    generation_time FLOAT DEFAULT 0.0
);

-- Ingestion logs for tracking data processing
CREATE TABLE ingestion_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    ingestion_type TEXT NOT NULL CHECK (ingestion_type IN ('youtube_transcripts', 'facebook_data', 'manual')),
    source_count INTEGER NOT NULL DEFAULT 0,
    processed_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed', 'partial')),
    
    -- Results tracking
    documents_created INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    embeddings_generated INTEGER DEFAULT 0,
    graph_nodes_created INTEGER DEFAULT 0
);

-- Indexes for performance optimization

-- Document indexes
CREATE INDEX idx_documents_source_type ON documents(source_type);
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_search_vector ON documents USING GIN(search_vector);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

-- Chunk indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_tiktok_keywords ON chunks USING GIN(tiktok_keywords);
CREATE INDEX idx_chunks_tiktok_relevance ON chunks(tiktok_relevance DESC);
CREATE INDEX idx_chunks_metadata ON chunks USING GIN(metadata);

-- Knowledge sources indexes
CREATE INDEX idx_knowledge_sources_type ON knowledge_sources(source_type);
CREATE INDEX idx_knowledge_sources_category ON knowledge_sources(category);
CREATE INDEX idx_knowledge_sources_created_at ON knowledge_sources(created_at);

-- Session and message indexes
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_messages_tools_used ON messages USING GIN(tools_used);

-- Ingestion log indexes
CREATE INDEX idx_ingestion_logs_user_id ON ingestion_logs(user_id);
CREATE INDEX idx_ingestion_logs_type ON ingestion_logs(ingestion_type);
CREATE INDEX idx_ingestion_logs_status ON ingestion_logs(status);
CREATE INDEX idx_ingestion_logs_started_at ON ingestion_logs(started_at);

-- Functions for vector similarity search

-- Function to match chunks by similarity
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        c.id as chunk_id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.metadata,
        d.title as document_title,
        d.source as document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Function for hybrid search (vector + text)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count int DEFAULT 10,
    text_weight float DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        c.id as chunk_id,
        c.document_id,
        c.content,
        (1 - (c.embedding <=> query_embedding)) * (1 - text_weight) + 
        ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) * text_weight as combined_score,
        c.metadata,
        d.title as document_title,
        d.source as document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE 
        to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
        OR (c.embedding <=> query_embedding) < 1.0
    ORDER BY combined_score DESC
    LIMIT match_count;
$$;

-- Triggers for maintaining updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger to update session activity and message count
CREATE OR REPLACE FUNCTION update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE sessions 
    SET 
        last_activity = CURRENT_TIMESTAMP,
        message_count = message_count + 1
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_session_on_message AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION update_session_activity();

-- Views for common queries

-- View for TikTok-specific document statistics
CREATE VIEW tiktok_document_stats AS
SELECT 
    source_type,
    category,
    COUNT(*) as document_count,
    AVG(LENGTH(content)) as avg_content_length,
    COUNT(DISTINCT DATE(created_at)) as days_with_content
FROM documents 
WHERE source_type IN ('youtube_transcript', 'facebook_post', 'facebook_comment')
GROUP BY source_type, category;

-- View for chunk statistics by category
CREATE VIEW tiktok_chunk_stats AS
SELECT 
    d.category,
    d.source_type,
    COUNT(c.id) as chunk_count,
    AVG(c.token_count) as avg_token_count,
    AVG(c.tiktok_relevance) as avg_relevance,
    COUNT(DISTINCT c.document_id) as document_count
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE d.source_type IN ('youtube_transcript', 'facebook_post', 'facebook_comment')
GROUP BY d.category, d.source_type;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;
