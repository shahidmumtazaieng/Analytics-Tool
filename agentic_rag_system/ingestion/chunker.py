"""
Document chunking utilities for TikTok Shop content.
Optimized for YouTube transcripts and Facebook group data.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    use_semantic_splitting: bool = True
    preserve_structure: bool = True


class TikTokContentChunker:
    """Specialized chunker for TikTok Shop content."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the chunker with configuration."""
        self.config = config
        
        # TikTok-specific patterns for better chunking
        self.section_patterns = [
            r'\n\n+',  # Double newlines
            r'\n(?=\d+\.)',  # Numbered lists
            r'\n(?=[-•*])',  # Bullet points
            r'\n(?=[A-Z][^a-z]*:)',  # Headers with colons
            r'(?<=\.)\s+(?=[A-Z])',  # Sentence boundaries
        ]
        
        # YouTube transcript specific patterns
        self.youtube_patterns = [
            r'\[\d+:\d+\]',  # Timestamps [00:00]
            r'\(\d+:\d+\)',  # Timestamps (00:00)
            r'(?<=\.)\s+(?=So|Now|Next|First|Second|Third)',  # Transition words
        ]
        
        # Facebook post specific patterns
        self.facebook_patterns = [
            r'\n(?=Comment:)',  # Comment separators
            r'\n(?=Reply:)',    # Reply separators
            r'\n(?=Post:)',     # Post separators
            r'(?<=\.)\s+(?=Has anyone|Does anyone|Can someone)',  # Question patterns
        ]
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document based on its type and content.
        
        Args:
            content: Document content to chunk
            metadata: Document metadata including source type
        
        Returns:
            List of chunks with metadata
        """
        try:
            source_type = metadata.get('source_type', 'general')
            
            if source_type == 'youtube_transcript':
                return self._chunk_youtube_transcript(content, metadata)
            elif source_type in ['facebook_post', 'facebook_comment']:
                return self._chunk_facebook_content(content, metadata)
            else:
                return self._chunk_general_content(content, metadata)
                
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Fallback to simple chunking
            return self._simple_chunk(content, metadata)
    
    def _chunk_youtube_transcript(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk YouTube transcript content."""
        chunks = []
        
        # Remove or normalize timestamps
        content = re.sub(r'\[?\d+:\d+\]?', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Split by semantic boundaries
        if self.config.use_semantic_splitting:
            sections = self._split_by_patterns(content, self.youtube_patterns + self.section_patterns)
        else:
            sections = [content]
        
        # Create chunks from sections
        for i, section in enumerate(sections):
            section_chunks = self._create_chunks_from_text(section, metadata, start_index=len(chunks))
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_facebook_content(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk Facebook post/comment content."""
        chunks = []
        
        # Split by Facebook-specific patterns
        if self.config.use_semantic_splitting:
            sections = self._split_by_patterns(content, self.facebook_patterns + self.section_patterns)
        else:
            sections = [content]
        
        # Create chunks from sections
        for i, section in enumerate(sections):
            section_chunks = self._create_chunks_from_text(section, metadata, start_index=len(chunks))
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_general_content(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk general content."""
        if self.config.use_semantic_splitting:
            sections = self._split_by_patterns(content, self.section_patterns)
        else:
            sections = [content]
        
        chunks = []
        for section in sections:
            section_chunks = self._create_chunks_from_text(section, metadata, start_index=len(chunks))
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_by_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Split text by multiple regex patterns."""
        sections = [text]
        
        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections
        
        return sections
    
    def _create_chunks_from_text(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        start_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text."""
        chunks = []
        words = text.split()
        
        if len(words) <= self.config.chunk_size:
            # Text is small enough to be a single chunk
            chunks.append({
                'content': text,
                'chunk_index': start_index,
                'metadata': {
                    **metadata,
                    'word_count': len(words),
                    'char_count': len(text)
                }
            })
            return chunks
        
        # Create overlapping chunks
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk_words = words[i:i + self.config.chunk_size]
            
            # Ensure we don't exceed max chunk size
            if len(chunk_words) > self.config.max_chunk_size:
                chunk_words = chunk_words[:self.config.max_chunk_size]
            
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'chunk_index': start_index + len(chunks),
                'metadata': {
                    **metadata,
                    'word_count': len(chunk_words),
                    'char_count': len(chunk_text),
                    'start_word': i,
                    'end_word': i + len(chunk_words)
                }
            })
            
            # Break if we've covered all words
            if i + len(chunk_words) >= len(words):
                break
        
        return chunks
    
    def _simple_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple fallback chunking method."""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.config.chunk_size):
            chunk_words = words[i:i + self.config.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'chunk_index': i // self.config.chunk_size,
                'metadata': {
                    **metadata,
                    'word_count': len(chunk_words),
                    'char_count': len(chunk_text)
                }
            })
        
        return chunks


def create_chunker(config: Optional[ChunkingConfig] = None) -> TikTokContentChunker:
    """Create a TikTok content chunker with optional configuration."""
    if config is None:
        config = ChunkingConfig()
    
    return TikTokContentChunker(config)


def optimize_chunks_for_tiktok(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Optimize chunks specifically for TikTok Shop content."""
    optimized_chunks = []
    
    for chunk in chunks:
        content = chunk['content']
        metadata = chunk.get('metadata', {})
        
        # Add TikTok-specific metadata
        tiktok_keywords = [
            'tiktok shop', 'product hunting', 'viral', 'trending', 'algorithm',
            'compliance', 'violation', 'banned', 'suspended', 'appeal',
            'strategy', 'marketing', 'engagement', 'influencer'
        ]
        
        content_lower = content.lower()
        found_keywords = [kw for kw in tiktok_keywords if kw in content_lower]
        
        # Enhance metadata
        enhanced_metadata = {
            **metadata,
            'tiktok_keywords': found_keywords,
            'keyword_count': len(found_keywords),
            'tiktok_relevance': len(found_keywords) / len(tiktok_keywords)
        }
        
        # Determine category based on keywords
        if any(kw in found_keywords for kw in ['product hunting', 'trending', 'viral']):
            enhanced_metadata['category'] = 'product_hunting'
        elif any(kw in found_keywords for kw in ['compliance', 'violation', 'policy']):
            enhanced_metadata['category'] = 'compliance'
        elif any(kw in found_keywords for kw in ['banned', 'suspended', 'appeal']):
            enhanced_metadata['category'] = 'reinstatement'
        elif any(kw in found_keywords for kw in ['strategy', 'marketing', 'growth']):
            enhanced_metadata['category'] = 'strategy'
        elif any(kw in found_keywords for kw in ['trending', 'viral', 'algorithm']):
            enhanced_metadata['category'] = 'trends'
        else:
            enhanced_metadata['category'] = 'general'
        
        optimized_chunks.append({
            **chunk,
            'metadata': enhanced_metadata
        })
    
    return optimized_chunks
