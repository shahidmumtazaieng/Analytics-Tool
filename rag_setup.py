<xaiArtifact>
<artifact_id>rag-tiktok-learning</artifact_id>
<title>RAG Setup for TikTok Learning Chatbot</title>
<contentType>text/python</contentType>

"""
RAG (Retrieval-Augmented Generation) Setup for TikTok Learning Chatbot
This module provides the infrastructure for fine-tuning and RAG implementation
for the TikTok learning chatbot with e-commerce data.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TikTokLearningData:
    """Data structure for TikTok learning content"""
    question: str
    answer: str
    category: str
    source: str
    confidence_score: float
    tags: List[str]
    created_at: datetime

class TikTokDataset(Dataset):
    """Custom dataset for TikTok learning fine-tuning"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input for fine-tuning
        input_text = f"Question: {item['question']}\nContext: {item.get('context', '')}\nAnswer:"
        target_text = item['answer']
        
        # Tokenize input and target
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class TikTokRAGSystem:
    """RAG system for TikTok learning chatbot"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.knowledge_base = []
        self.embeddings = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def add_knowledge(self, knowledge_items: List[Dict[str, Any]]):
        """Add knowledge items to the RAG system"""
        for item in knowledge_items:
            self.knowledge_base.append({
                'content': item['content'],
                'metadata': item.get('metadata', {}),
                'embedding': None
            })
        
        # Generate embeddings for all knowledge items
        texts = [item['content'] for item in self.knowledge_base]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        
        # Store embeddings with knowledge items
        for i, item in enumerate(self.knowledge_base):
            item['embedding'] = self.embeddings[i]
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a given query"""
        if not self.knowledge_base:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)
        
        # Get top-k most similar items
        top_indices = torch.topk(similarities, min(top_k, len(self.knowledge_base))).indices[0]
        
        relevant_context = []
        for idx in top_indices:
            relevant_context.append({
                'content': self.knowledge_base[idx]['content'],
                'metadata': self.knowledge_base[idx]['metadata'],
                'similarity_score': similarities[0][idx].item()
            })
        
        return relevant_context
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], model) -> str:
        """Generate response using retrieved context"""
        # Format context for the model
        context_text = "\n".join([item['content'] for item in context])
        
        # Create prompt with context
        prompt = f"""Context: {context_text}

Question: {query}

Answer:"""
        
        # Generate response (this would be integrated with your LLM)
        # For now, return a placeholder
        return f"Based on the context provided, here's what I found about '{query}': [Generated response would go here]"

class TikTokLearningFineTuner:
    """Fine-tuning system for TikTok learning model"""
    
    def __init__(self, base_model: str = "microsoft/DialoGPT-medium"):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        # Add special tokens for TikTok learning
        special_tokens = {
            'additional_special_tokens': [
                '[TIKTOK]', '[PRODUCT]', '[TREND]', '[STRATEGY]', '[INSIGHT]'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_training_data(self, data: List[TikTokLearningData]) -> List[Dict[str, Any]]:
        """Prepare data for fine-tuning"""
        training_data = []
        
        for item in data:
            # Create training examples
            training_data.append({
                'question': item.question,
                'answer': item.answer,
                'context': f"[TIKTOK] Category: {item.category} | Source: {item.source}",
                'confidence_score': item.confidence_score,
                'tags': item.tags
            })
        
        return training_data
    
    def fine_tune(self, training_data: List[Dict[str, Any]], output_dir: str = "./tiktok-learning-model"):
        """Fine-tune the model on TikTok learning data"""
        
        # Create dataset
        dataset = TikTokDataset(training_data, self.tokenizer)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # Start fine-tuning
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuned model saved to {output_dir}")
        return output_dir

def create_sample_tiktok_dataset() -> List[TikTokLearningData]:
    """Create sample TikTok learning dataset"""
    
    sample_data = [
        TikTokLearningData(
            question="How do I find trending products on TikTok Shop?",
            answer="To find trending products on TikTok Shop, use the 'Trending' tab, check hashtag challenges, monitor viral videos, and analyze the 'For You' page. Look for products with high engagement rates and growing sales velocity.",
            category="product_research",
            source="tiktok_shop_analytics",
            confidence_score=0.95,
            tags=["trending", "product_research", "tiktok_shop"],
            created_at=datetime.now()
        ),
        TikTokLearningData(
            question="What's the best time to post on TikTok for e-commerce?",
            answer="The best times to post on TikTok for e-commerce are typically 6-10 PM local time on weekdays and 2-4 PM on weekends. However, analyze your specific audience demographics and use TikTok Analytics to find your optimal posting times.",
            category="marketing_strategy",
            source="tiktok_analytics",
            confidence_score=0.92,
            tags=["posting_times", "marketing", "analytics"],
            created_at=datetime.now()
        ),
        TikTokLearningData(
            question="How do I optimize my TikTok Shop listings?",
            answer="Optimize TikTok Shop listings by using high-quality product images, compelling video content, relevant hashtags, competitive pricing, detailed product descriptions, and customer reviews. Focus on creating engaging content that showcases your products in action.",
            category="listing_optimization",
            source="tiktok_shop_guide",
            confidence_score=0.89,
            tags=["optimization", "listings", "content"],
            created_at=datetime.now()
        ),
        TikTokLearningData(
            question="What are the most profitable product categories on TikTok Shop?",
            answer="The most profitable categories on TikTok Shop include beauty and skincare, fashion accessories, home decor, fitness equipment, and tech gadgets. These categories have high engagement rates and strong conversion potential.",
            category="category_analysis",
            source="market_research",
            confidence_score=0.87,
            tags=["profitable_categories", "market_analysis"],
            created_at=datetime.now()
        ),
        TikTokLearningData(
            question="How do I handle customer service on TikTok Shop?",
            answer="Provide excellent customer service on TikTok Shop by responding quickly to messages, offering clear return policies, providing detailed product information, using video responses when possible, and maintaining a positive brand voice.",
            category="customer_service",
            source="customer_service_guide",
            confidence_score=0.94,
            tags=["customer_service", "support"],
            created_at=datetime.now()
        )
    ]
    
    return sample_data

def setup_tiktok_learning_system():
    """Setup the complete TikTok learning system"""
    
    # Create sample dataset
    logger.info("Creating sample TikTok learning dataset...")
    sample_data = create_sample_tiktok_dataset()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system = TikTokRAGSystem()
    
    # Add knowledge to RAG system
    knowledge_items = [
        {
            'content': item.answer,
            'metadata': {
                'category': item.category,
                'source': item.source,
                'confidence_score': item.confidence_score,
                'tags': item.tags
            }
        }
        for item in sample_data
    ]
    rag_system.add_knowledge(knowledge_items)
    
    # Initialize fine-tuner
    logger.info("Initializing fine-tuner...")
    fine_tuner = TikTokLearningFineTuner()
    
    # Prepare training data
    training_data = fine_tuner.prepare_training_data(sample_data)
    
    # Fine-tune model (commented out for demo - would take significant time)
    # output_dir = fine_tuner.fine_tune(training_data)
    
    logger.info("TikTok learning system setup complete!")
    
    return {
        'rag_system': rag_system,
        'fine_tuner': fine_tuner,
        'sample_data': sample_data
    }

if __name__ == "__main__":
    # Setup the TikTok learning system
    system = setup_tiktok_learning_system()
    
    # Test RAG system
    test_query = "How do I find trending products on TikTok Shop?"
    relevant_context = system['rag_system'].retrieve_relevant_context(test_query)
    
    print(f"Query: {test_query}")
    print(f"Retrieved {len(relevant_context)} relevant contexts:")
    for i, context in enumerate(relevant_context):
        print(f"{i+1}. Similarity: {context['similarity_score']:.3f}")
        print(f"   Content: {context['content'][:100]}...")
        print() 