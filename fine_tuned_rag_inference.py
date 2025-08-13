"""
Fine-tuned RAG Inference Pipeline
This script loads the fine-tuned model and integrates it with the RAG system.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTunedRAGInference:
    """Fine-tuned RAG inference system"""
    
    def __init__(self, 
                 model_path: str = "./tiktok-fine-tuned-model",
                 base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.model_path = model_path
        self.base_model = base_model
        self.embedding_model = embedding_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Sample knowledge base (in production, this would come from a vector database)
        self.knowledge_base = self._load_sample_knowledge_base()
        
        logger.info("Fine-tuned RAG inference system initialized successfully")
    
    def _load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer with fallback"""
        
        try:
            # Check if fine-tuned model exists
            if os.path.exists(self.model_path):
                # Find checkpoint directories
                checkpoints = [d for d in os.listdir(self.model_path) 
                             if d.startswith("checkpoint-") and 
                             os.path.isdir(os.path.join(self.model_path, d))]
                
                if checkpoints:
                    # Get the latest checkpoint by number
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                    model_name = os.path.join(self.model_path, latest_checkpoint)
                    logger.info(f"Found checkpoints: {checkpoints}")
                    logger.info(f"Loading fine-tuned model from {model_name}")
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    # Load from the main directory
                    logger.info(f"Loading fine-tuned model from {self.model_path}")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                raise FileNotFoundError(f"Fine-tuned model directory not found at {self.model_path}")
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}")
            logger.info(f"Falling back to base model: {self.base_model}")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _load_sample_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load sample knowledge base for demonstration"""
        
        knowledge_base = [
            {
                "content": "TikTok Shop best practices include creating high-quality videos (15-60 seconds), using trending hashtags, showcasing products in action, including clear pricing, and engaging with comments quickly.",
                "metadata": {"category": "best_practices", "source": "tiktok_guide"},
                "embedding": None
            },
            {
                "content": "To identify trending products on TikTok, monitor the 'For You' page, track hashtag performance, use TikTok's Creative Center analytics, and follow trending sounds and challenges.",
                "metadata": {"category": "trending", "source": "tiktok_analytics"},
                "embedding": None
            },
            {
                "content": "The optimal posting schedule for TikTok Shop is 2-3 times per day, with peak times being 6-10 AM, 7-11 PM, and weekends.",
                "metadata": {"category": "scheduling", "source": "tiktok_timing"},
                "embedding": None
            },
            {
                "content": "To increase TikTok Shop conversion rates, create authentic content, use user-generated content, offer limited-time promotions, and provide detailed product information.",
                "metadata": {"category": "conversion", "source": "tiktok_optimization"},
                "embedding": None
            },
            {
                "content": "The most profitable product categories on TikTok Shop include beauty and skincare, fashion and accessories, home and lifestyle, fitness and wellness, and tech gadgets.",
                "metadata": {"category": "categories", "source": "tiktok_research"},
                "embedding": None
            }
        ]
        
        # Compute embeddings for knowledge base
        for item in knowledge_base:
            item["embedding"] = self.embedding_model.encode(item["content"]).tolist()
        
        return knowledge_base
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context from knowledge base"""
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for item in self.knowledge_base:
            similarity = np.dot(query_embedding, item["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
            )
            similarities.append((similarity, item))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        relevant_context = [item for _, item in similarities[:top_k]]
        
        return relevant_context
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using fine-tuned model with RAG context"""
        
        # Prepare context string
        context_text = "\n".join([item["content"] for item in context])
        
        # Create prompt with context
        prompt = f"### Instruction:\nBased on the following context, answer this question: {query}\n\nContext: {context_text}\n\n### Response:"
        
        # Generate response
        start_time = time.time()
        result = self.generator(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        generation_time = time.time() - start_time
        
        # Extract response
        generated_text = result[0]["generated_text"]
        response = generated_text.split("### Response:")[-1].strip()
        
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        
        return response
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        
        start_time = time.time()
        
        # Retrieve relevant context
        context = self.retrieve_relevant_context(query)
        
        # Generate response
        response = self.generate_response(query, context)
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "context_sources": [item["metadata"]["source"] for item in context],
            "context_categories": [item["metadata"]["category"] for item in context],
            "processing_time": total_time,
            "model_used": "fine-tuned" if "fine-tuned" in str(self.model_path) else "base"
        }
    
    def test_model(self):
        """Test the fine-tuned model with sample queries"""
        
        test_queries = [
            "What are the best practices for TikTok Shop?",
            "How do I find trending products on TikTok?",
            "What's the best time to post on TikTok Shop?",
            "How can I increase my conversion rates?",
            "Which product categories are most profitable?"
        ]
        
        print("🧪 Testing Fine-tuned RAG Model")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            result = self.process_query(query)
            print(f"🤖 Response: {result['response']}")
            print(f"📚 Sources: {result['context_sources']}")
            print(f"⏱️  Time: {result['processing_time']:.2f}s")
            print(f"🔧 Model: {result['model_used']}")
            print("-" * 50)

def main():
    """Main function to test the fine-tuned RAG system"""
    
    # Initialize the fine-tuned RAG system
    rag_system = FineTunedRAGInference(
        model_path="./tiktok-fine-tuned-model",
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    # Test the system
    rag_system.test_model()
    
    # Interactive mode
    print("\n🎯 Interactive Mode - Ask questions about TikTok Shop!")
    print("Type 'quit' to exit")
    
    while True:
        try:
            query = input("\n💬 Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                result = rag_system.process_query(query)
                print(f"\n🤖 Answer: {result['response']}")
                print(f"📚 Sources: {', '.join(result['context_sources'])}")
                print(f"⏱️  Response time: {result['processing_time']:.2f}s")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for using the Fine-tuned RAG System!")

if __name__ == "__main__":
    main() 