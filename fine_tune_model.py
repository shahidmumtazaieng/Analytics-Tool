"""
Fine-tuning script for TikTok RAG Pipeline
This script fine-tunes a base model on TikTok e-commerce data for RAG applications.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TikTokFineTuner:
    """Fine-tuning class for TikTok RAG model"""
    
    def __init__(self, 
                 base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 output_dir: str = "./tiktok-fine-tuned-model"):
        
        self.base_model = base_model
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing fine-tuner with base model: {base_model}")
        logger.info(f"Using device: {self.device}")
    
    def load_and_prepare_data(self, data_path: str) -> Dataset:
        """Load and prepare training data"""
        
        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"Loaded {len(raw_data)} training examples")
        
        # Format data for instruction-following
        formatted_data = []
        for item in raw_data:
            formatted_item = self._format_training_example(item)
            if formatted_item:
                formatted_data.append(formatted_item)
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        logger.info(f"Prepared {len(formatted_data)} formatted examples")
        return dataset
    
    def _format_training_example(self, item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Format a single training example"""
        
        # Handle different data formats
        if 'instruction' in item and 'output' in item:
            # Alpaca format
            return {
                "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            }
        elif 'question' in item and 'answer' in item:
            # Q&A format
            context = item.get('context', '')
            context_text = f"\nContext: {context}" if context else ""
            return {
                "text": f"### Instruction:\n{item['question']}{context_text}\n\n### Response:\n{item['answer']}"
            }
        elif 'prompt' in item and 'completion' in item:
            # OpenAI format
            return {
                "text": f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
            }
        else:
            logger.warning(f"Skipping item with unknown format: {list(item.keys())}")
            return None
    
    def create_tiktok_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample TikTok e-commerce dataset for fine-tuning"""
        
        tiktok_data = [
            {
                "instruction": "What are the best practices for TikTok Shop product listings?",
                "output": "For TikTok Shop product listings, focus on high-quality videos (15-60 seconds), use trending hashtags, showcase products in action, include clear pricing, add compelling descriptions, and engage with comments quickly. Use TikTok's native features like duets and stitches to increase visibility."
            },
            {
                "instruction": "How do I identify trending products on TikTok?",
                "output": "To identify trending products on TikTok, monitor the 'For You' page, track hashtag performance, use TikTok's Creative Center analytics, follow trending sounds and challenges, analyze competitor content, and use tools like TikTok Ads Manager to see what's gaining traction."
            },
            {
                "instruction": "What's the optimal posting schedule for TikTok Shop?",
                "output": "The optimal posting schedule for TikTok Shop is 2-3 times per day, with peak times being 6-10 AM, 7-11 PM, and weekends. Post during your target audience's active hours, maintain consistency, and use TikTok's analytics to track when your audience is most engaged."
            },
            {
                "instruction": "How can I increase TikTok Shop conversion rates?",
                "output": "To increase TikTok Shop conversion rates, create authentic content, use user-generated content, offer limited-time promotions, provide detailed product information, use clear call-to-actions, optimize your bio and links, and engage with your audience through comments and live streams."
            },
            {
                "instruction": "What are the most profitable product categories on TikTok Shop?",
                "output": "The most profitable product categories on TikTok Shop include beauty and skincare, fashion and accessories, home and lifestyle, fitness and wellness, tech gadgets, and food and beverages. These categories have high engagement rates and strong purchasing intent among TikTok users."
            },
            {
                "instruction": "How do I handle customer service for TikTok Shop?",
                "output": "For TikTok Shop customer service, respond to comments within 2 hours, use TikTok's messaging features, provide clear return policies, offer multiple contact methods, train your team on TikTok-specific issues, and use automation tools for common questions while maintaining personal touch."
            },
            {
                "instruction": "What's the best way to price products on TikTok Shop?",
                "output": "Price products competitively by researching similar items, considering your target audience's budget, factoring in TikTok's commission (2-5%), offering bundle deals, using psychological pricing ($19.99 vs $20), and testing different price points to find optimal conversion rates."
            },
            {
                "instruction": "How can I use TikTok analytics to improve sales?",
                "output": "Use TikTok analytics to track video performance, monitor audience demographics, analyze engagement rates, identify top-performing content, track click-through rates to your shop, measure conversion rates, and optimize your content strategy based on data-driven insights."
            },
            {
                "instruction": "What are the key metrics for TikTok Shop success?",
                "output": "Key TikTok Shop metrics include video views, engagement rate, click-through rate to shop, conversion rate, average order value, customer acquisition cost, return on ad spend, follower growth rate, and customer lifetime value. Track these regularly to optimize performance."
            },
            {
                "instruction": "How do I create viral TikTok Shop content?",
                "output": "To create viral TikTok Shop content, follow trending formats, use popular sounds and effects, tell authentic stories, showcase products in creative ways, collaborate with influencers, participate in challenges, post consistently, and engage with your community through comments and duets."
            }
        ]
        
        return tiktok_data
    
    def save_sample_dataset(self, filename: str = "tiktok_training_data.json"):
        """Save sample dataset to file"""
        data = self.create_tiktok_dataset()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved sample dataset to {filename}")
        return filename
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer) -> Dataset:
        """Tokenize the dataset"""
        
        def tokenize_function(sample):
            return tokenizer(
                sample["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        logger.info("Dataset tokenization completed")
        return tokenized_dataset
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        
        logger.info(f"Loading tokenizer from {self.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model from {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        return model, tokenizer
    
    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration"""
        
        lora_config = LoraConfig(
            r=8,  # Rank of the LoRA layers
            lora_alpha=32,  # Scaling factor for the LoRA layers
            lora_dropout=0.1,  # Dropout rate for the LoRA layers
            bias="none",  # Bias handling
            task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        logger.info("LoRA configuration created")
        return lora_config
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments"""
        
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_steps=10,
            evaluation_strategy="epoch",
            learning_rate=2e-4,
            warmup_steps=100,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=4,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        logger.info("Training arguments configured")
        return args
    
    def fine_tune(self, data_path: Optional[str] = None):
        """Main fine-tuning function"""
        
        # Create sample dataset if no data path provided
        if data_path is None:
            data_path = self.save_sample_dataset()
        
        # Load and prepare data
        dataset = self.load_and_prepare_data(data_path)
        
        # Split into train and validation
        train_val_dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = train_val_dataset['train']
        eval_dataset = train_val_dataset['test']
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Tokenize datasets
        train_tokenized = self.tokenize_dataset(train_dataset, tokenizer)
        eval_tokenized = self.tokenize_dataset(eval_dataset, tokenizer)
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Start training
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Fine-tuning completed! Model saved to {self.output_dir}")
        
        return self.output_dir

def main():
    """Main function to run fine-tuning"""
    
    # Initialize fine-tuner
    fine_tuner = TikTokFineTuner(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir="./tiktok-fine-tuned-model"
    )
    
    # Run fine-tuning
    model_path = fine_tuner.fine_tune()
    
    print(f"\n✅ Fine-tuning completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    print(f"🚀 You can now use this model in your RAG pipeline!")

if __name__ == "__main__":
    main() 