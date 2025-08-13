"""
Complete Setup Script for Fine-tuned RAG Pipeline
This script handles the entire process from data preparation to model deployment.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FineTunedRAGSetup:
    """Complete setup for fine-tuned RAG system"""
    
    def __init__(self, 
                 base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 output_dir: str = "./tiktok-fine-tuned-model",
                 data_dir: str = "./data"):
        
        self.base_model = base_model
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        # Create necessary directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Fine-tuned RAG Setup initialized")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        
        required_packages = [
            'torch', 'transformers', 'datasets', 'peft', 
            'accelerate', 'sentence_transformers', 'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} is missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.info("Install missing packages with: pip install -r requirements-fine-tuning.txt")
            return False
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install fine-tuning dependencies"""
        
        try:
            logger.info("Installing fine-tuning dependencies...")
            
            # Install from requirements file
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-fine-tuning.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def create_training_data(self, filename: str = "tiktok_training_data.json") -> str:
        """Create comprehensive training dataset"""
        
        # Import the fine-tuning module
        from fine_tune_model import TikTokFineTuner
        
        fine_tuner = TikTokFineTuner()
        
        # Create and save dataset
        data_path = os.path.join(self.data_dir, filename)
        fine_tuner.save_sample_dataset(data_path)
        
        logger.info(f"✅ Training data created: {data_path}")
        return data_path
    
    def run_fine_tuning(self, data_path: Optional[str] = None) -> bool:
        """Run the fine-tuning process"""
        
        try:
            logger.info("🚀 Starting fine-tuning process...")
            
            # Import and run fine-tuning
            from fine_tune_model import TikTokFineTuner
            
            fine_tuner = TikTokFineTuner(
                base_model=self.base_model,
                output_dir=self.output_dir
            )
            
            # Run fine-tuning
            model_path = fine_tuner.fine_tune(data_path)
            
            logger.info(f"✅ Fine-tuning completed! Model saved to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            return False
    
    def test_fine_tuned_model(self) -> bool:
        """Test the fine-tuned model"""
        
        try:
            logger.info("🧪 Testing fine-tuned model...")
            
            # Import and test the model
            from fine_tuned_rag_inference import FineTunedRAGInference
            
            rag_system = FineTunedRAGInference(
                model_path=self.output_dir,
                base_model=self.base_model
            )
            
            # Run a quick test
            test_query = "What are the best practices for TikTok Shop?"
            result = rag_system.process_query(test_query)
            
            logger.info(f"✅ Test successful! Response: {result['response'][:100]}...")
            logger.info(f"⏱️  Response time: {result['processing_time']:.2f}s")
            logger.info(f"🔧 Model used: {result['model_used']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            return False
    
    def create_integration_script(self) -> str:
        """Create integration script for the main application"""
        
        integration_script = '''
"""
Integration script for fine-tuned RAG in main application
Add this to your main app.py or relevant module
"""

from fine_tuned_rag_inference import FineTunedRAGInference

# Initialize the fine-tuned RAG system
rag_system = FineTunedRAGInference(
    model_path="./tiktok-fine-tuned-model",
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Example usage in your API endpoint
async def chat_with_fine_tuned_rag(message: str, user_id: str):
    """Chat endpoint using fine-tuned RAG"""
    try:
        result = rag_system.process_query(message)
        
        return {
            "response": result["response"],
            "sources": result["context_sources"],
            "processing_time": result["processing_time"],
            "model_used": result["model_used"]
        }
    except Exception as e:
        return {"error": str(e)}

# Example FastAPI endpoint
@app.post("/chat/fine-tuned")
async def chat_endpoint(request: ChatRequest):
    return await chat_with_fine_tuned_rag(request.message, request.user_id)
'''
        
        script_path = os.path.join(self.output_dir, "integration_example.py")
        with open(script_path, 'w') as f:
            f.write(integration_script)
        
        logger.info(f"✅ Integration script created: {script_path}")
        return script_path
    
    def create_config_file(self) -> str:
        """Create configuration file for the fine-tuned model"""
        
        config = {
            "model_config": {
                "base_model": self.base_model,
                "fine_tuned_path": self.output_dir,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_length": 512,
                "temperature": 0.7,
                "top_k": 3
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 2,
                "learning_rate": 2e-4,
                "lora_r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            },
            "rag_config": {
                "knowledge_base_path": "./knowledge_base",
                "vector_db_type": "pinecone",  # or "local"
                "similarity_threshold": 0.7
            }
        }
        
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Configuration file created: {config_path}")
        return config_path
    
    def setup_complete_pipeline(self) -> bool:
        """Run the complete setup pipeline"""
        
        logger.info("🚀 Starting complete fine-tuned RAG setup...")
        
        # Step 1: Check dependencies
        logger.info("\n📋 Step 1: Checking dependencies...")
        if not self.check_dependencies():
            logger.info("Installing missing dependencies...")
            if not self.install_dependencies():
                return False
        
        # Step 2: Create training data
        logger.info("\n📊 Step 2: Creating training data...")
        data_path = self.create_training_data()
        
        # Step 3: Run fine-tuning
        logger.info("\n🎯 Step 3: Running fine-tuning...")
        if not self.run_fine_tuning(data_path):
            return False
        
        # Step 4: Test the model
        logger.info("\n🧪 Step 4: Testing fine-tuned model...")
        if not self.test_fine_tuned_model():
            return False
        
        # Step 5: Create integration files
        logger.info("\n🔧 Step 5: Creating integration files...")
        self.create_integration_script()
        self.create_config_file()
        
        logger.info("\n🎉 Fine-tuned RAG setup completed successfully!")
        logger.info(f"📁 Model location: {self.output_dir}")
        logger.info("🚀 You can now integrate the fine-tuned model into your main application!")
        
        return True

def main():
    """Main setup function"""
    
    print("🤖 Fine-tuned RAG Pipeline Setup")
    print("=" * 50)
    
    # Initialize setup
    setup = FineTunedRAGSetup()
    
    # Run complete setup
    success = setup.setup_complete_pipeline()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\n📚 Next steps:")
        print("1. Review the generated files in the output directory")
        print("2. Integrate the fine-tuned model into your main application")
        print("3. Test the RAG pipeline with your own queries")
        print("4. Customize the training data for your specific use case")
    else:
        print("\n❌ Setup failed. Please check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 