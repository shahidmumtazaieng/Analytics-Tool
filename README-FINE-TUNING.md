# Fine-tuned RAG Pipeline for TikTok E-commerce

This directory contains a complete fine-tuning system for creating a custom RAG (Retrieval-Augmented Generation) model specifically optimized for TikTok e-commerce data.

## 🎯 Overview

The fine-tuning system allows you to:
- **Fine-tune a base model** (TinyLlama) on TikTok e-commerce data
- **Integrate with RAG pipeline** for enhanced responses
- **Customize training data** for your specific use case
- **Deploy the model** in your main application

## 📁 File Structure

```
Backend/
├── fine_tune_model.py              # Main fine-tuning script
├── fine_tuned_rag_inference.py     # Inference with RAG integration
├── setup_fine_tuned_rag.py         # Complete setup automation
├── requirements-fine-tuning.txt     # Fine-tuning dependencies
├── README-FINE-TUNING.md           # This file
└── data/                           # Training data directory
    └── tiktok_training_data.json   # Sample training dataset
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Backend
pip install -r requirements-fine-tuning.txt
```

### 2. Run Complete Setup

```bash
python setup_fine_tuned_rag.py
```

This will:
- ✅ Check and install dependencies
- 📊 Create training dataset
- 🎯 Run fine-tuning process
- 🧪 Test the fine-tuned model
- 🔧 Generate integration files

### 3. Test the Model

```bash
python fine_tuned_rag_inference.py
```

## 📚 Detailed Usage

### Fine-tuning Process

#### Step 1: Prepare Training Data

The system includes a sample TikTok e-commerce dataset with 10 high-quality Q&A pairs:

```python
from fine_tune_model import TikTokFineTuner

# Create sample dataset
fine_tuner = TikTokFineTuner()
data_path = fine_tuner.save_sample_dataset("my_tiktok_data.json")
```

#### Step 2: Customize Training Data

You can create your own training data in the following formats:

**Alpaca Format:**
```json
{
  "instruction": "What are the best practices for TikTok Shop?",
  "output": "For TikTok Shop, focus on high-quality videos..."
}
```

**Q&A Format:**
```json
{
  "question": "How do I increase conversion rates?",
  "answer": "To increase conversion rates, create authentic content...",
  "context": "Additional context if available"
}
```

**OpenAI Format:**
```json
{
  "prompt": "What are trending products on TikTok?",
  "completion": "To find trending products, monitor the For You page..."
}
```

#### Step 3: Run Fine-tuning

```python
from fine_tune_model import TikTokFineTuner

fine_tuner = TikTokFineTuner(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir="./my-fine-tuned-model"
)

# Run fine-tuning
model_path = fine_tuner.fine_tune("my_tiktok_data.json")
```

### RAG Integration

#### Load Fine-tuned Model

```python
from fine_tuned_rag_inference import FineTunedRAGInference

# Initialize RAG system with fine-tuned model
rag_system = FineTunedRAGInference(
    model_path="./tiktok-fine-tuned-model",
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
```

#### Process Queries

```python
# Single query
result = rag_system.process_query("What are the best practices for TikTok Shop?")

print(f"Response: {result['response']}")
print(f"Sources: {result['context_sources']}")
print(f"Processing time: {result['processing_time']:.2f}s")
```

#### Interactive Mode

```python
# Start interactive chat
rag_system.test_model()
```

## ⚙️ Configuration

### Model Configuration

The system supports various configuration options:

```python
# Fine-tuning configuration
fine_tuner = TikTokFineTuner(
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Base model
    output_dir="./tiktok-fine-tuned-model"            # Output directory
)

# RAG configuration
rag_system = FineTunedRAGInference(
    model_path="./tiktok-fine-tuned-model",           # Fine-tuned model path
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Fallback model
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
)
```

### Training Parameters

You can customize training parameters in `fine_tune_model.py`:

```python
# LoRA configuration
lora_config = LoraConfig(
    r=8,                    # Rank of LoRA layers
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.1,       # Dropout rate
    bias="none",            # Bias handling
    task_type=TaskType.CAUSAL_LM
)

# Training arguments
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available()
)
```

## 🔧 Integration with Main Application

### FastAPI Integration

Add this to your main `app.py`:

```python
from fine_tuned_rag_inference import FineTunedRAGInference

# Initialize RAG system
rag_system = FineTunedRAGInference()

@app.post("/chat/fine-tuned")
async def chat_with_fine_tuned_rag(request: ChatRequest):
    """Chat endpoint using fine-tuned RAG"""
    try:
        result = rag_system.process_query(request.message)
        
        return {
            "response": result["response"],
            "sources": result["context_sources"],
            "processing_time": result["processing_time"],
            "model_used": result["model_used"]
        }
    except Exception as e:
        return {"error": str(e)}
```

### Frontend Integration

Update your frontend to use the fine-tuned endpoint:

```javascript
// Example frontend integration
async function chatWithFineTunedRAG(message) {
    const response = await fetch('/chat/fine-tuned', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, user_id: 'user123' })
    });
    
    return await response.json();
}
```

## 📊 Performance Optimization

### GPU Acceleration

The system automatically detects and uses GPU if available:

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Memory Optimization

For limited memory environments:

```python
# Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    device_map="auto"
)
```

### Batch Processing

For multiple queries:

```python
# Process multiple queries efficiently
queries = ["Query 1", "Query 2", "Query 3"]
results = [rag_system.process_query(q) for q in queries]
```

## 🧪 Testing and Evaluation

### Automated Testing

```python
# Run comprehensive tests
rag_system.test_model()
```

### Custom Evaluation

```python
# Evaluate on custom test set
test_queries = [
    "What are TikTok Shop best practices?",
    "How to increase conversion rates?",
    "Best time to post on TikTok?"
]

for query in test_queries:
    result = rag_system.process_query(query)
    print(f"Query: {query}")
    print(f"Response: {result['response']}")
    print(f"Time: {result['processing_time']:.2f}s")
    print("-" * 50)
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   per_device_train_batch_size=1
   ```

2. **Model Loading Errors**
   ```python
   # Check model path
   if os.path.exists(model_path):
       print("Model found")
   else:
       print("Model not found")
   ```

3. **Dependencies Missing**
   ```bash
   pip install -r requirements-fine-tuning.txt
   ```

### Performance Issues

1. **Slow Inference**
   - Use GPU acceleration
   - Reduce model size
   - Optimize batch processing

2. **Poor Response Quality**
   - Increase training epochs
   - Improve training data quality
   - Adjust temperature and top_k parameters

## 📈 Advanced Features

### Custom Knowledge Base

```python
# Add custom knowledge to RAG system
custom_knowledge = [
    {
        "content": "Your custom content here",
        "metadata": {"category": "custom", "source": "your_source"},
        "embedding": None  # Will be computed automatically
    }
]

rag_system.knowledge_base.extend(custom_knowledge)
```

### Model Comparison

```python
# Compare fine-tuned vs base model
base_result = base_rag_system.process_query(query)
fine_tuned_result = fine_tuned_rag_system.process_query(query)

print(f"Base model response: {base_result['response']}")
print(f"Fine-tuned response: {fine_tuned_result['response']}")
```

## 🚀 Deployment

### Production Deployment

1. **Model Optimization**
   ```python
   # Quantize model for production
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **Docker Deployment**
   ```dockerfile
   FROM python:3.9
   COPY requirements-fine-tuning.txt .
   RUN pip install -r requirements-fine-tuning.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

3. **Cloud Deployment**
   - AWS SageMaker
   - Google Cloud AI Platform
   - Azure Machine Learning

## 📚 Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Sentence Transformers](https://www.sbert.net/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. 