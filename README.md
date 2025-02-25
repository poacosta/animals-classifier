# Animals Classifier 🐾

This project combines computer vision and natural language processing to identify animals from images, then enriches the identification with detailed information from Wikipedia. It's a practical exploration of modern AI capabilities wrapped in a simple, user-friendly interface.

![image](https://github.com/user-attachments/assets/a73c6591-6472-4fa1-a9fa-4419c4fc5f5f)

## Architecture Overview

The system operates through three core components:
- **Vision Recognition**: Uses OpenAI's multimodal capabilities to identify animals in uploaded images
- **Knowledge Retrieval**: Leverages LlamaIndex to fetch and process relevant Wikipedia articles
- **Interactive Interface**: Presents findings through a clean Gradio UI

## Getting Started

### Prerequisites

You'll need the following on your development machine:
- Python 3.12+ (earlier versions might work but weren't tested)
- pip (for package management)
- virtualenv (for isolated environments)

### Installation & Setup

```bash
# Clone this repository
git clone 

# Navigate to project directory
cd animals-classifier

# Set up virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."  # On Windows: set OPENAI_API_KEY=sk-...

# Launch the application
python main.py
```

Once running, open your browser and navigate to http://127.0.0.1:7860 to interact with the classifier.

## Technical Implementation

The project is structured around three main Python modules:
- `agent.py`: Implements the Wikipedia interaction layer using LlamaIndex
- `classifier.py`: Handles image processing and OpenAI API integration
- `main.py`: Provides the Gradio interface and application entry point

## Testing Resources

The `images` directory contains sample animal images from Unsplash (free license) for testing the classifier's capabilities.

## Tech Stack

The project leverages several powerful technologies:
- **OpenAI**: For multimodal vision and language capabilities
- **LlamaIndex**: For structured retrieval from Wikipedia
- **Gradio**: For rapid interface development
- **Pillow**: For image processing

## Future Improvements

Potential enhancements to consider:
- Caching previously processed animals to reduce API costs
- Expanding beyond Wikipedia to other knowledge sources
- Adding capability to compare multiple animals
- Implementing offline recognition for common species

## Documentation References

- [LlamaIndex VectorStoreIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/)
- [OpenAI Embeddings](https://docs.llamaindex.ai/en/stable/examples/embeddings/OpenAI/)
- [WikipediaReader](https://docs.llamaindex.ai/en/stable/api_reference/readers/wikipedia/)

---

This project represents a practical intersection of multiple AI capabilities - image recognition, knowledge retrieval, and natural language processing. While relatively simple in scope, it demonstrates how powerful AI interfaces can be created with surprisingly little code.
