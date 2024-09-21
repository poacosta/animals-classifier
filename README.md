# Animals Classifier

This is a simple image classifier that receives an image and if it is an animal, provides the name, description, and
determines if the animal is dangerous based on Wikipedia info.

![Captura de pantalla 2024-09-21 a las 08 43 02-fullpage](https://github.com/user-attachments/assets/5420b392-1cc8-42d7-81e0-e3716542b02d)

## Getting Started

### Prerequisites

You need to have the following installed on your machine:

- [Python 3.12](https://www.python.org/downloads/release/python-3124/) or later
- [Pip](https://pypi.org/project/pip/)
- [Virtualenv](https://pypi.org/project/virtualenv/)

## How to use it

### Running the script

```bash
# Clone the repository
git clone

# Change directory
cd animals-classifier

# Install virtualenv (in case you don't have it)
pip install virtualenv

# Create a virtual environment
python -m venv my_virtual_environment

# Activate the virtual environment
source my_virtual_environment/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

Open the browser and go to the address shown in the terminal: http://127.0.0.1:7860

**Important:** Make sure to set your OpenAI key before running: `export OPENAI_API_KEY-="sk-..."`

## Resources

- Images:
    - See some images for testing in the `images` folder.
    - Taken from [Unsplash](https://unsplash.com/), under Free License.

## Built with

- [Python](https://www.python.org/)
- [Gradio](https://www.gradio.app/)
- [Pillow](https://python-pillow.org/)
- [OpenAI](https://www.openai.com/)
- [LlamaIndex](https://llamaindex.ai/)

## Related Docs

- [VectorStoreIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/)
- [OpenAI Embeddings](https://docs.llamaindex.ai/en/stable/examples/embeddings/OpenAI/)
- [WikipediaReader](https://docs.llamaindex.ai/en/stable/api_reference/readers/wikipedia/)
