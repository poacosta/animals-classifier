# Animals Classifier

This is a simple image classifier that receives an image and if it is an animal, provides the name, description, and
determines if the animal is dangerous based on Wikipedia info.

## Getting Started

### Prerequisites

You need to have the following installed on your machine:

- [Python 3.12](https://www.python.org/downloads/release/python-3124/) or later
- [Pip](https://pypi.org/project/pip/)
- [Virtualenv](https://pypi.org/project/virtualenv/)

## How to use it

### Running the Agent

```bash
# Clone the repository
git clone

# Change directory
cd animals-classifier

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Run the agent
python main.py
```

Make sure to set your OpenAI key before running: `export OPENAI_API_KEY-="sk-..."`

## Resources

- Images: [Unsplash](https://unsplash.com/)
  - See some images for testing in the `images` folder

## Built With

- [Python](https://www.python.org/)
- [Gradio](https://www.gradio.app/)
- [Pillow](https://python-pillow.org/)
- [OpenAI](https://www.openai.com/)
