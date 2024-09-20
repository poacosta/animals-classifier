import base64
import os
from io import BytesIO

import gradio as gr
import requests
from PIL import Image
from openai import OpenAI


class WikiFetcher:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"

    def get_info(self, animal_name):
        url = f"{self.base_url}?action=query&format=json&titles={animal_name}&prop=extracts&exintro&explaintext"
        response = requests.get(url)
        data = response.json()
        page = next(iter(data['query']['pages'].values()))
        description = page.get('extract', 'No description found.')
        return description


class AnimalClassifier:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.wiki_fetcher = WikiFetcher()

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def classify(self, image):
        if image is None:
            return "Please upload an image."

        base64_image = self.encode_image(Image.fromarray(image))

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "What kind of animal is this? Respond with the name of the animal. For example, 'cat'."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        animal_name = response.choices[0].message.content.split(" ")[-1]
        description = self.wiki_fetcher.get_info(animal_name)
        is_dangerous = any(keyword in description.lower() for keyword in ["danger", "attack", "kill"])

        return f"{response.choices[0].message.content}\n\nDescription: {description}\nIs Dangerous: {is_dangerous}"


def clear_result():
    output.value = None


classifier = AnimalClassifier()

with gr.Blocks() as demo:
    gr.Markdown("# Animals Classifier")
    gr.Markdown("Upload an image of an animal and the AI will try to classify it")
    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload an image", width=600, height=600)
        output = gr.TextArea(label="Result")
    classify_btn = gr.Button("Classify")
    classify_btn.click(fn=classifier.classify, inputs=image_input, outputs=output)
    image_input.clear(fn=clear_result, inputs=[], outputs=output)

demo.launch()
