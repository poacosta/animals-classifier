import base64
import os
from datetime import datetime
from io import BytesIO

from PIL import Image
from openai import OpenAI

from agent import WikipediaQueryEngine

CLASSIFICATION_PROMPT = "What kind of animal is this? If it is an animal, respond with the name (for example, 'Cat') or 'Invalid' if not."
CLASSIFICATION_MODEL = "gpt-4o-mini"


class WikiFetcher:
    def __init__(self):
        self.wiki_engine = WikipediaQueryEngine()

    def load_wikipedia_page(self, animal_name):
        """
        Loads the Wikipedia page for a given animal.

        Args:
            animal_name (str): The name of the animal to load the Wikipedia page for.

        This method appends (animal) to the animal name to specify the Wikipedia page for the animal.
        """
        self.wiki_engine.load_wikipedia_page(animal_name + " (animal)")

    def get_description(self):
        """
        Retrieves the summary of the loaded Wikipedia page.

        Returns:
            str: The summary of the Wikipedia page.
        """
        return self.wiki_engine.summary

    def is_dangerous(self, animal_name):
        """
        Queries the Wikipedia page to determine if the specified animal is dangerous.

        Args:
            animal_name (str): The name of the animal to query.

        Returns:
            str: The response from the query engine indicating if the animal is dangerous.
        """
        return self.wiki_engine.query(
            f"Is a {animal_name} dangerous? Respond with 'yes' or 'no' with an explanation of the answer.")


class AnimalClassifier:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.wiki_fetcher = WikiFetcher()

    def encode_image(self, image):
        """
        Encodes a PIL Image object to a base64 string.

        Args:
            image (PIL.Image.Image): The image to encode.

        Returns:
            str: The base64 encoded string of the image.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def classify(self, image):
        """
        Classifies the given image to determine the type of animal.

        Args:
            image (numpy.ndarray): The image to classify.

        Returns:
            str: The classification result, including the animal name, description, and danger status.
        """
        if image is None:
            return "Please upload an image."

        base64_image = self.encode_image(Image.fromarray(image))

        response = self.client.chat.completions.create(
            model=CLASSIFICATION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": CLASSIFICATION_PROMPT
                        },
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

        if response.choices[0].message.content == "Invalid":
            return "## The image does not contain an animal."

        # OpenAI response
        animal_name = response.choices[0].message.content.split(" ")[-1]

        # Load Wikipedia page
        self.wiki_fetcher.load_wikipedia_page(animal_name)

        # Query Wikipedia content
        description = self.wiki_fetcher.get_description()
        is_dangerous = self.wiki_fetcher.is_dangerous(animal_name)
        reference = f"Wikipedia, The Free Encyclopedia. Retrieved {datetime.today().strftime("%B %d, %Y")}, from https://en.wikipedia.org"

        result = [f"# {animal_name}", "## Description", str(description), "## Dangerous?", str(is_dangerous),
                  "## References", f"* {reference}"]

        return "\n".join(result)
