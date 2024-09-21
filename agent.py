from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader


class WikipediaQueryEngine:
    def __init__(self, llm_model="gpt-4o-mini", temperature=0, chunk_size=512):
        self.llm = OpenAI(temperature=temperature, model=llm_model)
        self.embed_model = OpenAIEmbedding()

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = chunk_size

        self.reader = WikipediaReader()
        self.index = None
        self.summary = None
        self.query_engine = None

    def load_wikipedia_page(self, page_title):
        """
        Loads a Wikipedia page and initializes the query engine.

        Args:
            page_title (str): The title of the Wikipedia page to load.

        This method performs the following steps:
        1. Load the Wikipedia page data using the WikipediaReader.
        2. Create a VectorStoreIndex from the loaded documents.
        3. Initializes the query engine from the index.
        4. Extracts and stores the summary of the Wikipedia page.

        Raises:
            ValueError: If the Wikipedia page cannot be loaded.
        """
        documents = self.reader.load_data(pages=[page_title])
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine()
        self.summary = documents[0].text.split("==")[0]

    def query(self, question):
        """
        Queries the loaded Wikipedia page using the initialized query engine.

        Args:
            question (str): The question to query against the Wikipedia page.

        Returns:
            str: The response from the query engine.

        Raises:
            ValueError: If the Wikipedia page is not loaded.
        """
        if not self.query_engine:
            raise ValueError("Wikipedia page not loaded. Call load_wikipedia_page() first.")
        response = self.query_engine.query(question)
        return response
