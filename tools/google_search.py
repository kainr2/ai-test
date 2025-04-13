import requests

from langchain.agents import Tool, initialize_agent, AgentType


class GoogleSearch:

    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id

    # https://programmablesearchengine.google.com/about/
    def search(self, query: str) -> str:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extract the top few search snippets
        results = data.get("items", [])
        snippets = [item["snippet"] for item in results[:3]]
        return "\n".join(snippets) if snippets else "No results found."

    def create_tool(self) -> Tool:
        return Tool(
            name="GoogleSearch",
            func=self.search,
            description="Use this tool to search the web using Google Custom Search API"
        )
